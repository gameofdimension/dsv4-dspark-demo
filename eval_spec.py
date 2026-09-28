"""Batch evaluation of dspark speculative decoding on an instruction dataset.

Samples N prompts (English alpaca + Chinese Wizard-LM evol, half and half by
default), runs generate_spec on each, and aggregates acceptance statistics:
mean acceptance length, per-position acceptance rates, avg draft acceptance
rate, commit-length distribution and throughput.

Results are written incrementally (one JSON line per prompt + a rolling
aggregate JSON), so partial runs still give usable numbers.

Usage:
  torchrun --nproc-per-node 8 eval_spec.py --ckpt-path ... --config config-fp8.json \
      --n 1000 --max-new-tokens 48 --out eval_results
"""
import json
import os
import random
import sys
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
encoding_dir = os.path.join(current_dir, 'encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages

from model import Transformer, ModelArgs
from generate import generate_spec, SpecMetrics

ALPACA_PARQUET = "/tmp/alpaca.parquet"
WIZARD_JSONL = "/tmp/wizard_zh.jsonl"
MAX_PROMPT_TOKENS = 512


def load_prompts(n: int, seed: int, alpaca_path: str = None, wizard_path: str = None):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    half = n // 2
    prompts = []
    alpaca = pq.read_table(alpaca_path or ALPACA_PARQUET).to_pylist()
    for r in rng.sample(alpaca, half):
        text = r["instruction"] + (f"\n{r['input']}" if r["input"] else "")
        prompts.append({"lang": "en", "text": text})
    with open(wizard_path or WIZARD_JSONL) as f:
        wizard = [json.loads(line) for line in f]
    for r in rng.sample(wizard, n - half):
        prompts.append({"lang": "zh", "text": r["instruction_zh"]})
    rng.shuffle(prompts)
    return prompts


def counters(m: SpecMetrics):
    return (m.steps, m.accepted, m.drafted, m.committed, list(m.per_pos_accepted))


def aggregate(m: SpecMetrics, elapsed: float, prompts_done: int, gen_tokens: int):
    rates = [c / max(m.steps, 1) for c in m.per_pos_accepted]
    return {
        "prompts_done": prompts_done,
        "steps": m.steps,
        "drafted": m.drafted,
        "accepted": m.accepted,
        "committed": m.committed,
        "gen_tokens": gen_tokens,
        "mean_acceptance_length": m.committed / max(m.steps, 1),
        "mean_accepted_drafts": m.accepted / max(m.steps, 1),
        "avg_draft_acceptance_rate": m.accepted / max(m.drafted, 1),
        "per_position_acceptance_rate": [round(r, 4) for r in rates],
        "commit_length_hist": m.commit_hist,
        "generated_throughput_tps": gen_tokens / max(elapsed, 1e-9),
        "wall_seconds": elapsed,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="eval_results")
    parser.add_argument("--start", type=int, default=0, help="resume: skip first N prompts")
    parser.add_argument("--alpaca", type=str, default="", help="path to alpaca parquet (default /tmp/alpaca.parquet)")
    parser.add_argument("--wizard", type=str, default="", help="path to wizard-lm zh jsonl (default /tmp/wizard_zh.jsonl)")
    args = parser.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with open(args.config) as f:
        margs = ModelArgs(**json.load(f))
        margs.temperature = args.temperature
        margs.max_batch_size = 1
    with torch.device("cuda"):
        model = Transformer(margs)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    n_spec = model.mtp[-1].block_size
    print(f"eval: {args.num_prompts} prompts, max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, n_spec={n_spec}")

    prompts = load_prompts(args.num_prompts, args.seed, args.alpaca or None, args.wizard or None)
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "per_prompt.jsonl")
    agg_path = os.path.join(args.out, "aggregate.json")

    metrics = SpecMetrics(n_spec)
    gen_tokens = 0
    t_start = time.time()
    skip = args.start
    for i, p in enumerate(prompts):
        prompt_tokens = tokenizer.encode(encode_messages([{"role": "user", "content": p["text"]}], thinking_mode="chat"))
        if not (2 <= len(prompt_tokens) <= MAX_PROMPT_TOKENS):
            continue
        if skip > 0:
            skip -= 1
            continue
        before = counters(metrics)
        t0 = time.time()
        error = None
        try:
            completion = generate_spec(model, [prompt_tokens], args.max_new_tokens, tokenizer.eos_token_id, metrics=metrics)
        except Exception as e:  # noqa: BLE001 - keep the batch eval alive
            error = repr(e)
            completion = None
        dt = time.time() - t0
        steps, accepted, drafted, committed, _ = counters(metrics)
        d_steps, d_acc, d_dra, d_com = steps - before[0], accepted - before[1], drafted - before[2], committed - before[3]
        n_gen = len(completion[0]) - 1 if completion else 0
        gen_tokens += n_gen
        text = ""
        if completion is not None:
            text = tokenizer.decode(completion[0]).replace("<｜end▁of▁sentence｜>", "").strip()
        rec = {
            "i": i, "lang": p["lang"], "prompt_len": len(prompt_tokens), "gen_tokens": n_gen,
            "steps": d_steps, "drafted": d_dra, "accepted": d_acc, "committed": d_com,
            "seconds": round(dt, 3), "text": text[:240], "error": error,
        }
        if rank == 0:
            with open(results_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done = i - args.start + 1
            if done % 25 == 0 or done == 1:
                elapsed = time.time() - t_start
                print(f"[{done}/{len(prompts) - args.start}] {elapsed:.0f}s elapsed, ETA {(len(prompts) - args.start - done) * (elapsed / done):.0f}s")
                with open(agg_path, "w") as f:
                    json.dump(aggregate(metrics, elapsed, done, gen_tokens), f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    report = aggregate(metrics, elapsed, len(prompts) - args.start, gen_tokens)
    if rank == 0:
        with open(agg_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
