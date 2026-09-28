"""Correctness check on the real checkpoint: greedy (temperature=0) outputs of
dspark speculative decoding must exactly match plain autoregressive decoding,
and the decoded text must be coherent (spot-checked by eye in the log)."""
import json
import os
import sys

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
from generate import generate, generate_spec
from eval_spec import load_prompts


def reset(model):
    for mod in model.modules():
        if hasattr(mod, "kv_cache") and isinstance(mod.kv_cache, torch.Tensor):
            mod.kv_cache.zero_()
        if hasattr(mod, "kv_state") and isinstance(mod.kv_state, torch.Tensor):
            mod.kv_state.zero_()
        if hasattr(mod, "score_state") and isinstance(mod.score_state, torch.Tensor):
            mod.score_state.fill_(float("-inf"))
        if hasattr(mod, "_snapshots"):
            mod._snapshots = None


def main():
    ckpt = sys.argv[1]
    config = sys.argv[2]
    n_prompts = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 64

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
    with open(config) as f:
        margs = ModelArgs(**json.load(f))
        margs.temperature = 0.0  # greedy: spec output must equal plain output exactly
        margs.max_batch_size = 1
    with torch.device("cuda"):
        model = Transformer(margs)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    load_model(model, os.path.join(ckpt, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")

    prompts = load_prompts(1000, 1234)[:n_prompts]
    n_match = 0
    for i, p in enumerate(prompts):
        ptoks = tokenizer.encode(encode_messages([{"role": "user", "content": p["text"]}], thinking_mode="chat"))
        if not (2 <= len(ptoks) <= 512):
            continue
        reset(model)
        out_plain = generate(model, [ptoks], max_new, tokenizer.eos_token_id)[0]
        reset(model)
        out_spec = generate_spec(model, [ptoks], max_new, tokenizer.eos_token_id)[0]
        ok = out_plain == out_spec
        n_match += ok
        text = tokenizer.decode(out_spec).replace("<｜end▁of▁sentence｜>", "")
        status = "MATCH" if ok else "MISMATCH"
        print(f"[{i}] {p['lang']} {status} len={len(out_spec)}")
        if not ok:
            for k, (a, b) in enumerate(zip(out_plain, out_spec)):
                if a != b:
                    print(f"    first diff at {k}: plain={a} spec={b} (lens {len(out_plain)}/{len(out_spec)})")
                    break
        print(f"    text: {text[:160]}")
    print(f"CORRECTNESS: {n_match}/{n_prompts} prompts exact match")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
