import os
import json
import time
from argparse import ArgumentParser
from typing import List, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs, sample
from encoding.encoding_dsv4 import encode_messages, parse_message_from_completion_text


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
) -> List[List[int]]:
    """Batch generation with left-padded prompts.

    The first forward pass processes [min_prompt_len:] tokens (prefill phase).
    Subsequent passes generate one token at a time (decode phase). For positions
    still within a prompt, the ground-truth token overrides the model's prediction.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)[0]
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


class SpecMetrics:
    """DSpark speculation statistics, printed in the same format as vLLM's
    'SpecDecoding metrics' log line."""

    def __init__(self, num_spec_tokens: int):
        self.num_spec_tokens = num_spec_tokens
        self.steps = 0
        self.accepted = 0
        self.drafted = 0
        self.committed = 0
        self.per_pos_accepted = [0] * num_spec_tokens
        self.commit_hist = [0] * (num_spec_tokens + 2)  # committed per step, 0..N+1

    def update(self, num_accepted: int, num_committed: int):
        self.steps += 1
        self.accepted += num_accepted
        self.drafted += self.num_spec_tokens
        self.committed += num_committed
        for i in range(min(num_accepted, self.num_spec_tokens)):
            self.per_pos_accepted[i] += 1
        self.commit_hist[min(num_committed, self.num_spec_tokens + 1)] += 1

    def summary(self, elapsed: float) -> str:
        mean_len = self.committed / max(self.steps, 1)
        rates = " ".join(f"{c / max(self.steps, 1):.3f}" for c in self.per_pos_accepted)
        return (
            f"SpecDecoding metrics: Mean acceptance length: {mean_len:.2f}, "
            f"Generated throughput: {self.committed / max(elapsed, 1e-9):.2f} tokens/s, "
            f"Accepted: {self.accepted}, Drafted: {self.drafted}, "
            f"Per-position acceptance rate: {rates}, "
            f"Avg Draft acceptance rate: {100.0 * self.accepted / max(self.drafted, 1):.1f}%"
        )


def verify_and_sample(
    target_logits: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Probabilistic rejection sampling (Leviathan et al., 2023), mirroring vLLM's
    dspark verifier with draft_sample_method="probabilistic".

    target_logits: [b, N+1, V] target logits over [anchor, d1..dN]
    draft_ids:     [b, N+1] (anchor, d1..dN)
    draft_logits:  [b, N, V] the draft distributions that produced d1..dN
    Returns (new_tokens [b, m], num_accepted [b]) where each row holds its accepted
    drafts followed by one resampled (rejection) or bonus (all accepted) token;
    m = max(num_accepted) + 1.
    """
    bsz, num_logits, _ = target_logits.shape
    n_spec = num_logits - 1
    draft = draft_ids[:, 1:]
    if temperature == 0:
        target_tokens = target_logits.argmax(dim=-1)
        accepted = draft == target_tokens[:, :n_spec]
        n_acc = accepted.int().cumprod(dim=1).sum(dim=1)
        new_tokens = draft.new_empty(bsz, int(n_acc.max()) + 1)
        for r in range(bsz):
            i = int(n_acc[r])
            new_tokens[r, :i] = draft[r, :i]
            new_tokens[r, i] = target_tokens[r, i]
        return new_tokens, n_acc
    scale = max(temperature, 1e-5)
    target = target_logits / scale
    draft_l = draft_logits.float() / scale
    p = target[:, :n_spec].gather(-1, draft.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(target[:, :n_spec], dim=-1)
    q = draft_l.gather(-1, draft.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(draft_l, dim=-1)
    # float32 explicitly: the process default dtype is bf16, too coarse for the
    # uniform draw behind the probability ratio test.
    u = torch.rand(bsz, n_spec, device=target_logits.device, dtype=torch.float32).clamp_min_(1e-20)
    accepted = p > u.log() + q
    n_acc = accepted.int().cumprod(dim=1).sum(dim=1)
    new_tokens = draft.new_empty(bsz, int(n_acc.max()) + 1)
    for r in range(bsz):
        i = int(n_acc[r])
        new_tokens[r, :i] = draft[r, :i]
        if i == n_spec:
            # All drafts accepted: one bonus token from the target's last position.
            new_tokens[r, i] = sample(target[r, -1].unsqueeze(0), temperature)[0]
        else:
            # Rejected at position i: resample from the residual norm((p - q)+).
            residual = (target[r, i].softmax(dim=-1) - draft_l[r, i].softmax(dim=-1)).clamp_min_(0)
            new_tokens[r, i] = residual.div_(torch.empty_like(residual).exponential_(1)).argmax()
    return new_tokens, n_acc


@torch.inference_mode()
def generate_spec(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    metrics: "SpecMetrics" = None,
) -> List[List[int]]:
    """DSpark speculative decoding (batch size 1), mirroring vLLM's dspark method.

    Each step the draft model proposes num_spec_tokens tokens in one parallel pass
    (backbone forward + sequential Markov sampling); the target model then verifies
    them in a single forward over [anchor + drafts]. Rejection sampling keeps the
    output distribution identical to plain autoregressive decoding. Rejected draft
    suffixes are rolled back (compressor state snapshots; stale KV entries are
    position-indexed and rewritten by the next block).
    """
    assert len(prompt_tokens) == 1, "dspark speculative decoding supports batch size 1"
    prompt_len = len(prompt_tokens[0])
    assert prompt_len >= 2, "dspark speculative decoding needs a prompt of at least 2 tokens"
    assert prompt_len <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    n_spec = model.mtp[-1].block_size
    temperature = model.temperature
    total_len = min(model.max_seq_len, max_new_tokens + prompt_len)
    tokens = torch.zeros(1, total_len, dtype=torch.long)
    tokens[0, :prompt_len] = torch.tensor(prompt_tokens[0], dtype=torch.long)

    # Prefill: target forward over the prompt (first sampled token becomes the
    # anchor), then build the DSpark layers' context KV from the prompt hiddens.
    anchor, _, main_hidden = model.forward(tokens[:, :prompt_len], 0)
    anchor = anchor.unsqueeze(1)
    tokens[0, prompt_len] = anchor[0, 0]
    model.forward_spec(anchor, main_hidden, 0)
    pos = prompt_len  # absolute position of the anchor token
    ctx_hidden = main_hidden[:, -1:]  # hiddens inserted into the draft KV so far (re-inserting the last prompt position is idempotent)
    ctx_start = prompt_len - 1
    generated = 0
    finished = False
    own_metrics = metrics is None
    if own_metrics:
        metrics = SpecMetrics(n_spec)
    t_start = time.time()

    while generated < max_new_tokens and not finished:
        if pos + n_spec + 1 > total_len:
            break  # too close to the end for a full draft block; finish with plain decoding
        # 1) Draft: propose n_spec tokens after the anchor (block occupies the
        #    anchor position onward in the draft's frame; the DSpark context KV is
        #    extended with everything committed before the anchor).
        draft_ids, draft_logits, _ = model.forward_spec(anchor, ctx_hidden, ctx_start)
        # 2) Verify: one target forward over [anchor, d1..dN].
        block = torch.cat([anchor, draft_ids[:, 1:]], dim=1)
        _, target_logits, target_hidden = model.forward(block, pos, full_logits=True)
        # 3) Rejection sampling; roll the compressor state back past rejected drafts.
        new_tokens, n_acc = verify_and_sample(target_logits, draft_ids, draft_logits, temperature)
        i = int(n_acc[0])
        if i < n_spec:
            model.spec_rollback(i)
        m = min(i + 1, max_new_tokens - generated, total_len - 1 - pos)
        tokens[0, pos + 1: pos + 1 + m] = new_tokens[0, :m]
        metrics.update(i, m)
        generated += m
        finished = bool((tokens[0, pos + 1: pos + 1 + m] == eos_id).any())
        # Next anchor = last committed token; its hidden does not exist yet (it is
        # computed by the next target forward), so the draft context covers all
        # committed positions strictly before it.
        ctx_hidden = target_hidden[:, :m]
        ctx_start = pos
        pos += m
        anchor = tokens[:, pos:pos + 1].clone()

    # Plain single-token decoding tail (sequence-end boundary or loop exit above).
    while generated < max_new_tokens and not finished and pos + 1 < total_len:
        next_token, _, _ = model.forward(tokens[:, pos:pos + 1], pos)
        tokens[0, pos + 1] = next_token[0]
        pos += 1
        generated += 1
        finished = next_token[0].item() == eos_id

    if own_metrics and metrics.steps:
        print(metrics.summary(time.time() - t_start))
    completion_tokens = []
    toks = tokens[0, prompt_len:prompt_len + max_new_tokens].tolist()
    if eos_id in toks:
        toks = toks[:toks.index(eos_id)]
    toks.append(eos_id)
    completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    spec: bool = False,
) -> None:
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
        args = ModelArgs(**json.load(f))
        args.temperature = temperature
    if interactive:
        args.max_batch_size = 1
        args.max_seq_len = 64 * 1024
    if spec:
        assert args.dspark_block_size > 0 and len(args.dspark_target_layer_ids) > 0, \
            "speculative decoding needs a dspark checkpoint (dspark_block_size / dspark_target_layer_ids in config)"
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            if spec:
                completion_tokens = generate_spec(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id)
            else:
                completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        if spec:
            completion_tokens = []
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat"))
                completion_tokens.extend(generate_spec(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id))
        else:
            prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
            completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--spec", action="store_true",
                        help="enable dspark speculative decoding (draft_sample_method=probabilistic; batch size 1, prompts are processed one by one)")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.spec)
