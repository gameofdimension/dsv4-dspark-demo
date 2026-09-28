"""Smoke test for dspark speculative decoding in flash-0731-demo.

1) Greedy (temperature=0): generate_spec output must exactly match plain generate
   (rejection sampling with greedy drafts is exact), which validates the whole
   multi-token decode path (window/compression/indexer masks, KV writes,
   compressor rollback, draft context insert).
2) Probabilistic (temperature=1): just runs and prints metrics.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(0)

from model import ModelArgs, Transformer
from generate import generate, generate_spec, verify_and_sample

N_LAYERS, DIM, VOCAB = 6, 1024, 8192
args = ModelArgs(
    n_layers=N_LAYERS, n_hash_layers=0, n_mtp_layers=2, dspark_block_size=5,
    dspark_target_layer_ids=(N_LAYERS - 2, N_LAYERS - 1),
    dim=DIM, moe_inter_dim=512, vocab_size=VOCAB,
    n_heads=8, head_dim=512, q_lora_rank=256, o_lora_rank=256, o_groups=4,
    n_routed_experts=8, n_activated_experts=2,
    dtype="bf16", expert_dtype=None, scale_dtype="fp32", scale_fmt=None,
    compress_ratios=(0, 0, 4, 128, 4, 128, 0, 0),
    index_n_heads=4, index_head_dim=128, index_topk=64,
    max_seq_len=2048, max_batch_size=1,
    dspark_noise_token_id=VOCAB - 1, dspark_markov_rank=64,
)

EOS = 7777  # arbitrary; random model hits it ~never


def build(temperature):
    args.temperature = temperature
    torch.manual_seed(1234)
    return Transformer(args)


def reset(model):
    for mod in model.modules():
        if hasattr(mod, "kv_cache") and isinstance(mod.kv_cache, torch.Tensor):
            mod.kv_cache.zero_()
        if hasattr(mod, "kv_state") and isinstance(mod.kv_state, torch.Tensor):
            mod.kv_state.zero_()
        if hasattr(mod, "score_state") and isinstance(mod.score_state, torch.Tensor):
            mod.score_state.fill_(float("-inf"))
        mod._snapshots = None if hasattr(mod, "_snapshots") else None


# ---- 1) greedy exactness ----
torch.manual_seed(42)
prompt = torch.randint(0, VOCAB, (200,)).tolist()
model = build(0.0)
out_plain = generate(model, [prompt], 120, EOS)[0]
reset(model)
out_spec = generate_spec(model, [prompt], 120, EOS)[0]
print("greedy plain:", len(out_plain), out_plain[:20])
print("greedy spec :", len(out_spec), out_spec[:20])
assert out_plain == out_spec, f"GREEDY MISMATCH:\n{out_plain}\n{out_spec}"
print("GREEDY EXACTNESS: PASS")

# ---- 2) prompt crossing window (win=128) + compression boundaries ----
torch.manual_seed(7)
prompt = torch.randint(0, VOCAB, (600,)).tolist()  # > 4*window, exercises rotation
out_plain_long = generate(model, [prompt], 150, EOS)[0]
reset(model)
out_spec_long = generate_spec(model, [prompt], 150, EOS)[0]
assert out_plain_long == out_spec_long, "GREEDY MISMATCH (long prompt)"
print("GREEDY EXACTNESS (long prompt, window rotation): PASS")

# ---- 3) probabilistic run ----
reset(model)
model.temperature = 1.0
model.mtp[-1].temperature = 1.0
out = generate_spec(model, [prompt], 150, EOS)[0]
print("probabilistic run ok, tokens:", len(out), out[:16])

# ---- 4) broken draft: every proposal rejected -> rollback every step, output
#         must still equal plain greedy decoding exactly ----
reset(model)
model.temperature = 0.0
model.mtp[-1].temperature = 0.0

from model import DSparkBlock
orig_forward_head = DSparkBlock.forward_head
def bad_forward_head(self, x, input_ids):
    output_ids, logits, confidence = orig_forward_head(self, x, input_ids)
    # corrupt every drafted token; keep shapes/dtypes intact
    output_ids[:, 1:] = (output_ids[:, 1:] + 1 + torch.randint_like(output_ids[:, 1:], 0, 10)) % 8192
    return output_ids, logits, confidence
DSparkBlock.forward_head = bad_forward_head
out_bad = generate_spec(model, [prompt], 120, EOS)[0]
DSparkBlock.forward_head = orig_forward_head
if out_bad != out_plain:
    for k, (a, b) in enumerate(zip(out_plain, out_bad)):
        if a != b:
            print(f"BROKEN-DRAFT MISMATCH at output index {k}: plain={a} spec={b} (lens {len(out_plain)} vs {len(out_bad)})")
            break
    else:
        print(f"BROKEN-DRAFT length mismatch only: {len(out_plain)} vs {len(out_bad)}")
    raise SystemExit(1)
print("BROKEN-DRAFT ROLLBACK EXACTNESS: PASS")

# ---- 5) verify_and_sample distribution check (pure math, no model) ----
torch.manual_seed(0)
V, N = 64, 3
target_logits = torch.randn(1, N + 1, V).cuda().float() * 2
draft_logits = torch.randn(1, N, V).cuda().float() * 2
p0 = torch.softmax(target_logits[0, 0], -1)
draft_ids = torch.randint(0, V, (1, N + 1)).cuda()
counts = torch.zeros(V, dtype=torch.float32, device="cuda")
trials = 100000
for t in range(trials):
    # Re-sample the first draft token from q each trial: the marginal of the
    # committed token equals p only when draft sampling randomness is included.
    draft_ids[0, 1] = torch.multinomial(torch.softmax(draft_logits[0, 0].float(), -1), 1)[0]
    new_tokens, n_acc = verify_and_sample(target_logits, draft_ids, draft_logits, 1.0)
    counts[new_tokens[0, 0]] += 1
emp = counts / trials
tv = 0.5 * (emp - p0).abs().sum().item()
print(f"distribution check: TV distance to target = {tv:.4f} (expect ~< 0.02 with 100k samples)")
assert tv < 0.02, f"TV distance too large: {tv}"
print("VERIFICATION DISTRIBUTION: PASS")
print("ALL TESTS PASS")
