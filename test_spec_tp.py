"""TP smoke test for dspark speculative decoding (run with torchrun --nproc-per-node 2)."""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
torch.manual_seed(0)

rank = int(os.getenv("RANK", "0"))
torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
if dist.is_initialized() is False and os.getenv("WORLD_SIZE", "1") != "1":
    dist.init_process_group("nccl")

from model import ModelArgs, Transformer
from generate import generate, generate_spec

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
    temperature=0.0,
)
model = Transformer(args)
torch.manual_seed(42)
prompt = torch.randint(0, VOCAB, (200,)).tolist()
out_plain = generate(model, [prompt], 80, 7777)[0]
for mod in model.modules():
    if hasattr(mod, "kv_cache") and isinstance(mod.kv_cache, torch.Tensor):
        mod.kv_cache.zero_()
    if hasattr(mod, "kv_state") and isinstance(mod.kv_state, torch.Tensor):
        mod.kv_state.zero_()
    if hasattr(mod, "score_state") and isinstance(mod.score_state, torch.Tensor):
        mod.score_state.fill_(float("-inf"))
out_spec = generate_spec(model, [prompt], 80, 7777)[0]
assert out_plain == out_spec, "TP GREEDY MISMATCH"
if rank == 0:
    print("TP GREEDY EXACTNESS: PASS")
dist.destroy_process_group()
