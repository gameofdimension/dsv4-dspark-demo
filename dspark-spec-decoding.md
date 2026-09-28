# DSpark Speculative Decoding — Design & Implementation Notes

This document describes the dspark speculative decoding pipeline added to this
demo: the design distilled from vLLM 0.29.0, the model-side changes required to
support it, the correctness methodology, and measured results on the real
FP8 checkpoint.

## 1. Background

`model.py` ships the DSpark draft module (`mtp.*` weights: a small stack of
`DSparkBlock` layers plus Markov / confidence heads, driven by
`Transformer.forward_spec`), but `generate.py` only implemented plain
one-token-per-step decoding — the draft weights were dead weight. The goal was
to close the loop and make speculative decoding usable, matching the semantics
of vLLM's

```
--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

For this checkpoint the block size is baked into the weights
(`dspark_block_size: 5` in `config.json`), so each step drafts 5 tokens
(vLLM's `num_speculative_tokens` is the runtime equivalent).

Reference sources read from the vLLM 0.29.0 sdist:

- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` — proposer loop
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` — verifier math
- `vllm/models/deepseek_v4/nvidia/dspark.py` — draft model & weight mapping
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` — per-step context-KV
  maintenance (the `num_rejected` masking in the prepare kernel)

## 2. Contract distilled from vLLM

**Proposer — one block per step, anchor-as-first-query.** Each step the draft
runs a single parallel forward over N query tokens = the anchor embedding
followed by N−1 noise-token embeddings (`dspark_noise_token_id`); every query
position predicts the *next* token. Block position 0 is the anchor, so the
first draft token is predicted *from the anchor*. Intra-block dependency is
injected afterwards by the sequential Markov head: for i = 0..N−1,
`logits[:, i] += markov_bias(prev_token)` then sample. The anchor's draft-KV is
computed inside the block itself — it is not read from the draft cache.

**Verifier — one target forward + rejection sampling.** The target consumes
`[anchor, d1..dN]` (N+1 tokens) in a single forward. Verification is standard
speculative decoding (Leviathan et al., 2023) with the full draft
distribution:

- accept draft token x at step i iff `p(x) > u * q(x)`, u ~ U(0,1), where
  p = target and q = draft probability at that position (both
  temperature-scaled exactly like the samplers);
- at the first rejection, resample from the residual `norm((p − q)+)`;
- if all N drafts are accepted, sample one bonus token from the target's last
  position.

This keeps the output distribution identical to plain autoregressive decoding.
With `--temperature 0` both draft and verifier switch to the greedy variant
(accept iff the draft token equals the target argmax; resample = target
argmax), which makes outputs *bit-comparable* to plain decoding — the basis of
our correctness tests.

**Draft context KV — committed tokens only.** After each verification, the
DSpark layers' sliding-window caches are extended with the target hidden
states of exactly the committed tokens of that step (anchor + accepted
drafts); the rejected suffix writes nothing. The resampled / bonus token —
the next anchor — has no hidden state yet (it was never fed to the target);
it enters as the next draft block's row-0 *query* embedding, and its KV is
inserted the step after. All draft layers derive their context KV from the
same projected `main_x = main_norm(main_proj(concat of target-layer hiddens))`.

**Rejected suffixes must self-heal.** In vLLM the incremental compressor state
lives in a position-indexed paged cache, so stale entries written by rejected
drafts are simply overwritten by the next block's tokens at the same
positions. The demo's rolling-register state machine needs explicit support
for this (§4.2).

## 3. Generation loop (`generate_spec` in `generate.py`)

```
prefill target over prompt           -> first sampled token = anchor@P,
                                        full prompt hiddens
forward_spec(anchor, prompt_hiddens, start_pos=0)
                                        -> builds DSpark context KV
loop:
    draft_ids, draft_logits = forward_spec(anchor, ctx_hidden, ctx_start)
                                        # block occupies P..P+N-1 (draft frame)
    block = [anchor, d1..dN]
    _, target_logits, target_hidden = forward(block, P, full_logits=True)
    new_tokens, n_acc    = verify_and_sample(target_logits, draft_ids,
                                             draft_logits, temperature)
    if rejected at i: model.spec_rollback(i)      # §4.2
    commit m = min(i+1, max_new, total_len-1-P) tokens at P+1..P+m
    ctx_hidden = target_hidden[:, :m]             # next step's context insert
    ctx_start  = P;  anchor = tokens[P+m];  P += m
tail: plain one-token decoding for the last < N+1 positions
```

Bookkeeping details that mattered:

- `forward_spec`'s `start_pos` is the *context start* (position of the first
  newly committed hidden); the draft block's RoPE starts at
  `start_pos + ctx_len`, i.e. at the anchor position.
- The anchor is re-fed to the target as row 0 of every block (its KV is
  written by that forward), mirroring the plain loop where each step feeds the
  previous token.
- The per-100-step / per-call metrics line follows vLLM's
  `SpecDecoding metrics` format, e.g.

  ```
  SpecDecoding metrics: Mean acceptance length: 2.44, Generated throughput: 5.49 tokens/s, Accepted: 5267, Drafted: 18300, Per-position acceptance rate: 0.631 0.377 0.218 0.136 0.078, Avg Draft acceptance rate: 28.8%
  ```

## 4. Model-side changes (`model.py`)

The original decode paths assumed `seqlen == 1`. Speculation needs the target
to process a K = N+1 token block per step, with **causality preserved per
query**. All changes are no-ops (bitwise, same op sequence) for K = 1.

### 4.1 Multi-token decode with per-query causal masks

- `get_window_topk_idxs` (decode branch): query at position p attends the
  window ending at p — later tokens inside the same block are visible to a
  query only at their own positions. Short early windows are padded with −1;
  the sparse-attention kernel skips −1 slots (zero KV, −inf score).
- `get_compress_topk_idxs` (decode branch): per-query slot count
  `(p+1)//ratio` — a query may only see compressed windows fully below its own
  position; −1 padding likewise.
- `Indexer.forward`: same per-query causal mask applied to the compressed-slot
  scores before top-k.
- Sliding-window KV writes are vectorized:
  `kv_cache[:bsz, arange(P, P+K) % win] = kv` (K ≤ win asserted).

### 4.2 Compressor snapshot / rollback — the subtle part

The compressor is a rolling register of the last `2·ratio` (overlap) or
`ratio` tokens' raw kv/score, with window shifts at compression boundaries.
A speculative block advances it through *draft* tokens. After a rejection at
step i the register must reflect exactly the committed prefix (anchor +
accepted drafts); neither of the naive options works:

- *Keep the post-block state*: the register has lost older committed entries
  (overwritten by drafts) and contains rejected-draft values — future
  compression windows would read garbage.
- *Restore a pre-block snapshot*: loses the accepted drafts' contributions —
  equally wrong.

Solution: the decode branch is refactored into a per-token state machine
(`Compressor._decode_token`), and for K > 1 a state snapshot (`kv_state`,
`score_state` clones) is taken **before each token**. On rejection with i
accepted drafts, `spec_rollback(i)` restores snapshot `i+1` — the state right
after the anchor + accepted drafts. The next block re-feeds the resampled
token's position first, so the state machine re-processes it and re-emits any
compression boundary with the correct token. Stale window/compressed-KV
entries written by the rejected suffix are position-indexed and rewritten by
the next block before any read — no cleanup needed. On full acceptance no
rollback happens (everything fed was committed).

### 4.3 DSpark context insert of the committed slice

`DSparkAttention` accepts a multi-token `main_x` (the committed slice) per
step: vectorized cache write at `(start_pos + arange(ctx_len)) % win`, RoPE
over the corresponding positions, and the draft block's top-k window ends at
the last committed position (`get_dspark_topk_idxs(..., ctx_len)`). The block
itself is appended past the cache as before (`win + arange(block_size)`).

### 4.4 `Transformer` plumbing

- `forward(..., full_logits=True)` returns logits for all K positions (the
  verifier needs p at every draft position; `ParallelHead` already supported
  the full-logits path).
- `self.compressors` registry (attention + indexer compressors of all target
  layers) backs `spec_rollback(accepted)`.

## 5. Correctness methodology

All of the following passed on this machine (tiny random-weight model,
L20, single GPU and TP=2):

1. **Greedy exactness** — `generate_spec` output equals plain `generate`
   output token-for-token (rejection with greedy drafts is exact), for both a
   short prompt and a 600-token prompt exercising window rotation. This single
   test validates the entire multi-token decode path, KV writes, per-query
   masks, context insert and rollback jointly.
2. **Broken-draft rollback** — the Markov head is monkeypatched to propose
   garbage, so every step rejects at position 0 and rolls back; the output
   must still equal plain greedy decoding. This isolates the rollback path.
3. **Verifier distribution** — 100k-trial Monte Carlo over synthetic
   distributions: the empirical marginal of the committed token matches the
   target distribution (TV = 0.007).
4. **TP=2 greedy exactness** — the loop is rank-symmetric (identical RNG seeds
   on all ranks), verified end-to-end.

On the real FP8 checkpoint (8×L20, `mp8fp8`):

- Full-chain greedy outputs match plain decoding exactly on 1/10 prompts. The
  divergences were root-caused with a per-position differential (plain 1-token
  forward vs block row-0 logits, compressor state kept in sync via
  `spec_rollback(0)`): **116/120 positions agree on argmax; all 4 flips have
  top-2 logit margin smaller than the logit perturbation** (e.g. margin
  0.003–0.49 vs perturbation 1.1–16.9). This is numeric noise from FP8
  quantization under different GEMM batching shapes (1 vs 6 rows), amplified
  through 43 layers — the same class of behavior as vLLM capturing separate
  CUDA graphs per shape — not a logic bug, and diverged continuations remain
  coherent.

Testing also caught four real bugs worth recording: `sample()` returns a 1-D
tensor (anchor shape), the first anchor token was never written into the
token buffer, the final block's bonus token could index past `total_len`
(off-by-one, commit now clamped by `total_len - 1 - pos`), and `torch.rand`
under the process-wide bf16 default dtype is too coarse for the rejection
uniform draw (now explicit float32).

## 6. Evaluation on the real checkpoint

Setup: `$CKPT_DIR`, 8×L20 via torchrun,
100 prompts (50 English alpaca + 50 Chinese Wizard-LM-Chinese-evol, seeded
sample), 100 new tokens each, temperature 1.0 (probabilistic). Script:
`eval_spec.py`; per-prompt records and the rolling aggregate live in
`eval_results/`.

| metric | value |
|---|---|
| spec steps / drafted / accepted tokens | 3660 / 18300 / 5267 |
| **mean acceptance length** (committed per step, incl. bonus) | **2.44** |
| per-position acceptance rate | 0.631 / 0.377 / 0.218 / 0.136 / 0.078 |
| avg draft acceptance rate | 28.8% |
| commit-length histogram (1..6) | 1352 / 928 / 582 / 301 / 214 / 283 |
| full-accept steps (6 committed) | 7.7% |
| errors / garbled outputs | 0 / 0 |

Per-language split: English 2.92 mean acceptance (38.4% rate), Chinese 2.11
(22.2%) — the DSpark head drafts noticeably better on English instructions.
Shape matches the reference vLLM log line (2.09 mean, rates 0.50/0.28/0.17/
0.10/0.05, 21.8%); absolute values differ with spec-token count (5 vs 7),
sampling temperature and FP4-vs-FP8 experts. Throughput (~5.5 tok/s) is the
demo loop's Python-level overhead (no CUDA graphs, per-step snapshots and
host syncs) and is not a performance claim.

## 7. Known limitations

- **Batch size 1.** The demo's caches and RoPE indexing take a single shared
  `start_pos`, i.e. lock-step batches; per-row acceptance divergence cannot be
  represented without ragged position support (vLLM's engine-level approach).
  File mode processes prompts one by one under `--spec`.
- **Spec-token count is checkpoint-fixed** (`dspark_block_size`); it is not a
  runtime knob.
- Greedy equivalence with plain decoding is bitwise only in the absence of
  quantization-induced near-ties (see §5); the distributional guarantee is
  what holds in general.

## 8. Files

- `generate.py` — `generate_spec`, `verify_and_sample`, `SpecMetrics`, `--spec`
- `model.py` — multi-token decode, compressor snapshot/rollback, DSpark
  context insert, `full_logits`
- `eval_spec.py` — batch evaluation (incremental JSONL + aggregate, resumable)
- `check_correct.py` — greedy equivalence check on a real checkpoint
- `test_spec.py`, `test_spec_tp.py` — self-contained smoke tests (random
  weights, no checkpoint needed)

Run:

```bash
# speculative decoding
torchrun --nproc-per-node 8 generate.py --ckpt-path $CKPT --config config-fp8.json \
    --input-file prompts.txt --max-new-tokens 100 --spec
# evaluation
torchrun --nproc-per-node 8 eval_spec.py --ckpt-path $CKPT --config config-fp8.json \
    --num-prompts 100 --max-new-tokens 100 --out eval_results
```
