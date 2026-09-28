# Inference code for DeepSeek models

Standalone inference demo for DeepSeek-V4-Flash extracted from the internal
the monorepo repo: the model/inference code plus the bundled `encoding/`
chat-format dependency. Nothing outside this directory is required.

Layout: `model.py` / `kernel.py` (model + tilelang kernels), `generate.py`
(inference entry, incl. dspark speculative decoding, see
`dspark-spec-decoding.md`), `convert.py` / `run-fp8.sh` (checkpoint
conversion), `encoding/` (DeepSeek-V4 chat encoding, self-contained),
`eval_spec.py` / `check_correct.py` / `test_spec*.py` (spec-decoding
evaluation, correctness checks, smoke tests).

Environment: Python 3.12 + CUDA 13 torch 2.10, pinned in `requirements.txt`
(`tilelang==0.1.14` + `apache-tvm-ffi==0.1.11` is the only combination
verified to JIT-compile the kernels). `fast_hadamard_transform` has no wheel
on the internal mirror — pass one in externally:

```bash
uv venv && uv pip install -r requirements.txt
uv pip install $FAST_HADAMARD_WHEEL --no-build-isolation
```

`run-fp8.sh` bootstraps the same `.venv` automatically (conversion +
8-GPU inference in one go). `DSPARK=1` enables dspark speculative decoding
(interactive and batch modes; see `dspark-spec-decoding.md`).

First convert huggingface model weight files to the format of this project.
```bash
export EXPERTS=256
export MP=4
export CONFIG=config.json
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Then chat with DeepSeek model at will!
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```

Or batch inference from file.
```bash
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

Or multi nodes inference.
```bash
torchrun --nnodes ${NODES} --nproc-per-node $((MP / NODES)) --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --input-file ${FILE}
```

If you want to use fp8, just remove `"expert_dtype": "fp4"` in `config.json` and specify `--expert-dtype fp8` in `convert.py`.
