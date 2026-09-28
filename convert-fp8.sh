#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731: HF 权重 -> FP8 布局转换 + 8 卡推理
# H20(Hopper) 无 FP4 张量核心, expert 必须转 fp8: convert.py 带 --expert-dtype fp8,
# 推理用本目录 config-fp8.json(= config.json 去掉 "expert_dtype": "fp4")。
# 用法: HF_CKPT_PATH=... SAVE_PATH=... FAST_HADAMARD_WHEEL=... bash convert-fp8.sh (三者必传, 无默认值)
# 推理模式: 默认 batch(读 prompts.txt, 验证跑通); INTERACTIVE=1 进交互模式;
#           INPUT_FILE / MAX_NEW_TOKENS 可覆盖批量输入与新 token 数
set -xeuo pipefail

# 依赖(.venv/requirements.txt/config-fp8.json/convert.py)均为相对路径, 切到脚本所在目录
cd "$(dirname "$0")"
command -v uv >/dev/null 2>&1 || pip install uv

if [ ! -d ".venv" ]; then
    uv venv
fi

# requirements.txt 已钉 tilelang==0.1.14 + apache-tvm-ffi==0.1.11:
# tilelang 0.1.8 + tvm-ffi>=0.1.12 import 即崩; 0.1.8 + 0.1.11 JIT 编译内核崩;
# 0.1.14 + 0.1.11 为唯一验证可用组合
uv pip install -r requirements.txt

# fast_hadamard_transform 为必需依赖(model.py 的 Indexer 每层前向都会调用
# rotate_activation -> 惰性 import; 内网源只有 sdist 无预编译 wheel, 需外部传入 wheel 路径)
if [ -f "$FAST_HADAMARD_WHEEL" ]; then
    uv pip install "$FAST_HADAMARD_WHEEL" --no-build-isolation
else
    echo "[convert-fp8] FATAL: fast_hadamard_transform wheel 不存在: $FAST_HADAMARD_WHEEL" >&2
    echo "[convert-fp8]        请检查 FAST_HADAMARD_WHEEL 指向的 wheel 路径" >&2
    exit 1
fi
.venv/bin/python -c "import fast_hadamard_transform" \
  || { echo "[convert-fp8] FATAL: fast_hadamard_transform 安装后仍无法导入" >&2; exit 1; }

source .venv/bin/activate

EXPERTS=256
MP=8
CONFIG="./config-fp8.json"

# 恰好 $MP 个 shard 才视为已转换完成(转换耗时久, 跳过); 目录不存在/不完整则重新转换
n_safetensors=0
if [ -d "$SAVE_PATH" ]; then
    n_safetensors=$(find "$SAVE_PATH" -maxdepth 1 -name '*.safetensors' | wc -l)
fi
if [ "$n_safetensors" -ne "$MP" ]; then
    python convert.py --hf-ckpt-path "${HF_CKPT_PATH}" --save-path "${SAVE_PATH}" \
        --n-experts "${EXPERTS}" --model-parallel "${MP}" --expert-dtype fp8
fi

if [ "${INTERACTIVE:-0}" = "1" ]; then
    torchrun --nproc-per-node "${MP}" generate.py --ckpt-path "${SAVE_PATH}" --config "${CONFIG}" --interactive
else
    INPUT_FILE=${INPUT_FILE:-"$(cd "$(dirname "$0")" && pwd)/prompts.txt"}
    torchrun --nproc-per-node "${MP}" generate.py --ckpt-path "${SAVE_PATH}" --config "${CONFIG}" \
        --input-file "${INPUT_FILE}" --max-new-tokens "${MAX_NEW_TOKENS:-64}"
fi
