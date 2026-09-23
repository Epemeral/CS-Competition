#!/usr/bin/env bash
set -Eeuo pipefail

# DCU quick-start runner. It deliberately does not install/replace torch or DTK.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SOURCE="${MODEL_SOURCE:-auto}"
MODEL_CACHE="${MODEL_CACHE:-$ROOT/model_cache}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/dcu}"
SKIP_KERNELS="${SKIP_KERNELS:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "找不到 Python：$PYTHON_BIN"
mkdir -p "$RESULT_DIR" "$MODEL_CACHE"

log "检查 Python、PyTorch、ROCm/HIP 和显存"
"$PYTHON_BIN" - <<'PY'
import platform, sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"无法导入 PyTorch: {exc}")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("HIP:", getattr(torch.version, "hip", None))
print("CUDA runtime:", getattr(torch.version, "cuda", None))
print("Platform:", platform.platform())
if not torch.cuda.is_available():
    raise SystemExit("未检测到 GPU；请在 DCU 计算节点运行，而不是登录节点。")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {p.total_memory/2**30:.1f} GiB")
PY

if [[ "$SKIP_KERNELS" != "1" ]]; then
  log "运行 T4/Triton 算子正确性"
  "$PYTHON_BIN" kernels/tests/test_correctness.py
  log "运行算子基准（结果写入 results/t4）"
  "$PYTHON_BIN" kernels/benchmarks/run_bench.py
fi

if [[ "$SKIP_BASELINE" != "1" ]]; then
  log "运行 Qwen2.5-7B FP16 Transformers 基线"
  # 64 GiB 显存下先扫 batch；输入和输出长度可由环境变量覆盖。
  "$PYTHON_BIN" scripts/qwen25_amdgpu_benchmark.py \
    --model "$MODEL" \
    --model-source "$MODEL_SOURCE" \
    --cache-dir "$MODEL_CACHE" \
    --dtype "${DTYPE:-float16}" \
    --attention "${ATTENTION:-sdpa}" \
    --batch-sizes "${BATCH_SIZES:-1,2,4,8,16}" \
    --input-tokens "${INPUT_TOKENS:-512}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
    --warmup "${WARMUP:-3}" \
    --repeats "${REPEATS:-7}" \
    --output "$RESULT_DIR/qwen25_baseline_$(date +%Y%m%d_%H%M%S).json"
fi

log "完成。请保留 results/t4/*.json 和 results/dcu/*.json，并记录当前 git commit。"
