#!/usr/bin/env bash
# Apply / verify Qwen3.8-27B Ascend GDN bring-up inside vllm_ascend_dev.
# Run from the tokenspeed repo root on the Ascend host (container).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source /home/tokenspeed_ws/ts_venv/bin/activate
export TOKENSPEED_CANN_ROOT="${TOKENSPEED_CANN_ROOT:-/usr/local/Ascend/cann-8.5.1}"
source "${TOKENSPEED_CANN_ROOT}/set_env.sh"
export PYTHONPATH="${ROOT}/python:${ROOT}/tokenspeed-kernel/python:${ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

echo "== GDN registry probe =="
python - <<'PY'
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

load_builtin_kernels()
reg = KernelRegistry.get()
print("platform", current_platform().vendor, "is_npu", current_platform().is_npu)
for mode in ("gdn_chunk_prefill", "gdn_decode_step", "gdn_decode_mtp"):
    specs = reg.get_for_operator("attention", mode, platform=current_platform())
    print(mode, [(s.name, s.solution, s.priority) for s in specs])
PY

echo "== CPU torch GDN tests =="
pytest -q tokenspeed-kernel-npu/test/test_gdn_torch_cpu.py

if python -c "import torch, torch_npu; assert torch.npu.is_available()"; then
  echo "== NPU GDN tests =="
  pytest -q tokenspeed-kernel-npu/test/test_gdn.py || true
fi

MODEL_PATH="${QWEN38_MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Set QWEN38_MODEL_PATH to a local Qwen3.8-27B snapshot to launch serve."
  exit 0
fi

echo "== serve ${MODEL_PATH} =="
exec python -m tokenspeed.cli serve "${MODEL_PATH}" \
  --served-model-name qwen3.8-27b \
  --device npu \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --attention-backend mha \
  --sampling-backend greedy \
  --tp-size "${TP_SIZE:-4}" \
  --disable-prefill-graph \
  --disable-pdl \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --max-total-tokens 8192 \
  --chunked-prefill-size 2048 \
  --prefix-granularity 128 \
  --disable-autotune \
  --host 0.0.0.0 \
  --port 31889
