#!/usr/bin/env bash
# Reproduce Qwen3.8-27B Ascend text-smoke (kernel probe + optional serve).
#
# Usage (inside vllm_ascend_dev / nsenter shell):
#   cd /home/tokenspeed_ws/tokenspeed
#   bash scripts/ascend_qwen38_smoke.sh              # probe + unit tests only
#   bash scripts/ascend_qwen38_smoke.sh --serve      # also launch HTTP serve
#   bash scripts/ascend_qwen38_verify_chat.sh        # curl chat (serve must be up)
#
# Env overrides:
#   QWEN38_MODEL_PATH  model snapshot dir
#   ASCEND_RT_VISIBLE_DEVICES  default 0,1,2,3
#   WORLD_SIZE         default 4
#   PORT               default 31891
#   MAX_TOTAL_TOKENS   default 131072
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DO_SERVE=0
for arg in "$@"; do
  case "${arg}" in
    --serve) DO_SERVE=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

if [[ -f /home/tokenspeed_ws/ts_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/tokenspeed_ws/ts_venv/bin/activate
fi
export TOKENSPEED_CANN_ROOT="${TOKENSPEED_CANN_ROOT:-/usr/local/Ascend/cann-8.5.1}"
# shellcheck disable=SC1091
source "${TOKENSPEED_CANN_ROOT}/set_env.sh"
export PYTHONPATH="${ROOT}/python:${ROOT}/tokenspeed-kernel/python:${ROOT}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "python/python3 not found; source ts_venv or set PYTHON_BIN" >&2
    exit 1
  fi
fi

echo "== patch presence checks =="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

root = Path(".")
checks = [
    (
        root / "tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py",
        "_TND_SUPPORTED_HEAD_DIMS",
        "eager MHA for head_dim=256",
    ),
    (
        root / "python/tokenspeed/runtime/models/qwen3_5.py",
        "Ascend Triton cannot compile fused_qk_rmsnorm_rope_gate",
        "unfused qk_rmsnorm+rope on NPU",
    ),
    (
        root / "tokenspeed-kernel/python/tokenspeed_kernel/ops/kvcache/triton.py",
        "backing.device.type == \"npu\"",
        "zero_byte_ranges NPU fallback",
    ),
    (
        root / "python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py",
        "def num_lcm_blocks",
        "QwenGDN parent sizing",
    ),
]
ok = True
for path, needle, label in checks:
    text = path.read_text(encoding="utf-8", errors="replace")
    hit = needle in text
    print(("OK " if hit else "MISSING "), label, "->", path)
    ok = ok and hit
if not ok:
    raise SystemExit(1)
print("all critical patches present")
PY

echo "== GDN registry probe =="
"${PYTHON_BIN}" - <<'PY'
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

load_builtin_kernels()
reg = KernelRegistry.get()
plat = current_platform()
print("platform", plat.vendor, "is_npu", plat.is_npu)
for mode in ("gdn_chunk_prefill", "gdn_decode_step", "mha_prefill"):
    specs = reg.get_for_operator("attention", mode, platform=plat)
    print(mode, [(s.name, s.solution, int(s.priority)) for s in specs[:4]])
PY

echo "== CPU torch GDN tests =="
"${PYTHON_BIN}" -m pytest -q tokenspeed-kernel-npu/test/test_gdn_torch_cpu.py

if "${PYTHON_BIN}" -c "import torch, torch_npu; assert torch.npu.is_available()"; then
  echo "== NPU GDN tests (best-effort) =="
  "${PYTHON_BIN}" -m pytest -q tokenspeed-kernel-npu/test/test_gdn.py || true
  echo "== NPU MHA eager path smoke =="
  "${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.mha import mha_prefill, _needs_eager

q = torch.randn(4, 2, 256, device="npu", dtype=torch.bfloat16)
k = torch.randn(4, 1, 256, device="npu", dtype=torch.bfloat16)
v = torch.randn(4, 1, 256, device="npu", dtype=torch.bfloat16)
assert _needs_eager(q), "head_dim=256 must use eager"
out = mha_prefill(
    q, k, v,
    cu_seqlens=torch.tensor([0, 4], device="npu", dtype=torch.int32),
    cu_seqlens_cpu=[0, 4],
    max_seqlen=4,
)
assert out.shape == q.shape, out.shape
print("mha_prefill eager ok", tuple(out.shape), float(out.float().abs().mean()))
PY
else
  echo "NPU not available; skipped device tests"
fi

MODEL_PATH="${QWEN38_MODEL_PATH:-/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
PORT="${PORT:-31891}"
WORLD_SIZE="${WORLD_SIZE:-4}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"

if [[ "${DO_SERVE}" -ne 1 ]]; then
  echo
  echo "Probe/tests done. To launch serve:"
  echo "  QWEN38_MODEL_PATH=${MODEL_PATH} bash scripts/ascend_qwen38_smoke.sh --serve"
  echo "Perf A/B examples:"
  echo "  WORLD_SIZE=2 MAX_NUM_SEQS=4 MAX_MODEL_LEN=2048 PORT=31891 bash scripts/ascend_qwen38_smoke.sh --serve"
  echo "Then in another shell:"
  echo "  PORT=${PORT} bash scripts/ascend_qwen38_verify_chat.sh"
  echo "  PORT=${PORT} bash scripts/ascend_qwen38_bench_http.sh"
  exit 0
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "model path missing: ${MODEL_PATH}"
  exit 1
fi

echo "== serve ${MODEL_PATH} port=${PORT} world_size=${WORLD_SIZE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} =="
exec "${PYTHON_BIN}" -m tokenspeed.cli serve "${MODEL_PATH}" \
  --served-model-name qwen3.8-27b \
  --device npu \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --attention-backend mha \
  --sampling-backend greedy \
  --world-size "${WORLD_SIZE}" \
  --language-model-only \
  --enforce-eager \
  --disable-prefill-graph \
  --disable-pdl \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --prefix-granularity 128 \
  --disable-autotune \
  --host 0.0.0.0 \
  --port "${PORT}"
