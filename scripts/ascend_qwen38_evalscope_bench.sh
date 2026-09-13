#!/usr/bin/env bash
# Run TokenSpeed-CI-aligned EvalScope benchmarks against Ascend OpenAI serve.
#
# Datasets (same as test/ci/eval):
#   aime25       -> math-ai/aime25          (full ~30; default limit 10)
#   gpqa_diamond -> Idavidrein/gpqa diamond (full 198; default limit 20)
#
# Prerequisites:
#   - serve already up (e.g. PORT=31891 bash scripts/ascend_qwen38_smoke.sh --serve)
#   - network to Hugging Face (or set HF_ENDPOINT / use modelscope for GPQA)
#
# Examples:
#   bash scripts/ascend_qwen38_evalscope_bench.sh
#   DATASETS=aime25 LIMIT_AIME=5 bash scripts/ascend_qwen38_evalscope_bench.sh
#   DATASETS=gpqa_diamond LIMIT_GPQA=10 bash scripts/ascend_qwen38_evalscope_bench.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f /home/tokenspeed_ws/ts_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/tokenspeed_ws/ts_venv/bin/activate
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31891}"
MODEL="${MODEL:-qwen3.8-27b}"
API_URL="http://${HOST}:${PORT}/v1"
WORK_ROOT="${WORK_ROOT:-/tmp/qwen38_evalscope}"
DATASETS="${DATASETS:-aime25,gpqa_diamond}"
LIMIT_AIME="${LIMIT_AIME:-10}"
LIMIT_GPQA="${LIMIT_GPQA:-20}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_cache}"
EVALSCOPE_VENV="${EVALSCOPE_VENV:-/tmp/evalscope-ascend}"

export HF_HOME
mkdir -p "${WORK_ROOT}" "${HF_HOME}"

if [[ ! -x "${EVALSCOPE_VENV}/bin/evalscope" ]]; then
  echo "== install evalscope into ${EVALSCOPE_VENV} =="
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --clear "${EVALSCOPE_VENV}"
    uv pip install --python "${EVALSCOPE_VENV}/bin/python" 'evalscope[perf]'
  else
    python3 -m venv "${EVALSCOPE_VENV}"
    "${EVALSCOPE_VENV}/bin/pip" install -U pip
    "${EVALSCOPE_VENV}/bin/pip" install 'evalscope[perf]'
  fi
fi
EVALSCOPE_BIN="${EVALSCOPE_VENV}/bin/evalscope"

echo "== probe ${API_URL}/models =="
curl -sS --fail "${API_URL}/models" | head -c 400
echo

GEN_CFG="$(python3 - <<PY
import json
print(json.dumps({
    "do_sample": False,
    "temperature": 0.0,
    "max_tokens": int("${MAX_TOKENS}"),
}))
PY
)"

run_one() {
  local dataset="$1"
  local limit="$2"
  local hub="$3"
  local dataset_args="$4"
  local work_dir="${WORK_ROOT}/${dataset}_limit${limit}"
  mkdir -p "${work_dir}"

  echo
  echo "== evalscope ${dataset} limit=${limit} hub=${hub} work_dir=${work_dir} =="
  "${EVALSCOPE_BIN}" eval \
    --model "${MODEL}" \
    --api-url "${API_URL}" \
    --api-key EMPTY_TOKEN \
    --datasets "${dataset}" \
    --dataset-hub "${hub}" \
    --dataset-args "${dataset_args}" \
    --limit "${limit}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --generation-config "${GEN_CFG}" \
    --work-dir "${work_dir}" \
    2>&1 | tee "${work_dir}/console.log"

  echo "finished ${dataset}; logs/results under ${work_dir}"
}

IFS=',' read -r -a DS_ARR <<< "${DATASETS}"
for ds in "${DS_ARR[@]}"; do
  ds="$(echo "${ds}" | tr -d '[:space:]')"
  case "${ds}" in
    aime25)
      run_one "aime25" "${LIMIT_AIME}" "huggingface" \
        '{"aime25":{"dataset_id":"math-ai/aime25"}}'
      ;;
    gpqa_diamond)
      # Prefer HF; if blocked, re-run with:
      #   GPQA_HUB=modelscope bash scripts/ascend_qwen38_evalscope_bench.sh DATASETS=gpqa_diamond
      GPQA_HUB="${GPQA_HUB:-huggingface}"
      if [[ "${GPQA_HUB}" == "modelscope" ]]; then
        run_one "gpqa_diamond" "${LIMIT_GPQA}" "modelscope" \
          '{"gpqa_diamond":{"dataset_id":"AI-ModelScope/gpqa_diamond"}}'
      else
        run_one "gpqa_diamond" "${LIMIT_GPQA}" "huggingface" \
          '{"gpqa_diamond":{"dataset_id":"Idavidrein/gpqa","subset_list":["gpqa_diamond"]}}'
      fi
      ;;
    *)
      echo "unknown dataset: ${ds} (supported: aime25, gpqa_diamond)" >&2
      exit 2
      ;;
  esac
done

echo
echo "ALL REQUESTED EVALSCOPE RUNS FINISHED. Root: ${WORK_ROOT}"
echo "Look for mean_acc / Accuracy in each */console.log or EvalScope report tables."
