#!/usr/bin/env bash
# Wrapper for scripts/ascend_qwen38_bench_http.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f /home/tokenspeed_ws/ts_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/tokenspeed_ws/ts_venv/bin/activate
fi

if [[ -x /home/tokenspeed_ws/ts_venv/bin/python ]]; then
  PYTHON_BIN=/home/tokenspeed_ws/ts_venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "python not found; source ts_venv or set PATH" >&2
  exit 1
fi

URL="${URL:-http://127.0.0.1:${PORT:-31891}/v1}"
exec "${PYTHON_BIN}" "${ROOT}/scripts/ascend_qwen38_bench_http.py" --url "${URL}" "$@"
