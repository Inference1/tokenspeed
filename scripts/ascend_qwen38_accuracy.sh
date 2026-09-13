#!/usr/bin/env bash
# Wrapper for scripts/ascend_qwen38_accuracy_compare.py (item-1 skeleton).
# Prefer Absolute ts_venv python when present (containers often lack `python`).
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

exec "${PYTHON_BIN}" "${ROOT}/scripts/ascend_qwen38_accuracy_compare.py" "$@"
