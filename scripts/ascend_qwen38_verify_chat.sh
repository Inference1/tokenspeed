#!/usr/bin/env bash
# Verify TokenSpeed OpenAI chat API for qwen3.8-27b (serve must already be up).
#
#   PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31891}"
MODEL="${MODEL:-qwen3.8-27b}"
BASE="http://${HOST}:${PORT}"

if [[ -f /home/tokenspeed_ws/ts_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /home/tokenspeed_ws/ts_venv/bin/activate
fi
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

export BASE MODEL
"${PYTHON_BIN}" - <<'PY'
import json
import os
import urllib.error
import urllib.request

base = os.environ["BASE"].rstrip("/")
model = os.environ["MODEL"]


def post_chat(messages, max_tokens=64):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


print("== models ==")
try:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as resp:
        models = json.loads(resp.read().decode())
    print(json.dumps(models, ensure_ascii=False)[:400])
except urllib.error.URLError as exc:
    raise SystemExit(f"cannot reach {base}: {exc}") from exc

print("\n== chat #1 (zh greeting) ==")
data = post_chat([{"role": "user", "content": "你好，用一句话介绍你自己"}])
content = data["choices"][0]["message"]["content"]
assert content and content.strip(), data
print("model:", data.get("model"))
print("content:", content[:240].replace("\n", "\\n"))
print("usage:", data.get("usage"))
print("PASS chat#1")

print("\n== chat #2 (math) ==")
data = post_chat([{"role": "user", "content": "1+1等于几？只回答数字"}])
content = data["choices"][0]["message"]["content"]
assert "2" in content, content
print("content:", content[:160].replace("\n", "\\n"))
print("PASS chat#2")

print("\n== chat #3 (longer prefill) ==")
long_prompt = "请用不超过两句话总结下面这段话的中心思想。" + ("人工智能推理需要正确性优先。" * 40)
data = post_chat([{"role": "user", "content": long_prompt}])
content = data["choices"][0]["message"]["content"]
usage = data.get("usage") or {}
assert content and content.strip(), data
assert int(usage.get("prompt_tokens", 0)) > 50, usage
print("prompt_tokens:", usage.get("prompt_tokens"))
print("content:", content[:200].replace("\n", "\\n"))
print("PASS chat#3")

print(f"\nALL CHAT CHECKS PASSED against {base} model={model}")
PY
