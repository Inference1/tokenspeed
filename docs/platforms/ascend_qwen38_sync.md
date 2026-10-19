# Ascend Qwen3.8-27B — bring-up notes (TokenSpeed / vLLM-Ascend)

## Current lab layout (2026-09-28)

| Item | Value |
|------|--------|
| Host | `178.136.2.2` |
| Container | `vllm_ascend_dev` |
| Enter container | Prefer `docker exec -it vllm_ascend_dev bash`. If host Docker returns `OCI runtime state … init_process_start`, use `nsenter` (see below). |
| Active API | **vLLM-Ascend** · `http://127.0.0.1:31911/v1` · served name `qwen3.8-27b` |
| NPUs | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` (TP=4 inside the container) |
| Client | Must share the serve network namespace (`docker exec` / `nsenter`), or use the container IP. Host-side `curl 127.0.0.1:31911` fails if the server is not published on the host netns. |
| Client Python | `aisbench_venv` / `ts_venv` for HTTP clients and EvalScope/AISBench only — do **not** start vLLM from `ts_venv`. |

```bash
# Host → container
ssh root@178.136.2.2
docker start vllm_ascend_dev
docker exec -it vllm_ascend_dev bash
# Fallback if docker exec is broken on this host:
# PID=$(docker inspect -f '{{.State.Pid}}' vllm_ascend_dev)
# nsenter -t "$PID" -m -u -i -n -p bash

export PATH="/usr/local/python3.12.13/bin:$PATH"
source /usr/local/Ascend/cann-8.5.1/set_env.sh
cd /home/tokenspeed_ws/tokenspeed   # or /home/AISBench for AISBench

curl -sS http://127.0.0.1:31911/v1/models | head
PORT=31911 bash scripts/ascend_qwen38_verify_chat.sh
```

Host-side alternative (when not sharing netns):

```bash
CIP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' vllm_ascend_dev)
curl -sS "http://${CIP}:31911/v1/models" | head
# Point EvalScope/AISBench api_url at http://${CIP}:31911/v1
```

Aligned AISBench accuracy client (thinking / `reasoning_effort=xhigh` top-level,
`max_out_len=32768`, server `max-model-len=131072`) lives under the lab AISBench
checkout; see measured scores in `docs/recipes/models.md`.

---

## TokenSpeed path (port 31891, optional)

Earlier TokenSpeed smoke (2026-09-13) remains valid on NPUs `0,1,2,3`, port **31891**:

| Item | Result |
|------|--------|
| Engine | TokenSpeed (`--device npu`, eager by default) |
| Model | `Qwen3.8-27B` (HF snapshot `1d4bf0f2…`) |
| Parallel | `world-size=4` (`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`) |
| Mode | `--language-model-only` |

Responses may include thinking / `</think>`; smoke checks content, not HF greedy
token match.

**Not claimed here:** formal HF-aligned accuracy or peak tok/s — run the accuracy
and HTTP bench scripts separately.

---

## Suggested evidence screenshots

Keep under e.g. `docs/evidence/ascend_qwen38_YYYY-MM-DD/`:

1. **Environment** — `npu-smi info`; `docker ps` showing `vllm_ascend_dev`
2. **Serve ready** — serve log; `curl http://127.0.0.1:31911/v1/models` (or `:31891`)
3. **Functional smoke** — `PORT=31911` (or `31891`) `verify_chat` / one chat JSON
4. **Accuracy** — AISBench/EvalScope summary, or `/tmp/qwen38_acc_compare.json`
5. **Perf** — `ascend_qwen38_bench_http` `SUMMARY`

---

## Daily start / stop

### Enter container

```bash
docker start vllm_ascend_dev
docker exec -it vllm_ascend_dev bash
# Fallback:
# PID=$(docker inspect -f '{{.State.Pid}}' vllm_ascend_dev)
# nsenter -t "$PID" -m -u -i -n -p bash
```

### A) Current: vLLM-Ascend (31911, NPUs 4–7)

Inside the container (system Python / image vLLM, not `ts_venv`):

```bash
export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# See lab nohup recipe: --max-model-len 131072, TP=4, port 31911
```

In another container shell:

```bash
curl -sS http://127.0.0.1:31911/v1/models | head
```

### B) TokenSpeed (31891, NPUs 0–3)

```bash
source /home/tokenspeed_ws/ts_venv/bin/activate
export TOKENSPEED_CANN_ROOT=/usr/local/Ascend/cann-8.5.1
source "${TOKENSPEED_CANN_ROOT}/set_env.sh"
cd /home/tokenspeed_ws/tokenspeed
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export QWEN38_MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
PORT=31891 bash scripts/ascend_qwen38_smoke.sh --serve
```

Verify:

```bash
PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh
```

---

## Key Ascend patches / behavior (PR checklist)

| Location | Role |
|----------|------|
| `tokenspeed-kernel-npu/.../ops/mha.py` | Prefer BNSD FIA for `head_dim=256`; eager fallback |
| `tokenspeed-kernel-npu/.../ops/gdn_npu.py` | Ascend GDN CANN/vectorized path; Torch fallback kept |
| `tokenspeed-kernel/.../attention/ascend.py` | Register `ascend_npu_gdn_*` (PERFORMANT+2) |
| `python/.../models/qwen3_5.py` | Skip `fused_qk_rmsnorm_rope_gate` on NPU when needed |
| `tokenspeed-kernel/.../ops/kvcache/triton.py` | NPU `zero_byte_ranges` via `tensor.zero_()` |
| `python/.../kv_cache/recipes/qwen35.py` | Correct GDN parent / `token_capacity` sizing |
| `python/.../layers/rotary_embedding.py` | Disable RoPE `torch.compile` on NPU |
| `python/.../execution/model_executor.py` | Skip RSAG prewarm on NPU |
| `python/.../layers/logits_processor.py` | Do not call `is_current_stream_capturing` without CUDA |
| Installed `tokenspeed-scheduler` | Must match Python ABI (`block_granularity`); reinstall editable after C++ changes |

When syncing the tree, ship together: `python/tokenspeed`, `tokenspeed-kernel/python`,
`tokenspeed-kernel-npu/python`, `tokenspeed-scheduler` (rebuild if ABI changes),
and `scripts`. Do not scp a single `triton.py` in isolation.

---

## Accuracy / performance

- **Preferred full accuracy**: AISBench against vLLM-Ascend (`gpqa_gen_0_shot_str`,
  `aime2025_gen_0_shot_chat_prompt`) with `reasoning_effort=xhigh` (request top-level).
- **CI-aligned small EvalScope**: `aime25` + `gpqa_diamond` — see
  `scripts/README_ascend_qwen38.md`.
- Token match vs HF: CUDA HF greedy → Ascend `ascend_qwen38_accuracy.sh` →
  `mean_token_match_rate`.
- Perf: `scripts/ascend_qwen38_bench_http.sh` (do not treat accuracy jsonl as perf).

---

## Known limits

- Default recipe stays **correctness-first** (`--enforce-eager` on TokenSpeed).
  GDN / MHA `D=256` NPU paths exist; use `ASCEND_ALLOW_GRAPH=1` on a free port for
  ACL-graph A/B without interrupting a long accuracy run.
- Chat may emit thinking text; disable thinking or strip `</think>` before scoring
  when the harness expects final answers only.
- Do not claim HF-matched accuracy from smoke alone.
