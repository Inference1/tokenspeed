# Ascend Qwen3.8-27B — 推理适配说明（TokenSpeed / vLLM-Ascend）

## 当前正确位置（2026-09-16）

| 项 | 正确值 |
|----|--------|
| 机器 | `178.136.2.2` |
| 容器 | `vllm_ascend_dev` |
| 进容器 | **`nsenter`（不要 `docker exec`）** — 本机 Docker 会报 `OCI runtime state … init_process_start` |
| 当前 API | **vLLM-Ascend** · `http://127.0.0.1:31911/v1` · 模型名 `qwen3.8-27b` |
| 卡 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`（容器内 TP=4） |
| 客户端 | 必须与 serve **同一网络命名空间**（再 `nsenter` 一次，或用容器 IP）；宿主机上 `curl 127.0.0.1:31911` 会 `Failed to connect` |
| 客户端 Python | `ts_venv` 可以（只做 HTTP / EvalScope）；**不要**在 `ts_venv` 里起 vLLM |

```bash
# 宿主机 → 容器（客户端 / 再开终端都用这个）
ssh root@178.136.2.2
PID=$(docker inspect -f '{{.State.Pid}}' vllm_ascend_dev)
nsenter -t "$PID" -m -u -i -n -p bash

export PATH="/usr/local/python3.11.14/bin:$PATH"
source /usr/local/Ascend/cann-8.5.1/set_env.sh
# 若只跑 EvalScope / curl，可用 TokenSpeed venv：
# source /home/tokenspeed_ws/ts_venv/bin/activate
cd /home/tokenspeed_ws/tokenspeed

curl -sS http://127.0.0.1:31911/v1/models | head
PORT=31911 bash scripts/ascend_qwen38_verify_chat.sh   # 脚本只认 PORT，与引擎无关
```

宿主机侧备选（不进 netns 时）：

```bash
CIP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' vllm_ascend_dev)
curl -sS "http://${CIP}:31911/v1/models" | head
# EvalScope: 把 api_url 设为 http://${CIP}:31911/v1
```

说明：当前 vLLM 路径因系统 Triton NPU driver=0，已打本地绕路（Torch gated LN + `patch_qwen3_5` 强制 eager full-attn）。能推理；加速需重装可用的 `triton-ascend`。

---

## TokenSpeed 路径（31891，可选）

此前 TokenSpeed 冒烟（2026-09-13）仍有效，卡 `0,1,2,3`、端口 **31891**：

| 项 | 结果 |
|----|------|
| 引擎 | TokenSpeed（`--device npu`，eager） |
| 模型 | `Qwen3.8-27B`（HF snapshot `1d4bf0f2…`） |
| 并行 | `world-size=4`（`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`） |
| API | `http://127.0.0.1:31891/v1` |
| 形态 | `--language-model-only` |

- `PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh` → **`ALL CHAT CHECKS PASSED`**

回复中常有 thinking + `</think>`；冒烟按内容判定，不等于 HF greedy 对齐。

**尚未声称的正式精度/性能数字**：需 HF greedy 基线 + HTTP bench。

---

## 建议截图保存（交付证据）

请在本地建目录（示例）`docs/evidence/ascend_qwen38_2026-09-13/`，至少保存：

1. **环境** — `npu-smi info`；`docker ps` 含 `vllm_ascend_dev`
2. **服务就绪** — serve 日志；`curl http://127.0.0.1:31911/v1/models`（当前）或 `:31891`（TokenSpeed）
3. **功能冒烟** — `PORT=31911`（或 `31891`）`verify_chat` / 一次 chat JSON
4. **精度** — `/tmp/qwen38_acc_compare.json` 的 `summary`
5. **性能** — `ascend_qwen38_bench_http` 的 `SUMMARY`

---

## 日常起停

### 进容器（唯一推荐）

```bash
ssh root@178.136.2.2
PID=$(docker inspect -f '{{.State.Pid}}' vllm_ascend_dev)
nsenter -t "$PID" -m -u -i -n -p bash
# 不要用: docker exec -it vllm_ascend_dev bash   # 本机会 OCI 失败
```

### A) 当前：vLLM-Ascend（31911，卡 4–7）

在 **nsenter 后的终端**（系统 Python，非 ts_venv 起服）：

```bash
export PATH="/usr/local/python3.11.14/bin:$PATH"
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export QWEN38_MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

vllm serve "$QWEN38_MODEL_PATH" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 31911 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --language-model-only \
  2>&1 | tee /tmp/vllm_qwen38_serve.log
```

另一终端同样 `nsenter` 后：

```bash
curl -sS http://127.0.0.1:31911/v1/models | head
```

### B) TokenSpeed（31891，卡 0–3）

```bash
source /home/tokenspeed_ws/ts_venv/bin/activate
cd /home/tokenspeed_ws/tokenspeed
export TOKENSPEED_CANN_ROOT=/usr/local/Ascend/cann-8.5.1
source "${TOKENSPEED_CANN_ROOT}/set_env.sh"
export PYTHONPATH="${PWD}/python:${PWD}/tokenspeed-kernel/python:${PWD}/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export QWEN38_MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

PORT=31891 bash scripts/ascend_qwen38_smoke.sh --serve
```

验证：

```bash
source /home/tokenspeed_ws/ts_venv/bin/activate
cd /home/tokenspeed_ws/tokenspeed
curl -sS http://127.0.0.1:31891/v1/models | head
PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh
```

---

## 关键 Ascend 补丁 / 行为（合 PR 时对照）

| 位置 | 作用 |
|------|------|
| `tokenspeed-kernel-npu/.../ops/mha.py` | `head_dim=256` 优先 BNSD FIA，失败回退 eager |
| `tokenspeed-kernel-npu/.../ops/gdn_npu.py` | Ascend GDN CANN/vectorized；Torch fallback 保留 |
| `tokenspeed-kernel/.../attention/ascend.py` | 注册 `ascend_npu_gdn_*` (PERFORMANT+2) |
| `python/.../models/qwen3_5.py` | NPU 跳过 `fused_qk_rmsnorm_rope_gate` |
| `tokenspeed-kernel/.../ops/kvcache/triton.py` | NPU `zero_byte_ranges` 用 `tensor.zero_()` |
| `python/.../kv_cache/recipes/qwen35.py` | GDN parent/`token_capacity` 正确 sizing |
| `python/.../layers/rotary_embedding.py` | NPU 关闭 RoPE `torch.compile`（Inductor KeyError） |
| `python/.../execution/model_executor.py` | NPU 跳过 RSAG prewarm |
| `python/.../layers/logits_processor.py` | 禁止在无 CUDA 时调用 `is_current_stream_capturing` |
| 已安装 `tokenspeed-scheduler` | 须与 Python 同 ABI（`block_granularity`）；改 C++ 后需 `pip install -e` |

**整树同步**时请同时带上：`python/tokenspeed`、`tokenspeed-kernel/python`、`tokenspeed-kernel-npu/python`、`tokenspeed-scheduler`（若 ABI 变了要重编）、`scripts`。不要只 scp 单个 `triton.py`。

---

## 精度 / 性能

- **标准 benchmark（推荐）**：EvalScope `aime25` + `gpqa_diamond`（与 CI 同数据集，默认小 limit）  
  `PORT=31891 bash scripts/ascend_qwen38_evalscope_bench.sh`  
- 精度对齐：CUDA 机 HF greedy → 昇腾 `ascend_qwen38_accuracy.sh` → `mean_token_match_rate`  
- 性能：`scripts/ascend_qwen38_bench_http.sh`（勿用 accuracy jsonl 当 perf）  
- 详见 `scripts/README_ascend_qwen38.md`

---

## 已知限制

- 默认仍 **正确性优先**（`--enforce-eager`）。GDN/`mha` D=256 已接 NPU 高性能路径；用 `ASCEND_ALLOW_GRAPH=1` + 另端口 `bench_http` 做 ACL graph A/B，勿打断 31891 全量评测。  

- Chat 可能带 thinking 风格文本；判分/比 token 前建议关 thinking 或剥 `</think>`。  
- 输出质量与 HF 对齐数字需单独测，不能仅凭冒烟通过声称「精度达标」。
