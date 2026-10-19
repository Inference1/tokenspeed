# Ascend Qwen3.8-27B — 推理适配说明（TokenSpeed）

## 结论（当前状态）

在实验室机器 `178.136.2.2`、容器 `vllm_ascend_dev` 上，**Qwen3.8-27B 文本推理已跑通**：

| 项 | 结果 |
|----|------|
| 引擎 | TokenSpeed（`--device npu`，eager） |
| 模型 | `Qwen3.8-27B`（HF snapshot `1d4bf0f2…`） |
| 并行 | `world-size=4`（`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`） |
| API | OpenAI 兼容 `http://127.0.0.1:31891/v1` |
| 服务名 | `qwen3.8-27b` |
| 形态 | `--language-model-only`（文本；非 VLM） |
| 架构路径 | hybrid GDN：16 full-attn + 48 linear/GDN |

冒烟（已在实验室复现，2026-09-13）：

- `/v1/models` 返回 `qwen3.8-27b`
- `PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh` → **`ALL CHAT CHECKS PASSED`**
  - chat#1 中文自我介绍 PASS
  - chat#2 数学含最终答案 `2` PASS
  - chat#3 较长 prefill（~342 prompt tokens）PASS

回复中常出现 thinking 草稿 + `</think>` 后再给最终答案；冒烟按「内容非空 / 含期望子串」判定，不代表已与 HF greedy 对齐。

**尚未声称的正式精度/性能数字**：需 HF greedy 基线对比 + 独立 HTTP bench（见下文「证据清单」）。

---

## 建议截图保存（交付证据）

请在本地建目录（示例）`docs/evidence/ascend_qwen38_2026-09-13/`，至少保存：

1. **环境**  
   - `npu-smi info`（卡型号 / 健康）  
   - `docker ps` 含 `vllm_ascend_dev`

2. **服务就绪**  
   - serve 日志：权重加载完成、`Created hybrid_linear_attn backend: 16 full … 48 linear …`  
   - `curl http://127.0.0.1:31891/v1/models` 输出（含 `qwen3.8-27b`）

3. **功能冒烟**  
   - `PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh` 全文，尤其最后一行 `ALL CHAT CHECKS PASSED`  
   - 或一次 `chat/completions` curl 的完整 JSON（提问 + 回复）

4. **精度**  
   - `/tmp/qwen38_acc_compare.json` 的 `summary`：`mean_token_match_rate` / `exact_token_match_frac`

5. **性能**  
   - `ascend_qwen38_bench_http` 的 `SUMMARY` 行（固定 prompt、`max_tokens=128`、thinking off）

每张图文件名建议带日期与内容，例如：`01_npu_smi.png`、`02_v1_models.png`、`03_verify_chat_pass.png`。

---

## 日常起停（对齐完成后）

### 进容器

```bash
ssh root@178.136.2.2
PID=$(docker inspect -f '{{.State.Pid}}' vllm_ascend_dev)
nsenter -t "$PID" -m -u -i -n -p bash
```

### 起 serve（单独占一个终端）

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

### 另一终端验证

```bash
# 进容器后务必 cd 到仓库
source /home/tokenspeed_ws/ts_venv/bin/activate
cd /home/tokenspeed_ws/tokenspeed

curl -sS http://127.0.0.1:31891/v1/models | head
PORT=31891 bash scripts/ascend_qwen38_verify_chat.sh
```

---

## 关键 Ascend 补丁 / 行为（合 PR 时对照）

| 位置 | 作用 |
|------|------|
| `tokenspeed-kernel-npu/.../ops/mha.py` | `head_dim=256` 走 eager MHA |
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

- 当前为 **正确性优先**：`--enforce-eager`，GDN/MHA 大量 torch/eager，性能未优化。  
- Chat 可能带 thinking 风格文本；判分/比 token 前建议关 thinking 或剥 `</think>`。  
- 输出质量与 HF 对齐数字需单独测，不能仅凭冒烟通过声称「精度达标」。
