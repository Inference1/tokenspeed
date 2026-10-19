# Ascend Qwen3.8-27B scripts

## Smoke / serve / verify

| Script | Role |
|--------|------|
| `ascend_qwen38_smoke.sh` | Patch checks + optional `--serve` |
| `ascend_qwen38_verify_chat.sh` | HTTP models + chat smoke (`PORT=31891`) |
| `ascend_qwen38_bench_http.sh` | E2E latency / tok-s (not accuracy) |

Model path on lab:

```bash
export QWEN38_MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

## Item 1 — greedy HF vs Ascend HTTP

| File | Role |
|------|------|
| `data/qwen38_accuracy_prompts.jsonl` | Shared prompts |
| `ascend_qwen38_accuracy_compare.py` | Driver |
| `ascend_qwen38_accuracy.sh` | Wrapper |

```bash
# A) HF baseline (CUDA machine)
bash scripts/ascend_qwen38_accuracy.sh \
  --prompts scripts/data/qwen38_accuracy_prompts.jsonl \
  --hf-model "$QWEN38_MODEL_PATH" \
  --skip-ascend \
  --out /tmp/qwen38_hf_baseline.json

# B) Ascend compare
bash scripts/ascend_qwen38_accuracy.sh \
  --prompts scripts/data/qwen38_accuracy_prompts.jsonl \
  --baseline /tmp/qwen38_hf_baseline.json \
  --skip-hf \
  --tokenizer-model "$QWEN38_MODEL_PATH" \
  --ascend-url http://127.0.0.1:31891/v1 \
  --served-model qwen3.8-27b \
  --out /tmp/qwen38_acc_compare.json
```

## Item 2 — EvalScope AIME25 + GPQA Diamond（与 CI 同数据集）

与 `test/ci/eval/*-evalscope-aime25.yaml` / `*-gpqa-diamond.yaml` 对齐：

| Dataset | Hub id | 全量约 | 默认 limit（小跑） |
|---------|--------|--------|-------------------|
| `aime25` | `math-ai/aime25` | 30 | 10 |
| `gpqa_diamond` | `Idavidrein/gpqa` subset `gpqa_diamond` | 198 | 20 |

```bash
# serve 已在 31891
PORT=31891 bash scripts/ascend_qwen38_evalscope_bench.sh

# 只跑更小子集
DATASETS=aime25 LIMIT_AIME=5 bash scripts/ascend_qwen38_evalscope_bench.sh
DATASETS=gpqa_diamond LIMIT_GPQA=10 bash scripts/ascend_qwen38_evalscope_bench.sh

# HF 拉不了 GPQA 时改 ModelScope
GPQA_HUB=modelscope DATASETS=gpqa_diamond bash scripts/ascend_qwen38_evalscope_bench.sh
```

结果目录默认：`/tmp/qwen38_evalscope/{dataset}_limitN/`。注意当前 serve 多为 `max-model-len=4096`，脚本默认 `MAX_TOKENS=1024`；AIME 长推理不够可酌情加大并相应提高 `max-model-len`。

Bring-up notes + screenshot checklist: `docs/platforms/ascend_qwen38_sync.md`.
