# Ascend Qwen3.8-27B scripts

## Smoke / serve / verify

| Script | Role |
|--------|------|
| `ascend_qwen38_smoke.sh` | Patch checks + optional `--serve` (`ASCEND_ALLOW_GRAPH=1` drops `--enforce-eager`) |
| `ascend_qwen38_verify_chat.sh` | HTTP `/v1/models` + chat smoke (`PORT=31911` = current vLLM; `31891` = TokenSpeed) |
| `ascend_qwen38_bench_http.sh` | E2E latency / tok/s (not accuracy) |

Perf A/B (use a free port; do not collide with a long accuracy job on 31891/31911):

```bash
PORT=31901 ASCEND_ALLOW_GRAPH=0 bash scripts/ascend_qwen38_smoke.sh --serve
PORT=31901 bash scripts/ascend_qwen38_bench_http.sh
# Other NPUs / other session:
PORT=31902 ASCEND_ALLOW_GRAPH=1 bash scripts/ascend_qwen38_smoke.sh --serve
PORT=31902 bash scripts/ascend_qwen38_bench_http.sh
```

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

## Item 2 — EvalScope AIME25 + GPQA Diamond (CI-aligned datasets)

Aligned with `test/ci/eval/*-evalscope-aime25.yaml` / `*-gpqa-diamond.yaml`:

| Dataset | Hub id | Full size (approx.) | Default small limit |
|---------|--------|---------------------|---------------------|
| `aime25` | `math-ai/aime25` | 30 | 10 |
| `gpqa_diamond` | `Idavidrein/gpqa` subset `gpqa_diamond` | 198 | 20 |

```bash
# Enter container, then:
# Current vLLM serve is on 31911 (TokenSpeed historically used 31891)
PORT=31911 bash scripts/ascend_qwen38_evalscope_bench.sh
# PORT=31891 bash scripts/ascend_qwen38_evalscope_bench.sh

# Smaller subsets
DATASETS=aime25 LIMIT_AIME=5 bash scripts/ascend_qwen38_evalscope_bench.sh
DATASETS=gpqa_diamond LIMIT_GPQA=10 bash scripts/ascend_qwen38_evalscope_bench.sh

# If Hugging Face is unreachable for GPQA, use ModelScope
GPQA_HUB=modelscope DATASETS=gpqa_diamond bash scripts/ascend_qwen38_evalscope_bench.sh
```

Results default to `/tmp/qwen38_evalscope/{dataset}_limitN/`. Many older serves used
`max-model-len=4096` with script default `MAX_TOKENS=1024`. For long thinking AIME/GPQA,
raise both (lab AISBench path uses `max-model-len=131072` and `max_out_len=32768`).

For full-suite AISBench numbers and protocol notes, see `docs/recipes/models.md` and
`docs/platforms/ascend_qwen38_sync.md`.
