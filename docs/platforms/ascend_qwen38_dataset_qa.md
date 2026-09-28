# Ascend Qwen3.8-27B — small-dataset Q&A log

- **Historical serve**: TokenSpeed Ascend · `http://127.0.0.1:31891/v1` · model `qwen3.8-27b`
- **Current (lab)**: vLLM-Ascend · container `vllm_ascend_dev` · `http://127.0.0.1:31911/v1` · NPUs `4,5,6,7`; client must share serve netns — see `ascend_qwen38_sync.md`
- **Prompt set**: `scripts/data/qwen38_accuracy_prompts.jsonl` (8 items)
- **Measured run**: long generation `max_tokens≥256` (`/tmp/qwen38_prompts_eval.jsonl`)
- **Result file**: `/tmp/qwen38_ascend_dataset_long.json`
- **Summary**: scored 7/7 · `ascend_expect_accuracy = 1.0` · `errors = []`
- **Notes**: replies often include thinking then `</think>` and a final answer. Model answers below are from lab logs (some truncated). No HF greedy token-match numbers here.

---

## 1. short_zh_hello (short chat, unscored)

**Tags**: `short`, `chat`  
**Expect**: none (smoke only)  
**Pass**: —

**Prompt**

> 你好，用一句话介绍你自己

**Model answer (excerpt)**

> (thinking) 用户要求用一句话介绍自己。我需要用中文回复，简洁地介绍我是通义千问，由阿里巴巴通义实验室开发的大语言模型。  
> `</think>`  
> 你好！我是通义千问（Qwen），…

---

## 2. short_math(short, scored)

**Tags**：`short`, `accuracy`  
**Expect contains**：`2`  
**Pass**：True

**Prompt**

> 1+1等于几？只回答数字

**Model answer (excerpt)**

> (thinking) We need answer user's request: "1+1等于几？只回答数字" means "What is 1+1? Only answer nu…  
> (full output contains最终数字 `2`，故 expect 通过）

---

## 3. short_en_fact(short, scored)

**Tags**：`short`, `accuracy`  
**Expect contains**：`Paris`  
**Pass**：True

**Prompt**

> What is the capital of France? Reply with one word only.

**Model answer (excerpt)**

> We need answer one word only. User asks capital of France. Answer: Paris. Ensure…

---

## 4. short_arith(short, scored)

**Tags**：`short`, `accuracy`  
**Expect contains**：`10`  
**Pass**：True

**Prompt**

> 若 x=3, y=x\*x+1，y 等于多少？只回答数字。

**Model answer (excerpt)**

> We need answer in Chinese? User: 若 x=3, y=x\*x+1，y 等于多少？只回答数字。 Need only number. …  
> (full output contains `10`）

---

## 5. long_repeat_summary(long prefill)

**Tags**：`long`, `prefill`  
**Expect contains**：`正确`  
**Pass**：True

**Prompt**

> 请用不超过两句话总结下面这段话的中心思想。  
> （「人工智能推理需要正确性优先。」重复多遍）

**Model answer (excerpt)**

> (thinking) 用户要求用不超过两句话总结给定文本的中心思想。文本重复同一句：「人工智能推理需要正确性优先。」需要简洁中文回答，不超过两句。可以直接说：中心思想是强调人工智能…

---

## 6. multi_turn_basic(multi-turn, scored)

**Tags**：`multi_turn`, `chat`, `accuracy`  
**Expect contains**：`小明`  
**Pass**：True

**Dialog**

| role | content |
|------|---------|
| user | 我叫小明。 |
| assistant | 好的，小明你好。 |
| user | 我叫什么名字？只回答名字。 |

**Model answer**

> (thinking) 用户问「我叫什么名字？只回答名字。」前文用户说「我叫小明。」因此应只回答名字：小明。需要严格只输出名字，不加标点或解释。  
> `</think>`  
> **小明**

---

## 7. layer_mix_code(hybrid layers, code)

**Tags**：`short`, `full_attn_gdn_mix`  
**Expect contains**：`return`  
**Pass**：True

**Prompt**

> 用 Python 写一个函数 add(a, b) 返回两数之和，只要函数体。

**Model answer (excerpt)**

> We need answer user in Chinese likely. User: "用 Python 写一个函数 add(a, b) 返回两数之和，只要…  
> (full output contains `return`）

---

## 8. layer_mix_reason(hybrid layers, reason)

**Tags**：`short`, `full_attn_gdn_mix`, `accuracy`  
**Expect contains**：`10`  
**Pass**：True

**Prompt**

> 3 的平方加 1 等于几？只回答数字。

**Model answer (excerpt)**

> We need answer user: "3 的平方加 1 等于几？只回答数字。" Chinese: 3 squared plus 1 equals what…  
> (full output contains `10`）

---

## Summary table

| id | Prompt (short) | Expect | Result |
|----|----------------|--------|--------|
| short_zh_hello | introduce yourself | — | valid Chinese reply |
| short_math | 1+1=? | `2` | True |
| short_en_fact | capital of France | `Paris` | True |
| short_arith | x=3, y=x²+1 | `10` | True |
| long_repeat_summary | summarize "correctness first" | `正确` | True |
| multi_turn_basic | what is my name | `小明` | True |
| layer_mix_code | Python add | `return` | True |
| layer_mix_reason | 3²+1=? | `10` | True |

**Conclusion**: with longer generation, all 7 `expect_contains` items passed. This is a
functional Ascend smoke log, not an HF greedy token-match claim.

---

## Standard benchmarks (AIME / GPQA Diamond)

Same datasets as CI via **EvalScope** (small limits) or full **AISBench** against
vLLM-Ascend (see `docs/recipes/models.md` for measured full-suite scores):

| Dataset | Source | Default small limit |
|---------|--------|---------------------|
| `aime25` | `math-ai/aime25` | `LIMIT_AIME=10` |
| `gpqa_diamond` | `Idavidrein/gpqa` / `gpqa_diamond` | `LIMIT_GPQA=20` |

```bash
PORT=31911 bash scripts/ascend_qwen38_evalscope_bench.sh
# or PORT=31891 for TokenSpeed
```

Details: `scripts/README_ascend_qwen38.md`.

---

## Export full answers from the server (optional)

To dump full model text from the JSON into markdown, inside the container:

```bash
python - <<'PY'
import json
from pathlib import Path
d = json.load(open("/tmp/qwen38_ascend_dataset_long.json", encoding="utf-8"))
lines = ["# Ascend dataset full Q&A\n"]
for c in d["cases"]:
    asc = (c.get("ascend") or {})
    text = asc.get("text") or ""
    lines.append(f"## {c['id']}\n")
    lines.append(f"- expect_ok: {c.get('ascend_expect_ok')}\n")
    lines.append(f"- error: {c.get('error')}\n\n")
    lines.append("### Model output\n\n```\n" + text + "\n```\n\n")
Path("/tmp/qwen38_qa_full.md").write_text("".join(lines), encoding="utf-8")
print("wrote /tmp/qwen38_qa_full.md")
PY
```

Then `scp` the file back and merge if needed.
