# Ascend Qwen3.8-27B 小数据集问答整理

- **服务**：TokenSpeed Ascend · `http://127.0.0.1:31891/v1` · 模型 `qwen3.8-27b`
- **任务集**：`scripts/data/qwen38_accuracy_prompts.jsonl`（8 题）
- **实测轮次**：加长生成 `max_tokens≥256`（`/tmp/qwen38_prompts_eval.jsonl`）
- **结果文件**：`/tmp/qwen38_ascend_dataset_long.json`
- **汇总**：计分 7/7 · `ascend_expect_accuracy = 1.0` · `errors = []`
- **说明**：回复常含 thinking，再出现 `</think>` 与最终答案；下列「模型回答」取自实测日志（部分为前缀摘要）。无 HF greedy 对齐数字。

---

## 1. short_zh_hello（短聊，不计分）

**标签**：`short`, `chat`  
**期望**：无（仅冒烟）  
**判分**：—

**问题**

> 你好，用一句话介绍你自己

**模型回答（摘要）**

> （thinking）用户要求用一句话介绍自己。我需要用中文回复，简洁地介绍我是通义千问，由阿里巴巴通义实验室开发的大语言模型。  
> `</think>`  
> 你好！我是通义千问（Qwen），…

---

## 2. short_math（短题 · 准确率）

**标签**：`short`, `accuracy`  
**期望包含**：`2`  
**判分**：True

**问题**

> 1+1等于几？只回答数字

**模型回答（摘要）**

> （thinking）We need answer user's request: "1+1等于几？只回答数字" means "What is 1+1? Only answer nu…  
> （完整输出中含最终数字 `2`，故 expect 通过）

---

## 3. short_en_fact（短题 · 准确率）

**标签**：`short`, `accuracy`  
**期望包含**：`Paris`  
**判分**：True

**问题**

> What is the capital of France? Reply with one word only.

**模型回答（摘要）**

> We need answer one word only. User asks capital of France. Answer: Paris. Ensure…

---

## 4. short_arith（短题 · 准确率）

**标签**：`short`, `accuracy`  
**期望包含**：`10`  
**判分**：True

**问题**

> 若 x=3, y=x\*x+1，y 等于多少？只回答数字。

**模型回答（摘要）**

> We need answer in Chinese? User: 若 x=3, y=x\*x+1，y 等于多少？只回答数字。 Need only number. …  
> （完整输出中含 `10`）

---

## 5. long_repeat_summary（长 prefill）

**标签**：`long`, `prefill`  
**期望包含**：`正确`  
**判分**：True

**问题**

> 请用不超过两句话总结下面这段话的中心思想。  
> （「人工智能推理需要正确性优先。」重复多遍）

**模型回答（摘要）**

> （thinking）用户要求用不超过两句话总结给定文本的中心思想。文本重复同一句：「人工智能推理需要正确性优先。」需要简洁中文回答，不超过两句。可以直接说：中心思想是强调人工智能…

---

## 6. multi_turn_basic（多轮 · 准确率）

**标签**：`multi_turn`, `chat`, `accuracy`  
**期望包含**：`小明`  
**判分**：True

**对话**

| role | content |
|------|---------|
| user | 我叫小明。 |
| assistant | 好的，小明你好。 |
| user | 我叫什么名字？只回答名字。 |

**模型回答**

> （thinking）用户问「我叫什么名字？只回答名字。」前文用户说「我叫小明。」因此应只回答名字：小明。需要严格只输出名字，不加标点或解释。  
> `</think>`  
> **小明**

---

## 7. layer_mix_code（混合层 · 代码）

**标签**：`short`, `full_attn_gdn_mix`  
**期望包含**：`return`  
**判分**：True

**问题**

> 用 Python 写一个函数 add(a, b) 返回两数之和，只要函数体。

**模型回答（摘要）**

> We need answer user in Chinese likely. User: "用 Python 写一个函数 add(a, b) 返回两数之和，只要…  
> （完整输出中含 `return`）

---

## 8. layer_mix_reason（混合层 · 推理）

**标签**：`short`, `full_attn_gdn_mix`, `accuracy`  
**期望包含**：`10`  
**判分**：True

**问题**

> 3 的平方加 1 等于几？只回答数字。

**模型回答（摘要）**

> We need answer user: "3 的平方加 1 等于几？只回答数字。" Chinese: 3 squared plus 1 equals what…  
> （完整输出中含 `10`）

---

## 汇总表

| id | 问题（缩略） | 期望 | 结果 |
|----|--------------|------|------|
| short_zh_hello | 一句话介绍自己 | — | 有正常中文回复 |
| short_math | 1+1=? | `2` | True |
| short_en_fact | capital of France | `Paris` | True |
| short_arith | x=3, y=x²+1 | `10` | True |
| long_repeat_summary | 总结「正确性优先」 | `正确` | True |
| multi_turn_basic | 我叫什么名字 | `小明` | True |
| layer_mix_code | Python add | `return` | True |
| layer_mix_reason | 3²+1=? | `10` | True |

**结论**：加长生成后，7 道带 `expect_contains` 的题全部命中；适合作为 Ascend 适配的小数据集功能实测记录。正式与 HF 的 token 级精度需另跑 baseline 对比。

---

## 标准 benchmark（AIME / GPQA Diamond）

与仓库 CI 相同，走 **EvalScope**：

| 数据集 | 来源 | 小跑默认 |
|--------|------|----------|
| `aime25` | `math-ai/aime25` | `LIMIT_AIME=10` |
| `gpqa_diamond` | `Idavidrein/gpqa` / `gpqa_diamond` | `LIMIT_GPQA=20` |

```bash
PORT=31891 bash scripts/ascend_qwen38_evalscope_bench.sh
```

详见 `scripts/README_ascend_qwen38.md`。

---

## 从服务器导出完整问答（可选）

若需要把 JSON 里的**全文**答案写进文档，在容器中执行：

```bash
python - <<'PY'
import json
from pathlib import Path
d = json.load(open("/tmp/qwen38_ascend_dataset_long.json", encoding="utf-8"))
lines = ["# Ascend 数据集完整问答\n"]
for c in d["cases"]:
    asc = (c.get("ascend") or {})
    text = asc.get("text") or ""
    lines.append(f"## {c['id']}\n")
    lines.append(f"- expect_ok: {c.get('ascend_expect_ok')}\n")
    lines.append(f"- error: {c.get('error')}\n\n")
    lines.append("### 模型全文\n\n```\n" + text + "\n```\n\n")
Path("/tmp/qwen38_qa_full.md").write_text("".join(lines), encoding="utf-8")
print("wrote /tmp/qwen38_qa_full.md")
PY
```

再 `scp` 回本机合并即可。
