#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
#
# Item 1 — greedy accuracy compare skeleton:
#   HF (or CUDA HF) baseline  vs  Ascend TokenSpeed OpenAI HTTP
#   shared jsonl prompts, same seed, temperature=0
#   metrics: token prefix match rate, exact token match, string match,
#            optional expect_contains accuracy, weak KL placeholder
#
# Follow-ups (NOT this skeleton):
#   - short/long/multi-turn suite expansion beyond the starter jsonl
#   - full-vocab KL (needs top-k / full logprobs both sides)
#   - per-layer full-attn vs GDN activation dumps
#
# Examples:
#   # A) HF baseline only (GPU/CPU machine with weights)
#   python scripts/ascend_qwen38_accuracy_compare.py \
#     --prompts scripts/data/qwen38_accuracy_prompts.jsonl \
#     --hf-model /path/to/Qwen3.8-27B \
#     --out /tmp/acc_hf.json --skip-ascend
#
#   # B) Ascend HTTP vs saved HF baseline (Ascend host, serve already up)
#   python scripts/ascend_qwen38_accuracy_compare.py \
#     --prompts scripts/data/qwen38_accuracy_prompts.jsonl \
#     --baseline /tmp/acc_hf.json \
#     --ascend-url http://127.0.0.1:31891/v1 \
#     --served-model qwen3.8-27b \
#     --tokenizer-model /path/to/Qwen3.8-27B \
#     --out /tmp/acc_compare.json
#
#   # C) Both sides in one process
#   python scripts/ascend_qwen38_accuracy_compare.py \
#     --prompts scripts/data/qwen38_accuracy_prompts.jsonl \
#     --hf-model /path/to/Qwen3.8-27B \
#     --ascend-url http://127.0.0.1:31891/v1 \
#     --served-model qwen3.8-27b \
#     --out /tmp/acc_compare.json

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_THINK_RE = re.compile(
    r"<think>.*?</think>\s*|"
    r".*?</think>\s*",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class PromptCase:
    id: str
    messages: list[dict[str, str]]
    max_tokens: int = 64
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    # Soft accuracy labels for "若干题准确率" (substring / regex on final text).
    expect_contains: list[str] = field(default_factory=list)
    expect_regex: str | None = None


@dataclass
class SideResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    # Per-generated-token logprobs (natural log), aligned with token_ids when present.
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseReport:
    id: str
    tags: list[str]
    prompt_tokens: int | None
    hf: SideResult | None
    ascend: SideResult | None
    token_match_prefix: int | None = None
    token_match_rate: float | None = None
    exact_token_match: bool | None = None
    string_exact_match: bool | None = None
    mean_abs_logprob_diff: float | None = None
    kl_placeholder: float | None = None
    hf_expect_ok: bool | None = None
    ascend_expect_ok: bool | None = None
    error: str | None = None


def strip_thinking(text: str) -> str:
    """Drop Qwen thinking blocks so string / expect checks compare final answers."""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def load_prompts(path: Path) -> list[PromptCase]:
    cases: list[PromptCase] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            try:
                cases.append(
                    PromptCase(
                        id=str(obj["id"]),
                        messages=list(obj["messages"]),
                        max_tokens=int(obj.get("max_tokens", 64)),
                        tags=list(obj.get("tags", [])),
                        notes=str(obj.get("notes", "")),
                        expect_contains=[str(x) for x in obj.get("expect_contains", [])],
                        expect_regex=(
                            str(obj["expect_regex"]) if obj.get("expect_regex") else None
                        ),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no}: missing field {exc}") from exc
    if not cases:
        raise ValueError(f"no prompts in {path}")
    return cases


def check_expect(case: PromptCase, text: str) -> bool | None:
    if not case.expect_contains and not case.expect_regex:
        return None
    final = strip_thinking(text)
    for needle in case.expect_contains:
        if needle not in final:
            return False
    if case.expect_regex and re.search(case.expect_regex, final) is None:
        return False
    return True


def _http_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def ascend_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    seed: int,
    request_logprobs: bool,
) -> SideResult:
    """Call OpenAI-compatible /v1/chat/completions on Ascend TokenSpeed."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
    }
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    raw = _http_json(url, payload)
    choice = raw["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    token_ids: list[int] = []
    logprobs: list[float] = []

    if isinstance(choice.get("token_ids"), list):
        token_ids = [int(x) for x in choice["token_ids"]]
    lp = choice.get("logprobs")
    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
        for item in lp["content"]:
            if isinstance(item, dict) and item.get("logprob") is not None:
                logprobs.append(float(item["logprob"]))

    return SideResult(
        text=text,
        token_ids=token_ids,
        logprobs=logprobs,
        finish_reason=choice.get("finish_reason"),
        raw=raw,
    )


def hf_chat(
    model_path: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    seed: int,
    device: str,
    dtype: str,
) -> SideResult:
    """Greedy HF generate; used as the reference baseline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    cache = getattr(hf_chat, "_cache", None)
    if cache is None or cache[0] != model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True,
        )
        if device not in {"auto", "cpu"} and hasattr(model, "to"):
            model = model.to(device)
        model.eval()
        hf_chat._cache = (model_path, model, tokenizer)  # type: ignore[attr-defined]
    else:
        _, model, tokenizer = cache

    # Prefer disabling thinking for closer parity with non-reasoning smoke serves.
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seq = out.sequences[0]
    prompt_len = int(inputs["input_ids"].shape[-1])
    gen_ids = seq[prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    logprobs: list[float] = []
    if out.scores:
        for step_scores, token_id in zip(out.scores, gen_ids, strict=False):
            log_softmax = torch.log_softmax(step_scores[0].float(), dim=-1)
            logprobs.append(float(log_softmax[token_id].item()))

    return SideResult(
        text=text,
        token_ids=[int(x) for x in gen_ids],
        logprobs=logprobs,
        finish_reason="stop",
        raw={"prompt_len": prompt_len, "backend": "hf"},
    )


def tokenize_text(model_path: str, text: str) -> list[int]:
    from transformers import AutoTokenizer

    cache = getattr(tokenize_text, "_tok", None)
    if cache is None or cache[0] != model_path:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenize_text._tok = (model_path, tok)  # type: ignore[attr-defined]
    else:
        _, tok = cache
    return tok.encode(text, add_special_tokens=False)


def prefix_match(a: list[int], b: list[int]) -> tuple[int, float]:
    n = min(len(a), len(b))
    matched = 0
    for i in range(n):
        if a[i] != b[i]:
            break
        matched += 1
    denom = max(len(a), len(b), 1)
    return matched, matched / denom


def mean_abs_diff(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n == 0:
        return None
    return sum(abs(a[i] - b[i]) for i in range(n)) / n


def kl_from_chosen_logprobs(p: list[float], q: list[float]) -> float | None:
    """Weak KL diagnostic from chosen-token logprobs only (not full vocab).

    True step KL needs full distributions / aligned top-k. Until both stacks
    expose that, report mean ``exp(lp_p) * (lp_p - lp_q)`` on the overlap.
    """
    n = min(len(p), len(q))
    if n == 0:
        return None
    total = 0.0
    for i in range(n):
        total += math.exp(p[i]) * (p[i] - q[i])
    return total / n


def load_baseline(path: Path) -> dict[str, SideResult]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, SideResult] = {}
    for case in obj.get("cases", []):
        hf = case.get("hf")
        if not hf:
            continue
        out[case["id"]] = SideResult(
            text=hf.get("text", ""),
            token_ids=list(hf.get("token_ids", [])),
            logprobs=list(hf.get("logprobs", [])),
            finish_reason=hf.get("finish_reason"),
            raw=dict(hf.get("raw", {})),
        )
    return out


def compare_case(
    case: PromptCase,
    hf: SideResult | None,
    ascend: SideResult | None,
    *,
    tokenizer_model: str | None,
) -> CaseReport:
    report = CaseReport(
        id=case.id,
        tags=list(case.tags),
        prompt_tokens=None,
        hf=hf,
        ascend=ascend,
    )
    if hf is not None:
        report.hf_expect_ok = check_expect(case, hf.text)
    if ascend is not None:
        report.ascend_expect_ok = check_expect(case, ascend.text)

    if hf is None or ascend is None:
        report.error = "missing hf or ascend side"
        return report

    if not ascend.token_ids and tokenizer_model and ascend.text:
        try:
            # Compare on final answer text when possible (thinking stripped).
            ascend.token_ids = tokenize_text(
                tokenizer_model, strip_thinking(ascend.text)
            )
            if hf.token_ids and hf.text:
                # Re-tokenize HF final text for a fairer string-derived id compare
                # when Ascend never returned token ids.
                hf_final_ids = tokenize_text(tokenizer_model, strip_thinking(hf.text))
                if hf_final_ids:
                    hf.token_ids = hf_final_ids
        except Exception as exc:  # noqa: BLE001
            report.error = f"retokenize failed: {exc}"

    if hf.token_ids and ascend.token_ids:
        matched, rate = prefix_match(hf.token_ids, ascend.token_ids)
        report.token_match_prefix = matched
        report.token_match_rate = rate
        report.exact_token_match = hf.token_ids == ascend.token_ids
    report.string_exact_match = strip_thinking(hf.text) == strip_thinking(ascend.text)
    report.mean_abs_logprob_diff = mean_abs_diff(hf.logprobs, ascend.logprobs)
    report.kl_placeholder = kl_from_chosen_logprobs(hf.logprobs, ascend.logprobs)
    return report


def summarize(reports: list[CaseReport]) -> dict[str, Any]:
    rates = [r.token_match_rate for r in reports if r.token_match_rate is not None]
    exact = [r.exact_token_match for r in reports if r.exact_token_match is not None]
    by_tag: dict[str, list[float]] = {}
    for r in reports:
        if r.token_match_rate is None:
            continue
        for tag in r.tags or ["untagged"]:
            by_tag.setdefault(tag, []).append(r.token_match_rate)

    def _acc(side: str) -> dict[str, Any]:
        vals = [
            getattr(r, f"{side}_expect_ok")
            for r in reports
            if getattr(r, f"{side}_expect_ok") is not None
        ]
        if not vals:
            return {"num_scored": 0, "accuracy": None}
        return {
            "num_scored": len(vals),
            "accuracy": sum(1 for x in vals if x) / len(vals),
        }

    return {
        "num_cases": len(reports),
        "num_with_token_rate": len(rates),
        "mean_token_match_rate": (sum(rates) / len(rates)) if rates else None,
        "exact_token_match_frac": (sum(1 for x in exact if x) / len(exact))
        if exact
        else None,
        "mean_token_match_rate_by_tag": {
            tag: (sum(vals) / len(vals)) for tag, vals in sorted(by_tag.items())
        },
        "hf_expect_accuracy": _acc("hf"),
        "ascend_expect_accuracy": _acc("ascend"),
        "errors": [r.id for r in reports if r.error],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompts", type=Path, required=True, help="jsonl prompt suite")
    p.add_argument("--out", type=Path, required=True, help="write full JSON report")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hf-model", type=str, default=None, help="HF checkpoint for baseline")
    p.add_argument("--hf-device", type=str, default="auto")
    p.add_argument(
        "--hf-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument("--baseline", type=Path, default=None, help="reuse prior HF results JSON")
    p.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="HF tokenizer path for retokenizing Ascend text (defaults to --hf-model)",
    )
    p.add_argument("--ascend-url", type=str, default=None, help="e.g. http://127.0.0.1:31891/v1")
    p.add_argument("--served-model", type=str, default="qwen3.8-27b")
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--skip-ascend", action="store_true")
    p.add_argument(
        "--request-logprobs",
        action="store_true",
        help="ask Ascend for logprobs if supported",
    )
    p.add_argument("--limit", type=int, default=0, help="optional first-N cases")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_prompts(args.prompts)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    baseline = load_baseline(args.baseline) if args.baseline else {}
    do_hf = bool(not args.skip_hf and args.hf_model)
    do_ascend = bool(not args.skip_ascend and args.ascend_url)
    tokenizer_model = args.tokenizer_model or args.hf_model

    if not do_hf and not baseline and not do_ascend:
        print(
            "nothing to do: provide --hf-model and/or --ascend-url (or --baseline)",
            file=sys.stderr,
        )
        return 2

    reports: list[CaseReport] = []
    for case in cases:
        print(f"== case {case.id} tags={case.tags} ==")
        hf_res = baseline.get(case.id)
        ascend_res = None
        err: str | None = None
        try:
            if do_hf:
                assert args.hf_model
                hf_res = hf_chat(
                    args.hf_model,
                    case.messages,
                    case.max_tokens,
                    seed=args.seed,
                    device=args.hf_device,
                    dtype=args.hf_dtype,
                )
                print(f"  hf tokens={len(hf_res.token_ids)} text={hf_res.text[:80]!r}")
            if do_ascend:
                assert args.ascend_url
                ascend_res = ascend_chat(
                    args.ascend_url,
                    args.served_model,
                    case.messages,
                    case.max_tokens,
                    seed=args.seed,
                    request_logprobs=args.request_logprobs,
                )
                print(f"  ascend text={ascend_res.text[:80]!r}")
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
            err = str(exc)
            print(f"  ERROR {err}", file=sys.stderr)

        if hf_res is not None and ascend_res is not None:
            report = compare_case(
                case,
                hf_res,
                ascend_res,
                tokenizer_model=tokenizer_model,
            )
        else:
            report = CaseReport(
                id=case.id,
                tags=list(case.tags),
                prompt_tokens=None,
                hf=hf_res,
                ascend=ascend_res,
                hf_expect_ok=check_expect(case, hf_res.text) if hf_res else None,
                ascend_expect_ok=check_expect(case, ascend_res.text)
                if ascend_res
                else None,
            )
        if err:
            report.error = err
        if report.token_match_rate is not None:
            print(
                f"  match_rate={report.token_match_rate:.3f} "
                f"exact={report.exact_token_match} "
                f"kl_placeholder={report.kl_placeholder} "
                f"ascend_expect={report.ascend_expect_ok}"
            )
        reports.append(report)

    payload = {
        "meta": {
            "prompts": str(args.prompts),
            "seed": args.seed,
            "hf_model": args.hf_model,
            "ascend_url": args.ascend_url,
            "served_model": args.served_model,
            "item": 1,
            "notes": (
                "Item-1 skeleton: greedy HF vs Ascend HTTP. "
                "Full-vocab KL and per-layer GDN/full-attn dumps are TODO (items 2–3)."
            ),
        },
        "summary": summarize(reports),
        "cases": [
            {
                "id": r.id,
                "tags": r.tags,
                "prompt_tokens": r.prompt_tokens,
                "token_match_prefix": r.token_match_prefix,
                "token_match_rate": r.token_match_rate,
                "exact_token_match": r.exact_token_match,
                "string_exact_match": r.string_exact_match,
                "mean_abs_logprob_diff": r.mean_abs_logprob_diff,
                "kl_placeholder": r.kl_placeholder,
                "hf_expect_ok": r.hf_expect_ok,
                "ascend_expect_ok": r.ascend_expect_ok,
                "error": r.error,
                "hf": asdict(r.hf) if r.hf else None,
                "ascend": asdict(r.ascend) if r.ascend else None,
            }
            for r in reports
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if not payload["summary"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
