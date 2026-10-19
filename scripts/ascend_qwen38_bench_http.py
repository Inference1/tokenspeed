#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
"""Non-stream HTTP latency / tok-s probe for Ascend serve (not accuracy).

Example:
  python scripts/ascend_qwen38_bench_http.py \\
    --url http://127.0.0.1:31891/v1 --model qwen3.8-27b \\
    --max-tokens 128 --warmup 1 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def post_chat(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool | None,
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Best-effort; servers that ignore unknown fields still work.
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return time.perf_counter() - t0, body


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:31891/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument(
        "--prompt",
        default="请用一句话介绍人工智能。不要输出思考过程，只给最终答案。",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--enable-thinking",
        choices=("default", "true", "false"),
        default="false",
        help="try chat_template_kwargs.enable_thinking (default: false)",
    )
    args = p.parse_args(argv)

    thinking: bool | None
    if args.enable_thinking == "default":
        thinking = None
    else:
        thinking = args.enable_thinking == "true"

    print(
        f"url={args.url} model={args.model} max_tokens={args.max_tokens} "
        f"warmup={args.warmup} rounds={args.rounds} enable_thinking={args.enable_thinking}"
    )

    try:
        for i in range(max(args.warmup, 0)):
            dt, body = post_chat(
                args.url,
                args.model,
                args.prompt,
                args.max_tokens,
                args.temperature,
                thinking,
                args.timeout,
            )
            usage = body.get("usage") or {}
            print(
                f"warmup {i + 1}: {dt:.3f}s "
                f"completion_tokens={usage.get('completion_tokens')}"
            )

        rows: list[tuple[float, int, int, float]] = []
        for i in range(max(args.rounds, 1)):
            dt, body = post_chat(
                args.url,
                args.model,
                args.prompt,
                args.max_tokens,
                args.temperature,
                thinking,
                args.timeout,
            )
            usage = body.get("usage") or {}
            pin = int(usage.get("prompt_tokens") or 0)
            pout = int(usage.get("completion_tokens") or 0)
            tps = (pout / dt) if dt > 0 else 0.0
            text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            rows.append((dt, pin, pout, tps))
            print(
                f"round {i + 1}: latency={dt:.3f}s prompt={pin} "
                f"completion={pout} ~{tps:.1f} tok/s text[:80]={text[:80]!r}"
            )

        lats = [r[0] for r in rows]
        tpss = [r[3] for r in rows]
        print(
            "\nSUMMARY "
            f"avg_latency={statistics.mean(lats):.3f}s "
            f"p50_latency={statistics.median(lats):.3f}s "
            f"avg_tok_s={statistics.mean(tpss):.1f} "
            f"(e2e non-stream; not true TTFT)"
        )
        return 0
    except urllib.error.URLError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
