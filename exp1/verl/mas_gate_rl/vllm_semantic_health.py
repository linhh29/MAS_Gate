#!/usr/bin/env python3
"""Fail-fast deterministic semantic health check for local Qwen vLLM servers."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def complete(url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, str | None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    choice = body["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", default="vllm")
    parser.add_argument("--stage", default="startup")
    args = parser.parse_args()

    checks = [
        ("exact_health", "Reply with exactly: HEALTHY", "HEALTHY", 32),
        ("arithmetic", "What is 2+2? Reply with exactly: 4", "4", 16),
        ("json", 'Return exactly this JSON object with no markdown: {"status":"ok","value":7}', '{"status":"ok","value":7}', 48),
    ]
    report = {"role": args.role, "stage": args.stage, "checks": []}
    passed = True
    for name, prompt, expected, max_tokens in checks:
        try:
            content, finish_reason = complete(args.base_url, args.model, prompt, max_tokens)
            ok = content == expected and finish_reason == "stop"
            report["checks"].append({
                "name": name, "expected": expected, "content": content,
                "finish_reason": finish_reason, "passed": ok,
            })
            passed &= ok
        except Exception as exc:  # fail closed: rollout must never consume an unverified server
            report["checks"].append({"name": name, "passed": False, "error": repr(exc)})
            passed = False
    report["passed"] = passed
    print(json.dumps(report, ensure_ascii=False))
    if not passed:
        print(f"[mas_gate_rl] FATAL: semantic health failed for {args.role} at {args.stage}", file=sys.stderr)
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
