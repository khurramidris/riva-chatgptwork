#!/usr/bin/env python3
"""Run one frozen Rival case or the complete 30-call preflight securely.

The API key is read from the process environment or requested with hidden
input. It is never accepted as a command-line argument and is removed from the
environment before this process exits. This launcher never evaluates outcomes.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rival.live_pilot import BudgetGuard, make_openai_provider, run_live_pilot


DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "rival" / "studies" / "twin2k_live_v2" / "protocol.json"
)


def _safe_error(value: object, api_key: str) -> str:
    text = " ".join(str(value).split())
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)\b(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:1_000]


def _expected_success(summary: dict[str, object], max_calls: int) -> bool:
    target = min(max_calls, int(summary["selected_cases"]))
    return (
        int(summary["successful_cases"]) >= target
        and int(summary["errors_this_run"]) == 0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-calls",
        type=int,
        choices=(1, 30),
        default=1,
        help="1 validates one frozen case; 30 completes/resumes the preflight",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--results", type=Path, default=Path("reports/live_pilot_v2_results.jsonl")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/live_pilot_v2_summary.json")
    )
    parser.add_argument("--budget-usd", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--expiry-minutes", type=int, default=120)
    args = parser.parse_args(argv)

    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    if args.expiry_minutes < 1:
        parser.error("--expiry-minutes must be positive")

    api_key = os.getenv("RIVAL_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            api_key = getpass.getpass("OpenRouter API key (input hidden): ")
        except (EOFError, KeyboardInterrupt):
            print("\nFAIL: no API key supplied", file=sys.stderr)
            return 2
    if not api_key:
        print("FAIL: no API key supplied", file=sys.stderr)
        return 2

    prior_rival_key = os.environ.get("RIVAL_API_KEY")
    os.environ["RIVAL_API_KEY"] = api_key
    try:
        provider = make_openai_provider(
            model=args.model,
            base_url=None,
            timeout_seconds=args.timeout_seconds,
            temperature=0.0,
            max_retries=1,
            history_limit=16,
            max_output_tokens=300,
            use_response_format=True,
        )
        guard = BudgetGuard(
            budget_usd=args.budget_usd,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            max_calls=args.max_calls,
            not_after=datetime.now(timezone.utc)
            + timedelta(minutes=args.expiry_minutes),
        )
        summary = run_live_pilot(
            args.protocol,
            args.results,
            provider,
            guard,
            phase="preflight",
            max_errors=1,
            summary_path=args.summary,
        )
    except Exception as exc:
        print(
            f"FAIL: {type(exc).__name__}: {_safe_error(exc, api_key)}",
            file=sys.stderr,
        )
        return 2
    finally:
        if prior_rival_key is None:
            os.environ.pop("RIVAL_API_KEY", None)
        else:
            os.environ["RIVAL_API_KEY"] = prior_rival_key
        api_key = ""

    passed = _expected_success(summary, args.max_calls)
    summary["verification_status"] = (
        "ONE_FROZEN_CASE_PASS"
        if passed and args.max_calls == 1
        else "PREFLIGHT_COMPLETE"
        if passed
        else "FAIL"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
