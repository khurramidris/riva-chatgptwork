#!/usr/bin/env python3
"""Run or resume the frozen 1,500-case Rival pilot in safe checkpoints."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rival.live_pilot import BudgetGuard, make_openai_provider, run_live_pilot


DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "rival" / "studies" / "twin2k_live_v2" / "protocol.json"
)
CHECKPOINTS = (300, 600, 900, 1200, 1500)


def _safe_error(value: object, api_key: str) -> str:
    text = " ".join(str(value).split())
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(r"(?i)\b(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:1_000]


def _checkpoint_status(summary: dict[str, object], target: int) -> str:
    successes = int(summary["successful_cases"])
    errors = int(summary["errors_this_run"])
    if errors:
        return "PILOT_PAUSED_PROVIDER_ERROR"
    if successes < target:
        return "PILOT_CHECKPOINT_INCOMPLETE"
    if target == 1500 and summary["status"] == "COMPLETE":
        return "PILOT_COMPLETE"
    return "PILOT_CHECKPOINT_COMPLETE"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, choices=CHECKPOINTS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--input-cost-per-million", type=float, required=True)
    parser.add_argument("--output-cost-per-million", type=float, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--results", type=Path, default=Path("reports/live_pilot_v2_results.jsonl")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/live_pilot_v2_pilot_summary.json")
    )
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--expiry-minutes", type=int, default=120)
    args = parser.parse_args(argv)

    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    if args.input_cost_per_million < 0 or args.output_cost_per_million < 0:
        parser.error("model prices must be nonnegative")
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

    def progress(event: dict[str, object]) -> None:
        if event["event"] == "error":
            print(
                f"PAUSED after {event['successful_cases']} successes: "
                f"{event['error_type']}",
                flush=True,
            )
            return
        new_successes = int(event["new_successes"])
        successful_cases = int(event["successful_cases"])
        if new_successes % 10 == 0 or successful_cases == args.target_total:
            print(
                f"Progress: {successful_cases}/{args.target_total} successes; "
                f"spent ${float(event['spent_usd']):.6f}",
                flush=True,
            )

    prior_rival_key = os.environ.get("RIVAL_API_KEY")
    os.environ["RIVAL_API_KEY"] = api_key
    try:
        # These transport settings intentionally match the completed preflight's
        # provider identity so its 30 successful rows remain resumable.
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
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
            max_calls=args.target_total,
            not_after=datetime.now(timezone.utc)
            + timedelta(minutes=args.expiry_minutes),
        )
        summary = run_live_pilot(
            args.protocol,
            args.results,
            provider,
            guard,
            phase="pilot",
            max_errors=1,
            summary_path=args.summary,
            progress_callback=progress,
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

    verification_status = _checkpoint_status(summary, args.target_total)
    summary["verification_status"] = verification_status
    summary["checkpoint_target"] = args.target_total
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verification_status in {
        "PILOT_CHECKPOINT_COMPLETE",
        "PILOT_COMPLETE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
