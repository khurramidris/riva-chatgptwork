#!/usr/bin/env python3
"""Run or resume the frozen Mega-Study A/B/C/D benchmark with hard guards."""

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

from rival.mega_study.constants import MODEL_CONFIG
from rival.mega_study.runner import run_benchmark


DEFAULT_STAGE = REPOSITORY_ROOT / ".rival-data" / "mega-study-v1"
DEFAULT_RESULTS = REPOSITORY_ROOT / "reports" / "mega_study" / "results.jsonl"
DEFAULT_SUMMARY = REPOSITORY_ROOT / "reports" / "mega_study" / "run_summary.json"


def _safe_error(value: object, api_key: str) -> str:
    detail = " ".join(str(value).split())
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    detail = re.sub(r"(?i)\b(bearer\s+)[^\s\"']+", r"\1[REDACTED]", detail)
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", detail)
    return detail[:1000]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "pilot"), required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--expiry-minutes", type=int, required=True)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--max-errors", type=int, default=1)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    if args.expiry_minutes < 1:
        parser.error("--expiry-minutes must be positive")
    if args.max_new_calls is not None and args.max_new_calls < 1:
        parser.error("--max-new-calls must be positive")
    if args.max_errors < 1:
        parser.error("--max-errors must be positive")
    if args.phase == "preflight" and args.max_new_calls not in (None, 4):
        parser.error("preflight is exactly one participant-study case × four variants")

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

    not_after = datetime.now(timezone.utc) + timedelta(minutes=args.expiry_minutes)

    def progress(event: dict[str, object]) -> None:
        new_calls = int(event["new_calls"])
        if args.phase == "preflight" or new_calls % 10 == 0:
            print(
                f"Progress: {event['terminal_work']}/{1200 if args.phase == 'pilot' else 4} "
                f"terminal; ${float(event['spent_usd']):.6f} spent; "
                f"latest={event['status']}",
                flush=True,
            )

    print(
        "Frozen route: "
        f"{MODEL_CONFIG['api_provider']} / {MODEL_CONFIG['upstream_provider']} / "
        f"{MODEL_CONFIG['model']}",
        flush=True,
    )
    try:
        summary = run_benchmark(
            args.stage_root,
            args.results,
            api_key=api_key,
            phase=args.phase,
            budget_usd=args.budget_usd,
            not_after=not_after.isoformat(),
            max_new_calls=(4 if args.phase == "preflight" else args.max_new_calls),
            max_errors=args.max_errors,
            summary_path=args.summary,
            progress=progress,
        )
    except Exception as exc:
        print(
            f"FAIL: {type(exc).__name__}: {_safe_error(exc, api_key)}",
            file=sys.stderr,
        )
        return 2
    finally:
        api_key = ""

    if args.phase == "preflight":
        verification = (
            "PREFLIGHT_PASS"
            if summary["status"] == "COMPLETE"
            and int(summary["successful_work_items"]) == 4
            else "PREFLIGHT_FAIL"
        )
    elif summary["status"] == "COMPLETE":
        verification = "PILOT_PREDICTIONS_COMPLETE"
    elif (
        args.max_new_calls is not None
        and int(summary["new_calls"]) == args.max_new_calls
        and int(summary["errors_this_run"]) == 0
    ):
        verification = "PILOT_CHECKPOINT_COMPLETE"
    else:
        verification = "PILOT_PAUSED"
    summary["verification_status"] = verification
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verification in {
        "PREFLIGHT_PASS",
        "PILOT_PREDICTIONS_COMPLETE",
        "PILOT_CHECKPOINT_COMPLETE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
