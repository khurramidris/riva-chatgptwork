#!/usr/bin/env python3
"""Audit a partial Mega-Study ledger without opening or comparing outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rival.mega_study.checkpoint import audit_checkpoint
from rival.mega_study.utils import MegaStudyError, atomic_json


DEFAULT_STAGE = REPOSITORY_ROOT / ".rival-data" / "mega-study-v1"
DEFAULT_RESULTS = REPOSITORY_ROOT / "reports" / "mega_study" / "results.jsonl"
DEFAULT_REPORT = (
    REPOSITORY_ROOT / "reports" / "mega_study" / "checkpoint_audit.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expect-terminal", type=int)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = audit_checkpoint(
            args.stage_root,
            args.results,
            expected_terminal=args.expect_terminal,
            budget_usd=args.budget_usd,
            max_failures=args.max_failures,
        )
        atomic_json(args.report, report)
    except (MegaStudyError, OSError, ValueError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summary = {
        "status": report["status"],
        "verification_status": report["verification_status"],
        "terminal_work_items": report["ledger"]["terminal_work_items"],
        "successful_work_items": report["ledger"]["successful_work_items"],
        "failure_work_items": report["ledger"]["failure_work_items"],
        "remaining_work_items": report["ledger"]["remaining_work_items"],
        "spent_usd": report["operations"]["spent_usd"],
        "scientific_accuracy_status": report["scientific_accuracy_status"],
        "report": str(args.report.resolve()),
        "audit_sha256": report["audit_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
