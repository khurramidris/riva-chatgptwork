#!/usr/bin/env python3
"""Freeze all Mega-Study predictions before any human outcome is opened."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rival.mega_study.runner import freeze_predictions
from rival.mega_study.utils import MegaStudyError


DEFAULT_STAGE = REPOSITORY_ROOT / ".rival-data" / "mega-study-v1"
DEFAULT_RESULTS = REPOSITORY_ROOT / "reports" / "mega_study" / "results.jsonl"
DEFAULT_MARKER = REPOSITORY_ROOT / "reports" / "mega_study" / "prediction_freeze.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    args = parser.parse_args(argv)
    try:
        marker = freeze_predictions(args.stage_root, args.results, args.marker)
    except MegaStudyError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(marker, indent=2, sort_keys=True))
    print("Human outcomes remain unopened. The prediction ledger is now immutable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
