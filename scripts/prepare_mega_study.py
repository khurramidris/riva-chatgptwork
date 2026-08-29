#!/usr/bin/env python3
"""Download, stage, and audit the frozen Mega-Study pilot without API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rival.mega_study.audit import audit_readiness
from rival.mega_study.stage import prepare_stage
from rival.mega_study.utils import MegaStudyError, atomic_json


DEFAULT_STAGE = REPOSITORY_ROOT / ".rival-data" / "mega-study-v1"
DEFAULT_CACHE = REPOSITORY_ROOT / ".rival-data" / "mega-study-source"
DEFAULT_REPORT = REPOSITORY_ROOT / "reports" / "mega_study" / "preparation_audit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="require every checksum-pinned source to exist in the cache",
    )
    args = parser.parse_args(argv)

    def progress(event: dict[str, object]) -> None:
        if event.get("event") == "download":
            print(f"Downloading and verifying {event['source_id']}...", flush=True)

    try:
        stage = prepare_stage(
            args.stage_root,
            source_cache=args.source_cache,
            allow_download=not args.no_download,
            progress=progress,
        )
        print("Rendering and auditing all 1,200 frozen prompts...", flush=True)
        audit = audit_readiness(args.stage_root, args.source_cache)
        payload = {"stage": stage, "readiness_audit": audit}
        atomic_json(args.report, payload)
    except MegaStudyError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summary = {
        "status": "PASS",
        "real_api_calls_made": 0,
        "stage_root": str(args.stage_root.resolve()),
        "source_cache": str(args.source_cache.resolve()),
        "report": str(args.report.resolve()),
        "protocol_sha256": stage["protocol_sha256"],
        "stage_sha256": stage["stage_sha256"],
        "cases": audit["cases"],
        "planned_calls": audit["planned_calls"],
        "official_replication": audit["official_replication"]["status"],
        "leakage_firewall": audit["leakage_firewall"]["status"],
        "protected_outcomes_materialized": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
