#!/usr/bin/env python3
"""Open frozen Mega-Study outcomes and produce the complete baseline report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rival.mega_study.evaluation import evaluate_benchmark, markdown_report
from rival.mega_study.stage import materialize_outcomes
from rival.mega_study.utils import MegaStudyError, atomic_json


DEFAULT_STAGE = REPOSITORY_ROOT / ".rival-data" / "mega-study-v1"
DEFAULT_CACHE = REPOSITORY_ROOT / ".rival-data" / "mega-study-source"
DEFAULT_RESULTS = REPOSITORY_ROOT / "reports" / "mega_study" / "results.jsonl"
DEFAULT_MARKER = REPOSITORY_ROOT / "reports" / "mega_study" / "prediction_freeze.json"
DEFAULT_JSON = REPOSITORY_ROOT / "reports" / "mega_study" / "baseline_report.json"
DEFAULT_MARKDOWN = REPOSITORY_ROOT / "reports" / "mega_study" / "baseline_report.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--freeze-marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    def progress(event: dict[str, object]) -> None:
        if event.get("event") == "download":
            print(f"Downloading and verifying {event['source_id']}...", flush=True)

    try:
        sealed = materialize_outcomes(
            args.stage_root,
            args.results,
            args.freeze_marker,
            source_cache=args.source_cache,
            allow_download=not args.no_download,
            progress=progress,
        )
        report = evaluate_benchmark(
            args.stage_root,
            args.results,
            args.freeze_marker,
        )
        atomic_json(args.json_report, report)
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(
            markdown_report(report), encoding="utf-8", newline="\n"
        )
    except MegaStudyError as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summary = {
        "status": report["status"],
        "development_only": True,
        "confirmation_claim_allowed": False,
        "outcome_manifest_sha256": sealed["outcome_manifest_sha256"],
        "json_report": str(args.json_report.resolve()),
        "markdown_report": str(args.markdown_report.resolve()),
        "summary_table": report["summary_table"],
        "primary_rival_contrasts": {
            key: report["paired_lifts"][key]["macro"]
            for key in report["primary_rival_contrasts"]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
