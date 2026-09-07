"""Explicit v2 operator commands. Available in verified source and wheel installations."""

import argparse
import json
import sys
from pathlib import Path

from ..execution import ExecutionError, AttemptJournal
from ..mega_study.utils import MegaStudyError, atomic_json, canonical_hash, read_jsonl, file_hash
from .freeze import freeze_predictions
from . import RUNTIME_REVISION
from .evaluation import evaluate_benchmark, markdown_report
from .runner import run_benchmark, exclusive_run_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-resources", help="verify archived v1 witnesses and the current resource layout")
    prepare = commands.add_parser("prepare", help="stage answer-free inputs using verified source files")
    prepare.add_argument("--stage-root", type=Path, required=True)
    prepare.add_argument("--source-cache", type=Path, required=True)
    prepare.add_argument("--no-download", action="store_true")
    reveal = commands.add_parser("materialize-outcomes", help="open outcomes only after prediction freeze")
    reveal.add_argument("--stage-root", type=Path, required=True)
    reveal.add_argument("--results", type=Path, required=True)
    reveal.add_argument("--freeze-marker", type=Path, required=True)
    reveal.add_argument("--source-cache", type=Path, required=True)
    reveal.add_argument("--no-download", action="store_true")
    run = commands.add_parser("run", help="new v2 execution; requires a separate results path")
    run.add_argument("--stage-root", type=Path, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--phase", choices=["preflight", "pilot"], default="preflight")
    run.add_argument("--budget-usd", type=float, required=True)
    run.add_argument("--not-after", required=True, help="ISO-8601 authorization expiry")
    run.add_argument("--max-new-attempts", type=int, required=True)
    run.add_argument("--summary", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="freeze completed v2 results")
    freeze.add_argument("--stage-root", type=Path, required=True)
    freeze.add_argument("--results", type=Path, required=True)
    freeze.add_argument("--freeze-marker", type=Path, required=True)
    score = commands.add_parser("reanalyze", help="corrected report from already frozen and revealed outcomes")
    score.add_argument("--stage-root", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--freeze-marker", type=Path, required=True)
    score.add_argument("--json-report", type=Path, required=True)
    score.add_argument("--markdown-report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-resources":
            from .resources import load_manifest
            manifest = load_manifest()
            report = {"status": "PASS", "runtime_revision": RUNTIME_REVISION,
                      "archived_protocol_sha256": manifest["manifest_sha256"],
                      "scope": "archived v1 scientific inputs and implementation; corrected runtime is v2"}
        elif args.command == "prepare":
            from .resources import prepare_stage
            report = prepare_stage(args.stage_root, source_cache=args.source_cache, allow_download=not args.no_download)
        elif args.command == "materialize-outcomes":
            from .resources import materialize_outcomes
            report = materialize_outcomes(args.stage_root, args.results, args.freeze_marker,
                source_cache=args.source_cache, allow_download=not args.no_download)
        elif args.command == "run":
            if args.max_new_attempts < 1:
                parser.error("max-new-attempts must be positive")
            protected = {args.results.resolve(), args.results.with_suffix(args.results.suffix + ".attempts.sqlite3").resolve(),
                         args.results.with_suffix(args.results.suffix + ".lock").resolve()}
            if args.summary.resolve() in protected:
                raise ExecutionError("summary must have a separate output path")
            if args.summary.exists():
                previous = json.loads(args.summary.read_text(encoding="utf-8"))
                if previous.get("runtime_revision") != RUNTIME_REVISION:
                    raise ExecutionError("refusing to overwrite a non-v2 summary artifact")
            report = run_benchmark(args.stage_root, args.results, phase=args.phase,
                                   budget_usd=args.budget_usd, not_after=args.not_after,
                                   max_new_calls=args.max_new_attempts)
            atomic_json(args.summary, report)
        elif args.command == "freeze":
            if args.freeze_marker.exists():
                raise ExecutionError("freeze marker exists; use a new v2 artifact path")
            with exclusive_run_lock(args.results.with_suffix(args.results.suffix + ".lock")):
                rows = read_jsonl(args.results)
                if not rows or any(row.get("runtime_revision") != RUNTIME_REVISION for row in rows):
                    raise ExecutionError("freeze requires a v2 result ledger")
                journal_path = args.results.with_suffix(args.results.suffix + ".attempts.sqlite3")
                if not journal_path.is_file():
                    raise ExecutionError("attempt journal is missing")
                scope = {key: rows[0][key] for key in ("runtime_revision", "protocol_sha256", "stage_sha256")}
                journal = AttemptJournal(journal_path, canonical_hash(scope))
                try:
                    accounting = journal.summary()
                    if accounting["unresolved_attempts"]:
                        raise ExecutionError("cannot freeze with unresolved billing or remote outcomes")
                    journal.connection.execute("PRAGMA wal_checkpoint(FULL)")
                finally:
                    journal.close()
                report = freeze_predictions(args.stage_root, args.results, args.freeze_marker)
                report.pop("freeze_sha256")
                report["runtime_revision"] = RUNTIME_REVISION
                report["attempt_journal_sha256"] = file_hash(journal_path)
                report["accounting"] = accounting
                report["freeze_sha256"] = canonical_hash(report)
                atomic_json(args.freeze_marker, report)
        else:
            if args.json_report.resolve() == args.markdown_report.resolve():
                raise ExecutionError("JSON and Markdown reports need distinct paths")
            if args.json_report.exists() or args.markdown_report.exists():
                raise ExecutionError("report path exists; corrected reanalysis needs new artifact paths")
            report = evaluate_benchmark(args.stage_root, args.results, args.freeze_marker)
            atomic_json(args.json_report, report)
            args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_report.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(report if args.command != "reanalyze" else {
            "runtime_revision": RUNTIME_REVISION, "status": report["status"],
            "json_report": str(args.json_report), "markdown_report": str(args.markdown_report)
        }, indent=2))
        return 0 if report.get("status") != "STOPPED" else 2
    except (ExecutionError, MegaStudyError, ValueError, OSError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
