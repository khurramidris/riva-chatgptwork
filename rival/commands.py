"""Current CLI; the hash-locked v1 CLI is retained as a historical witness."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import cli
from .readiness import release_claims
from .version import __version__


def qualify_all_command(args):
    from .research.qualification import run_all
    summary = run_all(args.output_dir, compact=args.compact)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary.get("status") == "PASS" else 1


def simulate_managed(args):
    from .engine import RivalEngine
    from .managed_execution import ExecutionSession
    from .mega_study_v2.runner import exclusive_run_lock
    from .providers import OpenAICompatibleProvider
    from .schemas import PopulationRecord, PopulationTargets, ScenarioSpec
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not payload.get("scenario", {}).get("scenario_id"):
        raise ValueError("input must declare a stable scenario_id for safe recovery")
    scenario = ScenarioSpec.model_validate({**payload["scenario"], "model_family": "managed"})
    records = [PopulationRecord.model_validate(item) for item in payload["records"]]
    targets = PopulationTargets.model_validate(payload["targets"]) if payload.get("targets") else None
    paths = [args.output, args.journal, args.database, args.input]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("input, output, journal and evidence database need distinct paths")
    if args.database.exists() and not args.journal.exists():
        raise ValueError("existing evidence database has no attempt journal; restore accounting before resuming")
    with exclusive_run_lock(args.output.with_suffix(args.output.suffix + ".lock")):
        if args.output.exists():
            raise ValueError("output already exists; use it or choose a new export path")
        with ExecutionSession(args.journal, scope_id=args.scope_id, budget_usd=args.budget_usd,
                not_after=datetime.fromisoformat(args.not_after.replace("Z", "+00:00")),
                reservation_usd=args.reservation_usd, max_total_attempts=args.max_attempts) as execution:
            provider = OpenAICompatibleProvider(model=args.model, base_url=args.base_url, execution=execution)
            args.database.parent.mkdir(parents=True, exist_ok=True)
            engine = RivalEngine(store_path=args.database)
            try:
                engine.register_provider("managed", provider)
                result = engine.simulate(records, scenario, targets)
                report = {"simulation": result.model_dump(mode="json"),
                          "accounting": execution.journal.summary(), "release_claims": release_claims()}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
                print(json.dumps({"output": str(args.output.resolve()), "accounting": report["accounting"]}))
            finally:
                engine.store.close()
    return 0


def build_parser():
    parser = cli.build_parser()
    parser.add_argument("--version", action="version", version=__version__)
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    # Frozen v1 execution is not a supported managed entry point. Retain its
    # files for reproducibility, and expose the explicitly versioned runtime.
    for name in ("mega-study", "run-live-pilot"):
        commands.choices.pop(name, None)
    commands._choices_actions = [action for action in commands._choices_actions
                                if action.dest not in {"mega-study", "run-live-pilot"}]
    commands.choices["qualify-all"].set_defaults(func=qualify_all_command)
    claims = commands.add_parser("status", help="show current release claims")
    claims.set_defaults(func=lambda args: print(json.dumps(release_claims(), indent=2)) or 0)
    commands.add_parser("mega-v2", help="portable corrected benchmark commands; use mega-v2 --help")
    commands.add_parser("study", help="prepare, run, resume and report a complete study; use study --help")
    commands.add_parser("evidence", help="import and verify pinned evidence; use evidence --help")
    commands.add_parser("uncertainty", help="plan and assess study-level error bounds; use uncertainty --help")
    commands.add_parser("calibration", help="build protected reference banks and fit calibration; use calibration --help")
    run = commands.add_parser("simulate-managed", help="research simulation with durable model request accounting")
    for name in ("input", "output", "journal", "database"):
        run.add_argument("--" + name, type=Path, required=True)
    for name in ("scope-id", "model", "not-after"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--base-url")
    run.add_argument("--budget-usd", type=float, required=True)
    run.add_argument("--reservation-usd", type=float, required=True, help="conservative per-attempt estimate; provider spending limits still apply")
    run.add_argument("--max-attempts", type=int, required=True, help="physical attempts over the journal lifetime")
    run.set_defaults(func=simulate_managed)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "uncertainty":
        from .uncertainty_commands import main as uncertainty_main
        return uncertainty_main(argv[1:])
    if argv and argv[0] == "calibration":
        from .calibration_commands import main as calibration_main
        return calibration_main(argv[1:])
    if argv and argv[0] == "evidence":
        from .evidence_commands import main as evidence_main
        return evidence_main(argv[1:])
    if argv and argv[0] == "study":
        from .study_commands import main as study_main
        return study_main(argv[1:])
    if argv and argv[0] == "mega-v2":
        from .mega_study_v2.__main__ import main as mega_main
        return mega_main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
