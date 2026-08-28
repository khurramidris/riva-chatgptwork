from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .demo import run_demo
from .reporting import markdown_report
from .schemas import EvaluationResult, HybridResult, SimulationResult
from .server import serve


_LIVE_STUDY_DIR = Path(__file__).resolve().parent / "studies" / "twin2k_live_v1"


def write_text(path: str | None, content: str) -> None:
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        print(destination.resolve())
    else:
        print(content)


def demo_command(args: argparse.Namespace) -> int:
    result = run_demo(
        sample_size=args.sample_size, human_anchor_size=args.human_anchor_size
    )
    if args.json:
        write_text(args.json, json.dumps(result, indent=2, sort_keys=True))
    if args.markdown:
        simulation = SimulationResult.model_validate(result["simulation"])
        hybrid = HybridResult.model_validate(result["hybrid"])
        evaluation = EvaluationResult.model_validate(result["synthetic_evaluation"])
        write_text(args.markdown, markdown_report(simulation, hybrid, evaluation))
    if not args.json and not args.markdown:
        summary = {
            "synthetic": result["simulation"]["distribution"],
            "hybrid": result["hybrid"]["corrected_distribution"],
            "protected_outcome": result["protected_outcome"],
            "synthetic_tvd": result["synthetic_evaluation"]["metrics"]["tvd"],
            "hybrid_tvd": result["hybrid_evaluation"]["metrics"]["tvd"],
            "relative_tvd_reduction": result["improvement"]["relative_tvd_reduction"],
        }
        print(json.dumps(summary, indent=2))
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    rows = []
    for index in range(args.repeats):
        result = run_demo(
            sample_size=args.sample_size,
            human_anchor_size=args.human_anchor_size + index,
        )
        rows.append(
            {
                "repeat": index + 1,
                "synthetic_tvd": result["synthetic_evaluation"]["metrics"]["tvd"],
                "hybrid_tvd": result["hybrid_evaluation"]["metrics"]["tvd"],
            }
        )
    payload = {
        "runs": rows,
        "mean_synthetic_tvd": mean(row["synthetic_tvd"] for row in rows),
        "mean_hybrid_tvd": mean(row["hybrid_tvd"] for row in rows),
        "all_hybrid_better": all(
            row["hybrid_tvd"] < row["synthetic_tvd"] for row in rows
        ),
    }
    write_text(args.output, json.dumps(payload, indent=2))
    return 0


def qualify_opinionqa_command(args: argparse.Namespace) -> int:
    from .research.opinionqa import benchmark_opinionqa

    report = benchmark_opinionqa(
        folds=args.folds,
        max_iter=args.iterations,
        include_question_results=not args.compact,
    )
    write_text(args.output, json.dumps(report, indent=2, sort_keys=True))
    return 0


def qualify_twin2k_command(args: argparse.Namespace) -> int:
    from .research.twin2k import benchmark_twin2k

    report = benchmark_twin2k(
        ridge_alpha=args.ridge_alpha,
        anchor_size=args.anchor_size,
        include_question_results=not args.compact,
    )
    write_text(args.output, json.dumps(report, indent=2, sort_keys=True))
    return 0


def qualify_all_command(args: argparse.Namespace) -> int:
    from .research.qualification import run_all

    summary = run_all(args.output_dir, compact=args.compact)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def qualify_integrity_command(args: argparse.Namespace) -> int:
    from .research.integrity_qualification import run_integrity_qualification

    report = run_integrity_qualification()
    write_text(args.output, json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def qualify_research_components_command(args: argparse.Namespace) -> int:
    from .research.integration_qualification import (
        run_research_integration_qualification,
    )

    report = run_research_integration_qualification()
    write_text(args.output, json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def verify_release_command(args: argparse.Namespace) -> int:
    from .release import verify_release_manifest

    report = verify_release_manifest(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def prepare_live_pilot_command(args: argparse.Namespace) -> int:
    from .live_pilot import prepare_twin2k_live_pilot

    report = prepare_twin2k_live_pilot(
        args.output_dir,
        dataset_root=args.dataset_root,
        cohort_size=args.cohort_size,
        target_count=args.target_count,
        anchor_size=args.anchor_size,
        history_items=args.history_items,
        minimum_history_items=args.minimum_history_items,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "protocol": str(Path(args.output_dir, "protocol.json").resolve()),
                "cases": str(Path(args.output_dir, "cases.jsonl").resolve()),
                "protocol_sha256": report["protocol_sha256"],
                "total_cases": report["cases"]["total"],
                "preflight_cases": report["cases"]["preflight"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def rehearse_live_pilot_command(args: argparse.Namespace) -> int:
    from .live_pilot import (
        BudgetGuard,
        DeterministicRehearsalProvider,
        evaluate_live_pilot,
        load_and_verify_protocol,
        run_live_pilot,
    )

    protocol, _, _ = load_and_verify_protocol(
        args.protocol, cases_path=args.cases, dataset_root=args.dataset_root
    )
    guard = BudgetGuard(
        budget_usd=1.0,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        max_calls=int(protocol["cases"]["total"]),
    )
    summary = run_live_pilot(
        args.protocol,
        args.results,
        DeterministicRehearsalProvider(),
        guard,
        phase="pilot",
        cases_path=args.cases,
        dataset_root=args.dataset_root,
        summary_path=args.summary,
    )
    evaluation = evaluate_live_pilot(
        args.protocol,
        args.results,
        cases_path=args.cases,
        dataset_root=args.dataset_root,
    )
    write_text(args.evaluation, json.dumps(evaluation, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETE" else 1


def run_live_pilot_command(args: argparse.Namespace) -> int:
    from .live_pilot import BudgetGuard, make_openai_provider, parse_not_after, run_live_pilot

    provider = make_openai_provider(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        max_retries=args.max_retries,
        history_limit=args.history_limit,
        max_output_tokens=args.max_output_tokens,
        use_response_format=not args.no_response_format,
    )
    guard = BudgetGuard(
        budget_usd=args.budget_usd,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
        max_calls=args.max_calls,
        not_after=parse_not_after(args.not_after),
    )
    summary = run_live_pilot(
        args.protocol,
        args.results,
        provider,
        guard,
        phase=args.phase,
        cases_path=args.cases,
        dataset_root=args.dataset_root,
        max_errors=args.max_errors,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETE" else 2


def evaluate_live_pilot_command(args: argparse.Namespace) -> int:
    from .live_pilot import evaluate_live_pilot

    report = evaluate_live_pilot(
        args.protocol,
        args.results,
        cases_path=args.cases,
        dataset_root=args.dataset_root,
    )
    write_text(args.output, json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "PARTIAL_UNEVALUABLE" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rival")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="run the calibrated concept-test demo")
    demo_parser.add_argument("--sample-size", type=int, default=1200)
    demo_parser.add_argument("--human-anchor-size", type=int, default=80)
    demo_parser.add_argument("--json")
    demo_parser.add_argument("--markdown")
    demo_parser.set_defaults(func=demo_command)

    benchmark_parser = subparsers.add_parser("benchmark", help="repeat the protected demo benchmark")
    benchmark_parser.add_argument("--sample-size", type=int, default=1200)
    benchmark_parser.add_argument("--human-anchor-size", type=int, default=80)
    benchmark_parser.add_argument("--repeats", type=int, default=5)
    benchmark_parser.add_argument("--output")
    benchmark_parser.set_defaults(func=benchmark_command)

    opinion_parser = subparsers.add_parser(
        "qualify-opinionqa",
        help="run the real-data, family-held-out OpinionQA benchmark",
    )
    opinion_parser.add_argument("--folds", type=int, default=5)
    opinion_parser.add_argument("--iterations", type=int, default=150)
    opinion_parser.add_argument("--compact", action="store_true")
    opinion_parser.add_argument("--output")
    opinion_parser.set_defaults(func=qualify_opinionqa_command)

    twin_parser = subparsers.add_parser(
        "qualify-twin2k",
        help="run the real-data Twin-2K longitudinal benchmark",
    )
    twin_parser.add_argument("--ridge-alpha", type=float, default=10.0)
    twin_parser.add_argument("--anchor-size", type=int, default=80)
    twin_parser.add_argument("--compact", action="store_true")
    twin_parser.add_argument("--output")
    twin_parser.set_defaults(func=qualify_twin2k_command)

    all_parser = subparsers.add_parser(
        "qualify-all", help="run and write every real-data qualification track"
    )
    all_parser.add_argument("--output-dir", default="reports")
    all_parser.add_argument("--compact", action="store_true")
    all_parser.set_defaults(func=qualify_all_command)

    integrity_parser = subparsers.add_parser(
        "qualify-integrity",
        help="verify prospective locking, sealing, phase-chain, and outcome-vault controls",
    )
    integrity_parser.add_argument("--output")
    integrity_parser.set_defaults(func=qualify_integrity_command)

    components_parser = subparsers.add_parser(
        "qualify-research-components",
        help="verify licensed research runtimes, adapters, and numerical parity",
    )
    components_parser.add_argument("--output")
    components_parser.set_defaults(func=qualify_research_components_command)

    release_parser = subparsers.add_parser(
        "verify-release", help="verify every file hash in RELEASE_MANIFEST.json"
    )
    release_parser.add_argument("--manifest", default="RELEASE_MANIFEST.json")
    release_parser.set_defaults(func=verify_release_command)

    live_prepare = subparsers.add_parser(
        "prepare-live-pilot",
        help="freeze the outcome-free Twin-2K live-provider pilot",
    )
    live_prepare.add_argument("--output-dir", default=str(_LIVE_STUDY_DIR))
    live_prepare.add_argument("--dataset-root")
    live_prepare.add_argument("--cohort-size", type=int, default=50)
    live_prepare.add_argument("--target-count", type=int, default=15)
    live_prepare.add_argument("--anchor-size", type=int, default=10)
    live_prepare.add_argument("--history-items", type=int, default=16)
    live_prepare.add_argument("--minimum-history-items", type=int, default=8)
    live_prepare.add_argument("--seed", type=int, default=20260828)
    live_prepare.set_defaults(func=prepare_live_pilot_command)

    rehearsal = subparsers.add_parser(
        "rehearse-live-pilot",
        help="run the frozen pilot end to end without network calls",
    )
    rehearsal.add_argument("--protocol", default=str(_LIVE_STUDY_DIR / "protocol.json"))
    rehearsal.add_argument("--cases")
    rehearsal.add_argument("--dataset-root")
    rehearsal.add_argument("--results", default="reports/live_pilot_rehearsal.jsonl")
    rehearsal.add_argument("--summary", default="reports/live_pilot_rehearsal_summary.json")
    rehearsal.add_argument("--evaluation", default="reports/live_pilot_rehearsal_evaluation.json")
    rehearsal.set_defaults(func=rehearse_live_pilot_command)

    live_run = subparsers.add_parser(
        "run-live-pilot",
        help="run or resume the frozen pilot against an OpenAI-compatible model",
    )
    live_run.add_argument("--protocol", default=str(_LIVE_STUDY_DIR / "protocol.json"))
    live_run.add_argument("--cases")
    live_run.add_argument("--dataset-root")
    live_run.add_argument("--results", default="reports/live_pilot_results.jsonl")
    live_run.add_argument("--summary", default="reports/live_pilot_summary.json")
    live_run.add_argument("--phase", choices=["preflight", "pilot"], default="preflight")
    live_run.add_argument("--model", required=True)
    live_run.add_argument("--base-url")
    live_run.add_argument("--budget-usd", type=float, required=True)
    live_run.add_argument("--input-cost-per-million", type=float, required=True)
    live_run.add_argument("--output-cost-per-million", type=float, required=True)
    live_run.add_argument("--max-calls", type=int, required=True)
    live_run.add_argument("--not-after", required=True, help="ISO-8601 UTC run expiry")
    live_run.add_argument("--timeout-seconds", type=int, default=60)
    live_run.add_argument("--temperature", type=float, default=0.0)
    live_run.add_argument("--max-retries", type=int, default=1)
    live_run.add_argument("--history-limit", type=int, default=16)
    live_run.add_argument("--max-output-tokens", type=int, default=300)
    live_run.add_argument("--max-errors", type=int, default=3)
    live_run.add_argument("--no-response-format", action="store_true")
    live_run.set_defaults(func=run_live_pilot_command)

    live_evaluate = subparsers.add_parser(
        "evaluate-live-pilot",
        help="reveal Twin-2K outcomes only after calls and evaluate the frozen run",
    )
    live_evaluate.add_argument("--protocol", default=str(_LIVE_STUDY_DIR / "protocol.json"))
    live_evaluate.add_argument("--cases")
    live_evaluate.add_argument("--dataset-root")
    live_evaluate.add_argument("--results", default="reports/live_pilot_results.jsonl")
    live_evaluate.add_argument("--output", default="reports/live_pilot_evaluation.json")
    live_evaluate.set_defaults(func=evaluate_live_pilot_command)

    server_parser = subparsers.add_parser("serve", help="start the API and study interface")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8080)
    server_parser.add_argument("--database", default="rival.sqlite3")
    server_parser.set_defaults(
        func=lambda args: serve(args.host, args.port, args.database) or 0
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
