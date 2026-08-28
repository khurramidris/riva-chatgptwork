from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .demo import run_demo
from .reporting import markdown_report
from .schemas import EvaluationResult, HybridResult, SimulationResult
from .server import serve


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
