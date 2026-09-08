"""Operator commands for the versioned study workflow."""

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys

from .study_contract import StudyRequest, StudyRequestV2, load_study_request
from .study_evidence import bind_evidence
from .study_support import UnsupportedStudy
from .evidence_catalog import read_bounded, strict_json
from .study_workflow import check_study, evaluate_study, export_study, prepare_study, run_study, study_status


def example_request():
    """Generated data, deliberately labelled as an engineering example."""
    return {
        "schema_version": "rival.study-request.v1",
        "brief": {"study_id": "grocery-concepts-example-v1", "title": "Grocery delivery concept example",
                  "decision": "Explore how two delivery plans compare in a generated audience.",
                  "question": "Which grocery delivery plan would you choose?", "context": "Monthly grocery delivery plans.",
                  "choices": [{"choice_id": "standard", "label": "Standard delivery", "features": {"value": 0.9, "convenience": 0.3}},
                              {"choice_id": "flexible", "label": "Flexible delivery", "features": {"value": 0.3, "convenience": 0.9}},
                              {"choice_id": "neither", "label": "Neither plan", "features": {"value": 0.2}}],
                  "information_cutoff": "2026-09-01T00:00:00Z", "sample_size": 1000, "seed": 42},
        "audience": {"description": "Six generated example records; no real consumers represented.",
                     "geography": ["example-only"],
                     "records": [{"person_id": f"generated-{i}", "attributes": {"segment": "value" if i < 3 else "convenience"},
                                  "preferences": {"value": 0.9 if i < 3 else 0.2, "convenience": 0.2 if i < 3 else 0.9},
                                  "evidence_ids": ["generated-example"]} for i in range(6)],
                     "sources": [{"source_id": "generated-example", "name": "Rival generated workflow example",
                                  "source_type": "synthetic", "collected_at": "2026-08-01T00:00:00Z",
                                  "geography": ["example-only"], "rights_reference": "Generated in Rival for testing.",
                                  "permitted_uses": ["simulation"]}],
                     "targets": {"controls": {"segment": {"value": 0.5, "convenience": 0.5}}}},
        "evidence": {"role": "development", "group_id": "generated-workflow-example",
                     "source_reference": "Generated engineering example; no independent human evidence."},
        "execution": {"mode": "offline"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rival study", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("example", "schema"):
        command = commands.add_parser(name, help=f"write the study {name}")
        command.add_argument("--output", type=Path, required=True)
        if name == "schema":
            command.add_argument("--version", choices=("v1", "v2"), default="v2")
    bind = commands.add_parser("bind-evidence", help="bind verified imports and a support policy to a study brief")
    for name in ("input", "catalog", "support", "output"):
        bind.add_argument("--" + name, type=Path, required=True)
    bind.add_argument("--bundle", action="append", required=True)
    prepare = commands.add_parser("prepare", help="validate and save a study without model calls")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path)
    check = commands.add_parser("check", help="inspect evidence and audience support without executing or saving a study")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--catalog", type=Path)
    for name in ("run", "status", "export", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--vault", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {"example", "schema"}:
            model = StudyRequest if getattr(args, "version", None) == "v1" else StudyRequestV2
            payload = example_request() if args.command == "example" else model.model_json_schema()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            result = {"output": str(args.output.resolve())}
        elif args.command == "bind-evidence":
            request = bind_evidence(strict_json(read_bounded(args.input)), args.catalog, args.bundle,
                                    strict_json(read_bounded(args.support)))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(request.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n")
            result = {"output": str(args.output.resolve()), "study_id": request.brief.study_id}
        elif args.command == "prepare":
            result = prepare_study(args.workspace, load_study_request(args.input), catalog_root=args.catalog)
        elif args.command == "check":
            result = check_study(load_study_request(args.input), catalog_root=args.catalog)
        elif args.command == "run":
            result = run_study(args.workspace)
        elif args.command == "status":
            result = study_status(args.workspace)
        elif args.command == "export":
            result = export_study(args.workspace, args.output)
        else:
            key = os.getenv("RIVAL_OUTCOME_KEY")
            if not key:
                raise ValueError("set RIVAL_OUTCOME_KEY to open the existing outcome vault")
            result = evaluate_study(args.workspace, args.vault, key_material=key)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except UnsupportedStudy as exc:
        print(json.dumps(exc.report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    except (ValueError, RuntimeError, OSError, sqlite3.Error, csv.Error) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
