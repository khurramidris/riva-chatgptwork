"""Freeze and audit a qualification experiment against human-only baselines."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from .evidence_catalog import json_bytes, read_bounded, strict_json
from .integrity import IntegrityError
from .qualification_catalog import QualificationCatalog
from .qualification_contract import QualificationProtocol
from .qualification_report import qualification_markdown


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rival qualification", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="print the outcome-free qualification protocol schema")
    for name in ("fit", "plan", "seal", "assess", "inspect", "export"):
        command = commands.add_parser(name)
        command.add_argument("--catalog", type=Path, required=True)
        if name in {"fit", "plan", "seal", "assess"}:
            command.add_argument("--workspace", type=Path, action="append", required=name != "assess")
        if name == "fit":
            command.add_argument("--design", type=Path, required=True)
            command.add_argument("--settings", type=Path, required=True)
        elif name == "plan":
            command.add_argument("--protocol", type=Path, required=True)
            command.add_argument("--baselines", required=True)
        elif name in {"seal", "assess"}:
            command.add_argument("--plan", required=True)
            if name == "seal":
                command.add_argument("--anchors", type=Path, required=True)
            else:
                command.add_argument("--outcomes", type=Path, required=True)
                command.add_argument("--operations", type=Path, required=True)
        else:
            command.add_argument("--assessment", required=True)
            if name == "export":
                command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def read(path):
        return strict_json(read_bounded(path))
    try:
        if args.command == "schema":
            result = QualificationProtocol.model_json_schema()
        else:
            catalog = QualificationCatalog(args.catalog)
            if args.command == "fit":
                result = catalog.fit(args.workspace, read(args.design), read(args.settings))
            elif args.command == "plan":
                result = catalog.plan(args.workspace, read(args.protocol), args.baselines)
            elif args.command == "seal":
                if len(args.workspace) != 1:
                    raise ValueError("seal takes exactly one workspace and its separate anchor sample")
                result = catalog.seal(args.plan, args.workspace[0], read(args.anchors))
            elif args.command == "assess":
                result = catalog.assess(args.plan, args.workspace or [], read(args.outcomes), read(args.operations))
                result.update(catalog.load_assessment(result["assessment_sha256"])["report"])
            else:
                result = catalog.load_assessment(args.assessment)["report"]
                if args.command == "export":
                    if args.output.resolve().is_relative_to(catalog.root) or catalog.root.is_relative_to(args.output.resolve()):
                        raise ValueError("export outside the private qualification catalog")
                    # Existing exports are never silently overwritten.
                    args.output.mkdir(parents=True, exist_ok=False)
                    (args.output / "report.json").write_bytes(json_bytes(result))
                    (args.output / "report.md").write_text(qualification_markdown(result), encoding="utf-8")
                    (args.output / "manifest.json").write_bytes(json_bytes({"assessment_sha256": args.assessment,
                        "plan_sha256": result["plan_sha256"], "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                        for name in ("report.json", "report.md")}}))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 1 if args.command == "assess" and result["status"] != "PASS" else 0
    except (ValueError, IntegrityError, OSError, sqlite3.Error) as exc:
        print(f"qualification {args.command} blocked: {exc}", file=sys.stderr)
        return 2
