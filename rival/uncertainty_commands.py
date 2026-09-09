"""Freeze, fit, calibrate and assess a protected study-level uncertainty cohort."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .evidence_catalog import read_bounded, strict_json
from .integrity import IntegrityError
from .uncertainty_catalog import UncertaintyCatalog


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rival uncertainty", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, parent in (("plan", "settings"), ("fit", "plan"), ("calibrate", "fit"), ("evaluate", "bound")):
        command = commands.add_parser(name)
        command.add_argument("--catalog", type=Path, required=True)
        command.add_argument("--workspace", type=Path, action="append", required=True)
        command.add_argument("--" + parent, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--catalog", type=Path, required=True)
    inspect.add_argument("--assessment", required=True)
    args = parser.parse_args(argv)
    catalog = UncertaintyCatalog(args.catalog)
    try:
        if args.command == "plan":
            result = catalog.plan(args.workspace, strict_json(read_bounded(Path(args.settings))))
        elif args.command == "fit":
            result = catalog.fit(args.plan, args.workspace)
        elif args.command == "calibrate":
            result = catalog.calibrate(args.fit, args.workspace)
        elif args.command == "evaluate":
            result = catalog.evaluate(args.bound, args.workspace)
        else:
            artifact = catalog.load_assessment(args.assessment)
            result = {"assessment_sha256": args.assessment,
                      "customer_qualified": artifact["assessment"]["customer_qualified"],
                      "metrics": {k: v for k, v in artifact["assessment"]["metrics"].items() if k != "rows"},
                      "bound": artifact["bound"]["bound"], "settings": artifact["plan"]["settings"]}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, IntegrityError, OSError, sqlite3.Error) as exc:
        print(f"uncertainty {args.command} blocked: {exc}", file=sys.stderr)
        return 2
