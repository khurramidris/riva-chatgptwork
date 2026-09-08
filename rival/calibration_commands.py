"""Build a training reference bank and fit an immutable calibration adapter."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .calibration_catalog import CalibrationCatalog
from .evidence_catalog import read_bounded, strict_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rival calibration", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bank = commands.add_parser("bank", help="harvest complete protected training studies")
    bank.add_argument("--catalog", type=Path, required=True)
    bank.add_argument("--workspace", type=Path, action="append", required=True)
    fit = commands.add_parser("fit", help="fit using the bank only; creates no model calls")
    fit.add_argument("--catalog", type=Path, required=True)
    fit.add_argument("--bank", required=True)
    fit.add_argument("--settings", type=Path)
    args = parser.parse_args(argv)
    try:
        catalog = CalibrationCatalog(args.catalog)
        result = (catalog.create_bank(args.workspace) if args.command == "bank" else
                  catalog.fit(args.bank, strict_json(read_bounded(args.settings)) if args.settings else None))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
