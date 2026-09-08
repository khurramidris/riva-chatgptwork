"""Import, inspect and reuse pinned evidence files without network access."""

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

from .evidence_catalog import EvidenceCatalog, EvidenceImportSpec, json_bytes, read_bounded, sha256, strict_json


def example_files():
    from .study_commands import example_request
    data = ("id,geography,segment,value,convenience\n"
            "1,example-only,value,0.9,0.2\n2,example-only,value,0.8,0.3\n3,example-only,value,0.7,0.4\n"
            "4,example-only,convenience,0.2,0.9\n5,example-only,convenience,0.3,0.8\n6,example-only,convenience,0.4,0.7\n").encode()
    request = example_request()
    request["brief"]["study_id"] = "catalog-workflow-example-v1"
    spec = {"schema_version": "rival.evidence-import.v1", "source": request["audience"]["sources"][0],
        "origin": "generated:Rival/evidence-example-v1", "revision": "generated-example-v1",
        "released_at": "2026-08-02T00:00:00Z", "retrieved_at": "2026-08-03T00:00:00Z",
        "artifact_sha256": sha256(data), "format": "csv", "expected_records": 6,
        "geography_attribute": "geography", "conditions": ["grocery-delivery-plans"],
        "csv_mapping": {"id_column": "id", "attributes": {"geography": {"column": "geography"},
                        "segment": {"column": "segment"}}, "preferences": {"value": "value", "convenience": "convenience"}}}
    policy = {"geography_attribute": "geography", "conditions": ["grocery-delivery-plans"],
        "required_attributes": ["segment"], "min_seed_records": 6, "min_effective_seed_records": 5.,
        "cells": [{"label": "Value segment", "filters": {"segment": "value"}, "min_seed_records": 3},
                  {"label": "Convenience segment", "filters": {"segment": "convenience"}, "min_seed_records": 3}]}
    return {"people.csv": data, "import.json": json_bytes(spec), "support.json": json_bytes(policy),
            "brief.json": json_bytes(request)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rival evidence", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example", help="write a generated import and study example")
    example.add_argument("--output-dir", type=Path, required=True)
    fingerprint = commands.add_parser("fingerprint", help="hash local source bytes for an import specification")
    fingerprint.add_argument("--input", type=Path, required=True)
    importer = commands.add_parser("import", help="verify and import a CSV or native population JSONL file")
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--spec", type=Path, required=True)
    importer.add_argument("--catalog", type=Path, required=True)
    for name in ("inspect", "list"):
        command = commands.add_parser(name)
        command.add_argument("--catalog", type=Path, required=True)
        if name == "inspect":
            command.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "example":
            output = args.output_dir.resolve()
            if output.exists():
                raise ValueError("example destination already exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="evidence-example-", dir=output.parent) as directory:
                staged = Path(directory) / "example"
                staged.mkdir()
                for name, content in example_files().items():
                    (staged / name).write_bytes(content)
                staged.rename(output)
            result = {"output": str(output), "evidence": "generated engineering example; no real people"}
        elif args.command == "fingerprint":
            data = read_bounded(args.input)
            result = {"artifact_sha256": sha256(data), "size_bytes": len(data)}
        elif args.command == "import":
            spec = EvidenceImportSpec.model_validate(strict_json(read_bounded(args.spec)))
            manifest = EvidenceCatalog(args.catalog).import_file(args.input, spec)
            result = manifest.model_dump(mode="json")
        elif args.command == "inspect":
            manifest, _ = EvidenceCatalog(args.catalog).load(args.bundle)
            result = {"verified": True, "manifest": manifest.model_dump(mode="json")}
        else:
            catalog = EvidenceCatalog(args.catalog)
            if not catalog.root.is_dir():
                raise ValueError("evidence catalog does not exist")
            result = []
            for path in sorted(catalog.root.iterdir()):
                if len(path.name) != 64:
                    continue
                manifest, _ = catalog.load(path.name)
                result.append({"bundle_sha256": manifest.bundle_sha256, "source_id": manifest.spec.source.source_id,
                               "revision": manifest.spec.revision, "records": manifest.spec.expected_records})
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, csv.Error) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
