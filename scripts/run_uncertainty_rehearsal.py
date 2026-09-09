"""Prospective loopback HTTP rehearsal. All data are generated, with no LLM calls."""

import argparse
from datetime import timedelta
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from threading import Thread
import time

from run_calibration_rehearsal import Handler
from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, json_bytes, strict_json
from rival.evidence_commands import example_files
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.schemas import utc_now
from rival.study_contract import parse_study_request
from rival.study_evidence import bind_evidence
from rival.study_workflow import _workspace, evaluate_study, export_study, prepare_study, run_study
from rival.uncertainty_catalog import UncertaintyCatalog


SETTINGS = {"max_tvd": 0.2, "required_coverage": 0.85, "max_bad_acceptance_rate": 0.1,
            "max_failure_rate": 0.1, "nominal_coverage": 0.9}


def rehearse(root, output):
    if root.exists() or output.exists():
        raise ValueError("use new rehearsal workspace and output directories")
    root.mkdir(parents=True)
    output.mkdir(parents=True)
    files = example_files()
    source = root / "people.csv"
    source.write_bytes(files["people.csv"])
    evidence = EvidenceCatalog(root / "evidence")
    bundle = evidence.import_file(source, EvidenceImportSpec.model_validate(strict_json(files["import.json"])))
    catalog = UncertaintyCatalog(root / "uncertainty")
    Handler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    key = secrets.token_hex(32)
    measured = []
    started = time.perf_counter()

    def request_for(identifier, role, assessment=None):
        brief = strict_json(files["brief.json"])
        brief["schema_version"] = "rival.study-request.v3"
        brief["brief"].update(study_id=identifier, question="Which delivery plan fits " + identifier + "?",
            title="Generated uncertainty rehearsal: " + identifier, sample_size=40, information_cutoff=utc_now().isoformat())
        brief["evidence"].update(role=role, group_id=identifier)
        brief["execution"] = {"mode": "managed", "model": "generated-calibration-fixture", "generation_seed": 42,
            "base_url": "http://127.0.0.1:" + str(server.server_address[1]) + "/chat/completions",
            "budget_usd": 1, "reservation_usd": 0.01, "max_attempts": 6, "max_retries": 1,
            "not_after": (utc_now() + timedelta(hours=1)).isoformat(),
            "model_pin": {"kind": "hosted_endpoint", "revision": "generated-fixture-v1",
                "reference": "Generated loopback fixture; no model weights", "expected_response_model": "generated-calibration-fixture"}}
        if assessment:
            brief.update(schema_version="rival.study-request.v5", uncertainty=assessment)
        return bind_evidence(brief, evidence.root, [bundle.bundle_sha256], strict_json(files["support.json"]))

    def run_and_evaluate(path, *, shift=False, report_name=None):
        before = Handler.calls
        assert run_study(path, api_key="loopback-fixture-key")["complete"]
        calls = Handler.calls
        assert run_study(path)["complete"] and Handler.calls == calls
        if report_name:
            export_study(path, output / (report_name + "-before-outcomes"))
        with _workspace(path) as w, w.execution() as session:
            result, sealed = w.require_complete(w.journal_snapshot(session))
            request = w.request
        observed = dict(result.distribution)
        if shift:
            observed = {"standard": 0, "flexible": 0, "neither": 1}
        else:
            observed["standard"] -= 0.08
            observed["flexible"] += 0.08
        vault = OutcomeVault(root / (request.brief.study_id + "-vault.sqlite3"))
        try:
            vault.deposit(request.brief.study_id, canonical_hash(sealed), {"distribution": observed,
                "source_reference": "Generated controlled outcomes, not human data"}, key, utc_now() - timedelta(seconds=1))
        finally:
            vault.close()
        evaluate_study(path, vault.path, key_material=key)
        if report_name:
            export_study(path, output / report_name)
        measured.append({"study_id": request.brief.study_id, "role": request.evidence.role,
                         "completed_draws": 40, "http_fixture_calls": Handler.calls - before, "resume_calls": 0})

    try:
        partitions = {}
        for role, count in (("training", 5), ("calibration", 19), ("evaluation", 60)):
            partitions[role] = []
            for i in range(count):
                identifier = f"l11-{role}-{i:02d}"
                path = root / identifier
                prepare_study(path, request_for(identifier, role), catalog_root=evidence.root)
                partitions[role].append(path)
        plan = catalog.plan([p for paths in partitions.values() for p in paths], SETTINGS)
        for role, paths in partitions.items():
            for path in paths:
                run_and_evaluate(path)
            if role == "training":
                fit = catalog.fit(plan["plan_sha256"], paths)
            elif role == "calibration":
                bound = catalog.calibrate(fit["fit_sha256"], paths)
            else:
                assessment = catalog.evaluate(bound["bound_sha256"], paths)
        artifact = catalog.load_assessment(assessment["assessment_sha256"])
        assert artifact["assessment"]["metrics"]["statistical_gates_passed"]
        for name, shift in (("same-process", False), ("changed-process", True)):
            path = root / ("l11-" + name)
            prepare_study(path, request_for("l11-" + name, "evaluation", assessment),
                          catalog_root=evidence.root, uncertainty_catalog_root=catalog.root)
            run_and_evaluate(path, shift=shift, report_name=name)
        reports = {name: strict_json((output / name / "report.json").read_bytes())["uncertainty"]
                   for name in ("same-process", "changed-process")}
        assert reports["same-process"]["comparison"]["research_bound_covered"]
        assert not reports["changed-process"]["comparison"]["research_bound_covered"]
        assert all(r["prediction"]["confidence"]["abstain"] for r in reports.values())
        record = {"schema_version": "rival.uncertainty-rehearsal.v1", "created_at": utc_now().isoformat(),
            "scope": "generated independent engineering fixtures; no human accuracy or domain qualification",
            "real_llm_calls": 0, "api_cost_usd": 0, "http_fixture_calls": Handler.calls,
            "elapsed_seconds": time.perf_counter() - started, "plan": plan, "fit": fit, "bound": bound,
            "assessment": assessment, "evaluation": artifact["assessment"]["metrics"],
            "studies": measured, "target_comparisons": {name: r["comparison"] for name, r in reports.items()}}
        record["record_sha256"] = canonical_hash(record)
        (output / "rehearsal.json").write_bytes(json_bytes(record))
        return {"http_fixture_calls": Handler.calls, "real_llm_calls": 0,
                "statistical_gates_passed": artifact["assessment"]["metrics"]["statistical_gates_passed"],
                "customer_qualified": False, "target_comparisons": record["target_comparisons"]}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rehearse(args.workspace, args.output), indent=2, sort_keys=True))
