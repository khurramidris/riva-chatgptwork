"""Exercise qualification over loopback HTTP; generated data, no actual LLM."""

import argparse
from datetime import timedelta
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from threading import Thread
import time

from run_calibration_rehearsal import Handler
from rival.calibration_catalog import CalibrationCatalog
from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, json_bytes, strict_json
from rival.evidence_commands import example_files
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.qualification_baselines import summarize_sample
from rival.qualification_catalog import QualificationCatalog
from rival.qualification_commands import main as qualification_cli
from rival.schemas import utc_now
from rival.study_evidence import bind_evidence
from rival.study_workflow import _workspace, evaluate_study, prepare_study, run_study
from rival.uncertainty_catalog import UncertaintyCatalog


def rehearse(root, output):
    if root.exists() or output.exists():
        raise ValueError("use new rehearsal and export directories")
    root.mkdir(parents=True)
    files = example_files()
    source = root / "people.csv"
    source.write_bytes(files["people.csv"])
    evidence = EvidenceCatalog(root / "evidence")
    bundle = evidence.import_file(source, EvidenceImportSpec.model_validate(strict_json(files["import.json"])))
    calibration = CalibrationCatalog(root / "calibration")
    uncertainty = UncertaintyCatalog(root / "uncertainty")
    qualification = QualificationCatalog(root / "qualification")
    Handler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    key = secrets.token_hex(32)
    started = time.perf_counter()
    measured = []

    def request(identifier, role, adapter=None, assessment=None):
        brief = strict_json(files["brief.json"])
        brief["schema_version"] = "rival.study-request.v3"
        brief["brief"].update(study_id=identifier, question="Which delivery plan fits " + identifier + "?",
            title="Generated qualification fixture", sample_size=40, information_cutoff=utc_now().isoformat())
        brief["evidence"].update(role=role, group_id=identifier)
        brief["execution"] = {"mode": "managed", "model": "generated-calibration-fixture", "generation_seed": 42,
            "base_url": "http://127.0.0.1:" + str(server.server_address[1]) + "/chat/completions", "budget_usd": 1,
            "reservation_usd": 0.01, "max_attempts": 6, "max_retries": 1,
            "not_after": (utc_now() + timedelta(hours=1)).isoformat(), "model_pin": {"kind": "hosted_endpoint",
                "revision": "generated-fixture-v1", "reference": "Generated response fixture; no actual model",
                "expected_response_model": "generated-calibration-fixture"}}
        if adapter:
            brief.update(schema_version="rival.study-request.v4", calibration=adapter)
        if assessment:
            brief.update(schema_version="rival.study-request.v5", uncertainty=assessment)
        return bind_evidence(brief, evidence.root, [bundle.bundle_sha256], strict_json(files["support.json"]))

    def execute(path):
        before = Handler.calls
        assert run_study(path, api_key="generated-fixture-key")["complete"]
        calls = Handler.calls
        assert run_study(path)["complete"] and Handler.calls == calls
        measured.append({"study_id": path.name, "http_fixture_calls": calls - before, "resume_calls": 0})

    def reveal(path, observed=None):
        with _workspace(path) as w, w.execution() as session:
            _, sealed = w.require_complete(w.journal_snapshot(session))
        vault = OutcomeVault(root / (path.name + "-vault.sqlite3"))
        try:
            vault.deposit(path.name, canonical_hash(sealed), {"distribution": observed or
                {"standard": 0.25, "flexible": 0.5, "neither": 0.25}, "source": "Generated outcomes"},
                key, utc_now() - timedelta(seconds=1))
        finally:
            vault.close()
        evaluate_study(path, vault.path, key_material=key)

    try:
        train = []
        for i in range(3):
            path = root / ("l12-reference-" + str(i))
            prepare_study(path, request(path.name, "training"), catalog_root=evidence.root)
            execute(path)
            reveal(path)
            train.append(path)
        bank = calibration.create_bank(train)
        fitted = calibration.fit(bank["bank_sha256"])
        adapter = {"adapter_sha256": fitted["adapter_sha256"]}
        baselines = qualification.fit(train, [{"study_id": p.name, "features": [float(i)], "weight": 1.}
            for i, p in enumerate(train)], {"feature_names": ["price_level"], "feature_provenance": "Generated pre-outcome prices",
                "relevance": "Classical numerical fixture; not a business-domain claim"})
        groups = {}
        for role, count in (("training", 5), ("calibration", 1), ("evaluation", 1)):
            groups[role] = []
            for i in range(count):
                path = root / ("l12-uq-" + role + "-" + str(i))
                prepare_study(path, request(path.name, role, adapter), catalog_root=evidence.root,
                              calibration_catalog_root=calibration.root)
                groups[role].append(path)
        uq_plan = uncertainty.plan([p for group in groups.values() for p in group],
            {"max_tvd": 0.4, "required_coverage": 0.8, "max_bad_acceptance_rate": 0.1, "max_failure_rate": 0.1})
        for role, paths in groups.items():
            for path in paths:
                execute(path)
                reveal(path)
            if role == "training":
                fit = uncertainty.fit(uq_plan["plan_sha256"], paths)
            elif role == "calibration":
                bound = uncertainty.calibrate(fit["fit_sha256"], paths)
            else:
                assessed = uncertainty.evaluate(bound["bound_sha256"], paths)
        members, targets = [], []
        for label in ("benefit", "harm", "missing-response", "unfinished"):
            path = root / ("l12-" + label)
            prepare_study(path, request(path.name, "evaluation", adapter, assessed), catalog_root=evidence.root,
                calibration_catalog_root=calibration.root, uncertainty_catalog_root=uncertainty.root)
            targets.append(path)
            anchor_units = [canonical_hash(path.name + "-anchor-" + str(i)) for i in range(4)]
            evaluation_units = [canonical_hash(path.name + "-evaluation-" + str(i)) for i in range(4)]
            members.append({"study_id": path.name, "features": [1.], "source_reference": "Generated fixture " + path.name,
                "instrument_sha256": canonical_hash("generated " + label), "rights_reference": "Generated by Rival",
                "evidence_origin": "generated", "outcome_access": "inspected", "timing": "historical",
                "fieldwork_starts_at": "2026-01-01T00:00:00+00:00", "outcomes_available_at": "2026-01-02T00:00:00+00:00",
                "independence_rationale": "Controlled generated cases, not independent human evidence",
                "anchor_units": anchor_units, "evaluation_units": evaluation_units,
                "unit_weights": {u: 1. for u in anchor_units + evaluation_units}})
        with _workspace(targets[0]) as w:
            audience = w.request.audience
        protocol = {"study_family": "Generated delivery preference cases", "audience": audience.description,
            "geography": audience.geography, "intended_decision": "Verify qualification accounting",
            "threshold_rationale": "Generated software thresholds, not a customer contract",
            "subgroup_scope": "Aggregate generated evidence only", "max_tvd": 0.4, "minimum_baseline_improvement": 0.01,
            "required_coverage": 0.8, "minimum_acceptance_rate": 0.1, "max_bad_acceptance_rate": 0.1,
            "max_failure_rate": 0.1, "max_missing_rate": 0.2, "budget_per_study_usd": 4.,
            "human_unit_cost_usd": 1., "max_turnaround_seconds": 3600., "members": members}
        plan = qualification.plan(targets, protocol, baselines["baselines_sha256"])
        (root / "protocol.json").write_bytes(json_bytes(protocol))
        outcomes, operations = {}, {}
        for i, path in enumerate(targets[:3]):
            spec = members[i]
            execute(path)
            def sample(role, choices):
                return {"source_reference": spec["source_reference"], "rows": [
                    {"unit_sha256": u, "choice_id": c, "weight": 1.}
                    for u, c in zip(spec[role + "_units"], choices, strict=True)]}
            anchors = sample("anchor", ("standard", "flexible", "flexible", "neither"))
            qualification.seal(plan["plan_sha256"], path, anchors)
            responses = (("standard", "flexible", "flexible", "neither"),
                         ("standard", "standard", "flexible", "flexible"),
                         ("standard", None, "flexible", "neither"))[i]
            outcomes[path.name] = sample("evaluation", responses)
            distribution = summarize_sample(outcomes[path.name], spec["evaluation_units"], ["standard", "flexible", "neither"])["distribution"]
            reveal(path, dict(zip(["standard", "flexible", "neither"], distribution, strict=True)))
            operations[path.name] = {"candidate_other_cost_usd": 1., "human_only_total_cost_usd": 4.,
                "human_only_turnaround_seconds": 60., "receipt_reference": "Generated cost fixture, not measured market pricing"}
        record = qualification.assess(plan["plan_sha256"], targets, outcomes, operations)
        report = qualification.load_assessment(record["assessment_sha256"])["report"]
        assert report["planned_groups"] == 4 and report["failed_groups"] == 1 and report["scored_groups"] == 3
        by_id = {row["study_id"]: row for row in report["rows"]}
        assert by_id["l12-benefit"]["baseline_improvement"]["synthetic_only"] > 0
        assert by_id["l12-harm"]["baseline_improvement"]["synthetic_only"] < 0
        assert not by_id["l12-missing-response"]["missingness_gate"]
        assert report["status"] == "FAIL" and report["customer_qualified"] is False
        assert qualification.assess(plan["plan_sha256"], [], {}, {}) == record
        assert qualification_cli(["export", "--catalog", str(qualification.root), "--assessment", record["assessment_sha256"], "--output", str(output)]) == 0
        summary = {"schema_version": "rival.qualification-rehearsal.v1", "created_at": utc_now().isoformat(),
            "scope": "Generated pipeline and accounting fixtures. No human accuracy or business cost evidence.",
            "elapsed_seconds": time.perf_counter() - started, "http_fixture_calls": Handler.calls,
            "real_llm_calls": 0, "api_cost_usd": 0., "simulated_draws": len(measured) * 40,
            "qualification_plan": plan, "assessment": record, "planned_qualification_groups": 4,
            "scored_qualification_groups": 3, "failed_qualification_groups": 1,
            "status": report["status"], "customer_qualified": False, "executed_studies": measured}
        (output / "rehearsal.json").write_bytes(json_bytes(summary))
        return summary
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
