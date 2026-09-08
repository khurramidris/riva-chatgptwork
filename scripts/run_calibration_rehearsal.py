"""Local HTTP fixture rehearsal of calibration, including a harmful adjustment.

People, HTTP responses and outcomes are generated engineering fixtures, not LLM
inference or human accuracy evidence. Raw workspaces and keys remain private.
"""

import argparse
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from threading import Thread
import time

from rival.calibration_catalog import CalibrationCatalog
from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, json_bytes, strict_json
from rival.evidence_commands import example_files
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.schemas import utc_now
from rival.study_contract import parse_study_request
from rival.study_evidence import bind_evidence
from rival.study_workflow import (_workspace, evaluate_study, export_study, prepare_study, run_study)


class Handler(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        person = json.loads(payload["messages"][1]["content"])["person"]
        probabilities = ([0.65, 0.30, 0.05] if person["attributes"]["segment"] == "value" else [0.4, 0.5, 0.1])
        type(self).calls += 1
        body = json_bytes({"id": "rehearsal-" + str(type(self).calls), "model": "generated-calibration-fixture",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(dict(zip(
                ["standard", "flexible", "neither"], probabilities, strict=True)))}}],
            "usage": {"cost": 0, "prompt_tokens": 10, "completion_tokens": 5}})
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def rehearse(root, output):
    if root.exists() or output.exists():
        raise ValueError("use new rehearsal workspace and report directories")
    root.mkdir(parents=True)
    output.mkdir(parents=True)
    examples = example_files()
    source = root / "people.csv"
    source.write_bytes(examples["people.csv"])
    catalog = EvidenceCatalog(root / "evidence")
    bundle = catalog.import_file(source, EvidenceImportSpec.model_validate(strict_json(examples["import.json"])))
    calibration = CalibrationCatalog(root / "calibration")
    Handler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    key = secrets.token_hex(32)
    measured = []
    reference_roots = []
    try:
        for index, name in enumerate(("reference-a", "reference-b", "held-out-benefit", "held-out-harm")):
            started = time.perf_counter()
            brief = strict_json(examples["brief.json"])
            brief["schema_version"] = "rival.study-request.v3"
            brief["brief"].update(study_id="l10-" + name, question="Which delivery plan suits " + name + "?",
                title="Generated calibration rehearsal: " + name, sample_size=40, information_cutoff=utc_now().isoformat())
            brief["evidence"].update(role="training" if index < 2 else "evaluation", group_id="l10-" + name)
            brief["execution"] = {"mode": "managed", "model": "generated-calibration-fixture", "generation_seed": 42,
                "base_url": "http://127.0.0.1:" + str(server.server_address[1]) + "/chat/completions",
                "budget_usd": 1, "reservation_usd": 0.01, "max_attempts": 6, "max_retries": 1,
                "not_after": (utc_now() + timedelta(hours=1)).isoformat(),
                "model_pin": {"kind": "hosted_endpoint", "revision": "generated-fixture-v1",
                    "reference": "Deterministic generated HTTP responses; no actual model weights",
                    "expected_response_model": "generated-calibration-fixture"}}
            if index >= 2:
                brief.update(schema_version="rival.study-request.v4", calibration={"adapter_sha256": fitted["adapter_sha256"]})
            request = bind_evidence(brief, catalog.root, [bundle.bundle_sha256], strict_json(examples["support.json"]))
            workspace_root = root / name
            prepare_study(workspace_root, request, catalog_root=catalog.root,
                          calibration_catalog_root=calibration.root if index >= 2 else None)
            before_calls = Handler.calls
            status = run_study(workspace_root, api_key="loopback-generated-fixture")
            assert status["complete"] and Handler.calls - before_calls == 6
            calls = Handler.calls
            assert run_study(workspace_root)["complete"] and Handler.calls == calls
            if index >= 2:
                export_study(workspace_root, output / (name + "-before-outcomes"))
            with _workspace(workspace_root) as workspace, workspace.execution() as session:
                result, sealed = workspace.require_complete(workspace.journal_snapshot(session))
            # The final case deliberately has a different generated data process.
            # This exercises negative improvement reporting, not model selection.
            observed = result.distribution if name == "held-out-harm" else {"standard": 0.25, "flexible": 0.50, "neither": 0.25}
            vault = OutcomeVault(root / (name + "-outcomes.sqlite3"))
            try:
                vault.deposit(request.brief.study_id, canonical_hash(sealed), {"distribution": observed,
                    "source_reference": "Generated fixture outcomes, not people"}, key, utc_now() - timedelta(seconds=1))
            finally:
                vault.close()
            evaluate_study(workspace_root, vault.path, key_material=key)
            if index < 2:
                reference_roots.append(workspace_root)
                if index == 1:
                    bank = calibration.create_bank(reference_roots)
                    fitted = calibration.fit(bank["bank_sha256"])
                    assert fitted == calibration.fit(bank["bank_sha256"])
            else:
                export_study(workspace_root, output / name)
            measured.append({"study_id": request.brief.study_id, "role": request.evidence.role,
                "planned_seeds": 6, "completed_seeds": 6, "planned_draws": 40, "completed_draws": 40,
                "http_calls": Handler.calls - before_calls, "resume_calls": 0,
                "elapsed_seconds": time.perf_counter() - started})
        report = {"schema_version": "rival.calibration-rehearsal.v1", "scope": "generated engineering fixtures",
            "human_accuracy_tested": False, "real_llm_calls": 0, "api_cost_usd": 0,
            "http_fixture_calls": Handler.calls, "bank": bank, "adapter": fitted, "studies": measured,
            "comparisons": {name: strict_json((output / name / "report.json").read_bytes())["calibration"]["comparison"]
                            for name in ("held-out-benefit", "held-out-harm")}}
        assert report["comparisons"]["held-out-benefit"]["tvd_reduction_vs_raw"] > 0
        assert report["comparisons"]["held-out-harm"]["tvd_reduction_vs_raw"] < 0
        report["report_sha256"] = canonical_hash(report)
        (output / "rehearsal.json").write_bytes(json_bytes(report))
        return report
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = rehearse(args.workspace, args.output)
    print(json.dumps({"http_fixture_calls": report["http_fixture_calls"], "real_llm_calls": 0,
                      "comparisons": report["comparisons"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
