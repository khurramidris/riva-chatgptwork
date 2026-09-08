"""Install a wheel offline and exercise it from outside the source checkout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROBE = r'''
import json
from pathlib import Path
import rival
from rival.commands import main
from rival.demo import run_demo
from rival.mega_study_v2.resources import load_manifest, load_prediction_stage
from rival.mega_study_v2.freeze import freeze_predictions
from rival.mega_study.utils import atomic_json, write_jsonl, file_hash, canonical_hash
from rival.readiness import release_claims
from rival.research.qualification import load_bundled_summary
from rival.version import __version__

package = Path(rival.__file__).resolve().parent
install_root = Path(Path("INSTALL_ROOT").read_text(encoding="utf-8")).resolve()
assert package.is_relative_to(install_root), (package, install_root)
manifest = load_manifest()
assert manifest["manifest_sha256"] == "5fa5cdf8ee9e1e802a18f7c03b0fb756b0359011add037df42728a425aff05c0"
for filename in ("syn_digits_LICENSE.txt", "uq_survey_LICENSE.txt"):
    assert (package / "notices" / filename).read_text(encoding="utf-8").rstrip().endswith("SOFTWARE.")
assert (package / "notices/THIRD_PARTY_NOTICES.md").is_file()
assert (package / "studies/mega_study_syn_digits_v1/CALIBRATION_DESIGN.json").is_file()
demo = run_demo(sample_size=40, human_anchor_size=5)
assert demo["simulation"]["confidence"]["abstain"] is True
assert demo["simulation"]["confidence"]["label"] == "unqualified"
assert load_bundled_summary()["status"] == "HISTORICAL_ONLY"

# The new study commands must be usable from the installed package, including
# their generated example, saved execution state and aggregate report export.
from rival.study_commands import example_request
from rival.study_contract import StudyRequest
from rival.study_workflow import prepare_study, run_study, export_study
study_input = example_request()
study_input["brief"]["sample_size"] = 40
prepare_study("workflow-study", StudyRequest.model_validate(study_input))
assert run_study("workflow-study")["completed_draws"] == 40
assert run_study("workflow-study")["accounting"]["attempts"] == 0
export_study("workflow-study", "workflow-report")
study_report = json.loads(Path("workflow-report/report.json").read_text(encoding="utf-8"))
assert study_report["confidence"]["abstain"] is True
assert study_report["audience"]["simulation_draws"] == 40

from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, strict_json
from rival.evidence_commands import example_files
from rival.study_evidence import bind_evidence
from rival.study_workflow import check_study
example = example_files()
Path("evidence-source.csv").write_bytes(example["people.csv"])
catalog = EvidenceCatalog("workflow-catalog")
bundle = catalog.import_file("evidence-source.csv", EvidenceImportSpec.model_validate(strict_json(example["import.json"])))
brief = strict_json(example["brief.json"])
brief["brief"]["sample_size"] = 40
request = bind_evidence(brief, catalog.root, [bundle.bundle_sha256], strict_json(example["support.json"]))
assert check_study(request, catalog_root=catalog.root)["support_audit"]["passed"]
prepare_study("workflow-study-v2", request, catalog_root=catalog.root)
assert run_study("workflow-study-v2")["completed_draws"] == 40
assert run_study("workflow-study-v2")["accounting"]["attempts"] == 0
export_study("workflow-study-v2", "workflow-report-v2")
assert json.loads(Path("workflow-report-v2/report.json").read_text())["evidence"]["import_verification"]["raw_bytes_and_conversion_verified"]

# Exercise the real local HTTP surface, without model/network dependencies.
from http.server import ThreadingHTTPServer
from threading import Thread
import urllib.request
from rival.server import RivalApplication, make_handler
app = RivalApplication("http-evidence.sqlite3")
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
worker = Thread(target=server.serve_forever, daemon=True)
worker.start()
base = "http://127.0.0.1:" + str(server.server_address[1])
try:
    with urllib.request.urlopen(base + "/api/health") as response:
        assert json.load(response)["release_claims"]["customer_launch_ready"] is False
    with urllib.request.urlopen(base + "/") as response:
        assert "generates people, anchors and outcomes" in response.read().decode()
    request = urllib.request.Request(base + "/api/demo/run", method="POST",
        data=json.dumps({"sample_size": 40, "human_anchor_size": 5}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        assert json.load(response)["simulation"]["confidence"]["abstain"] is True
finally:
    server.shutdown()
    server.server_close()
    worker.join()
    app.engine.store.close()

# Exercise both installed v3 elicitation paths through real loopback HTTP.
# Responses here are generated fixtures, not model-accuracy evidence.
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from rival.schemas import utc_now
from rival.study_execution_audit import audit_study_execution
class ModelHandler(BaseHTTPRequestHandler):
    calls = 0
    def log_message(self, *args):
        pass
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).calls += 1
        text = ('{"standard":0.6,"flexible":0.3,"neither":0.1}' if 'response_format' in payload
                else 'I prefer standard delivery.')
        response = {'id': 'fixture-' + str(type(self).calls), 'model': 'fixture-model',
            'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
            'usage': {'cost': 0.0, 'prompt_tokens': 10, 'completion_tokens': 5}}
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
model_server = ThreadingHTTPServer(('127.0.0.1', 0), ModelHandler)
model_worker = Thread(target=model_server.serve_forever, daemon=True)
model_worker.start()
model_requests = {}
try:
    for method in ('direct', 'ssr'):
        brief = strict_json(example['brief.json'])
        brief['schema_version'] = 'rival.study-request.v3'
        brief['brief'].update(study_id='installed-v3-' + method, sample_size=40)
        brief['execution'] = {'mode': 'managed', 'model': 'fixture-model',
            'base_url': 'http://127.0.0.1:' + str(model_server.server_address[1]) + '/chat/completions',
            'budget_usd': 0.1, 'reservation_usd': 0.01, 'max_attempts': 6,
            'not_after': (utc_now() + timedelta(hours=1)).isoformat(), 'generation_seed': 42,
            'model_pin': {'kind': 'hosted_endpoint', 'revision': 'generated-fixture-v1',
                         'reference': 'Test responses, no actual model', 'expected_response_model': 'fixture-model'},
            'elicitation': {'method': method}}
        if method == 'ssr':
            brief['execution']['elicitation']['embedding'] = {'kind': 'hashing'}
        request = bind_evidence(brief, catalog.root, [bundle.bundle_sha256], strict_json(example['support.json']))
        model_requests[method] = request
        workspace = 'workflow-study-v3-' + method
        prepare_study(workspace, request, catalog_root=catalog.root)
        assert run_study(workspace, api_key='test-local-key')['complete']
        count = ModelHandler.calls
        assert run_study(workspace)['complete'] and ModelHandler.calls == count
        assert audit_study_execution(workspace)['accepted_seed_requests'] == 6
        export_study(workspace, 'workflow-report-v3-' + method)
    assert ModelHandler.calls == 12
    # Fit only verified training studies, then bind a v4 target before reveal.
    from rival.calibration_catalog import CalibrationCatalog
    from rival.mathx import canonical_hash as study_hash
    from rival.outcome_vault import OutcomeVault
    from rival.study_contract import parse_study_request
    from rival.study_workflow import _workspace, evaluate_study
    calibration = CalibrationCatalog('workflow-calibration')
    references = []
    for index in range(3):
        payload = model_requests['direct'].model_dump(mode='json')
        identifier = 'installed-calibration-' + str(index)
        payload['brief'].update(study_id=identifier, question='Which delivery plan fits scenario ' + str(index) + '?',
                                information_cutoff=utc_now().isoformat())
        payload['evidence'].update(role='training' if index < 2 else 'evaluation', group_id=identifier)
        if index == 2:
            payload.update(schema_version='rival.study-request.v4', calibration={'adapter_sha256': fitted['adapter_sha256']})
        request = parse_study_request(payload)
        prepare_study(identifier, request, catalog_root=catalog.root,
                      calibration_catalog_root=calibration.root if index == 2 else None)
        assert run_study(identifier, api_key='test-local-key')['complete']
        count = ModelHandler.calls
        assert run_study(identifier)['complete'] and ModelHandler.calls == count
        with _workspace(identifier) as workspace, workspace.execution() as session:
            _, sealed = workspace.require_complete(workspace.journal_snapshot(session))
        vault = OutcomeVault(identifier + '-vault.sqlite3')
        try:
            vault.deposit(identifier, study_hash(sealed), {'distribution': {'standard': 0.2, 'flexible': 0.5, 'neither': 0.3},
                          'source': 'Generated test outcomes'}, 'installed-generated-outcome-key', utc_now() - timedelta(seconds=1))
        finally:
            vault.close()
        evaluate_study(identifier, vault.path, key_material='installed-generated-outcome-key')
        if index < 2:
            references.append(identifier)
            if index == 1:
                bank = calibration.create_bank(references)
                fitted = calibration.fit(bank['bank_sha256'])
        else:
            export_study(identifier, 'workflow-calibration-report')
            report = json.loads(Path('workflow-calibration-report/report.json').read_text())
            assert report['schema_version'] == 'rival.study-report.v4'
            assert report['calibration']['comparison']['tvd_reduction_vs_raw'] > 0.3
            assert report['confidence']['label'] == 'unqualified'
    assert ModelHandler.calls == 30
finally:
    model_server.shutdown()
    model_server.server_close()
    model_worker.join()

# A tiny generated fixture exercises the actual installed stage loader/freeze
# without downloading survey answers or making provider requests.
root = Path("fixture-stage")
root.mkdir()
write_jsonl(root / "cases.jsonl", [{"case_id": "fixture-case", "pid": "fixture-person"}])
write_jsonl(root / "personas.jsonl", [{"pid": "fixture-person"}])
atomic_json(root / "audit.json", {"status": "PASS"})
stage = {"protocol_sha256": manifest["manifest_sha256"], "protected_outcomes_materialized": False,
    "prediction": {"cases_path": "cases.jsonl", "cases_sha256": file_hash(root / "cases.jsonl"),
                   "personas_path": "personas.jsonl", "personas_sha256": file_hash(root / "personas.jsonl")},
    "leakage_audit_path": "audit.json", "leakage_audit_sha256": file_hash(root / "audit.json")}
stage["stage_sha256"] = canonical_hash(stage)
atomic_json(root / "stage_manifest.json", stage)
assert len(load_prediction_stage(root)[1]) == 1
write_jsonl(root / "results.jsonl", [{"work_id": "fixture-case::" + variant, "status": "CONTEXT_FAILURE",
    "protocol_sha256": manifest["manifest_sha256"], "stage_sha256": stage["stage_sha256"],
    "provider": manifest["provider_identity"]} for variant in manifest["variants"]])
marker = freeze_predictions(root, root / "results.jsonl", root / "freeze.json")
assert marker["required_work_items"] == 4
print(json.dumps({"status": "PASS", "version": __version__, "package_location": str(package),
                  "archived_protocol_sha256": manifest["manifest_sha256"], "paid_calls": 0,
                  "checks": ["installed imports", "all frozen witnesses", "complete notices", "E/F design resources",
                             "offline demo", "research confidence", "historical claims separation", "real stage loader", "freeze algorithm",
                             "local HTTP health, demo page and demo execution",
                             "study preparation, execution, recovery and report export",
                             "pinned evidence import, population support and v2 study execution",
                             "pinned v3 direct/SSR HTTP workflow, measurements and cached recovery",
                             "protected training bank, runtime calibration, v4 sealing and held-out comparison"]}))
'''


def run_checked(command, **kwargs):
    """Preserve child diagnostics when an isolated installation check fails."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="rival-installed-") as directory:
        root = Path(directory)
        install = root / "installed"
        venv = root / "console-venv"
        cwd = root / "outside-checkout"
        cwd.mkdir()
        base_environment = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        base_environment.pop("PYTHONPATH", None)
        environment = {**base_environment, "PYTHONPATH": str(install)}
        run_checked([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                     "--target", str(install), str(wheel)], cwd=cwd, env=environment)
        (cwd / "INSTALL_ROOT").write_text(str(install), encoding="utf-8")
        result = run_checked([sys.executable, "-c", PROBE], cwd=cwd, env=environment)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        for command in (["--version"], ["status"], ["mega-v2", "verify-resources"],
                        ["mega-v2", "run", "--help"], ["simulate-managed", "--help"],
                        ["study", "status", "--workspace", "workflow-study"]):
            run_checked([sys.executable, "-m", "rival", *command], cwd=cwd, env=environment)
        report["checks"].append("installed module CLI commands")
        # ``pip --target`` intentionally installs import files without console
        # wrappers on Windows. Install the same wheel into a disposable venv so
        # the platform-specific launcher is exercised as well.
        run_checked([sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
                    cwd=cwd, env=base_environment)
        venv_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run_checked([str(venv_python), "-m", "pip", "install", "--ignore-installed", "--no-index",
                     "--no-deps", str(wheel)], cwd=cwd, env=base_environment)
        launcher = venv / ("Scripts/rival.exe" if os.name == "nt" else "bin/rival")
        run_checked([str(launcher), "--version"], cwd=cwd, env=base_environment)
        report["checks"].append("installed console entry point")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
