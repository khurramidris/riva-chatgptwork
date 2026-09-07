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
                             "local HTTP health, demo page and demo execution"]}))
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
        environment = {**base_environment, "PYTHONPATH": str(install)}
        run_checked([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                     "--target", str(install), str(wheel)], cwd=cwd, env=environment)
        (cwd / "INSTALL_ROOT").write_text(str(install), encoding="utf-8")
        result = run_checked([sys.executable, "-c", PROBE], cwd=cwd, env=environment)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        for command in (["--version"], ["status"], ["mega-v2", "verify-resources"],
                        ["mega-v2", "run", "--help"], ["simulate-managed", "--help"]):
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
