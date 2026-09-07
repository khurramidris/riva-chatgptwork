from __future__ import annotations

import hashlib
import json
import zipfile
import shutil
from pathlib import Path
from typing import Any

from .version import __version__


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_manifest(
    manifest_path: str | Path = "RELEASE_MANIFEST.json",
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def file_check(relative: str, expected: str) -> bool:
        candidate = (root / relative).resolve()
        within_root = candidate == root or root in candidate.parents
        if not within_root:
            checks.append(
                {"path": relative, "status": "FAIL", "reason": "path escapes root"}
            )
            return False
        if not candidate.is_file():
            checks.append(
                {"path": relative, "status": "FAIL", "reason": "missing"}
            )
            return False
        actual = sha256_file(candidate)
        checks.append(
            {
                "path": relative,
                "status": "PASS" if actual == expected else "FAIL",
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
        return actual == expected

    wheel = manifest["wheel"]
    wheel_valid = file_check(wheel["path"], wheel["sha256"])
    for relative, expected in manifest.get("qualification_artifacts", {}).items():
        file_check(relative, expected)
    source_inventory = manifest.get("source_inventory")
    inventory_valid = False
    if source_inventory:
        inventory_valid = file_check(source_inventory, manifest.get("source_inventory_sha256"))
    if manifest.get("schema_version") == "rival.release-manifest.v2":
        checks.append({"path": "required-release-inventory", "status": "PASS" if inventory_valid else "FAIL"})
    wheel_path = (root / wheel["path"]).resolve()
    if wheel_path.suffix == ".whl" and wheel_valid:
        wheel_version: str | None = None
        try:
            with zipfile.ZipFile(wheel_path) as archive:
                if inventory_valid:
                    inventory = json.loads((root / source_inventory).read_text(encoding="utf-8"))
                    observed = {name: hashlib.sha256(archive.read(name)).hexdigest()
                                for name in archive.namelist() if not name.endswith("/")}
                    checks.append({"path": "wheel-file-inventory", "status": "PASS" if (
                        len(observed) == len([name for name in archive.namelist() if not name.endswith("/")])
                        and inventory.get("wheel_files") == observed) else "FAIL"})
                metadata_names = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) == 1:
                    metadata = archive.read(metadata_names[0]).decode("utf-8")
                    for line in metadata.splitlines():
                        if line.startswith("Version: "):
                            wheel_version = line.removeprefix("Version: ").strip()
                            break
        except (OSError, ValueError, UnicodeDecodeError, zipfile.BadZipFile):
            wheel_version = None
        checks.append(
            {
                "path": "wheel-metadata-version",
                "status": "PASS" if manifest.get("release") == wheel_version else "FAIL",
                "expected": manifest.get("release"),
                "actual": wheel_version,
            }
        )
    else:
        # Non-wheel artifacts are supported for verifier unit tests. A real
        # release should always take the wheel-metadata path above.
        checks.append(
            {
                "path": "package-version",
                "status": "PASS" if manifest.get("schema_version") != "rival.release-manifest.v2"
                and manifest.get("release") == __version__ and wheel_valid else "FAIL",
                "expected": manifest.get("release"),
                "actual": __version__,
            }
        )
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "manifest": str(path),
        "release": manifest.get("release"),
        "checks": checks,
        "scope": "artifact integrity only; no predictive or customer qualification",
    }


def build_release_manifest(wheel_path: str | Path, output_dir: str | Path,
                           evidence_paths: list[str | Path] = ()) -> Path:
    """Create a reviewable wheel inventory and hash-bound verification bundle."""
    from .readiness import release_claims
    wheel_path = Path(wheel_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    wheel_copy = destination / wheel_path.name
    if wheel_copy != wheel_path:
        shutil.copy2(wheel_path, wheel_copy)
    with zipfile.ZipFile(wheel_copy) as archive:
        wheel_files = {name: hashlib.sha256(archive.read(name)).hexdigest()
                       for name in archive.namelist() if not name.endswith("/")}
    inventory = destination / "wheel_inventory.json"
    inventory.write_text(json.dumps({"release": __version__, "wheel_files": wheel_files},
                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {}
    for source in evidence_paths:
        source = Path(source).resolve()
        relative = "verification/" + source.name
        target = destination / relative
        target.parent.mkdir(exist_ok=True)
        if target != source:
            shutil.copy2(source, target)
        artifacts[relative] = sha256_file(target)
    manifest = {"schema_version": "rival.release-manifest.v2", "release": __version__,
                "release_claims": release_claims(), "wheel": {"path": wheel_copy.name, "sha256": sha256_file(wheel_copy)},
                "source_inventory": inventory.name, "source_inventory_sha256": sha256_file(inventory),
                "qualification_artifacts": artifacts}
    path = destination / "RELEASE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verify_release_manifest(path)["status"] != "PASS":
        raise ValueError("built release manifest does not verify")
    return path
