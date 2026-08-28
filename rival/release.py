from __future__ import annotations

import hashlib
import json
import zipfile
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

    def file_check(relative: str, expected: str) -> None:
        candidate = (root / relative).resolve()
        within_root = candidate == root or root in candidate.parents
        if not within_root:
            checks.append(
                {"path": relative, "status": "FAIL", "reason": "path escapes root"}
            )
            return
        if not candidate.is_file():
            checks.append(
                {"path": relative, "status": "FAIL", "reason": "missing"}
            )
            return
        actual = sha256_file(candidate)
        checks.append(
            {
                "path": relative,
                "status": "PASS" if actual == expected else "FAIL",
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )

    wheel = manifest["wheel"]
    file_check(wheel["path"], wheel["sha256"])
    for relative, expected in manifest.get("qualification_artifacts", {}).items():
        file_check(relative, expected)
    source_inventory = manifest.get("source_inventory")
    if source_inventory:
        candidate = root / source_inventory
        checks.append(
            {
                "path": source_inventory,
                "status": "PASS" if candidate.is_file() else "FAIL",
                "reason": "present" if candidate.is_file() else "missing",
            }
        )
    wheel_path = (root / wheel["path"]).resolve()
    if wheel_path.suffix == ".whl" and wheel_path.is_file():
        wheel_version: str | None = None
        try:
            with zipfile.ZipFile(wheel_path) as archive:
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
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile):
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
                "status": "PASS" if manifest.get("release") == __version__ else "FAIL",
                "expected": manifest.get("release"),
                "actual": __version__,
            }
        )
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "manifest": str(path),
        "release": manifest.get("release"),
        "checks": checks,
    }
