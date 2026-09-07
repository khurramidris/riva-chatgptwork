"""Verify archived scientific witnesses in either a checkout or installed wheel.

The archive describes v1. It does not assert that the corrected v2 runtime is
byte-identical to v1. Runtime files are inventoried separately in release builds.
"""

from pathlib import Path
import json

from ..mega_study import protocol, stage


def manifest_path() -> Path:
    package = Path(__file__).resolve().parents[1]
    if (package.parent / "pyproject.toml").is_file():
        return package / "studies/mega_study_v1/MEGA_STUDY_MANIFEST.json"
    return package / "frozen/mega_v1/rival/studies/mega_study_v1/MEGA_STUDY_MANIFEST.json"


def load_manifest():
    return protocol.load_manifest(manifest_path())


def load_prediction_stage(root):
    return stage.load_prediction_stage(root, manifest_file=manifest_path())


def prepare_stage(root, **kwargs):
    return stage.prepare_stage(root, manifest_file=manifest_path(), **kwargs)


def materialize_outcomes(root, results, marker, **kwargs):
    from . import RUNTIME_REVISION
    from ..mega_study.utils import ProtocolError, file_hash
    marker_payload = json.loads(Path(marker).read_text(encoding="utf-8"))
    journal = Path(results).with_suffix(Path(results).suffix + ".attempts.sqlite3")
    if (marker_payload.get("runtime_revision") != RUNTIME_REVISION or not journal.is_file()
            or file_hash(journal) != marker_payload.get("attempt_journal_sha256")):
        raise ProtocolError("v2 outcome opening requires the frozen v2 attempt journal")
    if (Path(root) / "sealed").exists():
        raise ProtocolError("outcomes already materialized; use the existing evidence")
    return stage.materialize_outcomes(root, results, marker, manifest_file=manifest_path(), **kwargs)
