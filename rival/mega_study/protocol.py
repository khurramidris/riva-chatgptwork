"""Verification of the frozen Mega-Study preregistration and cohort."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .constants import (
    EXPECTED_OUTCOMES,
    MODEL_CONFIG,
    RETRIEVAL_CONFIG,
    SCHEMA_VERSION,
    VARIANTS,
)
from .prompts import template_hashes
from .utils import ProtocolError, canonical_hash, file_hash, read_jsonl, text_hash


def study_directory() -> Path:
    return Path(str(files("rival").joinpath("studies", "mega_study_v1")))


def manifest_path() -> Path:
    return study_directory() / "MEGA_STUDY_MANIFEST.json"


def _portable_text_hash(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline=None) as handle:
        return text_hash(handle.read())


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_hash(payload)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else manifest_path()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read Mega-Study manifest {source}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("Mega-Study manifest schema version does not match code")
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise ProtocolError("Mega-Study manifest hash does not verify")
    if tuple(manifest.get("variants", ())) != VARIANTS:
        raise ProtocolError("frozen A/B/C/D variants drifted")
    if manifest.get("model_config") != MODEL_CONFIG:
        raise ProtocolError("frozen model/provider configuration drifted")
    if manifest.get("retrieval_config") != RETRIEVAL_CONFIG:
        raise ProtocolError("frozen retrieval configuration drifted")
    if manifest.get("prompt_template_sha256") != template_hashes():
        raise ProtocolError("frozen prompt templates drifted")
    if manifest.get("outcome_ids") != {
        key: list(value) for key, value in EXPECTED_OUTCOMES.items()
    }:
        raise ProtocolError("frozen outcome map drifted")
    cohort_path = source.parent / str(manifest["cohort"]["path"])
    if _portable_text_hash(cohort_path) != manifest["cohort"]["sha256"]:
        raise ProtocolError("frozen Mega-Study cohort hash does not verify")
    witness = manifest.get("preserved_wave4_witness", {})
    repository_root = source.parents[3]
    for relative, expected in witness.items():
        observed_path = repository_root / relative
        if not observed_path.exists() or _portable_text_hash(observed_path) != expected:
            raise ProtocolError(
                f"preserved Wave-4 artifact changed or is missing: {relative}"
            )
    implementation = manifest.get("implementation_witness", {})
    if not implementation:
        raise ProtocolError("Mega-Study manifest has no implementation witness")
    for relative, expected in implementation.items():
        observed_path = repository_root / relative
        if not observed_path.exists() or _portable_text_hash(observed_path) != expected:
            raise ProtocolError(
                f"frozen Mega-Study implementation changed or is missing: {relative}"
            )
    return manifest


def load_cohort(
    manifest: dict[str, Any], path: str | Path | None = None
) -> list[dict[str, Any]]:
    source = (
        Path(path)
        if path is not None
        else manifest_path().parent / str(manifest["cohort"]["path"])
    )
    rows = read_jsonl(source)
    if len(rows) != int(manifest["cohort"]["total_cases"]):
        raise ProtocolError("frozen cohort row count does not match manifest")
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ProtocolError("frozen cohort contains empty or duplicate case IDs")
    counts: dict[str, int] = {}
    for row in rows:
        study = str(row.get("study_id", ""))
        counts[study] = counts.get(study, 0) + 1
    if counts != manifest["cohort"]["cases_per_study"]:
        raise ProtocolError("frozen cohort study allocation does not match manifest")
    return rows


def verify_source_file(path: str | Path, source: dict[str, Any]) -> None:
    candidate = Path(path)
    if not candidate.is_file():
        raise ProtocolError(f"source file is missing: {candidate}")
    expected_size = int(source["size_bytes"])
    if candidate.stat().st_size != expected_size:
        raise ProtocolError(
            f"source size drifted for {source['source_id']}: "
            f"{candidate.stat().st_size} != {expected_size}"
        )
    if file_hash(candidate) != source["sha256"]:
        raise ProtocolError(f"source hash drifted for {source['source_id']}")


def protocol_identity(manifest: dict[str, Any]) -> str:
    return str(manifest["manifest_sha256"])
