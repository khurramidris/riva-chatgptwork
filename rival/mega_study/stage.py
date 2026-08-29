"""Download, verify, and capability-separate the frozen benchmark data."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .constants import EXPECTED_OUTCOMES, FORBIDDEN_PREDICTION_KEYS, SCHEMA_VERSION
from .outcomes import extract_outcome_cells, human_response_object, survey_question_chunks
from .protocol import load_cohort, load_manifest, verify_source_file
from .retrieval import demographics_text, evidence_items
from .utils import (
    LeakageError,
    ProtocolError,
    atomic_json,
    canonical_hash,
    file_hash,
    read_jsonl,
    text_hash,
    write_jsonl,
)


ProgressCallback = Callable[[dict[str, Any]], None]


def _download(source: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        str(source["url"]),
        headers={"User-Agent": "Rival-Mega-Study/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
            "wb"
        ) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProtocolError(f"failed to download {source['source_id']}") from exc
    try:
        verify_source_file(temporary, source)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)


def acquire_sources(
    manifest: dict[str, Any],
    cache_root: str | Path,
    *,
    allow_download: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Path]:
    root = Path(cache_root)
    paths: dict[str, Path] = {}
    for source in manifest["sources"]:
        destination = root / str(source["local_name"])
        try:
            verify_source_file(destination, source)
        except ProtocolError:
            if not allow_download:
                raise
            if progress:
                progress({"event": "download", "source_id": source["source_id"]})
            _download(source, destination)
        verify_source_file(destination, source)
        paths[str(source["source_id"])] = destination
    return paths


def _load_selected_personas(
    persona_paths: list[Path], participant_ids: set[str]
) -> dict[str, dict[str, Any]]:
    numeric_ids = sorted(pid.removeprefix("pid_") for pid in participant_ids)
    frames: list[pd.DataFrame] = []
    for path in persona_paths:
        frame = pd.read_parquet(
            path,
            columns=["pid", "persona_text", "persona_json"],
            filters=[("pid", "in", numeric_ids)],
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ProtocolError("no frozen participants mapped to original Twin-2K personas")
    combined = pd.concat(frames, ignore_index=True)
    combined["pid"] = combined["pid"].astype(str).map(lambda value: f"pid_{value}")
    if combined["pid"].duplicated().any():
        raise ProtocolError("original Twin-2K persona data contains duplicate PIDs")
    personas = {
        str(row.pid): {
            "pid": str(row.pid),
            "persona_text": str(row.persona_text),
            "persona_json": str(row.persona_json),
        }
        for row in combined.itertuples(index=False)
    }
    missing = sorted(participant_ids - set(personas))
    if missing:
        raise ProtocolError(f"{len(missing)} frozen PIDs have no original persona")
    return personas


def _prediction_rows(
    manifest: dict[str, Any],
    cohort: list[dict[str, Any]],
    sources: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_study: dict[str, pd.DataFrame] = {}
    for study in manifest["development_studies"]:
        # This column projection is the primary leakage firewall.  The
        # answer-bearing column is not read into the prediction process.
        frame = pd.read_parquet(
            sources[f"mega:{study}"], columns=["PID", "survey_text"]
        )
        frame["PID"] = frame["PID"].astype(str)
        if frame["PID"].duplicated().any():
            raise ProtocolError(f"{study} contains duplicate PIDs")
        by_study[study] = frame.set_index("PID")
    participant_ids = {str(row["pid"]) for row in cohort}
    persona_paths = [
        path for source_id, path in sources.items() if source_id.startswith("persona:")
    ]
    personas = _load_selected_personas(persona_paths, participant_ids)
    case_rows: list[dict[str, Any]] = []
    persona_rows: list[dict[str, Any]] = []
    for pid in sorted(personas, key=lambda value: int(value.removeprefix("pid_"))):
        persona = personas[pid]
        if text_hash(persona["persona_text"]) != next(
            str(row["persona_text_sha256"])
            for row in cohort
            if str(row["pid"]) == pid
        ):
            raise ProtocolError(f"persona_text hash drifted for {pid}")
        if text_hash(persona["persona_json"]) != next(
            str(row["persona_json_sha256"])
            for row in cohort
            if str(row["pid"]) == pid
        ):
            raise ProtocolError(f"persona_json hash drifted for {pid}")
        demographics = demographics_text(persona["persona_json"])
        history_count = len(evidence_items(persona["persona_json"]))
        persona_rows.append(
            {
                **persona,
                "demographics": demographics,
                "demographics_sha256": text_hash(demographics),
                "historical_evidence_count": history_count,
            }
        )
    for frozen in cohort:
        study, pid = str(frozen["study_id"]), str(frozen["pid"])
        try:
            survey_text = str(by_study[study].loc[pid, "survey_text"])
        except KeyError as exc:
            raise ProtocolError(f"frozen case {study}/{pid} is missing") from exc
        if text_hash(survey_text) != frozen["survey_text_sha256"]:
            raise ProtocolError(f"survey_text hash drifted for {study}/{pid}")
        question_count = len(survey_question_chunks(survey_text))
        case_rows.append(
            {
                "case_id": frozen["case_id"],
                "study_id": study,
                "pid": pid,
                "selection_rank": int(frozen["selection_rank"]),
                "survey_text": survey_text,
                "survey_text_sha256": frozen["survey_text_sha256"],
                "expected_outcome_ids": list(EXPECTED_OUTCOMES[study]),
                "answerable_question_count": question_count,
                "variant_order": list(frozen["variant_order"]),
            }
        )
    return case_rows, persona_rows, personas


def _sealed_outcome_rows(
    manifest: dict[str, Any],
    cohort: list[dict[str, Any]],
    sources: dict[str, Path],
) -> list[dict[str, Any]]:
    """Read the protected column only after prediction inputs have been frozen."""

    by_study: dict[str, pd.DataFrame] = {}
    for study in manifest["development_studies"]:
        frame = pd.read_parquet(
            sources[f"mega:{study}"],
            columns=["PID", "survey_text", "survey_json_with_human_response"],
        )
        frame["PID"] = frame["PID"].astype(str)
        by_study[study] = frame.set_index("PID")
    rows: list[dict[str, Any]] = []
    for frozen in cohort:
        study, pid = str(frozen["study_id"]), str(frozen["pid"])
        record = by_study[study].loc[pid]
        protected = str(record["survey_json_with_human_response"])
        try:
            survey = json.loads(protected)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"protected response is invalid JSON for {study}/{pid}") from exc
        response = human_response_object(survey)
        cells = extract_outcome_cells(study, str(record["survey_text"]), response)
        rows.append(
            {
                "case_id": frozen["case_id"],
                "study_id": study,
                "pid": pid,
                "cells": [cell.as_dict() for cell in cells],
                "protected_response_sha256": text_hash(protected),
            }
        )
    return rows


def _audit_prediction_package(
    cases_path: Path,
    personas_path: Path,
    *,
    expected_cases: int,
) -> dict[str, Any]:
    cases, personas = read_jsonl(cases_path), read_jsonl(personas_path)
    if len(cases) != expected_cases:
        raise LeakageError("prediction package case count drifted")
    allowed_case_keys = {
        "case_id",
        "study_id",
        "pid",
        "selection_rank",
        "survey_text",
        "survey_text_sha256",
        "expected_outcome_ids",
        "answerable_question_count",
        "variant_order",
    }
    allowed_persona_keys = {
        "pid",
        "persona_text",
        "persona_json",
        "demographics",
        "demographics_sha256",
        "historical_evidence_count",
    }
    for row in cases:
        if set(row) != allowed_case_keys:
            raise LeakageError("prediction case contains an unapproved field")
    for row in personas:
        if set(row) != allowed_persona_keys:
            raise LeakageError("prediction persona contains an unapproved field")
    lowered_keys = {
        str(key).casefold()
        for row in [*cases, *personas]
        for key in row
    }
    collision = lowered_keys & FORBIDDEN_PREDICTION_KEYS
    if collision:
        raise LeakageError(f"protected keys entered prediction package: {sorted(collision)}")
    return {
        "status": "PASS",
        "phase": "prediction_preparation",
        "prediction_source_projection": ["PID", "survey_text"],
        "excluded_source_field": "survey_json_with_human_response",
        "protected_outcomes_materialized": False,
        "prediction_case_keys": sorted(allowed_case_keys),
        "prediction_persona_keys": sorted(allowed_persona_keys),
        "prediction_cases": len(cases),
        "prediction_personas": len(personas),
        "cases_sha256": file_hash(cases_path),
        "personas_sha256": file_hash(personas_path),
    }


def prepare_stage(
    output_root: str | Path,
    *,
    manifest_file: str | Path | None = None,
    source_cache: str | Path | None = None,
    allow_download: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_file)
    cohort = load_cohort(manifest)
    root = Path(output_root)
    if (root / "sealed").exists():
        raise ProtocolError(
            "refusing to rebuild a prediction stage after outcomes were materialized; "
            "use a new output root"
        )
    cache = Path(source_cache) if source_cache else root / "source"
    sources = acquire_sources(
        manifest,
        cache,
        allow_download=allow_download,
        progress=progress,
    )
    prediction_dir = root / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # This phase reads only answer-free survey_text plus original personas.
    # Human responses are not extracted or materialized until after predictions
    # are frozen by ``materialize_outcomes`` below.
    cases, personas, _ = _prediction_rows(manifest, cohort, sources)
    cases_path = prediction_dir / "cases.jsonl"
    personas_path = prediction_dir / "personas.jsonl"
    write_jsonl(cases_path, cases)
    write_jsonl(personas_path, personas)

    audit = _audit_prediction_package(
        cases_path,
        personas_path,
        expected_cases=len(cohort),
    )
    audit_path = root / "leakage_firewall_audit.json"
    atomic_json(audit_path, audit)
    stage_manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["manifest_sha256"],
        "cohort_sha256": manifest["cohort"]["sha256"],
        "source_sha256": {
            source_id: file_hash(path) for source_id, path in sorted(sources.items())
        },
        "prediction": {
            "cases_path": "prediction/cases.jsonl",
            "cases_sha256": file_hash(cases_path),
            "personas_path": "prediction/personas.jsonl",
            "personas_sha256": file_hash(personas_path),
        },
        "protected_outcomes_materialized": False,
        "leakage_audit_path": audit_path.name,
        "leakage_audit_sha256": file_hash(audit_path),
        "case_count": len(cases),
        "persona_count": len(personas),
    }
    stage_manifest["stage_sha256"] = canonical_hash(stage_manifest)
    atomic_json(root / "stage_manifest.json", stage_manifest)
    return stage_manifest


def _verified_freeze_marker(
    stage: dict[str, Any], results_path: Path, marker_path: Path
) -> dict[str, Any]:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("prediction freeze marker is missing or invalid") from exc
    payload = dict(marker)
    digest = payload.pop("freeze_sha256", None)
    if digest != canonical_hash(payload):
        raise ProtocolError("prediction freeze marker hash does not verify")
    if marker.get("stage_sha256") != stage["stage_sha256"]:
        raise ProtocolError("prediction freeze marker belongs to another stage")
    if marker.get("results_sha256") != file_hash(results_path):
        raise ProtocolError("prediction ledger changed after freezing")
    if marker.get("outcomes_opened") is not False:
        raise ProtocolError("freeze marker does not certify an outcome-blind run")
    return marker


def materialize_outcomes(
    output_root: str | Path,
    results_path: str | Path,
    freeze_marker: str | Path,
    *,
    manifest_file: str | Path | None = None,
    source_cache: str | Path | None = None,
    allow_download: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Open protected human answers only after every prediction is frozen."""

    manifest = load_manifest(manifest_file)
    root = Path(output_root)
    stage, _, _ = load_prediction_stage(root, manifest_file=manifest_file)
    marker = _verified_freeze_marker(stage, Path(results_path), Path(freeze_marker))
    cohort = load_cohort(manifest)
    cache = Path(source_cache) if source_cache else root / "source"
    sources = acquire_sources(
        manifest,
        cache,
        allow_download=allow_download,
        progress=progress,
    )
    outcomes = _sealed_outcome_rows(manifest, cohort, sources)
    sealed_dir = root / "sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = sealed_dir / "outcomes.jsonl"
    write_jsonl(outcomes_path, outcomes)
    outcome_manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["manifest_sha256"],
        "stage_sha256": stage["stage_sha256"],
        "prediction_freeze_sha256": marker["freeze_sha256"],
        "results_sha256": marker["results_sha256"],
        "outcomes_path": "outcomes.jsonl",
        "outcomes_sha256": file_hash(outcomes_path),
        "case_count": len(outcomes),
        "outcome_cell_count": sum(len(row["cells"]) for row in outcomes),
        "protected_source_field": "survey_json_with_human_response",
        "materialized_after_prediction_freeze": True,
        "source_sha256": {
            source_id: file_hash(path)
            for source_id, path in sorted(sources.items())
            if source_id.startswith("mega:")
        },
    }
    outcome_manifest["outcome_manifest_sha256"] = canonical_hash(outcome_manifest)
    outcome_manifest_path = sealed_dir / "outcome_manifest.json"
    atomic_json(outcome_manifest_path, outcome_manifest)
    try:
        os.chmod(sealed_dir, 0o700)
        os.chmod(outcomes_path, 0o600)
        os.chmod(outcome_manifest_path, 0o600)
    except OSError:
        pass
    return outcome_manifest


def load_prediction_stage(
    root: str | Path, *, manifest_file: str | Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    base = Path(root)
    manifest = load_manifest(manifest_file)
    try:
        stage = json.loads((base / "stage_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("staged Mega-Study data is missing or invalid") from exc
    expected = dict(stage)
    digest = expected.pop("stage_sha256", None)
    if digest != canonical_hash(expected):
        raise ProtocolError("stage manifest hash does not verify")
    if stage.get("protocol_sha256") != manifest["manifest_sha256"]:
        raise ProtocolError("stage was built from a different frozen protocol")
    if stage.get("protected_outcomes_materialized") is not False:
        raise LeakageError("prediction stage does not certify outcome-free preparation")
    cases_path = base / stage["prediction"]["cases_path"]
    personas_path = base / stage["prediction"]["personas_path"]
    if file_hash(cases_path) != stage["prediction"]["cases_sha256"]:
        raise ProtocolError("staged prediction cases drifted")
    if file_hash(personas_path) != stage["prediction"]["personas_sha256"]:
        raise ProtocolError("staged personas drifted")
    audit_path = base / stage["leakage_audit_path"]
    if file_hash(audit_path) != stage["leakage_audit_sha256"]:
        raise ProtocolError("leakage audit drifted")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise LeakageError("staged leakage firewall did not pass")
    cases = read_jsonl(cases_path)
    personas = {str(row["pid"]): row for row in read_jsonl(personas_path)}
    return stage, cases, personas
