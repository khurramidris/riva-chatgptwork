"""V1 freeze checks with installed-resource resolution; v2 marker binding is added by the CLI."""
from pathlib import Path
from typing import Any
from ..mega_study.constants import SCHEMA_VERSION, VARIANTS
from ..mega_study.runner import _latest_ledger, work_id, utc_now
from ..mega_study.utils import ProtocolError, file_hash, canonical_hash, atomic_json
from .resources import load_manifest, load_prediction_stage


def freeze_predictions(
    stage_root: str | Path,
    results_path: str | Path,
    marker_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest()
    stage, cases, _ = load_prediction_stage(stage_root)
    results = Path(results_path)
    latest = _latest_ledger(results)
    required = {
        work_id(str(case["case_id"]), variant)
        for case in cases
        for variant in VARIANTS
    }
    missing = sorted(required - set(latest))
    if missing:
        raise ProtocolError(f"cannot freeze predictions: {len(missing)} work items missing")
    terminal = {"SUCCESS", "PARSE_FAILURE", "API_FAILURE", "CONTEXT_FAILURE"}
    nonterminal = sorted(
        identifier for identifier in required if latest[identifier].get("status") not in terminal
    )
    if nonterminal:
        raise ProtocolError(
            f"cannot freeze predictions: {len(nonterminal)} work items are nonterminal"
        )
    mismatched = sorted(
        identifier
        for identifier in required
        if latest[identifier].get("protocol_sha256") != manifest["manifest_sha256"]
        or latest[identifier].get("stage_sha256") != stage["stage_sha256"]
        or latest[identifier].get("provider") != manifest["provider_identity"]
    )
    if mismatched:
        raise ProtocolError(
            f"cannot freeze predictions: {len(mismatched)} work items have "
            "protocol, stage, or provider drift"
        )
    status_counts: dict[str, int] = {}
    for identifier in required:
        status = str(latest[identifier]["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["manifest_sha256"],
        "stage_sha256": stage["stage_sha256"],
        "results_sha256": file_hash(results),
        "required_work_items": len(required),
        "status_counts": status_counts,
        "frozen_at": utc_now(),
        "outcomes_opened": False,
    }
    payload["freeze_sha256"] = canonical_hash(payload)
    atomic_json(marker_path, payload)
    return payload
