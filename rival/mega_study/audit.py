"""No-API readiness audit for the frozen Mega-Study benchmark."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .constants import FORBIDDEN_PREDICTION_KEYS, SYSTEM_INSTRUCTION, VARIANTS
from .prompts import render_prompt, template_hashes
from .protocol import load_manifest
from .provider import OpenRouterSurveyProvider
from .replication import run_official_digital_certification_replication
from .runner import work_schedule
from .stage import load_prediction_stage
from .utils import LeakageError, ProtocolError, canonical_hash, file_hash


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def audit_readiness(
    stage_root: str | Path,
    source_cache: str | Path,
) -> dict[str, Any]:
    """Render every frozen prompt and reproduce one official result family."""

    manifest = load_manifest()
    stage, cases, personas = load_prediction_stage(stage_root)
    audit_path = Path(stage_root) / str(stage["leakage_audit_path"])
    leakage_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if leakage_audit.get("status") != "PASS":
        raise LeakageError("prediction-stage leakage audit did not pass")
    if leakage_audit.get("protected_outcomes_materialized") is not False:
        raise LeakageError("protected outcomes were materialized before prediction")
    forbidden = {value.casefold() for value in FORBIDDEN_PREDICTION_KEYS}
    for row in [*cases, *personas.values()]:
        collision = {str(key).casefold() for key in row} & forbidden
        if collision:
            raise LeakageError(f"protected fields entered a prompt source: {collision}")

    contexts = {variant: [] for variant in VARIANTS}
    retrieval_counts: list[int] = []
    for case in cases:
        persona = personas[str(case["pid"])]
        for variant in VARIANTS:
            prompt = render_prompt(case, persona, variant)
            if prompt.system != SYSTEM_INSTRUCTION:
                raise ProtocolError("variant system instruction drifted")
            if prompt.context_chars > int(manifest["model_context_policy"]["max_chars"]):
                raise ProtocolError(
                    f"{case['case_id']}/{variant} exceeds frozen context policy"
                )
            contexts[variant].append(prompt.context_chars)
            if variant == "rival_retrieval":
                retrieval_counts.append(len(prompt.retrieval_audit))
    observed_context = {name: _summary(values) for name, values in contexts.items()}
    if observed_context != manifest["model_context_policy"]["observed_context_chars"]:
        raise ProtocolError("full prompt context audit differs from frozen manifest")
    observed_retrieval = _summary(retrieval_counts)
    frozen_retrieval = manifest["model_context_policy"]["rival_retrieved_items"]
    for key in ("min", "median", "max"):
        if observed_retrieval[key] != frozen_retrieval[key]:
            raise ProtocolError("retrieval-count audit differs from frozen manifest")

    scheduled = work_schedule(cases, "pilot")
    if len(scheduled) != 1200:
        raise ProtocolError("A/B/C/D schedule does not contain exactly 1,200 calls")
    official = run_official_digital_certification_replication(
        Path(source_cache) / "reference" / "digital_certification"
    )
    if official["status"] != "PASS":
        raise ProtocolError("official-method replication did not pass")
    report = {
        "schema_version": manifest["schema_version"],
        "study_id": manifest["study_id"],
        "status": "PASS",
        "protocol_sha256": manifest["manifest_sha256"],
        "stage_sha256": stage["stage_sha256"],
        "cases": len(cases),
        "unique_participants": len(personas),
        "planned_calls": len(scheduled),
        "prompt_template_sha256": template_hashes(),
        "provider_identity": OpenRouterSurveyProvider.identity(),
        "context_chars": observed_context,
        "rival_retrieved_items": {
            key: observed_retrieval[key] for key in ("min", "median", "max")
        },
        "leakage_firewall": leakage_audit,
        "official_replication": official,
        "outcomes_materialized": False,
        "real_api_calls_made": 0,
    }
    report["audit_sha256"] = canonical_hash(report)
    return report
