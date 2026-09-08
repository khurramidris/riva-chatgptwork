"""Aggregate execution measurements and explicitly fresh repeat comparisons."""

from collections import Counter
import copy
import json
import math
import numpy as np
from .mathx import canonical_hash
from .providers import OpenAICompatibleProvider
from .study_contract import StudyRequestV3


def execution_audit(workspace, snapshot, simulation=None):
    if not isinstance(workspace.request, StudyRequestV3):
        raise ValueError("model execution audits require a v3 study")
    attempts = snapshot.get("attempts", [])
    requests = snapshot["requests"]
    planned = workspace.prepared["planned_unique_seeds"]
    latency, usages, outcomes, models, providers, fingerprints = [], [], Counter(), set(), set(), set()
    for attempt in attempts:
        payload = json.loads(attempt["payload"] or "{}")
        measured = payload.get("_rival_transport", {})
        duration = measured.get("latency_ms")
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
            latency.append(duration)
        usages.append(OpenAICompatibleProvider._usage_diagnostics(payload))
        reason = measured.get("rejection_type")
        error = payload.get("error", {})
        if isinstance(error, dict) and "http_status" in error:
            reason = "http_" + str(error["http_status"])
        outcomes[reason or attempt["state"]] += 1
        for key, values in (("model", models), ("provider", providers), ("system_fingerprint", fingerprints)):
            value = payload.get(key)
            if isinstance(value, str) and value:
                values.add(value)
    accepted = sum(row["state"] == "DONE" for row in requests)
    result = {"schema_version": "rival.model-execution-audit.v1",
        "study_id": workspace.request.brief.study_id,
        "planned_seed_requests": planned, "planned_draws": workspace.prepared["planned_draws"],
        "accepted_seed_requests": accepted, "unaccepted_seed_requests": planned - accepted,
        "complete": simulation is not None,
        "accounting": snapshot["accounting"], "attempt_outcomes": dict(sorted(outcomes.items())),
        "response_models": sorted(models), "response_providers": sorted(providers),
        "response_system_fingerprints": sorted(fingerprints),
        "transport_latency_ms": {"measured_attempts": len(latency),
            "unmeasured_attempts": len(attempts) - len(latency), "total": float(sum(latency)),
            "median": float(np.median(latency)) if latency else None,
            "p95": float(np.percentile(latency, 95)) if latency else None},
        "tokens": {key: {"reported": sum(usage.get(key, 0) for usage in usages),
                          "reporting_attempts": sum(key in usage for usage in usages)}
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "scope": "execution measurements, including billed failed attempts; no human-accuracy measurement",
        "latency_scope": "client monotonic HTTP duration; excludes backoff, local embedding, setup and unreturned requests"}
    if simulation is not None:
        predictions = {item.person_id.rsplit("__draw_", 1)[0]: item for item in simulation.predictions}
        result["elicitation"] = {key: sum(item.diagnostics.get(key, 0) for item in predictions.values())
            for key in ("ssr_degenerate", "ssr_tied_minima", "ssr_tied_maxima")}
    result["audit_sha256"] = canonical_hash(result)
    return result


def audit_study_execution(root):
    from .study_workflow import _workspace
    with _workspace(root) as workspace, workspace.execution() as session:
        snapshot = workspace.journal_snapshot(session)
        if workspace.read("completed"):
            workspace.require_complete(snapshot)
            recorded = workspace.read("model_execution_evidence", required=True)
            expected = execution_audit(workspace, snapshot, workspace.simulation())
            if canonical_hash(recorded) != canonical_hash(expected):
                raise ValueError("saved execution measurements no longer match the journal")
            return recorded
        return execution_audit(workspace, snapshot)


def compare_studies(first_root, second_root):
    """Read completed independent studies; compare stability without model calls."""
    from .study_workflow import _workspace

    def read(root):
        with _workspace(root) as workspace, workspace.execution() as session:
            if not isinstance(workspace.request, StudyRequestV3):
                raise ValueError("fresh model comparisons require v3 studies")
            snapshot = workspace.journal_snapshot(session)
            result, _ = workspace.require_complete(snapshot)
            payload = copy.deepcopy(workspace.prepared["request"])
            payload["brief"].pop("study_id")
            probabilities = {item.person_id.rsplit("__draw_", 1)[0]: item.probabilities for item in result.predictions}
            ids = {json.loads(row["payload"])["id"] for row in snapshot["requests"]}
            return workspace.request.brief.study_id, payload, probabilities, ids, result.distribution, snapshot

    first, second = read(first_root), read(second_root)
    if first[0] == second[0]:
        raise ValueError("fresh comparisons require distinct study IDs")
    if canonical_hash(first[1]) != canonical_hash(second[1]) or first[2].keys() != second[2].keys():
        raise ValueError("replicate inputs must match except for study_id")
    if first[3] & second[3]:
        raise ValueError("replicates share provider request IDs; independent completions are not established")
    tvds = [sum(abs(value - second[2][person][key]) for key, value in probabilities.items()) / 2
            for person, probabilities in first[2].items()]
    result = {"schema_version": "rival.model-repeat-comparison.v1", "study_ids": [first[0], second[0]],
        "matched_seed_requests": len(tvds), "exactly_equal_seed_distributions": sum(value == 0 for value in tvds),
        "mean_seed_tvd": float(np.mean(tvds)), "max_seed_tvd": float(max(tvds)),
        "aggregate_tvd": sum(abs(value - second[4][key]) for key, value in first[4].items()) / 2,
        "physical_attempts": [item[5]["accounting"]["attempts"] for item in (first, second)],
        "recorded_cost_usd": [item[5]["accounting"]["total_cost_usd"] for item in (first, second)],
        "scope": "descriptive fresh-run repeatability; disjoint returned request IDs; not human accuracy or cross-runtime determinism"}
    result["comparison_sha256"] = canonical_hash(result)
    return result
