"""Deterministic pre-outcome uncertainty records for v5 studies."""

from .integrity import IntegrityError
from .mathx import canonical_hash
from .runtime_calibration import comparison_metrics
from .runtime_uncertainty import assess_error
from .uncertainty_catalog import prediction_values, uncertainty_identity
from .calibration_catalog import protected_observation


def uncertainty_prediction(workspace, result):
    binding = workspace.prepared["uncertainty"]
    artifact = binding["artifact"]
    assessment, bound, fit, plan = (artifact[k] for k in ("assessment", "bound", "fit", "plan"))
    if (binding["assessment_sha256"] != workspace.request.uncertainty.assessment_sha256
            or canonical_hash(artifact) != binding["artifact_sha256"]
            or canonical_hash(assessment) != binding["assessment_sha256"]
            or canonical_hash(bound) != assessment["bound_sha256"]
            or canonical_hash(fit) != bound["fit_sha256"]
            or canonical_hash(plan) != fit["plan_sha256"]
            or canonical_hash(uncertainty_identity(workspace.request, workspace.prepared)) != canonical_hash(plan["identity"])):
        raise IntegrityError("prepared uncertainty artifact or prediction identity changed")
    distribution, x, request_ids = prediction_values(workspace, result)
    previous_ids = set(plan["distribution_request_sha256"])
    for stage in (fit, bound, assessment):
        for row in stage["rows"]:
            previous_ids.update(row.get("model_request_sha256", []))
    if previous_ids.intersection(request_ids):
        raise IntegrityError("target reused model response IDs from uncertainty evidence")
    decision = assess_error(fit["model"], bound["bound"], x, plan["settings"])
    reasons = list(decision["reasons"])
    if not assessment["metrics"]["statistical_gates_passed"]:
        reasons.append("held_out_statistical_gates_not_passed")
    reasons.append("customer_domain_qualification_pending_L12")
    payload = {"schema_version": "rival.uncertainty-prediction.v1",
        "study_id": workspace.request.brief.study_id, "raw_simulation_sha256": canonical_hash(result),
        "assessment_sha256": binding["assessment_sha256"], "prediction_kind": plan["identity"]["prediction"],
        "distribution": dict(zip(plan["identity"]["choice_ids"], distribution, strict=True)),
        "research_assessment": decision, "settings": plan["settings"],
        "statistical_gates_passed": assessment["metrics"]["statistical_gates_passed"],
        "evidence_counts": {role: sum(m["role"] == role for m in plan["members"])
                            for role in ("training", "calibration", "evaluation")},
        "held_out_metrics": {k: v for k, v in assessment["metrics"].items() if k != "rows"},
        "confidence": {"label": "unqualified", "abstain": True, "lower_tvd": 0.0, "upper_tvd": 1.0,
                       "reason_codes": reasons},
        "scope": "research error bound relative to observed study distributions; no individual, subgroup or causal guarantee"}
    return {**payload, "prediction_sha256": canonical_hash(payload)}


def verify_uncertainty_prediction(workspace, result, completed):
    saved = workspace.read("uncertainty_prediction", required=True)
    if (canonical_hash(saved) != canonical_hash(uncertainty_prediction(workspace, result))
            or completed.get("uncertainty_sha256") != canonical_hash(saved)):
        raise IntegrityError("saved uncertainty prediction differs from its pinned derivation")
    return saved


def uncertainty_comparison(workspace, result, sealed):
    prediction = verify_uncertainty_prediction(workspace, result, workspace.read("completed", required=True))
    observed, _, reveal_hash = protected_observation(workspace, sealed)
    choices = [c.choice_id for c in workspace.request.brief.choices]
    error = comparison_metrics([[prediction["distribution"][c] for c in choices]], [observed])["tvd"][0]
    research = prediction["research_assessment"]
    return {"prediction_sha256": prediction["prediction_sha256"], "outcome_evidence_sha256": reveal_hash,
            "observed_tvd": error, "research_bound_covered": error <= research["upper_tvd"],
            "candidate_accept": research["candidate_accept"],
            "bad_candidate_acceptance": research["candidate_accept"] and error > prediction["settings"]["max_tvd"],
            "refitted": False}
