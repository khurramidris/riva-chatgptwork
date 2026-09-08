"""Pre-outcome calibration outputs and protected comparisons for v4 studies."""

from .calibration_catalog import calibration_identity, protected_observation, seed_responses
from .integrity import IntegrityError
from .mathx import canonical_hash
from .runtime_calibration import apply_ensemble, comparison_metrics


def calibrated_prediction(workspace, result):
    binding = workspace.prepared["calibration"]
    adapter = binding["artifact"]
    if (canonical_hash(adapter) != binding["adapter_sha256"]
            or binding["adapter_sha256"] != workspace.request.calibration.adapter_sha256):
        raise IntegrityError("prepared calibration artifact changed")
    identity = calibration_identity(workspace.request, workspace.prepared)
    if canonical_hash(identity) != canonical_hash(adapter["identity"]):
        raise IntegrityError("calibration target identity changed")
    responses = seed_responses(result, identity)
    request_ids = {canonical_hash(prediction.provider_call.provider_request_id) for prediction in result.predictions
                   if prediction.provider_call and prediction.provider_call.provider_request_id}
    if (len(request_ids) != len(identity["seed_ids"])
            or request_ids.intersection(adapter["reference_request_sha256"])):
        raise IntegrityError("target model request IDs are duplicated or reuse reference responses")
    predicted = apply_ensemble(responses[None, :, :], adapter["fit"])[0]
    choices = identity["choice_ids"]
    payload = {"schema_version": "rival.calibrated-prediction.v1", "study_id": workspace.request.brief.study_id,
        "raw_simulation_sha256": canonical_hash(result), "adapter_sha256": binding["adapter_sha256"],
        "bank_sha256": adapter["bank_sha256"], "adapter_version": adapter["fit"]["adapter_version"],
        "raw_distribution": result.distribution,
        "calibrated_distribution": dict(zip(choices, predicted.tolist(), strict=True)),
        "uniform_panel_distribution": dict(zip(choices, responses.mean(axis=0).tolist(), strict=True)),
        "historical_mean_distribution": dict(zip(choices, adapter["historical_mean"], strict=True)),
        "training_groups": len(adapter["members"]), "seed_records": len(identity["seed_ids"]),
        "fit_diagnostics": adapter["fit"]["diagnostics"],
        "reference_execution": {"attempts": sum(member["attempts"] for member in adapter["members"]),
                                "known_api_cost_usd": sum(member["known_api_cost_usd"] for member in adapter["members"]),
                                "new_model_calls_for_fit": 0},
        "scope": "fixed panel and choice schema; aggregate research adjustment, not individual or qualified human probabilities"}
    return {**payload, "prediction_sha256": canonical_hash(payload)}


def verify_calibrated_prediction(workspace, result, completed=None):
    stored = workspace.read("calibrated_prediction", required=True)
    expected = calibrated_prediction(workspace, result)
    if canonical_hash(stored) != canonical_hash(expected):
        raise IntegrityError("saved calibration prediction differs from its pinned derivation")
    if completed is not None and completed.get("calibration_sha256") != canonical_hash(stored):
        raise IntegrityError("completed calibration seal changed")
    return stored


def calibration_comparison(workspace, result, sealed):
    prediction = verify_calibrated_prediction(workspace, result, workspace.read("completed", required=True))
    observed, _, reveal_hash = protected_observation(workspace, sealed)
    choices = [choice.choice_id for choice in workspace.request.brief.choices]
    metrics = {}
    for name in ("raw", "calibrated", "uniform_panel", "historical_mean"):
        values = [[prediction[name + "_distribution"][choice] for choice in choices]]
        metrics[name] = {key: value[0] for key, value in comparison_metrics(values, observed[None, :]).items()}
    payload = {"schema_version": "rival.calibration-comparison.v1",
        "study_id": workspace.request.brief.study_id,
        "evidence_role": workspace.request.evidence.role,
        "prediction_sha256": prediction["prediction_sha256"],
        "outcome_evidence_sha256": reveal_hash, "metrics": metrics,
        "tvd_reduction_vs_raw": metrics["raw"]["tvd"] - metrics["calibrated"]["tvd"],
        "tvd_reduction_vs_historical_mean": metrics["historical_mean"]["tvd"] - metrics["calibrated"]["tvd"],
        "held_out_from_this_adapter": True,
        "qualification": "descriptive comparison; semantic independence and future performance remain unqualified"}
    return {**payload, "comparison_sha256": canonical_hash(payload)}
