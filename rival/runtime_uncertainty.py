"""Study-level error bounds; no synthetic draw is an independent observation.

Split conformal uses the exact finite-sample order statistic, with [0, 1]
when the requested rank is unavailable. Coverage is marginal under exchangeable
study groups, not conditional on accepting a prediction or on any subgroup.
"""

from decimal import Decimal, ROUND_CEILING

import numpy as np
from pydantic import Field, StrictFloat, model_validator
from scipy.stats import beta

from .runtime_calibration import probability_array
from .schemas import StrictModel


METHOD = "rival.study-error-conformal.v1"
FEATURES = ("entropy", "largest_share", "top_gap", "panel_disagreement")


class UncertaintyPin(StrictModel):
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class UncertaintySettings(StrictModel):
    # Decision-specific thresholds must be declared before any cohort runs.
    max_tvd: StrictFloat = Field(gt=0, lt=1, allow_inf_nan=False)
    required_coverage: StrictFloat = Field(gt=0, lt=1, allow_inf_nan=False)
    max_bad_acceptance_rate: StrictFloat = Field(gt=0, lt=1, allow_inf_nan=False)
    max_failure_rate: StrictFloat = Field(gt=0, lt=1, allow_inf_nan=False)
    nominal_coverage: StrictFloat = Field(default=0.9, gt=0, lt=1, allow_inf_nan=False)
    family_error_rate: StrictFloat = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    ridge: StrictFloat = Field(default=1.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def coverage_order(self):
        if self.required_coverage > self.nominal_coverage:
            raise ValueError("required coverage cannot exceed nominal coverage")
        return self


def features(distribution, panel):
    p = probability_array([distribution], 2)[0]
    panel = probability_array(panel, 2)
    if panel.shape[1] != len(p):
        raise ValueError("panel and distribution choice counts differ")
    ordered = np.sort(p)
    nonzero = p[p > 0]
    return [float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(p))),
            float(ordered[-1]), float(ordered[-1] - ordered[-2]),
            float(np.abs(panel - panel.mean(axis=0)).sum(axis=1).mean() / 2)]


def _matrix(rows):
    raw = np.asarray(rows)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != len(FEATURES) or not len(raw):
        raise ValueError("expected nonempty numeric study feature rows")
    x = raw.astype(float)
    if not np.isfinite(x).all() or np.any((x < -1e-12) | (x > 1 + 1e-12)):
        raise ValueError("study features must be finite and in [0, 1]")
    return x


def _errors(values, n):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.shape != (n,):
        raise ValueError("expected one numeric observed error per study")
    result = raw.astype(float)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("observed errors must be finite and in [0, 1]")
    return result


def fit_error_model(rows, errors, settings):
    settings = UncertaintySettings.model_validate(settings)
    x = _matrix(rows)
    y = _errors(errors, len(x))
    if len(x) < 5:
        raise ValueError("at least five independent training study groups are required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.diag([0, *([settings.ridge] * len(FEATURES))])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"method": METHOD, "features": list(FEATURES), "mean": mean.tolist(),
            "scale": scale.tolist(), "coefficients": coefficients.tolist(),
            "minimum": x.min(axis=0).tolist(), "maximum": x.max(axis=0).tolist(),
            "training_groups": len(x)}


def expected_errors(model, rows):
    x = _matrix(rows)
    if model["method"] != METHOD or model["features"] != list(FEATURES):
        raise ValueError("unsupported uncertainty model")
    mean, scale, coefficients = (np.asarray(model[key], dtype=float)
                                 for key in ("mean", "scale", "coefficients"))
    if (mean.shape != (len(FEATURES),) or scale.shape != mean.shape
            or coefficients.shape != (len(FEATURES) + 1,)
            or not all(np.isfinite(a).all() for a in (mean, scale, coefficients))
            or np.any(scale <= 0)):
        raise ValueError("invalid uncertainty model coefficients")
    return np.clip(np.column_stack((np.ones(len(x)), (x - mean) / scale)) @ coefficients, 0, 1)


def conformal_bound(model, rows, errors, coverage):
    if isinstance(coverage, bool) or not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    predictions = expected_errors(model, rows)
    residuals = np.maximum(0, _errors(errors, len(predictions)) - predictions)
    # Decimal prevents 0.9 * 10 rounding just above an integer rank.
    rank = int((Decimal(len(residuals) + 1) * Decimal(str(coverage))).to_integral_value(rounding=ROUND_CEILING))
    informative = rank <= len(residuals)
    return {"calibration_groups": len(residuals), "rank": rank,
            "offset": float(np.sort(residuals)[rank - 1]) if informative else 1.0,
            "finite_rank": informative, "nominal_coverage": coverage}


def assess_error(model, bound, row, settings):
    settings = UncertaintySettings.model_validate(settings)
    x = _matrix([row])[0]
    expected = float(expected_errors(model, [row])[0])
    supported = bool(np.all(x >= np.asarray(model["minimum"]) - 1e-12)
                     and np.all(x <= np.asarray(model["maximum"]) + 1e-12))
    upper = min(1.0, expected + bound["offset"]) if supported else 1.0
    reasons = []
    if not supported:
        reasons.append("features_outside_training_range")
    if not bound["finite_rank"]:
        reasons.append("too_few_calibration_groups")
    if upper > settings.max_tvd:
        reasons.append("error_bound_exceeds_tolerance")
    return {"expected_tvd": expected, "lower_tvd": 0.0, "upper_tvd": upper,
            "within_feature_support": supported, "candidate_accept": not reasons,
            "reasons": reasons, "nominal_marginal_coverage": bound["nominal_coverage"]}


def binomial_bounds(successes, total, tail_error):
    """One-sided Clopper-Pearson limits, including an empty denominator."""
    if (type(successes) is not int or type(total) is not int or not 0 <= successes <= total
            or not 0 < tail_error < 1):
        raise ValueError("invalid binomial counts or error level")
    return {"lower": float(beta.ppf(tail_error, successes, total - successes + 1)) if successes else 0.0,
            "upper": float(beta.ppf(1 - tail_error, successes + 1, total - successes)) if successes < total else 1.0}


def evaluate_policy(model, bound, rows, settings):
    """Score every predeclared evaluation group; None rows count as failures.

    Three one-sided gates use Bonferroni family-error control for this ONE
    prespecified policy. Trying many plans does not retain that guarantee.
    """
    settings = UncertaintySettings.model_validate(settings)
    if not rows:
        raise ValueError("evaluation roster is empty")
    scored, failed = [], 0
    for row in rows:
        if row is None:
            failed += 1
            continue
        error = float(_errors([row["observed_tvd"]], 1)[0])
        decision = assess_error(model, bound, row["features"], settings)
        scored.append({**decision, "observed_tvd": error,
                       "covered": error <= decision["upper_tvd"],
                       "bad_acceptance": decision["candidate_accept"] and error > settings.max_tvd})
    covered = sum(row["covered"] for row in scored)
    accepted = [row for row in scored if row["candidate_accept"]]
    bad = sum(row["bad_acceptance"] for row in accepted)
    tail = settings.family_error_rate / 3
    coverage = binomial_bounds(covered, len(rows), tail)  # Failures are noncoverage.
    risk = binomial_bounds(bad, len(accepted), tail)
    failure = binomial_bounds(failed, len(rows), tail)
    gates = {"coverage": coverage["lower"] >= settings.required_coverage,
             "accepted_error_risk": bool(accepted) and risk["upper"] <= settings.max_bad_acceptance_rate,
             "failure_rate": failure["upper"] <= settings.max_failure_rate}
    return {"planned_groups": len(rows), "scored_groups": len(scored), "failed_groups": failed,
            "covered_groups": covered, "accepted_groups": len(accepted), "bad_accepted_groups": bad,
            "acceptance_rate": len(accepted) / len(rows), "coverage_rate": covered / len(rows),
            "mean_tvd": float(np.mean([r["observed_tvd"] for r in scored])) if scored else None,
            "accepted_mean_tvd": float(np.mean([r["observed_tvd"] for r in accepted])) if accepted else None,
            "coverage_lower": coverage["lower"], "bad_acceptance_rate_upper": risk["upper"],
            "failure_rate_upper": failure["upper"], "per_gate_tail_error": tail,
            "gates": gates, "statistical_gates_passed": all(gates.values()), "rows": scored,
            "interpretation": "fixed-policy study-level evidence; exchangeability and semantic independence are assumptions"}
