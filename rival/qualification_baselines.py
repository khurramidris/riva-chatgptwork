"""Weighted history and regularized multinomial regression on human outcomes.

These are Rival-written classical baselines, not extracted upstream models.
Training weights are study weights; they do not imply respondent-level precision.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .qualification_contract import BaselineSettings, HumanSample
from .runtime_calibration import probability_array


METHOD = "rival.human-multinomial-baseline.v1"


def matrix(values, columns):
    raw = np.asarray(values, dtype=object)
    if (raw.ndim != 2 or not len(raw) or raw.shape[1] != columns
            or any(type(v) not in (int, float) for v in raw.flat)):
        raise ValueError("baseline features must be finite numeric rows of the declared width")
    x = raw.astype(float)
    if not np.isfinite(x).all():
        raise ValueError("baseline features must be finite")
    return x


def objective_gradient(flat, design, outcomes, weights, ridge):
    coefficients = flat.reshape(design.shape[1], outcomes.shape[1])
    logits = design @ coefficients
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    # Penalize slopes only; an intercept-only fit has the weighted-history optimum.
    value = -np.sum(weights[:, None] * outcomes * log_probabilities) + ridge * np.sum(coefficients[1:] ** 2) / 2
    gradient = design.T @ (weights[:, None] * (np.exp(log_probabilities) - outcomes))
    gradient[1:] += ridge * coefficients[1:]
    return float(value), gradient.ravel()


def fit_baselines(features, outcomes, weights, settings):
    settings = BaselineSettings.model_validate(settings)
    x = matrix(features, len(settings.feature_names))
    y = probability_array(outcomes, 2)
    raw_weights = matrix([[v] for v in weights], 1)[:, 0]
    if len(x) != len(y) or len(x) != len(raw_weights) or len(x) < 3 or np.any(raw_weights <= 0):
        raise ValueError("baseline fit requires at least three human training study groups with positive weights")
    w = raw_weights / raw_weights.max()
    w /= w.sum()
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-9] = 1
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    if not np.isfinite(design).all() or not np.isfinite(scale).all():
        raise ValueError("baseline features exceed finite numerical range")
    history = w @ y
    start = np.zeros((design.shape[1], y.shape[1]))
    start[0] = np.log(np.maximum(history, 1e-12))
    result = minimize(objective_gradient, start.ravel(), args=(design, y, w, settings.ridge),
        jac=True, method="L-BFGS-B", options={"maxiter": settings.max_iter,
        "ftol": settings.tolerance, "gtol": settings.tolerance})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("nonfinite classical baseline fit")
    return {"method": METHOD, "settings": settings.model_dump(mode="json"),
        "history": history.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": result.x.reshape(start.shape).tolist(),
        "feature_minimum": x.min(axis=0).tolist(), "feature_maximum": x.max(axis=0).tolist(),
        "training_groups": len(x), "diagnostics": {"converged": bool(result.success),
        "iterations": int(result.nit), "objective": float(result.fun), "message": str(result.message)}}


def predict_baselines(model, features):
    if model["method"] != METHOD:
        raise ValueError("unsupported classical baseline model")
    x = matrix([features], len(model["settings"]["feature_names"]))
    coefficients = np.asarray(model["coefficients"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    if np.any(scale <= 0) or not np.isfinite(coefficients).all():
        raise ValueError("invalid classical baseline coefficients")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        design = np.column_stack((np.ones(len(x)), (x - np.asarray(model["mean"])) / scale))
        classical = probability_array(softmax(design @ coefficients, axis=1), 2)[0].tolist()
    history = probability_array([model["history"]], 2)[0].tolist()
    return {"weighted_history": history, "classical_multinomial": classical}


def summarize_sample(sample, units, choices, unit_weights=None):
    sample = HumanSample.model_validate(sample)
    actual = [r.unit_sha256 for r in sample.rows]
    if len(actual) != len(set(actual)) or set(actual) != set(units):
        raise ValueError("human sample must contain every frozen unit exactly once, including missing responses")
    counts = dict.fromkeys(choices, 0.0)
    observed = []
    for row in sample.rows:
        if unit_weights is not None and row.weight != unit_weights[row.unit_sha256]:
            raise ValueError("human sample weight differs from its frozen design")
        if row.choice_id is not None:
            if row.choice_id not in counts:
                raise ValueError("human response contains an undeclared choice")
            counts[row.choice_id] += row.weight
            observed.append(row.weight)
    total = sum(r.weight for r in sample.rows)
    valid = sum(observed)
    if not np.isfinite(total) or not np.isfinite(valid) or total <= 0:
        raise ValueError("human sample weights exceed finite numerical range")
    normalized = [w / max(observed) for w in observed] if observed else []
    return {"distribution": [counts[c] / valid for c in choices] if valid else None,
            "eligible_units": len(sample.rows), "observed_units": len(observed),
            "missing_units": len(sample.rows) - len(observed),
            "missing_rate": 1 - len(observed) / len(sample.rows),
            "missing_weight_fraction": max(0.0, min(1.0, 1 - valid / total)),
            "effective_observed_units": sum(normalized) ** 2 / sum(w * w for w in normalized) if valid else 0.0}
