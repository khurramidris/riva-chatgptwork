"""SYN-DIGITS ensemble objective, extended explicitly to probability responses.

The archived research implementation and published experiments are unchanged.
One-hot inputs recover the paper's distributional ensemble. Soft inputs retain
elicited probabilities instead of silently converting ties to a first-choice vote.
"""

from typing import Literal

import numpy as np
from pydantic import Field, StrictInt

from .schemas import StrictModel


ADAPTER_VERSION = "rival.ensemble-calibration.v1"
UPSTREAM_REVISION = "db891b6f821c914455b11763a96679864bf4fc48"


class CalibrationPin(StrictModel):
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationSettings(StrictModel):
    method: Literal["kl_persona_base"] = "kl_persona_base"
    max_iter: StrictInt = Field(default=500, ge=1, le=10_000)
    learning_rate: float = Field(default=1.0, gt=0, le=100, allow_inf_nan=False)
    reg_persona: float = Field(default=1e-6, ge=0, le=10, allow_inf_nan=False)
    reg_base: float = Field(default=1e-6, ge=0, le=10, allow_inf_nan=False)
    gradient_clip: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    tolerance: float = Field(default=1e-6, gt=0, le=0.01, allow_inf_nan=False)


def probability_array(value, ndim):
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise ValueError("probabilities must be numeric, not strings or booleans")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or any(size == 0 for size in array.shape) or array.shape[-1] < 2:
        raise ValueError("probability array has an empty or invalid shape")
    if (not np.isfinite(array).all() or (array < 0).any() or (array > 1).any()
            or not np.allclose(array.sum(axis=-1), 1.0, rtol=0, atol=1e-8)):
        raise ValueError("probabilities must be finite, nonnegative and sum to one")
    return array


def one_hot_answers(answers, choices):
    values = np.asarray(answers)
    if (values.dtype.kind not in "iu" or values.ndim != 2 or not values.size
            or not isinstance(choices, int) or isinstance(choices, bool) or choices < 2
            or (values < 0).any() or (values >= choices).any()):
        raise ValueError("answers must be a nonempty matrix of integer choice indices")
    return np.eye(choices)[values]


def _predict(responses, theta):
    personas = responses.shape[1]
    return np.einsum("mnk,n->mk", responses, theta[:personas], optimize=False) + theta[personas:]


def objective_gradient(responses, observed, theta, settings):
    """Mean KL + L2 penalty and its ambient simplex gradient; no MSE penalty."""
    predicted = np.maximum(_predict(responses, theta), 1e-12)
    positive = observed > 0
    divergence = np.zeros_like(observed)
    divergence[positive] = observed[positive] * np.log(observed[positive] / predicted[positive])
    personas = responses.shape[1]
    penalty = np.r_[np.full(personas, settings.reg_persona),
                    np.full(observed.shape[1], settings.reg_base)]
    objective = float(divergence.sum() / len(observed) + np.dot(penalty, theta * theta))
    derivative = -observed / predicted / len(observed)
    gradient = np.r_[np.einsum("mnk,mk->n", responses, derivative, optimize=False),
                     derivative.sum(axis=0)] + 2 * penalty * theta
    return objective, gradient


def fit_ensemble(responses, observed, settings=None):
    settings = CalibrationSettings.model_validate(settings or {})
    responses, observed = probability_array(responses, 3), probability_array(observed, 2)
    if responses.shape[0] != observed.shape[0] or responses.shape[2] != observed.shape[1]:
        raise ValueError("reference questions and choices must align")
    theta = np.full(responses.shape[1] + responses.shape[2],
                    1 / (responses.shape[1] + responses.shape[2]))
    initial, _ = objective_gradient(responses, observed, theta, settings)
    objectives = [initial]
    steps = 0
    for _ in range(settings.max_iter):
        objective, gradient = objective_gradient(responses, observed, theta, settings)
        gap = float(np.dot(theta, gradient) - gradient.min())
        if gap <= settings.tolerance:
            break
        direction = gradient * min(1.0, settings.gradient_clip / max(float(np.linalg.norm(gradient)), 1e-12))
        rate = settings.learning_rate
        # Backtracking protects the objective without using any held-out labels.
        for _ in range(40):
            logits = np.log(np.maximum(theta, 1e-300)) - rate * direction
            candidate = np.exp(np.clip(logits - logits.max(), -700, 0))
            candidate /= candidate.sum()
            value, _ = objective_gradient(responses, observed, candidate, settings)
            if value <= objective + 1e-14:
                break
            rate *= 0.5
        else:
            raise ValueError("calibration optimizer could not make a stable step")
        theta = candidate
        objectives.append(value)
        steps += 1
    final, gradient = objective_gradient(responses, observed, theta, settings)
    gap = float(np.dot(theta, gradient) - gradient.min())
    weights = theta[:responses.shape[1]]
    normalized = weights / weights.sum()
    return {"adapter_version": ADAPTER_VERSION, "upstream_revision": UPSTREAM_REVISION,
            "response_semantics": "probability-mixture extension; one-hot inputs recover the published ensemble",
            "settings": settings.model_dump(mode="json"), "coefficients": theta.tolist(),
            "diagnostics": {"iterations": steps, "initial_objective": initial,
                "final_objective": final, "simplex_gap": gap,
                "converged": gap <= settings.tolerance,
                "monotone_objective": bool(np.all(np.diff(objectives) <= 1e-12)),
                "effective_personas": float(1 / np.square(normalized).sum()),
                "base_mass": float(theta[responses.shape[1]:].sum())}}


def apply_ensemble(responses, fit):
    responses = probability_array(responses, 3)
    theta = np.asarray(fit["coefficients"], dtype=float)
    if (fit["adapter_version"] != ADAPTER_VERSION
            or theta.shape != (responses.shape[1] + responses.shape[2],)
            or not np.isfinite(theta).all() or (theta < 0).any()
            or not np.isclose(theta.sum(), 1.0, rtol=0, atol=1e-8)):
        raise ValueError("calibration coefficients or adapter version do not match")
    return probability_array(_predict(responses, theta), 2)


def comparison_metrics(predicted, observed):
    predicted, observed = probability_array(predicted, 2), probability_array(observed, 2)
    if predicted.shape != observed.shape:
        raise ValueError("evaluation matrices must align")
    midpoint = (predicted + observed) / 2
    js = np.zeros(len(predicted))
    for side in (predicted, observed):
        terms = np.zeros_like(side)
        positive = side > 0
        terms[positive] = side[positive] * np.log(side[positive] / midpoint[positive])
        js += 0.5 * terms.sum(axis=1)
    return {"tvd": (0.5 * np.abs(predicted - observed).sum(axis=1)).tolist(),
            "jensen_shannon": js.tolist(),
            "squared_error": np.square(predicted - observed).sum(axis=1).tolist()}
