from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from .mathx import normalize
from .schemas import EvaluationResult


def _aligned(
    predicted: dict[str, float], observed: dict[str, float]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    keys = sorted(set(predicted) | set(observed))
    p = normalize(predicted.get(key, 0.0) for key in keys)
    q = normalize(observed.get(key, 0.0) for key in keys)
    return keys, p, q


def total_variation_distance(
    predicted: dict[str, float], observed: dict[str, float]
) -> float:
    _, p, q = _aligned(predicted, observed)
    return float(0.5 * np.abs(p - q).sum())


def mean_absolute_percentage_point_error(
    predicted: dict[str, float], observed: dict[str, float]
) -> float:
    _, p, q = _aligned(predicted, observed)
    return float(np.mean(np.abs(p - q)) * 100.0)


def jensen_shannon_divergence(
    predicted: dict[str, float], observed: dict[str, float]
) -> float:
    _, p, q = _aligned(predicted, observed)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def spearman_rank(
    predicted: dict[str, float], observed: dict[str, float]
) -> float:
    _, p, q = _aligned(predicted, observed)
    if len(p) < 2 or np.allclose(p, p[0]) or np.allclose(q, q[0]):
        return 0.0
    p_rank = np.argsort(np.argsort(p)).astype(float)
    q_rank = np.argsort(np.argsort(q)).astype(float)
    return float(np.corrcoef(p_rank, q_rank)[0, 1])


def evaluate_distribution(
    run_id: str,
    predicted: dict[str, float],
    observed: dict[str, float],
    preregistration_hash: str | None = None,
    outcome_available_at: datetime | None = None,
) -> EvaluationResult:
    _, p, q = _aligned(predicted, observed)
    metrics = {
        "tvd": total_variation_distance(predicted, observed),
        "mae_percentage_points": mean_absolute_percentage_point_error(
            predicted, observed
        ),
        "jensen_shannon": jensen_shannon_divergence(predicted, observed),
        "spearman": spearman_rank(predicted, observed),
        "variance_ratio": float(np.var(p) / np.var(q)) if np.var(q) > 0 else math.nan,
    }
    return EvaluationResult(
        run_id=run_id,
        observed_distribution=dict(observed),
        metrics=metrics,
        preregistration_hash=preregistration_hash,
        outcome_available_at=outcome_available_at,
    )


def interval_coverage(
    intervals: dict[str, tuple[float, float]], observed: dict[str, float]
) -> float:
    checks = [
        intervals[key][0] <= observed[key] <= intervals[key][1]
        for key in observed
        if key in intervals
    ]
    return float(np.mean(checks)) if checks else 0.0

