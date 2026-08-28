from __future__ import annotations

import math

import numpy as np

from .schemas import ConfidenceAssessment


FEATURE_NAMES = [
    "average_entropy",
    "provider_disagreement",
    "population_margin_error",
    "population_ess_ratio",
    "scenario_novelty",
    "human_anchor_rate",
]


class ConfidenceModel:
    """Ridge error predictor with a conservative cold-start policy."""

    def __init__(self, ridge: float = 1.0, quality_threshold: float = 0.16):
        self.ridge = ridge
        self.quality_threshold = quality_threshold
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.residual_std: float = 0.12
        self.training_examples = 0

    @staticmethod
    def vectorize(features: dict[str, float]) -> np.ndarray:
        return np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES])

    def fit(self, feature_rows: list[dict[str, float]], observed_tvd: list[float]) -> None:
        if len(feature_rows) != len(observed_tvd) or len(feature_rows) < 5:
            raise ValueError("confidence fitting requires at least five aligned examples")
        matrix = np.vstack([self.vectorize(row) for row in feature_rows])
        target = np.asarray(observed_tvd, dtype=float)
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        standardized = (matrix - self.mean) / self.scale
        design = np.column_stack([np.ones(len(matrix)), standardized])
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )
        residuals = target - design @ self.coefficients
        self.residual_std = max(float(np.std(residuals, ddof=1)), 0.025)
        self.training_examples = len(feature_rows)

    def assess(self, features: dict[str, float]) -> ConfidenceAssessment:
        if self.coefficients is None or self.mean is None or self.scale is None:
            expected = self._cold_start_error(features)
            uncertainty = 0.14
            reason = "cold-start confidence; fewer than five protected outcomes"
        else:
            vector = (self.vectorize(features) - self.mean) / self.scale
            expected = float(np.dot(np.r_[1.0, vector], self.coefficients))
            uncertainty = 1.645 * self.residual_std
            reason = "ridge estimate from protected outcome features"

        expected = min(max(expected, 0.0), 1.0)
        lower = max(0.0, expected - uncertainty)
        upper = min(1.0, expected + uncertainty)
        if upper <= self.quality_threshold:
            label = "high"
        elif expected <= self.quality_threshold:
            label = "medium"
        else:
            label = "low"
        abstain = label == "low" or upper > max(self.quality_threshold * 1.75, 0.25)
        return ConfidenceAssessment(
            label=label,
            expected_tvd=expected,
            lower_tvd=lower,
            upper_tvd=upper,
            abstain=abstain,
            reason=reason,
            training_examples=self.training_examples,
            features={name: float(features.get(name, 0.0)) for name in FEATURE_NAMES},
        )

    @staticmethod
    def _cold_start_error(features: dict[str, float]) -> float:
        entropy = features.get("average_entropy", 0.5)
        disagreement = min(features.get("provider_disagreement", 0.0) * 10.0, 1.0)
        margin = min(features.get("population_margin_error", 0.0) * 10.0, 1.0)
        ess_penalty = 1.0 - min(max(features.get("population_ess_ratio", 0.0), 0.0), 1.0)
        novelty = features.get("scenario_novelty", 0.5)
        anchor_credit = min(features.get("human_anchor_rate", 0.0) * 2.0, 0.5)
        value = (
            0.06
            + 0.05 * entropy
            + 0.07 * disagreement
            + 0.05 * margin
            + 0.05 * ess_penalty
            + 0.10 * novelty
            - 0.08 * anchor_credit
        )
        return float(max(value, 0.03))

