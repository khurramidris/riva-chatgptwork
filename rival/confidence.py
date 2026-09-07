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
    """Research error predictor. Fitting cannot authorize operational claims."""

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
        values = np.asarray([float(features.get(name, 0.0)) for name in FEATURE_NAMES])
        if not np.isfinite(values).all():
            raise ValueError("confidence features must be finite")
        # Planned or caller-declared anchors are not protected evidence.
        values[FEATURE_NAMES.index("human_anchor_rate")] = 0.0
        return values

    def fit(self, feature_rows: list[dict[str, float]], observed_tvd: list[float]) -> None:
        if len(feature_rows) != len(observed_tvd) or len(feature_rows) < 5:
            raise ValueError("confidence fitting requires at least five aligned examples")
        matrix = np.vstack([self.vectorize(row) for row in feature_rows])
        target = np.asarray(observed_tvd, dtype=float)
        if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
            raise ValueError("observed TVD must be finite and between zero and one")
        if not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and positive")
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
        features = dict(zip(FEATURE_NAMES, self.vectorize(features), strict=True))
        if self.coefficients is None or self.mean is None or self.scale is None:
            expected = self._cold_start_error(features)
            reason = "unqualified cold-start heuristic; empirical error coverage not established"
        else:
            vector = (self.vectorize(features) - self.mean) / self.scale
            expected = float(np.dot(np.r_[1.0, vector], self.coefficients))
            reason = "research ridge fit; independent provenance and error coverage require qualification"

        expected = min(max(expected, 0.0), 1.0)
        return ConfidenceAssessment(
            label="unqualified",
            expected_tvd=expected,
            lower_tvd=0.0,
            upper_tvd=1.0,
            abstain=True,
            reason=reason + "; diagnostic estimate only; decision support withheld",
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
        value = (
            0.06
            + 0.05 * entropy
            + 0.07 * disagreement
            + 0.05 * margin
            + 0.05 * ess_penalty
            + 0.10 * novelty
        )
        return float(max(value, 0.03))
