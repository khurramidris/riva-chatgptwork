from __future__ import annotations

import math

import numpy as np

from .mathx import effective_sample_size, normalize, project_simplex
from .schemas import EstimateInterval, HumanObservation, HybridResult


class HybridEstimator:
    """Prediction-powered residual correction for categorical distributions.

    The implementation operationalizes the estimator family developed in
    AI-Augmented Estimation, SYN-DIGITS and related prediction-powered work.
    """

    def __init__(self, confidence_level: float = 0.95):
        if confidence_level != 0.95:
            raise ValueError("v0.1 currently supports 95% intervals")
        self.confidence_level = confidence_level
        self.z = 1.959963984540054

    def correct(
        self,
        synthetic_distribution: dict[str, float],
        observations: list[HumanObservation],
    ) -> HybridResult:
        choice_ids = list(synthetic_distribution)
        synthetic = normalize(synthetic_distribution[key] for key in choice_ids)
        if not observations:
            intervals = {
                key: EstimateInterval(
                    estimate=float(value), lower=0.0, upper=1.0, standard_error=0.5
                )
                for key, value in zip(choice_ids, synthetic, strict=True)
            }
            return HybridResult(
                synthetic_distribution=dict(synthetic_distribution),
                corrected_distribution={
                    key: float(value)
                    for key, value in zip(choice_ids, synthetic, strict=True)
                },
                residual_adjustment={key: 0.0 for key in choice_ids},
                intervals=intervals,
                human_sample_size=0,
                effective_human_sample_size=0.0,
                warnings=["no human anchor supplied; interval is intentionally uninformative"],
            )

        weights = normalize(observation.weight for observation in observations)
        residual_matrix = np.zeros((len(observations), len(choice_ids)), dtype=float)
        for row, observation in enumerate(observations):
            if observation.observed_choice not in choice_ids:
                raise ValueError(
                    f"unknown observed choice {observation.observed_choice!r}"
                )
            predicted = normalize(
                observation.synthetic_probabilities.get(key, 0.0) for key in choice_ids
            )
            observed = np.zeros(len(choice_ids), dtype=float)
            observed[choice_ids.index(observation.observed_choice)] = 1.0
            residual_matrix[row] = observed - predicted

        residual = np.sum(residual_matrix * weights[:, None], axis=0)
        raw_corrected = synthetic + residual
        corrected = project_simplex(raw_corrected)
        ess = effective_sample_size(weights)
        intervals: dict[str, EstimateInterval] = {}
        for column, key in enumerate(choice_ids):
            centered = residual_matrix[:, column] - residual[column]
            weighted_variance = float(np.sum(weights * np.square(centered)))
            standard_error = math.sqrt(weighted_variance / max(ess - 1.0, 1.0))
            intervals[key] = EstimateInterval(
                estimate=float(corrected[column]),
                lower=max(0.0, float(corrected[column] - self.z * standard_error)),
                upper=min(1.0, float(corrected[column] + self.z * standard_error)),
                standard_error=standard_error,
            )

        warnings: list[str] = []
        if len(observations) < 30:
            warnings.append("human anchor has fewer than 30 observations")
        if not np.allclose(raw_corrected, corrected, atol=1e-9):
            warnings.append("residual-corrected vector was projected onto the simplex")

        return HybridResult(
            synthetic_distribution={
                key: float(value) for key, value in zip(choice_ids, synthetic, strict=True)
            },
            corrected_distribution={
                key: float(value) for key, value in zip(choice_ids, corrected, strict=True)
            },
            residual_adjustment={
                key: float(value) for key, value in zip(choice_ids, residual, strict=True)
            },
            intervals=intervals,
            human_sample_size=len(observations),
            effective_human_sample_size=ess,
            warnings=warnings,
        )

