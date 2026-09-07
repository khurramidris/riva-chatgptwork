from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import t

from ..integrity import prepare_prediction_context
from ..mathx import validate_probabilities
from ..providers import PredictionProvider
from ..schemas import PopulationRecord, ScenarioSpec


@dataclass(frozen=True, slots=True)
class PairedPrediction:
    person_id: str
    control: float
    treatment: float
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PrePeriodAnchor:
    observed_control: float
    observed_treatment: float
    synthetic_control: float
    synthetic_treatment: float

    @property
    def residual_adjustment(self) -> float:
        return (self.observed_treatment - self.observed_control) - (
            self.synthetic_treatment - self.synthetic_control
        )


@dataclass(frozen=True, slots=True)
class SRCTEstimate:
    raw_effect: float
    residual_adjustment: float
    estimate: float
    standard_error: float
    lower: float
    upper: float
    effective_sample_size: float
    pairs: int
    confidence: float
    method: str = "paired-srct"


def estimate_paired_srct(
    predictions: Sequence[PairedPrediction],
    confidence: float = 0.95,
    pre_period: PrePeriodAnchor | None = None,
) -> SRCTEstimate:
    if len(predictions) < 2:
        raise ValueError("at least two independent pairs are required to estimate variance")
    if len({item.person_id for item in predictions}) != len(predictions):
        raise ValueError("paired predictions must have distinct person IDs")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    weights = np.asarray([item.weight for item in predictions], dtype=float)
    differences = np.asarray(
        [item.treatment - item.control for item in predictions], dtype=float
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("pair weights must be finite and positive")
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired predictions must be finite")
    weight_sum = float(weights.sum())
    raw = float(np.dot(weights, differences) / weight_sum)
    effective_n = weight_sum**2 / float(np.dot(weights, weights))
    if effective_n <= 1.0 + 1e-12:
        raise ValueError("effective pair count is insufficient to estimate variance")
    centered = differences - raw
    denominator = weight_sum - float(np.dot(weights, weights)) / weight_sum
    variance = float(np.dot(weights, centered**2) / denominator) if denominator > 0 else 0.0
    standard_error = math.sqrt(max(variance, 0.0) / effective_n)
    adjustment = pre_period.residual_adjustment if pre_period is not None else 0.0
    estimate = raw + adjustment
    degrees = max(1, int(math.floor(effective_n)) - 1)
    critical = float(t.ppf(0.5 + confidence / 2.0, degrees))
    margin = critical * standard_error
    return SRCTEstimate(
        raw_effect=raw,
        residual_adjustment=adjustment,
        estimate=estimate,
        standard_error=standard_error,
        lower=estimate - margin,
        upper=estimate + margin,
        effective_sample_size=effective_n,
        pairs=len(predictions),
        confidence=confidence,
    )


def simulate_paired_srct(
    provider: PredictionProvider,
    population: Sequence[PopulationRecord],
    control_scenario: ScenarioSpec,
    treatment_scenario: ScenarioSpec,
    outcome_choice_id: str,
    confidence: float = 0.95,
    pre_period: PrePeriodAnchor | None = None,
) -> SRCTEstimate:
    if len(population) < 2 or len({person.person_id for person in population}) != len(population):
        raise ValueError("simulation requires at least two distinct participants")
    control_choices = {choice.choice_id for choice in control_scenario.choices}
    treatment_choices = {choice.choice_id for choice in treatment_scenario.choices}
    if outcome_choice_id not in control_choices or outcome_choice_id not in treatment_choices:
        raise ValueError("outcome_choice_id must exist in both scenarios")
    pairs = []
    # Prepare both arms before the first request; either arm can expose leakage.
    control_context = prepare_prediction_context(
        list(population), control_scenario, None, provider.identity()
    )
    treatment_context = prepare_prediction_context(
        list(population), treatment_scenario, None, provider.identity()
    )
    for person, treated_person in zip(
        control_context.records, treatment_context.records, strict=True
    ):
        control = provider.predict(person, control_scenario)
        treatment = provider.predict(treated_person, treatment_scenario)
        for output, choices in ((control, control_choices), (treatment, treatment_choices)):
            if set(output.probabilities) != choices:
                raise ValueError("provider returned the wrong choice_id set")
            validate_probabilities(output.probabilities.values())
        pairs.append(
            PairedPrediction(
                person_id=person.person_id,
                control=float(control.probabilities[outcome_choice_id]),
                treatment=float(treatment.probabilities[outcome_choice_id]),
                weight=person.weight,
            )
        )
    return estimate_paired_srct(pairs, confidence=confidence, pre_period=pre_period)
