from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal, Sequence

import numpy as np


IntervalMethod = Literal["clt", "hoeffding", "bernstein"]


@dataclass(frozen=True, slots=True)
class SurveyInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float
    sample_size: int
    confidence: float
    method: str

    @property
    def width(self) -> float:
        return self.upper - self.lower


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")


def human_mean_interval(
    values: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> SurveyInterval:
    """CLT survey-mean interval matching the licensed UQ implementation."""

    _validate_confidence(confidence)
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        raise ValueError("at least two finite observations are required")
    estimate = float(data.mean())
    standard_error = float(data.std(ddof=1) / math.sqrt(len(data)))
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    margin = z * standard_error
    return SurveyInterval(
        estimate=estimate,
        lower=estimate - margin,
        upper=estimate + margin,
        standard_error=standard_error,
        sample_size=len(data),
        confidence=confidence,
        method="human-clt",
    )


def synthetic_mean_interval(
    values: Sequence[float] | np.ndarray,
    k: int | None = None,
    confidence: float = 0.95,
    scale: float = 2.0,
    method: IntervalMethod = "clt",
    minimum_k: int = 2,
    parameter_bounds: tuple[float, float] = (0.0, 1.0),
) -> SurveyInterval:
    """Conservative interval port of ``synthetic_CI`` from UQ Survey.

    ``scale`` is the paper/repository's C parameter. The implementation uses the
    actual available prefix length when ``k`` exceeds the number of responses.
    """

    _validate_confidence(confidence)
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    if minimum_k < 2:
        raise ValueError("minimum_k must be at least two")
    lower_bound, upper_bound = map(float, parameter_bounds)
    if not lower_bound < upper_bound:
        raise ValueError("parameter_bounds must be increasing")
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    requested = len(data) if k is None else int(k)
    if requested <= 0 or len(data) == 0:
        estimate = (lower_bound + upper_bound) / 2
        return SurveyInterval(
            estimate, lower_bound, upper_bound, math.inf, 0, confidence, method
        )
    used = data[: min(requested, len(data))]
    n = len(used)
    estimate = float(used.mean())
    if n <= minimum_k:
        return SurveyInterval(
            estimate, lower_bound, upper_bound, math.inf, n, confidence, method
        )

    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    sample_std = float(used.std(ddof=1))
    standard_error = sample_std / math.sqrt(n)
    if method == "clt":
        margin = z * standard_error * math.sqrt(scale)
    elif method == "hoeffding":
        span = upper_bound - lower_bound
        margin = span * math.sqrt(scale * math.log(2.0 / alpha) / (2.0 * n))
    elif method == "bernstein":
        span = upper_bound - lower_bound
        log_term = math.log(4.0 / alpha)
        margin = sample_std * math.sqrt(2.0 * scale * log_term / n)
        margin += (7.0 / 3.0) * scale * span * log_term / (n - 1)
    else:
        raise ValueError("method must be clt, hoeffding, or bernstein")
    return SurveyInterval(
        estimate=estimate,
        lower=estimate - margin,
        upper=estimate + margin,
        standard_error=standard_error,
        sample_size=n,
        confidence=confidence,
        method=f"synthetic-{method}",
    )


@dataclass(frozen=True, slots=True)
class CoverageDiagnostic:
    covered: int
    total: int
    coverage_rate: float
    target_coverage: float
    mean_width: float


def interval_coverage(
    intervals: Sequence[SurveyInterval], truths: Sequence[float]
) -> CoverageDiagnostic:
    if len(intervals) != len(truths) or not intervals:
        raise ValueError("intervals and truths must be non-empty and aligned")
    flags = [interval.lower <= float(truth) <= interval.upper for interval, truth in zip(intervals, truths, strict=True)]
    return CoverageDiagnostic(
        covered=sum(flags),
        total=len(flags),
        coverage_rate=float(np.mean(flags)),
        target_coverage=float(np.mean([item.confidence for item in intervals])),
        mean_width=float(np.mean([item.width for item in intervals])),
    )


def minimum_k_for_width(
    values: Sequence[float] | np.ndarray,
    maximum_width: float,
    **kwargs,
) -> int | None:
    if maximum_width <= 0:
        raise ValueError("maximum_width must be positive")
    data = np.asarray(values, dtype=float).reshape(-1)
    for k in range(1, len(data) + 1):
        if synthetic_mean_interval(data, k=k, **kwargs).width <= maximum_width:
            return k
    return None


@dataclass(frozen=True, slots=True)
class ResidualCorrectedInterval:
    synthetic: SurveyInterval
    residual: SurveyInterval
    estimate: float
    lower: float
    upper: float


def residual_corrected_interval(
    synthetic_values: Sequence[float] | np.ndarray,
    observed_anchor: Sequence[float] | np.ndarray,
    synthetic_anchor: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> ResidualCorrectedInterval:
    """Combine synthetic uncertainty with a held-out human residual anchor."""

    observed = np.asarray(observed_anchor, dtype=float).reshape(-1)
    predicted = np.asarray(synthetic_anchor, dtype=float).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError("observed and synthetic anchor arrays must align")
    synthetic = synthetic_mean_interval(
        synthetic_values, confidence=confidence, parameter_bounds=bounds
    )
    residual = human_mean_interval(observed - predicted, confidence=confidence)
    estimate = synthetic.estimate + residual.estimate
    lower = synthetic.lower + residual.lower
    upper = synthetic.upper + residual.upper
    return ResidualCorrectedInterval(
        synthetic=synthetic,
        residual=residual,
        estimate=float(np.clip(estimate, *bounds)),
        lower=float(np.clip(lower, *bounds)),
        upper=float(np.clip(upper, *bounds)),
    )
