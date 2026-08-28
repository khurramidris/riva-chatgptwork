from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln


DemandObjective = Literal["binomial", "truncated"]
DEMAND_UPSTREAM_COMMIT = "b56a7c0acad7406bff81b7cdf179314894b2fa97"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    output = np.exp(np.clip(shifted, -700, 0))
    return output / output.sum()


def _calibrate(q_matrix: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(q_matrix, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return np.clip(expit(intercept + slope * logits), 1e-9, 1.0 - 1e-9)


def _likelihood(
    demand: np.ndarray,
    exposure_n: int,
    probability: np.ndarray,
    objective: DemandObjective,
) -> float:
    q = np.clip(probability, 1e-12, 1.0 - 1e-12)
    if np.any(demand < 0) or np.any(demand > exposure_n):
        return math.inf
    log_choose = (
        gammaln(exposure_n + 1)
        - gammaln(demand + 1)
        - gammaln(exposure_n - demand + 1)
    )
    log_likelihood = log_choose + demand * np.log(q) + (exposure_n - demand) * np.log1p(-q)
    if objective == "truncated":
        if np.any(demand <= 0):
            return math.inf
        positive_mass = np.clip(1.0 - np.power(1.0 - q, exposure_n), 1e-300, None)
        log_likelihood -= np.log(positive_mass)
    return float(-np.sum(log_likelihood))


@dataclass(frozen=True, slots=True)
class PersonaDemandModel:
    exposure_n: int
    persona_weights: np.ndarray
    no_buy_weight: float
    intercept: float
    slope: float
    objective: DemandObjective
    objective_value: float
    converged: bool
    status: str

    @property
    def source(self) -> dict[str, str]:
        return {
            "method": "H&M persona-mixture demand and pricing",
            "upstream_commit": DEMAND_UPSTREAM_COMMIT,
            "integration": "adapted-scipy-runtime",
        }

    def purchase_probability(self, persona_probabilities: np.ndarray) -> np.ndarray:
        matrix = np.asarray(persona_probabilities, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != len(self.persona_weights):
            raise ValueError("persona probability matrix has the wrong width")
        calibrated = _calibrate(matrix, self.intercept, self.slope)
        return np.clip(calibrated @ self.persona_weights, 1e-9, 1.0 - 1e-9)

    def mean_demand(self, persona_probabilities: np.ndarray) -> np.ndarray:
        return self.exposure_n * self.purchase_probability(persona_probabilities)


def fit_persona_demand(
    persona_probabilities: np.ndarray,
    observed_demand: Sequence[int] | np.ndarray,
    exposure_n: int,
    objective: DemandObjective = "truncated",
    calibration_iterations: int = 6,
) -> PersonaDemandModel:
    """Fit calibrated persona/no-buy mixture without a CVXPY runtime dependency."""

    matrix = np.asarray(persona_probabilities, dtype=float)
    demand = np.asarray(observed_demand, dtype=int).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(demand) or matrix.shape[1] < 1:
        raise ValueError("persona probabilities and demand must be aligned matrices")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("persona probabilities must be finite and in [0, 1]")
    if exposure_n < 1 or np.any(demand > exposure_n) or np.any(demand < 0):
        raise ValueError("observed demand must be between zero and exposure_n")
    if objective not in {"binomial", "truncated"}:
        raise ValueError("objective must be binomial or truncated")
    if objective == "truncated" and np.any(demand <= 0):
        raise ValueError("truncated fitting requires strictly positive demand")
    if calibration_iterations < 1:
        raise ValueError("calibration_iterations must be positive")

    personas = matrix.shape[1]
    raw_mean = float(np.clip(matrix.mean(), 1e-6, 1.0 - 1e-6))
    target = float(np.clip(demand.mean() / exposure_n, 1e-6, 1.0 - 1e-6))
    persona_mass = float(np.clip(target / raw_mean, 1e-5, 0.99999))
    initial_weights = np.r_[np.full(personas, persona_mass / personas), 1.0 - persona_mass]
    raw_logit = math.log(raw_mean / (1.0 - raw_mean))
    target_logit = math.log(target / (1.0 - target))
    initial = np.r_[np.log(np.clip(initial_weights, 1e-12, None)), target_logit - raw_logit, 0.0]

    def unpack(params: np.ndarray) -> tuple[np.ndarray, float, float]:
        weights = _softmax(params[: personas + 1])
        intercept = float(params[-2])
        slope = float(np.exp(params[-1]))
        return weights, intercept, slope

    def loss(params: np.ndarray) -> float:
        weights, intercept, slope = unpack(params)
        calibrated = _calibrate(matrix, intercept, slope)
        probability = calibrated @ weights[:-1]
        value = _likelihood(demand, exposure_n, probability, objective)
        # The source algorithm alternates these blocks. A weak centering penalty
        # removes softmax's additive non-identifiability without changing the fit.
        return value + 1e-10 * float(np.square(params[: personas + 1].mean()))

    result = None
    params = initial
    for _ in range(calibration_iterations):
        result = minimize(
            loss,
            params,
            method="L-BFGS-B",
            bounds=[(-25.0, 25.0)] * (personas + 1) + [(-20.0, 20.0), (-5.0, 5.0)],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        params = np.asarray(result.x, dtype=float)
        if result.success:
            break
    assert result is not None
    weights, intercept, slope = unpack(params)
    return PersonaDemandModel(
        exposure_n=int(exposure_n),
        persona_weights=weights[:-1].copy(),
        no_buy_weight=float(weights[-1]),
        intercept=intercept,
        slope=slope,
        objective=objective,
        objective_value=float(loss(params)),
        converged=bool(result.success),
        status=str(result.message),
    )


def revenue_cvar(
    revenue_samples: Sequence[float] | np.ndarray,
    tail_probability: float = 0.1,
) -> float:
    values = np.asarray(revenue_samples, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("revenue_samples must contain finite values")
    if not 0 < tail_probability <= 1:
        raise ValueError("tail_probability must be in (0, 1]")
    tail_size = max(1, int(math.ceil(tail_probability * len(values))))
    return float(np.mean(np.partition(values, tail_size - 1)[:tail_size]))


@dataclass(frozen=True, slots=True)
class PriceDecision:
    price: float
    purchase_probability: float
    expected_demand: float
    expected_revenue: float
    lower_tail_revenue: float
    score: float
    candidates: tuple[dict[str, float], ...]


def optimize_price(
    prices: Sequence[float] | np.ndarray,
    purchase_probabilities: Sequence[float] | np.ndarray,
    market_size: int,
    unit_cost: float = 0.0,
    risk_weight: float = 0.0,
    tail_probability: float = 0.1,
    draws: int = 4000,
    seed: int = 20260828,
) -> PriceDecision:
    price_values = np.asarray(prices, dtype=float).reshape(-1)
    probabilities = np.asarray(purchase_probabilities, dtype=float).reshape(-1)
    if len(price_values) == 0 or price_values.shape != probabilities.shape:
        raise ValueError("prices and purchase probabilities must be non-empty and aligned")
    if market_size < 1 or draws < 20:
        raise ValueError("market_size and draws are too small")
    if np.any(price_values < 0) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("invalid prices or purchase probabilities")
    if not 0 <= risk_weight <= 1:
        raise ValueError("risk_weight must be in [0, 1]")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for price, probability in zip(price_values, probabilities, strict=True):
        demand_samples = rng.binomial(market_size, probability, size=draws)
        revenue_samples = (price - unit_cost) * demand_samples
        expected_demand = float(market_size * probability)
        expected_revenue = float((price - unit_cost) * expected_demand)
        lower_tail = revenue_cvar(revenue_samples, tail_probability)
        score = (1.0 - risk_weight) * expected_revenue + risk_weight * lower_tail
        rows.append(
            {
                "price": float(price),
                "purchase_probability": float(probability),
                "expected_demand": expected_demand,
                "expected_revenue": expected_revenue,
                "lower_tail_revenue": lower_tail,
                "score": score,
            }
        )
    best = max(rows, key=lambda row: (row["score"], row["expected_revenue"], -row["price"]))
    return PriceDecision(
        price=best["price"],
        purchase_probability=best["purchase_probability"],
        expected_demand=best["expected_demand"],
        expected_revenue=best["expected_revenue"],
        lower_tail_revenue=best["lower_tail_revenue"],
        score=best["score"],
        candidates=tuple(rows),
    )

