from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


def validate_probabilities(values: Iterable[float]) -> np.ndarray:
    """Validate an external PMF without silently repairing invalid evidence."""
    raw = list(values)
    if any(isinstance(value, (bool, str)) for value in raw):
        raise ValueError("probabilities must be numeric, not booleans or strings")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("probabilities must be a nonempty vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError("probabilities must be finite values between zero and one")
    if not np.isclose(array.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError("probabilities must sum to one")
    return array / array.sum()


def normalize(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    array = np.where(np.isfinite(array), array, 0.0)
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 0:
        return np.ones_like(array) / max(len(array), 1)
    return array / total


def stable_softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    array = np.asarray(list(values), dtype=float) / temperature
    array -= np.max(array)
    exp = np.exp(np.clip(array, -60, 60))
    return normalize(exp)


def project_simplex(values: Iterable[float]) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    vector = np.asarray(list(values), dtype=float)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("simplex projection expects a non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(vector) + 1)
    valid = ordered - (cumulative - 1.0) / indices > 0
    if not np.any(valid):
        return np.ones(len(vector)) / len(vector)
    rho = int(np.nonzero(valid)[0][-1])
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def shannon_entropy(probabilities: Iterable[float], normalized: bool = True) -> float:
    probs = normalize(probabilities)
    positive = probs[probs > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    if normalized and len(probs) > 1:
        entropy /= math.log(len(probs))
    return entropy


def effective_sample_size(weights: Iterable[float]) -> float:
    array = np.asarray(list(weights), dtype=float)
    total = float(array.sum())
    squared = float(np.square(array).sum())
    return (total * total / squared) if squared > 0 else 0.0


def canonical_hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_unit_interval(*parts: Any) -> float:
    digest = canonical_hash(parts)
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
