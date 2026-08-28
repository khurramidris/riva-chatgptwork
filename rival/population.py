from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .mathx import effective_sample_size
from .schemas import PopulationDiagnostics, PopulationRecord, PopulationTargets


class PopulationError(ValueError):
    pass


class PopulationCompiler:
    """Calibrate weighted seed records to declared population controls.

    The compact raking implementation follows the same broad statistical family
    used by PopulationSim/IPF systems, while keeping Rival's core dependency-light.
    """

    def __init__(self, max_iterations: int = 250, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    @staticmethod
    def _value(record: PopulationRecord, feature: str) -> str:
        if feature not in record.attributes:
            return "__missing__"
        return str(record.attributes[feature])

    @staticmethod
    def _target_proportions(categories: dict[str, float]) -> dict[str, float]:
        total = float(sum(categories.values()))
        if total <= 0:
            raise PopulationError("population control has no positive mass")
        return {str(category): float(value) / total for category, value in categories.items()}

    def calibrate(
        self,
        records: list[PopulationRecord],
        targets: PopulationTargets,
    ) -> tuple[list[PopulationRecord], PopulationDiagnostics]:
        if not records:
            raise PopulationError("cannot compile an empty population")

        weights = np.asarray([record.weight for record in records], dtype=float)
        total_weight = float(weights.sum())
        controls = {
            feature: self._target_proportions(categories)
            for feature, categories in targets.controls.items()
        }

        masks: dict[str, dict[str, np.ndarray]] = {}
        unsupported: dict[str, list[str]] = defaultdict(list)
        for feature, categories in controls.items():
            values = np.asarray([self._value(record, feature) for record in records])
            masks[feature] = {}
            for category, target in categories.items():
                mask = values == str(category)
                masks[feature][category] = mask
                if target > 0 and not bool(mask.any()):
                    unsupported[feature].append(category)

        if unsupported:
            details = "; ".join(
                f"{feature}: {', '.join(categories)}"
                for feature, categories in unsupported.items()
            )
            raise PopulationError(f"seed population lacks target categories: {details}")

        converged = False
        marginal_errors: dict[str, dict[str, float]] = {}
        max_error = float("inf")
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            for feature, categories in controls.items():
                for category, target_proportion in categories.items():
                    mask = masks[feature][category]
                    current_mass = float(weights[mask].sum())
                    desired_mass = target_proportion * total_weight
                    if current_mass > 0:
                        weights[mask] *= desired_mass / current_mass
                weights *= total_weight / float(weights.sum())

            marginal_errors = self.marginal_errors(records, weights, controls)
            max_error = max(
                abs(error)
                for categories in marginal_errors.values()
                for error in categories.values()
            )
            if max_error <= self.tolerance:
                converged = True
                break

        compiled = [
            record.model_copy(update={"weight": float(weight)})
            for record, weight in zip(records, weights, strict=True)
        ]
        ess = effective_sample_size(weights)
        diagnostics = PopulationDiagnostics(
            converged=converged,
            iterations=iteration,
            max_absolute_margin_error=float(max_error),
            effective_sample_size=ess,
            effective_sample_ratio=ess / len(records),
            marginal_errors=marginal_errors,
            unsupported_categories=dict(unsupported),
        )
        return compiled, diagnostics

    def marginal_errors(
        self,
        records: list[PopulationRecord],
        weights: np.ndarray,
        controls: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        total = float(weights.sum())
        result: dict[str, dict[str, float]] = {}
        for feature, categories in controls.items():
            values = np.asarray([self._value(record, feature) for record in records])
            result[feature] = {}
            for category, target in categories.items():
                actual = float(weights[values == str(category)].sum()) / total
                result[feature][category] = actual - target
        return result

    @staticmethod
    def filter_records(
        records: Iterable[PopulationRecord], filters: dict[str, Any]
    ) -> list[PopulationRecord]:
        def matches(record: PopulationRecord) -> bool:
            for feature, expected in filters.items():
                actual = record.attributes.get(feature)
                if isinstance(expected, list):
                    if actual not in expected:
                        return False
                elif isinstance(expected, dict):
                    minimum = expected.get("min")
                    maximum = expected.get("max")
                    if minimum is not None and (actual is None or actual < minimum):
                        return False
                    if maximum is not None and (actual is None or actual > maximum):
                        return False
                elif actual != expected:
                    return False
            return True

        return [record for record in records if matches(record)]

    @staticmethod
    def sample(
        records: list[PopulationRecord], sample_size: int, seed: int
    ) -> list[PopulationRecord]:
        if not records:
            raise PopulationError("no population records match the requested segment")
        rng = np.random.default_rng(seed)
        probabilities = np.asarray([record.weight for record in records], dtype=float)
        probabilities /= probabilities.sum()
        indices = rng.choice(len(records), size=sample_size, replace=True, p=probabilities)
        sample_weight = 1.0 / sample_size
        return [
            records[int(index)].model_copy(
                update={
                    "person_id": f"{records[int(index)].person_id}__draw_{position}",
                    "weight": sample_weight,
                }
            )
            for position, index in enumerate(indices)
        ]

