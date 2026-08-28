from __future__ import annotations

from collections import Counter

from .mathx import shannon_entropy
from .schemas import AgentPrediction, PopulationRecord


class AdaptiveSampler:
    """Select human anchors where uncertainty, weight and coverage are highest."""

    def select(
        self,
        predictions: list[AgentPrediction],
        records_by_id: dict[str, PopulationRecord],
        sample_size: int,
        coverage_attribute: str | None = None,
    ) -> list[str]:
        if sample_size <= 0:
            return []
        if sample_size >= len(predictions):
            return [prediction.person_id for prediction in predictions]

        group_counts: Counter[str] = Counter()
        selected: list[str] = []
        remaining = {prediction.person_id: prediction for prediction in predictions}
        while remaining and len(selected) < sample_size:
            best_id = max(
                remaining,
                key=lambda person_id: self._score(
                    remaining[person_id],
                    records_by_id.get(person_id),
                    coverage_attribute,
                    group_counts,
                ),
            )
            selected.append(best_id)
            record = records_by_id.get(best_id)
            if coverage_attribute and record:
                group = str(record.attributes.get(coverage_attribute, "__missing__"))
                group_counts[group] += 1
            del remaining[best_id]
        return selected

    @staticmethod
    def _score(
        prediction: AgentPrediction,
        record: PopulationRecord | None,
        coverage_attribute: str | None,
        group_counts: Counter[str],
    ) -> float:
        uncertainty = shannon_entropy(prediction.probabilities.values())
        disagreement = prediction.diagnostics.get("provider_disagreement", 0.0)
        coverage_bonus = 0.0
        if coverage_attribute and record:
            group = str(record.attributes.get(coverage_attribute, "__missing__"))
            coverage_bonus = 1.0 / (1.0 + group_counts[group])
        return (
            0.55 * uncertainty
            + 0.25 * min(disagreement * 10.0, 1.0)
            + 0.15 * coverage_bonus
            + 0.05 * min(prediction.weight, 1.0)
        )

