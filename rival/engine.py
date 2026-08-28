from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .behavior import BehaviorRouter
from .confidence import ConfidenceModel
from .evaluation import evaluate_distribution
from .hybrid import HybridEstimator
from .integrity import (
    PreparedPredictionContext,
    prepare_prediction_context as build_prediction_context,
    verify_locked_context,
)
from .mathx import canonical_hash, effective_sample_size, shannon_entropy
from .population import PopulationCompiler
from .providers import PredictionProvider
from .schemas import (
    AgentPrediction,
    EvaluationResult,
    HumanObservation,
    HybridResult,
    PopulationDiagnostics,
    PopulationRecord,
    PopulationTargets,
    PredictionContext,
    RetrievalAudit,
    ScenarioSpec,
    SimulationResult,
)
from .store import EvidenceStore


class RivalEngine:
    def __init__(
        self,
        store: EvidenceStore | None = None,
        store_path: str | Path = ":memory:",
    ):
        self.population = PopulationCompiler()
        self.router = BehaviorRouter()
        self.hybrid = HybridEstimator()
        self.confidence_model = ConfidenceModel()
        self.store = store or EvidenceStore(store_path)
        self._confidence_rows: list[dict[str, float]] = []
        self._confidence_errors: list[float] = []

    def register_provider(self, name: str, provider: PredictionProvider) -> None:
        self.router.register(name, provider)

    def _prepare(
        self,
        records: list[PopulationRecord],
        scenario: ScenarioSpec,
        targets: PopulationTargets | None,
    ) -> tuple[PreparedPredictionContext, PopulationDiagnostics | None, PredictionProvider]:
        filtered = self.population.filter_records(records, scenario.population_filter)
        if not filtered:
            raise ValueError("the population filter produced zero eligible records")
        diagnostics: PopulationDiagnostics | None = None
        compiled = filtered
        if targets:
            compiled, diagnostics = self.population.calibrate(filtered, targets)
        provider = self.router.get(scenario.model_family)
        prepared = build_prediction_context(
            compiled, scenario, targets, provider.identity()
        )
        return prepared, diagnostics, provider

    def prepare_prediction_context(
        self,
        records: list[PopulationRecord],
        scenario: ScenarioSpec,
        targets: PopulationTargets | None = None,
    ) -> tuple[PredictionContext, RetrievalAudit]:
        prepared, _, _ = self._prepare(records, scenario, targets)
        return prepared.context, prepared.audit

    def simulate(
        self,
        records: list[PopulationRecord],
        scenario: ScenarioSpec,
        targets: PopulationTargets | None = None,
        include_predictions: bool = True,
        locked_context: PredictionContext | None = None,
    ) -> SimulationResult:
        prepared, diagnostics, provider = self._prepare(records, scenario, targets)
        if locked_context is not None:
            verify_locked_context(locked_context, prepared.context)
        active_context = locked_context or prepared.context
        sampled = self.population.sample(
            prepared.records, scenario.sample_size, scenario.seed
        )
        rng = np.random.default_rng(scenario.seed + 17)

        totals = Counter({choice.choice_id: 0.0 for choice in scenario.choices})
        counts = Counter({choice.choice_id: 0 for choice in scenario.choices})
        predictions: list[AgentPrediction] = []
        entropies: list[float] = []
        disagreements: list[float] = []
        for person in sampled:
            output = provider.predict(person, scenario)
            provider_call = provider.call_identity(person, scenario, output)
            choice_ids = [choice.choice_id for choice in scenario.choices]
            probabilities = np.asarray(
                [output.probabilities[choice_id] for choice_id in choice_ids], dtype=float
            )
            probabilities /= probabilities.sum()
            sampled_choice = str(rng.choice(choice_ids, p=probabilities))
            for choice_id, probability in zip(choice_ids, probabilities, strict=True):
                totals[choice_id] += person.weight * float(probability)
            counts[sampled_choice] += 1
            entropies.append(shannon_entropy(probabilities))
            disagreements.append(output.diagnostics.get("provider_disagreement", 0.0))
            predictions.append(
                AgentPrediction(
                    person_id=person.person_id,
                    probabilities={
                        choice_id: float(probability)
                        for choice_id, probability in zip(
                            choice_ids, probabilities, strict=True
                        )
                    },
                    sampled_choice=sampled_choice,
                    weight=person.weight,
                    provider=provider.name,
                    diagnostics=output.diagnostics,
                    provider_call=provider_call,
                )
            )

        total_mass = float(sum(totals.values()))
        distribution = {
            choice_id: float(totals[choice_id] / total_mass) for choice_id in totals
        }
        ess = effective_sample_size(person.weight for person in sampled)
        features = {
            "average_entropy": float(np.mean(entropies)),
            "provider_disagreement": float(np.mean(disagreements)),
            "population_margin_error": (
                diagnostics.max_absolute_margin_error if diagnostics else 0.0
            ),
            "population_ess_ratio": (
                diagnostics.effective_sample_ratio if diagnostics else 1.0
            ),
            "scenario_novelty": scenario.novelty,
            "human_anchor_rate": scenario.human_anchor_size / scenario.sample_size,
        }
        confidence = self.confidence_model.assess(features)
        warnings: list[str] = []
        if diagnostics and not diagnostics.converged:
            warnings.append("population controls did not converge")
        if confidence.abstain:
            warnings.append("confidence policy recommends human verification or abstention")
        if scenario.interaction_mode != "independent":
            warnings.append("v0.1 supports independent-agent estimands only")

        lineage_hash = canonical_hash(
            {
                "prediction_context_sha256": active_context.context_sha256,
                "provider_calls": [
                    prediction.provider_call.cache_key
                    for prediction in predictions
                    if prediction.provider_call is not None
                ],
                "distribution": distribution,
            }
        )
        result = SimulationResult(
            scenario=scenario,
            distribution=distribution,
            sample_counts=dict(counts),
            population_size=len(prepared.records),
            effective_sample_size=ess,
            average_entropy=float(np.mean(entropies)),
            provider=provider.name,
            population_diagnostics=diagnostics,
            predictions=predictions if include_predictions else [],
            confidence=confidence,
            prediction_context=active_context,
            retrieval_audit=prepared.audit,
            lineage_hash=lineage_hash,
            warnings=warnings,
        )
        self.store.register_study(scenario)
        self.store.save_run(result)
        return result

    def correct(
        self,
        simulation: SimulationResult,
        observations: list[HumanObservation],
    ) -> HybridResult:
        return self.hybrid.correct(simulation.distribution, observations)

    def evaluate(
        self,
        simulation: SimulationResult,
        observed_distribution: dict[str, float],
        preregistration_hash: str | None = None,
        learn_confidence: bool = True,
    ) -> EvaluationResult:
        evaluation = evaluate_distribution(
            simulation.run_id,
            simulation.distribution,
            observed_distribution,
            preregistration_hash=preregistration_hash,
        )
        self.store.save_evaluation(evaluation)
        if learn_confidence and simulation.confidence:
            self._confidence_rows.append(simulation.confidence.features)
            self._confidence_errors.append(evaluation.metrics["tvd"])
            if len(self._confidence_rows) >= 5:
                self.confidence_model.fit(
                    self._confidence_rows, self._confidence_errors
                )
        return evaluation
