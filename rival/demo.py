from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import RivalEngine
from .evaluation import evaluate_distribution
from .mathx import normalize, stable_softmax
from .reporting import evidence_card
from .schemas import (
    ChoiceSpec,
    HumanObservation,
    PopulationRecord,
    PopulationTargets,
    ScenarioSpec,
)


def demo_population(size: int = 600, seed: int = 41) -> list[PopulationRecord]:
    rng = np.random.default_rng(seed)
    records: list[PopulationRecord] = []
    for index in range(size):
        age_band = str(rng.choice(["18-34", "35-54", "55+"], p=[0.48, 0.34, 0.18]))
        region = str(
            rng.choice(["north", "south", "central", "west"], p=[0.2, 0.35, 0.3, 0.15])
        )
        income = str(rng.choice(["lower", "middle", "upper"], p=[0.32, 0.5, 0.18]))
        household = str(rng.choice(["single", "family"], p=[0.54, 0.46]))
        price_sensitivity = float(np.clip(rng.normal(0.72 if income == "lower" else 0.46, 0.16), 0.05, 1.0))
        sustainability = float(np.clip(rng.beta(2.2 if age_band == "18-34" else 1.7, 2.0), 0, 1))
        quality = float(np.clip(rng.beta(2.5 if income == "upper" else 1.8, 1.8), 0, 1))
        convenience = float(np.clip(rng.beta(2.4 if household == "family" else 1.7, 1.9), 0, 1))
        records.append(
            PopulationRecord(
                person_id=f"seed_{index:04d}",
                attributes={
                    "age_band": age_band,
                    "region": region,
                    "income": income,
                    "household": household,
                },
                preferences={
                    "price_sensitivity": price_sensitivity,
                    "sustainability": sustainability,
                    "quality": quality,
                    "convenience": convenience,
                    "novelty": float(rng.beta(1.8, 2.2)),
                },
                evidence_ids=["src_demo_panel"],
            )
        )
    return records


def demo_targets() -> PopulationTargets:
    return PopulationTargets(
        label="Illustrative national household population",
        controls={
            "age_band": {"18-34": 0.32, "35-54": 0.40, "55+": 0.28},
            "region": {"north": 0.30, "south": 0.25, "central": 0.25, "west": 0.20},
            "income": {"lower": 0.27, "middle": 0.52, "upper": 0.21},
            "household": {"single": 0.45, "family": 0.55},
        },
    )


def demo_scenario(sample_size: int = 1200, human_anchor_size: int = 80) -> ScenarioSpec:
    return ScenarioSpec(
        name="Laundry refill concept test",
        question="Which laundry detergent format would you purchase next month?",
        context=(
            "A familiar household-care brand is considering three formats. "
            "All options deliver the same number of washes."
        ),
        choices=[
            ChoiceSpec(
                choice_id="value",
                label="Value bottle",
                description="Conventional bottle at $6",
                features={"price": 0.60, "quality": 0.45, "convenience": 0.65},
            ),
            ChoiceSpec(
                choice_id="refill",
                label="Eco refill pouch",
                description="80% less plastic at $8",
                features={
                    "price": 0.80,
                    "quality": 0.58,
                    "convenience": 0.50,
                    "sustainability": 0.95,
                    "novelty": 0.35,
                },
            ),
            ChoiceSpec(
                choice_id="premium",
                label="Premium concentrate",
                description="Compact high-performance bottle at $10",
                features={
                    "price": 1.0,
                    "quality": 1.0,
                    "convenience": 0.82,
                    "sustainability": 0.48,
                    "novelty": 0.25,
                },
            ),
        ],
        task_type="choice",
        model_family="heuristic",
        sample_size=sample_size,
        human_anchor_size=human_anchor_size,
        novelty=0.42,
        information_cutoff="2026-08-27",
        horizon="next purchase within 30 days",
        intended_use="illustrative consumer concept research",
    )


def _hidden_truth(probabilities: dict[str, float]) -> dict[str, float]:
    order = ["value", "refill", "premium"]
    base = np.asarray([probabilities[key] for key in order])
    # A deliberately hidden behavioral shift: real shoppers like refill more
    # than the synthetic baseline and premium less. It acts as the protected outcome.
    adjusted = normalize(base * np.asarray([0.96, 1.38, 0.72]))
    return {key: float(value) for key, value in zip(order, adjusted, strict=True)}


def run_demo(
    engine: RivalEngine | None = None,
    sample_size: int = 1200,
    human_anchor_size: int = 80,
    scenario_overrides: dict | None = None,
) -> dict:
    engine = engine or RivalEngine()
    scenario = demo_scenario(sample_size, human_anchor_size)
    if scenario_overrides:
        allowed = {"name", "question", "context", "horizon", "novelty"}
        updates = {
            key: value for key, value in scenario_overrides.items() if key in allowed
        }
        scenario = ScenarioSpec.model_validate(
            {**scenario.model_dump(mode="json"), **updates}
        )
    simulation = engine.simulate(demo_population(), scenario, demo_targets())
    choice_ids = [choice.choice_id for choice in scenario.choices]

    truth_rows = [_hidden_truth(row.probabilities) for row in simulation.predictions]
    truth_distribution = {
        key: float(np.mean([row[key] for row in truth_rows])) for key in choice_ids
    }
    rng = np.random.default_rng(scenario.seed + 99)
    anchor_indices = rng.choice(
        len(simulation.predictions),
        size=min(human_anchor_size, len(simulation.predictions)),
        replace=False,
    )
    observations: list[HumanObservation] = []
    for index in anchor_indices:
        prediction = simulation.predictions[int(index)]
        truth = truth_rows[int(index)]
        observed_choice = str(rng.choice(choice_ids, p=[truth[key] for key in choice_ids]))
        observations.append(
            HumanObservation(
                person_id=prediction.person_id,
                observed_choice=observed_choice,
                synthetic_probabilities=prediction.probabilities,
            )
        )
    hybrid = engine.correct(simulation, observations)
    synthetic_eval = engine.evaluate(
        simulation,
        truth_distribution,
        preregistration_hash="demo_locked_before_truth",
        learn_confidence=False,
    )
    hybrid_eval = evaluate_distribution(
        simulation.run_id,
        hybrid.corrected_distribution,
        truth_distribution,
        preregistration_hash="demo_locked_before_truth",
    )
    return {
        "simulation": simulation.model_dump(mode="json"),
        "hybrid": hybrid.model_dump(mode="json"),
        "protected_outcome": truth_distribution,
        "synthetic_evaluation": synthetic_eval.model_dump(mode="json"),
        "hybrid_evaluation": hybrid_eval.model_dump(mode="json"),
        "evidence_card": evidence_card(simulation, hybrid, synthetic_eval),
        "improvement": {
            "tvd_reduction": synthetic_eval.metrics["tvd"]
            - hybrid_eval.metrics["tvd"],
            "relative_tvd_reduction": (
                1.0 - hybrid_eval.metrics["tvd"] / synthetic_eval.metrics["tvd"]
                if synthetic_eval.metrics["tvd"] > 0
                else 0.0
            ),
        },
    }
