from __future__ import annotations

from typing import Any

from .schemas import EvaluationResult, HybridResult, SimulationResult
from .readiness import release_claims


def evidence_card(
    simulation: SimulationResult,
    hybrid: HybridResult | None = None,
    evaluation: EvaluationResult | None = None,
) -> dict[str, Any]:
    diagnostics = simulation.population_diagnostics
    if evaluation and evaluation.run_id != simulation.run_id:
        raise ValueError("evaluation belongs to a different simulation")
    return {
        "release_claims": release_claims(),
        "run_id": simulation.run_id,
        "lineage_hash": simulation.lineage_hash,
        "study": simulation.scenario.name,
        "estimand": "population choice distribution",
        "population": {
            "eligible_seed_records": simulation.population_size,
            "simulation_draws": simulation.scenario.sample_size,
            "effective_sample_size": simulation.effective_sample_size,
            "filters": simulation.scenario.population_filter,
            "control_error": (
                diagnostics.max_absolute_margin_error if diagnostics else None
            ),
        },
        "scenario": {
            "information_cutoff": simulation.scenario.information_cutoff,
            "horizon": simulation.scenario.horizon,
            "novelty": simulation.scenario.novelty,
            "interaction_mode": simulation.scenario.interaction_mode,
        },
        "result": {
            "synthetic": simulation.distribution,
            "hybrid": hybrid.corrected_distribution if hybrid else None,
            "intervals": (
                {
                    key: interval.model_dump(mode="json")
                    for key, interval in hybrid.intervals.items()
                }
                if hybrid
                else None
            ),
        },
        "confidence": (
            {**simulation.confidence.model_dump(mode="json"), "label": "unqualified",
             "lower_tvd": 0.0, "upper_tvd": 1.0, "abstain": True,
             "reason": "research diagnostic only; current release confidence is unqualified"}
            if simulation.confidence
            else None
        ),
        "validation": evaluation.model_dump(mode="json") if evaluation else None,
        "labels": {
            "quantitative_output": "simulated" if not hybrid else "hybrid estimate",
            "agent_quotes": "synthetic; never a literal participant quotation",
            "validation": "comparison only; protected provenance is not verified by this report",
            "intervals": "research intervals; empirical coverage not qualified",
            "confidence": "unqualified; decision support withheld",
        },
        "warnings": release_claims()["limitations"] + simulation.warnings + (hybrid.warnings if hybrid else []),
    }


def markdown_report(
    simulation: SimulationResult,
    hybrid: HybridResult | None = None,
    evaluation: EvaluationResult | None = None,
) -> str:
    card = evidence_card(simulation, hybrid, evaluation)
    result = hybrid.corrected_distribution if hybrid else simulation.distribution
    lines = [
        f"# {simulation.scenario.name}",
        "",
        f"**Run:** `{simulation.run_id}`  ",
        f"**Lineage:** `{simulation.lineage_hash}`  ",
        f"**Output label:** {card['labels']['quantitative_output']}",
        "",
        "## Result",
        "",
        "| Choice | Estimate | Research interval (coverage unqualified) |",
        "|---|---:|---:|",
    ]
    for choice in simulation.scenario.choices:
        estimate = result[choice.choice_id]
        interval_text = "—"
        if hybrid:
            interval = hybrid.intervals[choice.choice_id]
            interval_text = f"{interval.lower:.1%}–{interval.upper:.1%}"
        lines.append(f"| {choice.label} | {estimate:.1%} | {interval_text} |")
    lines.extend(
        [
            "",
            "## Confidence",
            "",
            (
                "Unqualified — research error diagnostic "
                f"{simulation.confidence.expected_tvd:.3f}. "
                "Decision support withheld; empirical error coverage is not established."
                if simulation.confidence
                else "Not assessed."
            ),
            "",
            "## Evidence and limitations",
            "",
            f"- Eligible seed records: {simulation.population_size}",
            f"- Simulation draws: {simulation.scenario.sample_size}",
            f"- Information cutoff: {simulation.scenario.information_cutoff or 'not supplied'}",
            f"- Interaction mode: {simulation.scenario.interaction_mode}",
        ]
    )
    if evaluation:
        lines.append(f"- Comparison TVD: {evaluation.metrics['tvd']:.4f}; protected provenance not verified by this report.")
    for warning in card["warnings"]:
        lines.append(f"- Warning: {warning}")
    lines.append("")
    lines.append("*Generated agent language is synthetic and is not a participant quotation.*")
    return "\n".join(lines)
