from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from ..elicitation import HashingTextEmbedder, SSRScale, SemanticSimilarityRater
from ..experiments.srct import PairedPrediction, PrePeriodAnchor, estimate_paired_srct
from ..personas import InterviewPersonaBuilder, InterviewTranscript, InterviewTurn
from ..pricing import fit_persona_demand, optimize_price
from ..providers import CentauriProvider, SocratesProvider
from ..uncertainty import human_mean_interval, synthetic_mean_interval
from ..vendor.uq_survey.evaluations import CI as upstream_ci
from ..vendor.uq_survey.evaluations import synthetic_CI as upstream_synthetic_ci
from .mad import source_identity, summary_mad
from .provenance import stable_hash
from .synthetic_control import ResearchSyntheticControl


def run_research_integration_qualification() -> dict[str, Any]:
    """Smoke/parity qualification for integrated research components.

    Passing means the components are wired and reproduce bounded numerical
    invariants. It is not evidence of customer-domain predictive validity.
    """

    checks: list[dict[str, Any]] = []

    def check(name: str, operation: Callable[[], Any]) -> None:
        try:
            detail = operation()
            checks.append({"name": name, "status": "PASS", "detail": detail})
        except Exception as exc:
            checks.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    def ssr_check() -> dict[str, Any]:
        rater = SemanticSimilarityRater(
            SSRScale(
                choice_ids=("disagree", "neutral", "agree"),
                anchors=("I disagree", "I am neutral", "I agree"),
            ),
            HashingTextEmbedder(256),
        )
        probabilities = rater.rate("I agree with this proposal")
        if not np.isclose(sum(probabilities.values()), 1.0) or not all(
            np.isfinite(value) and value >= 0 for value in probabilities.values()
        ):
            raise AssertionError("SSR did not return a valid PMF")
        # Explicitly exercise the upstream denominator-zero edge case.
        degenerate = SemanticSimilarityRater(
            SSRScale(choice_ids=("a", "b"), anchors=("same", "same")),
            HashingTextEmbedder(64),
            epsilon=0.0,
        ).rate("same")
        if not np.isclose(sum(degenerate.values()), 1.0):
            raise AssertionError("SSR degenerate guard failed")
        return {"pmf": probabilities, "degenerate_guard": degenerate}

    check("semantic_similarity_rating", ssr_check)

    def uq_check() -> dict[str, Any]:
        values = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        human = human_mean_interval(values)
        expected_human = upstream_ci(values, 0.95)
        if not np.allclose(
            [human.estimate, human.lower, human.upper], expected_human, atol=1e-12
        ):
            raise AssertionError("human CI differs from licensed implementation")
        methods = {}
        for method in ("clt", "hoeffding", "bernstein"):
            ours = synthetic_mean_interval(values, k=7, method=method)
            expected = upstream_synthetic_ci(
                values, 7, 0.05, C=2.0, CI_type=method
            )
            if not np.allclose(
                [ours.estimate, ours.lower, ours.upper], expected, atol=1e-12
            ):
                raise AssertionError(f"{method} CI differs from licensed implementation")
            methods[method] = asdict(ours)
        return methods

    check("uq_survey_numerical_parity", uq_check)

    def syn_digits_check() -> dict[str, Any]:
        matrix = np.array(
            [[1.0, 2.0, np.nan], [2.0, np.nan, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
        )
        completed = ResearchSyntheticControl.complete(
            matrix, method="hard_svd", rank=2, max_iter=30
        )
        observed = np.isfinite(matrix)
        if not np.all(np.isfinite(completed)) or not np.allclose(
            completed[observed], matrix[observed]
        ):
            raise AssertionError("matrix completion failed invariants")
        return {"shape": list(completed.shape), "missing_after": int(np.isnan(completed).sum())}

    check("syn_digits_full_component", syn_digits_check)

    def srct_check() -> dict[str, Any]:
        pairs = [
            PairedPrediction(str(index), 0.2 + index * 0.01, 0.3 + index * 0.01)
            for index in range(12)
        ]
        anchor = PrePeriodAnchor(0.22, 0.27, 0.20, 0.23)
        estimate = estimate_paired_srct(pairs, pre_period=anchor)
        if not np.isclose(estimate.raw_effect, 0.1) or not np.isclose(
            estimate.residual_adjustment, 0.02
        ):
            raise AssertionError("paired or pre-period effect is incorrect")
        return asdict(estimate)

    check("paired_srct_and_preperiod", srct_check)

    def pricing_check() -> dict[str, Any]:
        rng = np.random.default_rng(28)
        matrix = np.clip(rng.normal([0.20, 0.35, 0.50], 0.05, size=(18, 3)), 0.01, 0.95)
        demand = np.maximum(1, np.rint(100 * (matrix @ np.array([0.15, 0.25, 0.10])))).astype(int)
        model = fit_persona_demand(
            matrix, demand, 100, objective="truncated", calibration_iterations=2
        )
        probabilities = model.purchase_probability(matrix[:3])
        if not np.isclose(model.persona_weights.sum() + model.no_buy_weight, 1.0):
            raise AssertionError("persona/no-buy weights are not on the simplex")
        decision = optimize_price([10, 12, 15], [0.5, 0.4, 0.25], 1000, unit_cost=4, draws=500)
        return {
            "converged": model.converged,
            "probabilities": probabilities.tolist(),
            "decision": asdict(decision),
        }

    check("persona_demand_and_pricing", pricing_check)

    def persona_check() -> dict[str, Any]:
        transcript = InterviewTranscript(
            person_id="p1",
            attributes={"age_band": "35-49"},
            preferences={"quality": 0.8},
            turns=[
                InterviewTurn(speaker="interviewer", text="What matters to you?"),
                InterviewTurn(speaker="participant", text="Reliability and service."),
            ],
        )
        record = InterviewPersonaBuilder().build(transcript)
        try:
            InterviewTranscript(
                person_id="bad",
                metadata={"observed_outcome": "yes"},
                turns=[InterviewTurn(speaker="participant", text="hello")],
            )
        except ValueError:
            protected = "rejected"
        else:
            raise AssertionError("protected outcome entered a persona")
        return {"person_id": record.person_id, "history": len(record.history), "protected": protected}

    check("interview_grounded_personas", persona_check)

    def adapter_check() -> dict[str, Any]:
        providers = [
            CentauriProvider(model_revision="test-centauri-revision", model_license="test-authorized"),
            SocratesProvider(model_revision="test-socrates-revision", model_license="test-authorized"),
        ]
        identities = [provider.identity().model_dump(mode="json") for provider in providers]
        if any("api_key" in str(identity).casefold() for identity in identities):
            raise AssertionError("credential material entered provider identity")
        return identities

    check("behavioral_model_adapters", adapter_check)

    def mad_check() -> dict[str, Any]:
        summary = summary_mad([0.8, 0.9, 1.0, 0.7])
        if len(summary) != 4 or not np.isclose(summary[0], 0.85):
            raise AssertionError("Twin-2K MAD summary is invalid")
        return {"summary": summary, "source": source_identity()}

    check("twin2k_official_mad", mad_check)

    report: dict[str, Any] = {
        "schema_version": "rival.research-integration-qualification.v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "scope": "component wiring and numerical parity; not external predictive or causal validity",
        "checks": checks,
    }
    report["report_sha256"] = stable_hash(report)
    return report

