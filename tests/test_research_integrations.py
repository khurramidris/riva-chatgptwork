import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rival.demo import demo_population, demo_scenario
from rival.elicitation import (
    GeneratedText,
    HashingTextEmbedder,
    SSRScale,
    SemanticSimilarityRater,
    SSRElicitationProvider,
)
from rival.experiments.srct import (
    PairedPrediction,
    PrePeriodAnchor,
    estimate_paired_srct,
    simulate_paired_srct,
)
from rival.personas import (
    InterviewPersonaBuilder,
    InterviewTranscript,
    InterviewTurn,
    load_interview_csv,
    load_interview_jsonl,
)
from rival.pricing import fit_persona_demand, optimize_price, revenue_cvar
from rival.providers import CentauriProvider, PredictionProvider, ProviderPrediction, SocratesProvider
from rival.research.integration_qualification import run_research_integration_qualification
from rival.research.mad import summary_mad
from rival.research.synthetic_control import ResearchSyntheticControl
from rival.server import RivalApplication
from rival.schemas import ScenarioSpec
from rival.uncertainty import (
    human_mean_interval,
    residual_corrected_interval,
    synthetic_mean_interval,
)
from rival.vendor.uq_survey.evaluations import CI, synthetic_CI


class StaticTextGenerator:
    @property
    def identity(self):
        return {"provider": "test", "model": "static"}

    def generate(self, person, scenario):
        return GeneratedText("I strongly prefer the second option")


class TreatmentProvider(PredictionProvider):
    name = "treatment-test"

    def predict(self, person, scenario):
        value = 0.65 if scenario.scenario_id == "treatment" else 0.45
        return ProviderPrediction(probabilities={"yes": value, "no": 1 - value})


class SSRTests(unittest.TestCase):
    def test_hashing_embedder_is_deterministic(self):
        embedder = HashingTextEmbedder(64)
        first = embedder.encode(["same words"])
        second = embedder.encode(["same words"])
        np.testing.assert_array_equal(first, second)

    def test_ssr_returns_probability_mass(self):
        rater = SemanticSimilarityRater(
            SSRScale(("no", "maybe", "yes"), ("I disagree", "I am unsure", "I agree")),
            HashingTextEmbedder(128),
        )
        result = rater.rate("I agree")
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertTrue(all(value >= 0 for value in result.values()))

    def test_ssr_degenerate_embeddings_fail_safe(self):
        result = SemanticSimilarityRater(
            SSRScale(("a", "b"), ("same", "same")),
            HashingTextEmbedder(64),
            epsilon=0,
        ).rate("same")
        self.assertEqual(result, {"a": 0.5, "b": 0.5})

    def test_ssr_provider_maps_text_to_scenario_choices(self):
        provider = SSRElicitationProvider(StaticTextGenerator(), HashingTextEmbedder(128))
        result = provider.predict(demo_population(20)[0], demo_scenario(20, 0))
        self.assertEqual(set(result.probabilities), {"value", "refill", "premium"})
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)


class UncertaintyTests(unittest.TestCase):
    def test_human_interval_matches_upstream(self):
        values = np.array([0, 1, 1, 0, 1, 1], dtype=float)
        ours = human_mean_interval(values)
        np.testing.assert_allclose([ours.estimate, ours.lower, ours.upper], CI(values, 0.95))

    def test_synthetic_intervals_match_all_upstream_formulas(self):
        values = np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=float)
        for method in ("clt", "hoeffding", "bernstein"):
            with self.subTest(method=method):
                ours = synthetic_mean_interval(values, k=7, method=method)
                expected = synthetic_CI(values, 7, 0.05, C=2, CI_type=method)
                np.testing.assert_allclose([ours.estimate, ours.lower, ours.upper], expected)

    def test_residual_corrected_interval(self):
        result = residual_corrected_interval(
            [0.3, 0.4, 0.5, 0.4],
            [1, 0, 1, 1],
            [0.6, 0.4, 0.7, 0.6],
        )
        self.assertGreater(result.estimate, result.synthetic.estimate)
        self.assertLessEqual(result.lower, result.upper)


class SyntheticControlTests(unittest.TestCase):
    def test_all_completion_methods_return_finite_matrix(self):
        matrix = np.array([[1, 2, np.nan], [2, np.nan, 4], [3, 4, 5], [4, 5, 6]], dtype=float)
        observed = np.isfinite(matrix)
        for method in ("hard_svd", "soft_svd", "als"):
            with self.subTest(method=method):
                result = ResearchSyntheticControl.complete(matrix, method=method, rank=2, max_iter=30)
                self.assertTrue(np.all(np.isfinite(result)))
                if method != "als":
                    np.testing.assert_allclose(result[observed], matrix[observed])

    def test_full_synthetic_control_ridge_path(self):
        synthetic = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]], dtype=float)
        real = synthetic + np.array([0.1, -0.1, 0.2])
        result = ResearchSyntheticControl(real, synthetic, dataset_name="test").evaluate_column(2, method="ridge")
        self.assertIn("rival_source", result)
        self.assertEqual(result["rival_source"]["integration"], "vendored-with-portability-patches")


class ExperimentPricingPersonaTests(unittest.TestCase):
    def test_paired_srct_with_preperiod_adjustment(self):
        pairs = [PairedPrediction(str(i), 0.2, 0.3) for i in range(10)]
        result = estimate_paired_srct(pairs, pre_period=PrePeriodAnchor(0.2, 0.26, 0.2, 0.23))
        self.assertAlmostEqual(result.raw_effect, 0.1)
        self.assertAlmostEqual(result.residual_adjustment, 0.03)
        self.assertAlmostEqual(result.estimate, 0.13)

    def test_provider_driven_srct(self):
        payload = demo_scenario(20, 0).model_dump(mode="json")
        payload.update(
            {
                "scenario_id": "control",
                "choices": [
                    {"choice_id": "yes", "label": "Yes"},
                    {"choice_id": "no", "label": "No"},
                ],
            }
        )
        control = ScenarioSpec.model_validate(payload)
        treatment = control.model_copy(update={"scenario_id": "treatment"})
        result = simulate_paired_srct(TreatmentProvider(), demo_population(20), control, treatment, "yes")
        self.assertAlmostEqual(result.estimate, 0.2)

    def test_persona_demand_weights_and_predictions(self):
        rng = np.random.default_rng(8)
        matrix = np.clip(rng.normal([0.2, 0.4, 0.5], 0.04, size=(16, 3)), 0.01, 0.95)
        demand = np.maximum(1, np.rint(100 * matrix @ np.array([0.12, 0.25, 0.08]))).astype(int)
        model = fit_persona_demand(matrix, demand, 100, calibration_iterations=2)
        self.assertAlmostEqual(model.persona_weights.sum() + model.no_buy_weight, 1.0)
        predictions = model.purchase_probability(matrix[:2])
        self.assertTrue(np.all((predictions > 0) & (predictions < 1)))

    def test_price_optimization_and_cvar(self):
        decision = optimize_price([10, 12, 15], [0.5, 0.4, 0.25], 1000, unit_cost=4, draws=500)
        self.assertEqual(decision.price, 12)
        self.assertLessEqual(revenue_cvar([1, 2, 10, 20], 0.5), np.mean([1, 2, 10, 20]))

    def test_interview_builder_and_outcome_guard(self):
        transcript = InterviewTranscript(
            person_id="p1",
            turns=[InterviewTurn(speaker="participant", text="Quality matters most.")],
            preferences={"quality": 0.9},
        )
        record = InterviewPersonaBuilder().build(transcript)
        self.assertEqual(record.person_id, "p1")
        self.assertEqual(len(record.history), 1)
        with self.assertRaises(ValueError):
            InterviewTranscript(
                person_id="p2",
                turns=[InterviewTurn(speaker="participant", text="hello")],
                metadata={"observed_outcome": "yes"},
            )

    def test_interview_file_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "interviews.jsonl"
            jsonl.write_text(json.dumps({"person_id": "p1", "turns": [{"speaker": "participant", "text": "hello"}]}) + "\n")
            csv_path = root / "interviews.csv"
            csv_path.write_text("person_id,speaker,text\np2,participant,hello\n")
            self.assertEqual(load_interview_jsonl(jsonl)[0].person_id, "p1")
            self.assertEqual(load_interview_csv(csv_path)[0].person_id, "p2")


class AdapterSurfaceTests(unittest.TestCase):
    def test_behavioral_adapter_identities_are_revision_bound(self):
        providers = [
            CentauriProvider(model_revision="abc", model_license="authorized"),
            SocratesProvider(model_revision="def", model_license="authorized"),
        ]
        self.assertEqual(providers[0].identity().model, "socius-org/Centauri@abc")
        self.assertEqual(providers[1].identity().model, "Socrates@def")
        self.assertNotIn("api_key", str([provider.identity() for provider in providers]).lower())

    def test_official_mad_wrapper(self):
        result = summary_mad([0.8, 0.9, 1.0, 0.7])
        self.assertEqual(result[0], 0.85)
        self.assertEqual(len(result), 4)

    def test_server_research_surfaces_and_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            app = RivalApplication(str(Path(directory) / "ledger.sqlite3"))
            ssr = app.research_ssr(
                {
                    "response": "yes",
                    "choices": [
                        {"choice_id": "no", "anchor": "no"},
                        {"choice_id": "yes", "anchor": "yes"},
                    ],
                }
            )
            interval = app.research_uncertainty({"values": [0, 1, 1, 0, 1]})
            self.assertAlmostEqual(sum(ssr["probabilities"].values()), 1.0)
            self.assertEqual(interval["sample_size"], 5)
            app.engine.store.close()
        self.assertEqual(run_research_integration_qualification()["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
