import unittest

from rival.demo import demo_population, demo_scenario, demo_targets, run_demo
from rival.engine import RivalEngine
from rival.evaluation import jensen_shannon_divergence, total_variation_distance
from rival.hybrid import HybridEstimator
from rival.schemas import HumanObservation


class EngineTests(unittest.TestCase):
    def test_end_to_end_simulation(self):
        engine = RivalEngine()
        result = engine.simulate(
            demo_population(150), demo_scenario(200, 30), demo_targets()
        )
        self.assertAlmostEqual(sum(result.distribution.values()), 1.0, places=9)
        self.assertEqual(len(result.predictions), 200)
        self.assertTrue(result.population_diagnostics.converged)
        self.assertEqual(engine.store.get("runs", result.run_id)["run_id"], result.run_id)

    def test_demo_hybrid_reduces_protected_error(self):
        engine = RivalEngine()
        result = run_demo(engine=engine, sample_size=800, human_anchor_size=120)
        synthetic = result["synthetic_evaluation"]["metrics"]["tvd"]
        hybrid = result["hybrid_evaluation"]["metrics"]["tvd"]
        self.assertLess(hybrid, synthetic)
        self.assertGreater(result["improvement"]["relative_tvd_reduction"], 0.2)
        self.assertEqual(engine.confidence_model.training_examples, 0)


class HybridTests(unittest.TestCase):
    def test_residual_correction(self):
        observations = [
            HumanObservation(
                person_id=f"p{i}",
                observed_choice="b" if i < 7 else "a",
                synthetic_probabilities={"a": 0.6, "b": 0.4},
            )
            for i in range(10)
        ]
        result = HybridEstimator().correct({"a": 0.6, "b": 0.4}, observations)
        self.assertAlmostEqual(result.corrected_distribution["b"], 0.7, places=7)
        self.assertAlmostEqual(sum(result.corrected_distribution.values()), 1.0, places=9)

    def test_no_anchor_is_explicitly_uncertain(self):
        result = HybridEstimator().correct({"a": 0.4, "b": 0.6}, [])
        self.assertEqual(result.human_sample_size, 0)
        self.assertTrue(result.warnings)


class EvaluationTests(unittest.TestCase):
    def test_distribution_metrics(self):
        self.assertAlmostEqual(
            total_variation_distance({"a": 0.5, "b": 0.5}, {"a": 0.7, "b": 0.3}),
            0.2,
        )
        self.assertAlmostEqual(
            jensen_shannon_divergence({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
