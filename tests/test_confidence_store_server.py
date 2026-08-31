import tempfile
import unittest
from pathlib import Path

from rival.confidence import ConfidenceModel
from rival.demo import demo_scenario
from rival.schemas import EvidenceSource
from rival.server import RivalApplication
from rival.store import EvidenceStore, ImmutableConflict


class ConfidenceTests(unittest.TestCase):
    def test_cold_start_is_conservative(self):
        assessment = ConfidenceModel().assess(
            {
                "average_entropy": 0.8,
                "provider_disagreement": 0.03,
                "population_margin_error": 0.01,
                "population_ess_ratio": 0.7,
                "scenario_novelty": 0.8,
                "human_anchor_rate": 0.0,
            }
        )
        self.assertIn(assessment.label, {"medium", "low"})
        self.assertEqual(assessment.training_examples, 0)

    def test_ridge_fit(self):
        model = ConfidenceModel()
        rows = []
        errors = []
        for index in range(10):
            novelty = index / 10
            rows.append(
                {
                    "average_entropy": 0.5,
                    "provider_disagreement": 0.01,
                    "population_margin_error": 0.0,
                    "population_ess_ratio": 0.9,
                    "scenario_novelty": novelty,
                    "human_anchor_rate": 0.1,
                }
            )
            errors.append(0.05 + 0.2 * novelty)
        model.fit(rows, errors)
        low = model.assess({**rows[0], "scenario_novelty": 0.1})
        high = model.assess({**rows[0], "scenario_novelty": 0.9})
        self.assertLess(low.expected_tvd, high.expected_tvd)
        self.assertEqual(high.training_examples, 10)


class StoreTests(unittest.TestCase):
    def test_append_only_conflict(self):
        store = EvidenceStore()
        source = EvidenceSource(
            source_id="fixed",
            name="Source A",
            source_type="licensed",
            rights_reference="agreement-1",
        )
        store.register_evidence(source)
        store.register_evidence(source)
        changed = source.model_copy(update={"name": "Changed"})
        with self.assertRaises(ImmutableConflict):
            store.register_evidence(changed)


class ServerApplicationTests(unittest.TestCase):
    def test_health_and_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            app = RivalApplication(str(Path(directory) / "test.sqlite3"))
            try:
                self.assertEqual(app.health()["status"], "ok")
                result = app.demo_run(
                    {
                        "sample_size": 250,
                        "human_anchor_size": 60,
                        "scenario": {"name": "API test"},
                    }
                )
                self.assertEqual(result["simulation"]["scenario"]["name"], "API test")
                self.assertIn("evidence_card", result)
            finally:
                app.engine.store.close()


if __name__ == "__main__":
    unittest.main()

