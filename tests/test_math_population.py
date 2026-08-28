import unittest

import numpy as np

from rival.mathx import effective_sample_size, project_simplex, stable_softmax
from rival.population import PopulationCompiler, PopulationError
from rival.schemas import PopulationRecord, PopulationTargets


class MathTests(unittest.TestCase):
    def test_simplex_projection(self):
        projected = project_simplex([1.2, -0.1, 0.4])
        self.assertAlmostEqual(float(projected.sum()), 1.0, places=10)
        self.assertTrue(np.all(projected >= 0))

    def test_softmax_and_ess(self):
        values = stable_softmax([0, 1, 2])
        self.assertAlmostEqual(float(values.sum()), 1.0)
        self.assertGreater(values[2], values[1])
        self.assertAlmostEqual(effective_sample_size([1, 1, 1, 1]), 4.0)


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            PopulationRecord(person_id=f"p{i}", attributes={"region": region, "age": age})
            for i, (region, age) in enumerate(
                [
                    ("north", "young"),
                    ("north", "old"),
                    ("south", "young"),
                    ("south", "old"),
                    ("south", "old"),
                ]
            )
        ]

    def test_raking_hits_controls(self):
        targets = PopulationTargets(
            controls={
                "region": {"north": 0.65, "south": 0.35},
                "age": {"young": 0.4, "old": 0.6},
            }
        )
        compiled, diagnostics = PopulationCompiler().calibrate(self.records, targets)
        self.assertTrue(diagnostics.converged)
        self.assertLess(diagnostics.max_absolute_margin_error, 1e-5)
        north = sum(record.weight for record in compiled if record.attributes["region"] == "north")
        self.assertAlmostEqual(north / sum(record.weight for record in compiled), 0.65, places=5)

    def test_missing_target_support_fails(self):
        targets = PopulationTargets(controls={"region": {"north": 0.5, "west": 0.5}})
        with self.assertRaises(PopulationError):
            PopulationCompiler().calibrate(self.records, targets)

    def test_filter_and_sample_are_reproducible(self):
        compiler = PopulationCompiler()
        filtered = compiler.filter_records(self.records, {"region": "south"})
        first = compiler.sample(filtered, 10, 9)
        second = compiler.sample(filtered, 10, 9)
        self.assertEqual([row.person_id for row in first], [row.person_id for row in second])


if __name__ == "__main__":
    unittest.main()

