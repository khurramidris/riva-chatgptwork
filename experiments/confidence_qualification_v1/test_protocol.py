"""Checks for concrete leakage and coverage failure modes, not predictive validity."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('confidence_qualification', Path(__file__).with_name('run.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTests(unittest.TestCase):
    def test_repeated_questions_do_not_inflate_calibration_sample_size(self):
        # Nine identical rows in one family still supply only one independent score.
        q, groups = module.upper_residual_quantile(np.full(9, 0.2), np.full(9, 0.1), ['a'] * 9)
        self.assertEqual(groups, 1)
        self.assertEqual(q, 1.0)

    def test_finite_sample_rank_uses_worst_score_with_nine_families(self):
        q, groups = module.upper_residual_quantile(np.arange(1, 10) / 10,
                                                   np.zeros(9), list('abcdefghi'))
        self.assertEqual(groups, 9)
        self.assertAlmostEqual(q, 0.9)

    def test_temporal_family_purge_removes_earlier_sibling_labels(self):
        actual = module.purge_earlier_families([0, 0, 1, 2, 3], ['a', 'b', 'a', 'c', 'a'])
        np.testing.assert_array_equal(actual, [-1, 0, -1, 2, 3])

    def test_current_labels_cannot_be_passed_to_error_fit(self):
        with self.assertRaisesRegex(ValueError, 'only error_fit'):
            module.error_predictions(np.zeros((4, 7)), np.array(['word text'] * 4),
                                     np.zeros(4), np.array([True, True, False, False]))

    def test_invalid_calibration_does_not_produce_confident_bound(self):
        with self.assertRaises(ValueError):
            module.upper_residual_quantile([float('nan')], [0.1], ['a'])


if __name__ == '__main__':
    unittest.main()
