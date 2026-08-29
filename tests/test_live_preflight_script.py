import unittest

from scripts.run_live_preflight import _expected_success


class SecureLivePreflightScriptTests(unittest.TestCase):
    def test_one_case_is_an_expected_partial_success(self):
        summary = {
            "selected_cases": 30,
            "successful_cases": 1,
            "errors_this_run": 0,
        }
        self.assertTrue(_expected_success(summary, 1))

    def test_error_is_not_a_success(self):
        summary = {
            "selected_cases": 30,
            "successful_cases": 1,
            "errors_this_run": 1,
        }
        self.assertFalse(_expected_success(summary, 1))

    def test_full_preflight_requires_all_30(self):
        summary = {
            "selected_cases": 30,
            "successful_cases": 29,
            "errors_this_run": 0,
        }
        self.assertFalse(_expected_success(summary, 30))


if __name__ == "__main__":
    unittest.main()
