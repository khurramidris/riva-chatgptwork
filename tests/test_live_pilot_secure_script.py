import unittest

from scripts.run_live_pilot_secure import _checkpoint_status


class SecureLivePilotScriptTests(unittest.TestCase):
    def test_intermediate_checkpoint(self):
        summary = {"successful_cases": 300, "errors_this_run": 0, "status": "STOPPED"}
        self.assertEqual(
            _checkpoint_status(summary, 300), "PILOT_CHECKPOINT_COMPLETE"
        )

    def test_final_checkpoint(self):
        summary = {"successful_cases": 1500, "errors_this_run": 0, "status": "COMPLETE"}
        self.assertEqual(_checkpoint_status(summary, 1500), "PILOT_COMPLETE")

    def test_provider_error_pauses(self):
        summary = {"successful_cases": 417, "errors_this_run": 1, "status": "STOPPED"}
        self.assertEqual(
            _checkpoint_status(summary, 600), "PILOT_PAUSED_PROVIDER_ERROR"
        )

    def test_expiry_or_budget_stop_is_incomplete(self):
        summary = {"successful_cases": 417, "errors_this_run": 0, "status": "STOPPED"}
        self.assertEqual(
            _checkpoint_status(summary, 600), "PILOT_CHECKPOINT_INCOMPLETE"
        )


if __name__ == "__main__":
    unittest.main()
