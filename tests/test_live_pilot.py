import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rival.demo import demo_population, demo_scenario
from rival.live_pilot import (
    BudgetExceeded,
    BudgetGuard,
    DeterministicRehearsalProvider,
    PilotProtocolError,
    evaluate_live_pilot,
    load_and_verify_protocol,
    prepare_twin2k_live_pilot,
    run_live_pilot,
)
from rival.providers import OpenAICompatibleProvider, ProviderError


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ProviderAccountingTests(unittest.TestCase):
    def test_remote_provider_requires_https_and_url_credentials_are_rejected(self):
        with self.assertRaises(ProviderError):
            OpenAICompatibleProvider(
                model="test", api_key="dummy", base_url="http://example.com/v1/chat"
            )
        with self.assertRaises(ProviderError):
            OpenAICompatibleProvider(
                model="test",
                api_key="dummy",
                base_url="https://user:password@example.com/v1/chat",
            )
        local = OpenAICompatibleProvider(
            model="test", api_key="dummy", base_url="http://127.0.0.1:8000/v1/chat"
        )
        self.assertEqual(local.model, "test")

    def test_openai_provider_records_usage_without_recording_key(self):
        person = demo_population(20)[0]
        scenario = demo_scenario(20, 0)
        probabilities = {
            choice.choice_id: 1.0 / len(scenario.choices) for choice in scenario.choices
        }
        response = {
            "id": "request-1",
            "choices": [{"message": {"content": json.dumps(probabilities)}}],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 17,
                "total_tokens": 140,
                "cost": 0.0042,
            },
        }
        provider = OpenAICompatibleProvider(
            model="test-model", api_key="super-secret", max_retries=1
        )
        with patch("urllib.request.urlopen", return_value=_Response(response)):
            output = provider.predict(person, scenario)
        self.assertEqual(output.diagnostics["prompt_tokens"], 123)
        self.assertEqual(output.diagnostics["completion_tokens"], 17)
        self.assertEqual(output.diagnostics["provider_cost_usd"], 0.0042)
        self.assertNotIn("super-secret", str(provider.identity().model_dump()))

    def test_expired_budget_fails_before_a_call(self):
        guard = BudgetGuard(
            budget_usd=1,
            input_cost_per_million=1,
            output_cost_per_million=1,
            max_calls=1,
            not_after=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        with self.assertRaises(BudgetExceeded):
            guard.authorize(0.01)


class FrozenPilotTests(unittest.TestCase):
    def _prepare(self, root: Path):
        return prepare_twin2k_live_pilot(
            root,
            cohort_size=20,
            target_count=3,
            anchor_size=4,
            history_items=8,
            minimum_history_items=4,
            seed=99117,
        )

    def test_protocol_and_cases_verify_and_cases_are_outcome_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._prepare(root)
            loaded, cases, inputs = load_and_verify_protocol(root / "protocol.json")
            self.assertEqual(loaded["protocol_sha256"], protocol["protocol_sha256"])
            self.assertEqual(len(cases), loaded["cases"]["total"])
            self.assertGreater(loaded["cases"]["preflight"], 0)
            case_text = (root / "cases.jsonl").read_text(encoding="utf-8").lower()
            for forbidden in ("human_outcome", "ground_truth", "observed_choice"):
                self.assertNotIn(forbidden, case_text)
            sample = cases[0]
            target_family = next(
                target["family_id"]
                for target in loaded["targets"]
                if target["column"] == sample["target_column"]
            )
            from rival.research.firewall import twin_family_split

            split = twin_family_split(inputs.questions)
            families = dict(zip(split.item_ids, split.family_ids, strict=True))
            self.assertTrue(
                all(families[column] != target_family for column in sample["history_columns"])
            )

    def test_cases_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            cases = root / "cases.jsonl"
            cases.write_text(cases.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaises(PilotProtocolError):
                load_and_verify_protocol(root / "protocol.json")

    def test_budget_stop_resume_and_full_rehearsal_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._prepare(root)
            results = root / "results.jsonl"
            provider = DeterministicRehearsalProvider()
            stopped = run_live_pilot(
                root / "protocol.json",
                results,
                provider,
                BudgetGuard(1, 0, 0, 1),
                phase="preflight",
            )
            self.assertEqual(stopped["status"], "STOPPED")
            self.assertEqual(stopped["successful_cases"], 1)

            preflight = run_live_pilot(
                root / "protocol.json",
                results,
                provider,
                BudgetGuard(1, 0, 0, int(protocol["cases"]["preflight"])),
                phase="preflight",
            )
            self.assertEqual(preflight["status"], "COMPLETE")

            complete = run_live_pilot(
                root / "protocol.json",
                results,
                provider,
                BudgetGuard(1, 0, 0, int(protocol["cases"]["total"])),
                phase="pilot",
            )
            self.assertEqual(complete["status"], "COMPLETE")
            report = evaluate_live_pilot(root / "protocol.json", results)
            self.assertEqual(report["status"], "REHEARSAL_ONLY")
            self.assertEqual(report["successful_cases"], protocol["cases"]["total"])
            self.assertIn("generic", report["variants"])
            self.assertIn("twin", report["variants"])


if __name__ == "__main__":
    unittest.main()
