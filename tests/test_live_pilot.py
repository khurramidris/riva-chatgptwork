import io
import json
import tempfile
import unittest
import urllib.error
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
        with patch("urllib.request.urlopen", return_value=_Response(response)) as request_call:
            output = provider.predict(person, scenario)
        request = request_call.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        choice_ids = [choice.choice_id for choice in scenario.choices]
        self.assertEqual(request_payload["response_format"]["type"], "json_schema")
        schema = request_payload["response_format"]["json_schema"]["schema"]
        self.assertTrue(request_payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(schema["required"], choice_ids)
        self.assertEqual(set(schema["properties"]), set(choice_ids))
        self.assertEqual(
            request_payload["reasoning"], {"effort": "none", "exclude": True}
        )
        self.assertEqual(request_payload["provider"], {"require_parameters": True})
        self.assertEqual(output.diagnostics["prompt_tokens"], 123)
        self.assertEqual(output.diagnostics["completion_tokens"], 17)
        self.assertEqual(output.diagnostics["provider_cost_usd"], 0.0042)
        self.assertNotIn("super-secret", str(provider.identity().model_dump()))

    def test_openai_provider_persists_sanitized_http_error_detail(self):
        person = demo_population(20)[0]
        scenario = demo_scenario(20, 0)
        fake_key = "sk-or-v1-example-secret-value"
        response = io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "code": 403,
                        "message": f"Access denied for Bearer {fake_key}",
                    }
                }
            ).encode("utf-8")
        )
        failure = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            403,
            "Forbidden",
            None,
            response,
        )
        provider = OpenAICompatibleProvider(
            model="test-model", api_key=fake_key, max_retries=1
        )
        with patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(ProviderError) as captured:
                provider.predict(person, scenario)
        detail = str(captured.exception)
        self.assertIn("HTTP 403", detail)
        self.assertIn("Access denied", detail)
        self.assertIn("[REDACTED]", detail)
        self.assertNotIn(fake_key, detail)

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

    def test_windows_crlf_checkout_preserves_cases_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._prepare(root)
            cases = root / "cases.jsonl"
            lf_content = cases.read_bytes()
            self.assertNotIn(b"\r\n", lf_content)
            cases.write_bytes(lf_content.replace(b"\n", b"\r\n"))

            loaded, loaded_cases, _ = load_and_verify_protocol(root / "protocol.json")

            self.assertEqual(loaded["protocol_sha256"], protocol["protocol_sha256"])
            self.assertEqual(len(loaded_cases), protocol["cases"]["total"])

    def test_windows_crlf_dataset_checkout_preserves_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._prepare(root)
            source = Path(__file__).parents[1] / "rival" / "datasets" / "twin2k"
            dataset = root / "windows-dataset"
            dataset.mkdir()
            for name in (
                "human_history.csv",
                "human_outcomes.csv",
                "question_catalog.json",
                "wave4_mapping.json",
            ):
                content = (source / name).read_text(encoding="utf-8")
                (dataset / name).write_bytes(
                    content.replace("\n", "\r\n").encode("utf-8")
                )

            loaded, loaded_cases, _ = load_and_verify_protocol(
                root / "protocol.json", dataset_root=dataset
            )

            self.assertEqual(loaded["protocol_sha256"], protocol["protocol_sha256"])
            self.assertEqual(len(loaded_cases), protocol["cases"]["total"])

    def test_substantive_dataset_tampering_is_still_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            source = Path(__file__).parents[1] / "rival" / "datasets" / "twin2k"
            dataset = root / "tampered-dataset"
            dataset.mkdir()
            for name in (
                "human_history.csv",
                "human_outcomes.csv",
                "question_catalog.json",
                "wave4_mapping.json",
            ):
                (dataset / name).write_bytes((source / name).read_bytes())
            mapping = dataset / "wave4_mapping.json"
            mapping.write_text(
                mapping.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )

            with self.assertRaises(PilotProtocolError):
                load_and_verify_protocol(root / "protocol.json", dataset_root=dataset)

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

    def test_v2_strict_openrouter_contract_runs_one_frozen_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare(root)
            results = root / "results.jsonl"
            provider = OpenAICompatibleProvider(
                model="dots-studio/dots-3-note-preview:free",
                api_key="test-secret",
                max_retries=1,
                temperature=0.0,
                history_limit=8,
                max_output_tokens=300,
            )

            def response_for(request, timeout):
                del timeout
                payload = json.loads(request.data.decode("utf-8"))
                schema = payload["response_format"]["json_schema"]["schema"]
                keys = list(schema["required"])
                probabilities = {key: 1.0 / len(keys) for key in keys}
                return _Response(
                    {
                        "id": "frozen-case-1",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": json.dumps(probabilities)},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    }
                )

            with patch("urllib.request.urlopen", side_effect=response_for):
                summary = run_live_pilot(
                    root / "protocol.json",
                    results,
                    provider,
                    BudgetGuard(1, 0, 0, 1),
                    phase="preflight",
                )
            self.assertEqual(summary["successful_cases"], 1)
            self.assertEqual(summary["errors_this_run"], 0)
            self.assertNotIn("test-secret", results.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
