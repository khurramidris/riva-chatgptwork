import copy
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from rival.commands import main
from rival.execution import AuthorizationStopped, ExecutionError, ReconciliationRequired
from rival.integrity import IntegrityError
from rival.managed_execution import ExecutionSession
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.schemas import utc_now
from rival.store import EvidenceStore
from rival.study_commands import example_request
from rival.study_contract import StudyRequest, load_study_request
from rival.study_workflow import (_Workspace, evaluate_study, export_study,
                                  prepare_study, run_study, study_status)
from test_execution_journal import Response, completion


class StudyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workspace = self.root / "study"
        self.output = self.root / "export"
        self.payload = example_request()
        self.payload["brief"]["sample_size"] = 40
        self.payload["audience"]["records"] = self.payload["audience"]["records"][::3]
        self.request = StudyRequest.model_validate(self.payload)

    def managed(self, **limits):
        self.payload["execution"] = {"mode": "managed", "model": "pinned-test-model-r1",
            "base_url": "https://provider.example/v1/chat/completions", "budget_usd": 1.,
            "reservation_usd": .01, "max_attempts": 10,
            "not_after": (utc_now() + timedelta(hours=1)).isoformat(), **limits}
        self.request = StudyRequest.model_validate(self.payload)
        prepare_study(self.workspace, self.request)

    def response(self, cost=.01):
        return completion('{"standard": 0.5, "flexible": 0.3, "neither": 0.2}', cost)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.workspace / "study.sqlite3")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def test_offline_prepare_run_export_is_repeatable_and_keeps_personas_private(self):
        with patch("urllib.request.urlopen") as network:
            prepared = prepare_study(self.workspace, self.request)
            self.assertFalse(prepared["complete"])
            self.assertEqual(prepared["planned_unique_seeds"], 2)
            self.assertEqual(prepare_study(self.workspace, self.request), prepared)
            completed = run_study(self.workspace)
            self.assertEqual(completed["completed_draws"], 40)
            self.assertEqual(run_study(self.workspace), completed)
            export_study(self.workspace, self.output)
            first = {path.name: path.read_bytes() for path in self.output.iterdir()}
            export_study(self.workspace, self.output)
            self.assertEqual(first, {path.name: path.read_bytes() for path in self.output.iterdir()})
        network.assert_not_called()
        report = json.loads(first["report.json"])
        self.assertTrue(report["confidence"]["abstain"])
        self.assertEqual(report["audience"]["simulation_draws"], 40)
        self.assertEqual(report["audience"]["eligible_seed_records"], 2)
        self.assertEqual(report["audience"]["effective_seed_records"], 2)
        self.assertEqual(report["execution"]["accounting"]["attempts"], 0)
        self.assertFalse(report["release_claims"]["customer_launch_ready"])
        self.assertNotIn("generated-0", first["report.json"].decode())
        self.assertNotIn("manifest.key", first)
        self.assertIsNone(report["evaluation"])
        self.assertAlmostEqual(sum(row["simulated_share"] for row in report["choices"]), 1.)
        (self.output / "report.md").write_text("operator annotation", encoding="utf-8")
        with self.assertRaises(ValueError):
            export_study(self.workspace, self.output)
        self.assertEqual((self.output / "report.md").read_text(), "operator annotation")

    def test_invalid_inputs_are_rejected_before_workspace_creation_or_network(self):
        cases = []
        duplicate = copy.deepcopy(self.payload)
        duplicate["audience"]["records"][1]["person_id"] = duplicate["audience"]["records"][0]["person_id"]
        cases.append(duplicate)
        unknown_source = copy.deepcopy(self.payload)
        unknown_source["audience"]["records"][0]["evidence_ids"] = ["undeclared"]
        cases.append(unknown_source)
        future_source = copy.deepcopy(self.payload)
        future_source["audience"]["sources"][0]["collected_at"] = "2030-01-01T00:00:00Z"
        cases.append(future_source)
        outcome_source = copy.deepcopy(self.payload)
        outcome_source["audience"]["sources"][0]["source_type"] = "outcome"
        cases.append(outcome_source)
        infinite = copy.deepcopy(self.payload)
        infinite["audience"]["records"][0]["weight"] = float("inf")
        cases.append(infinite)
        no_timestamp = copy.deepcopy(self.payload)
        del no_timestamp["audience"]["sources"][0]["collected_at"]
        cases.append(no_timestamp)
        for payload in cases:
            with self.subTest(payload=payload), patch("urllib.request.urlopen") as network:
                with self.assertRaises(ValueError):
                    prepare_study(self.workspace, StudyRequest.model_validate(payload))
                network.assert_not_called()
                self.assertFalse(self.workspace.exists())

    def test_unsupported_population_and_outcome_fields_fail_before_any_calls(self):
        self.payload["audience"]["targets"]["controls"]["segment"]["missing"] = .5
        with patch("urllib.request.urlopen") as network, self.assertRaisesRegex(ValueError, "lacks target"):
            prepare_study(self.workspace, StudyRequest.model_validate(self.payload))
        network.assert_not_called()
        self.payload["audience"]["targets"] = None
        self.payload["audience"]["records"][0]["attributes"]["observed_choice"] = "standard"
        with patch("urllib.request.urlopen") as network, self.assertRaises(IntegrityError):
            prepare_study(self.workspace, StudyRequest.model_validate(self.payload))
        network.assert_not_called()

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_nested_values(self):
        path = self.root / "input.json"
        for data in ('{"brief":{},"brief":{}}', '{"unexpected":NaN}'):
            path.write_text(data)
            with self.assertRaises(ValueError):
                load_study_request(path)

    def test_managed_calls_only_unique_seeds_and_repeated_run_preserves_original_cost(self):
        with patch("urllib.request.urlopen") as network:
            self.managed()
        network.assert_not_called()
        with patch("urllib.request.urlopen", return_value=Response(self.response())) as network:
            result = run_study(self.workspace, api_key="test-provider-secret")
        self.assertEqual(network.call_count, 2)
        self.assertAlmostEqual(result["accounting"]["total_cost_usd"], .02)
        with patch("urllib.request.urlopen") as network:
            resumed = run_study(self.workspace)
            export_study(self.workspace, self.output)
        network.assert_not_called()
        self.assertEqual(result, resumed)
        self.assertNotIn("test-provider-secret", (self.output / "report.json").read_text())
        self.assertEqual(resumed["request_states"], {"DONE": 2})

    def test_partial_interrupt_requires_reconciliation_and_retains_full_denominator(self):
        self.managed()
        with patch("urllib.request.urlopen", side_effect=[Response(self.response()), KeyboardInterrupt()]) as network:
            with self.assertRaises(KeyboardInterrupt):
                run_study(self.workspace, api_key="test")
        self.assertEqual(network.call_count, 2)
        status = study_status(self.workspace)
        self.assertEqual(status["state"], "blocked")
        self.assertEqual(status["planned_draws"], 40)
        self.assertEqual(status["completed_draws"], 0)
        self.assertIsNone(status["accounting"]["total_cost_usd"])
        self.assertEqual(status["accounting"]["unresolved_attempts"], 1)
        with self.assertRaises(IntegrityError):
            export_study(self.workspace, self.output)
        with patch("urllib.request.urlopen") as network, self.assertRaises(ReconciliationRequired):
            run_study(self.workspace, api_key="test")
        network.assert_not_called()
        with self.connection() as connection:
            scope = json.loads(connection.execute("SELECT envelope FROM workflow_records WHERE name='prepared'").fetchone()[0])["payload"]["value"]["execution_scope_id"]
        with ExecutionSession(self.workspace / "attempts.sqlite3", scope_id=scope, budget_usd=1,
                reservation_usd=.01, max_total_attempts=10, not_after=utc_now() + timedelta(hours=1)) as execution:
            row = execution.journal.connection.execute("SELECT work_id, ordinal FROM attempts WHERE cost IS NULL").fetchone()
            execution.journal.reconcile(row[0], row[1], cost_usd=.01,
                evidence_reference="test recovered provider receipt", completed_payload=self.response())
        with patch("urllib.request.urlopen") as network:
            completed = run_study(self.workspace, api_key="test")
        network.assert_not_called()
        self.assertEqual(completed["completed_draws"], 40)
        self.assertAlmostEqual(completed["accounting"]["total_cost_usd"], .02)

    def test_budget_stop_does_not_replay(self):
        self.managed(budget_usd=.015)
        with patch("urllib.request.urlopen", return_value=Response(self.response())) as network:
            with self.assertRaises(AuthorizationStopped):
                run_study(self.workspace, api_key="test")
        self.assertEqual(network.call_count, 1)
        with patch("urllib.request.urlopen") as network, self.assertRaises(AuthorizationStopped):
            run_study(self.workspace, api_key="test")
        network.assert_not_called()
        self.assertFalse(study_status(self.workspace)["complete"])
        self.assertEqual(study_status(self.workspace)["planned_unique_seeds"], 2)

    def test_terminal_parse_failure_retains_cost_and_cannot_be_retried(self):
        self.managed(max_retries=1)
        with patch("urllib.request.urlopen", return_value=Response(completion("invalid JSON"))) as network:
            with self.assertRaises(ExecutionError):
                run_study(self.workspace, api_key="test")
        self.assertEqual(network.call_count, 1)
        with patch("urllib.request.urlopen") as network, self.assertRaises(ExecutionError):
            run_study(self.workspace, api_key="test")
        network.assert_not_called()
        status = study_status(self.workspace)
        self.assertEqual(status["request_states"], {"FAILED": 1})
        self.assertEqual(status["accounting"]["total_cost_usd"], .01)
        self.assertEqual(status["planned_draws"], 40)

    def test_missing_or_replaced_attempt_journal_stops_before_network(self):
        self.managed()
        journal = self.workspace / "attempts.sqlite3"
        journal.rename(self.workspace / "retained-journal.sqlite3")
        with patch("urllib.request.urlopen") as network, self.assertRaises(IntegrityError):
            run_study(self.workspace, api_key="test")
        network.assert_not_called()
        journal.touch()
        with patch("urllib.request.urlopen") as network, self.assertRaises((IntegrityError, sqlite3.Error)):
            run_study(self.workspace, api_key="test")
        network.assert_not_called()

    def test_changed_request_or_signed_checkpoint_is_rejected(self):
        prepare_study(self.workspace, self.request)
        changed = copy.deepcopy(self.payload)
        changed["brief"]["question"] = "A changed question"
        with self.assertRaises(IntegrityError):
            prepare_study(self.workspace, StudyRequest.model_validate(changed))
        with self.connection() as connection:
            envelope = json.loads(connection.execute("SELECT envelope FROM workflow_records WHERE name='prepared'").fetchone()[0])
            envelope["payload"]["value"]["request"]["brief"]["question"] = "Tampered"
            connection.execute("UPDATE workflow_records SET envelope=? WHERE name='prepared'", (json.dumps(envelope),))
        with patch("urllib.request.urlopen") as network, self.assertRaises(IntegrityError):
            run_study(self.workspace)
        network.assert_not_called()

    def test_crash_after_saved_run_recovers_same_run_without_api_key_or_new_calls(self):
        self.managed()
        with patch("urllib.request.urlopen", return_value=Response(self.response())), patch.object(_Workspace, "seal", side_effect=RuntimeError("simulated crash")):
            with self.assertRaises(RuntimeError):
                run_study(self.workspace, api_key="test")
        with self.connection() as connection:
            original = connection.execute("SELECT id FROM runs").fetchone()[0]
        with patch("urllib.request.urlopen") as network:
            run_study(self.workspace)
            export_study(self.workspace, self.output)
        network.assert_not_called()
        self.assertEqual(json.loads((self.output / "report.json").read_text())["run_id"], original)
        with self.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)

    def test_crash_after_saved_seal_recovers_existing_seal(self):
        prepare_study(self.workspace, self.request)
        with patch.object(EvidenceStore, "append_phase_event", side_effect=RuntimeError("simulated crash")):
            with self.assertRaises(RuntimeError):
                run_study(self.workspace)
        with self.connection() as connection:
            original = connection.execute("SELECT payload FROM sealed_manifests").fetchone()[0]
        self.assertTrue(run_study(self.workspace)["complete"])
        with self.connection() as connection:
            self.assertEqual(connection.execute("SELECT payload FROM sealed_manifests").fetchone()[0], original)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM phase_events").fetchone()[0], 1)

    def test_protected_evaluation_and_heldout_role_are_bound_and_idempotent(self):
        self.payload["evidence"]["role"] = "calibration"
        self.request = StudyRequest.model_validate(self.payload)
        prepare_study(self.workspace, self.request)
        run_study(self.workspace)
        with self.connection() as connection:
            sealed = json.loads(connection.execute("SELECT payload FROM sealed_manifests").fetchone()[0])
        path = self.root / "external-vault.sqlite3"
        vault = OutcomeVault(path)
        key = "unit-test-outcome-key-long-enough-for-aes"
        try:
            vault.deposit(self.request.brief.study_id, canonical_hash(sealed),
                {"standard": .2, "flexible": .5, "neither": .3}, key, utc_now())
        finally:
            vault.close()
        self.assertEqual(evaluate_study(self.workspace, path, key_material=key)["prediction_phase"], "evaluated")
        evaluate_study(self.workspace, path, key_material=key)
        export_study(self.workspace, self.output)
        report = json.loads((self.output / "report.json").read_text())
        self.assertTrue(report["evidence"]["protected_comparison_verified"])
        self.assertTrue(report["confidence"]["abstain"])
        with self.connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM confidence_assignments WHERE role='training'").fetchone()[0], 0)

    def test_premature_outcomes_cannot_be_revealed(self):
        self.payload["preregistration"] = {"outcome_not_before": (utc_now() + timedelta(days=1)).isoformat()}
        prepare_study(self.workspace, StudyRequest.model_validate(self.payload))
        run_study(self.workspace)
        path = self.root / "future-vault.sqlite3"
        vault = OutcomeVault(path)
        vault.close()
        with patch.object(OutcomeVault, "reveal") as reveal, self.assertRaises(IntegrityError):
            evaluate_study(self.workspace, path, key_material="test")
        reveal.assert_not_called()
        self.assertEqual(study_status(self.workspace)["prediction_phase"], "prediction_locked")

    def test_operator_cli_produces_a_complete_study_report(self):
        input_path = self.root / "input.json"
        with patch("sys.stdout", new_callable=io.StringIO), patch("urllib.request.urlopen") as network:
            self.assertEqual(main(["study", "example", "--output", str(input_path)]), 0)
            self.assertEqual(main(["study", "prepare", "--input", str(input_path), "--workspace", str(self.workspace)]), 0)
            self.assertEqual(main(["study", "run", "--workspace", str(self.workspace)]), 0)
            self.assertEqual(main(["study", "status", "--workspace", str(self.workspace)]), 0)
            self.assertEqual(main(["study", "export", "--workspace", str(self.workspace), "--output", str(self.output)]), 0)
        network.assert_not_called()
        self.assertTrue((self.output / "report.md").is_file())
        with patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["study", "example", "--output", str(input_path)]), 2)

    def test_actual_http_transport_retries_then_exports_without_replay(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        received = []
        response = self.response(.005)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                retry = len(received) == 1
                self.send_response(429 if retry else 200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"code": 429}, "usage": {"cost": 0.}}
                                           if retry else response).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.managed(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions")
            result = run_study(self.workspace, api_key="local-test-token")
            self.assertEqual(result["accounting"]["attempts"], 3)
            self.assertAlmostEqual(result["accounting"]["total_cost_usd"], .01)
            self.assertEqual(received[0], received[1])
            self.assertEqual(received[0]["model"], "pinned-test-model-r1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        # The server is stopped: an unintended replay would fail this recovery.
        self.assertEqual(run_study(self.workspace), result)
        export_study(self.workspace, self.output)


if __name__ == "__main__":
    unittest.main()
