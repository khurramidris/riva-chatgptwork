import io
import json
import tempfile
import threading
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rival.execution import AttemptJournal, AuthorizationStopped, ExecutionError, ReconciliationRequired, reported_cost
from rival.mega_study_v2.provider import ManagedSurveyProvider
from rival.mega_study_v2.runner import exclusive_run_lock, run_benchmark
from rival.mega_study.runner import balanced_variant_order
from rival.mega_study.utils import read_jsonl, write_jsonl, atomic_json, canonical_hash, file_hash
from rival.mega_study.protocol import load_manifest
from rival.mega_study_v2.__main__ import main
from test_mega_study_runtime import _small_stage


class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.payload).encode()


def completion(content="{}", cost=.01):
    return {"id": "provider-response", "choices": [{"message": {"content": content}}],
            "usage": {"cost": cost, "prompt_tokens": 100, "completion_tokens": 10}}


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "attempts.db"
        self.journal = AttemptJournal(self.path, "scope")
        self.provider = ManagedSurveyProvider(api_key="not-a-real-api-key")
        self.options = dict(work_id="w", journal=self.journal, budget_usd=1.,
                            not_after=datetime.now(timezone.utc) + timedelta(hours=1), reservation_usd=.01)

    def tearDown(self):
        self.journal.close()
        self.directory.cleanup()

    def call(self, **overrides):
        return self.provider.complete_managed("system", "user", **{**self.options, **overrides})

    def test_all_retry_costs_and_tokens_are_retained_and_resume_is_cached(self):
        with patch("urllib.request.urlopen", side_effect=[Response(completion("")), Response(completion("")), Response(completion())]) as send, patch("time.sleep"):
            result = self.call()
            cached = self.call()
        self.assertEqual(send.call_count, 3)
        self.assertAlmostEqual(result.usage["provider_cost_usd"], .03)
        self.assertEqual(result.usage["prompt_tokens"], 300)
        self.assertEqual(cached.attempts, 3)
        self.assertAlmostEqual(self.journal.summary()["total_cost_usd"], .03)

    def test_each_retry_requires_budget_and_expiry_authorization(self):
        with patch("urllib.request.urlopen", return_value=Response(completion(""))) as send, patch("time.sleep"):
            with self.assertRaises(AuthorizationStopped):
                self.call(budget_usd=.015)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.journal.summary()["known_cost_usd"], .01)
        with patch("urllib.request.urlopen") as send:
            with self.assertRaises(AuthorizationStopped):
                self.call(not_after=datetime.now(timezone.utc) - timedelta(seconds=1))
        send.assert_not_called()

    def test_expiry_is_rechecked_between_retries(self):
        now = datetime.now(timezone.utc)
        with patch("rival.execution.datetime") as clock, patch("time.sleep"), patch("urllib.request.urlopen", return_value=Response(completion(""))) as send:
            clock.now.side_effect = [now, now, now + timedelta(seconds=10)]
            with self.assertRaises(AuthorizationStopped):
                self.call(not_after=now + timedelta(seconds=5))
        self.assertEqual(send.call_count, 1)

    def test_attempt_limit_includes_internal_retries_and_can_resume_safely(self):
        with patch("time.sleep"), patch("urllib.request.urlopen", side_effect=[Response(completion("")), Response(completion())]) as send:
            with self.assertRaises(AuthorizationStopped):
                self.call(max_new_attempts=1)
            result = self.call(max_new_attempts=1)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.usage["provider_cost_usd"], .02)

    def test_unknown_billing_is_not_zero_and_does_not_replay(self):
        raw = completion()
        raw.pop("usage")
        with patch("urllib.request.urlopen", return_value=Response(raw)) as send:
            with self.assertRaises(ReconciliationRequired): self.call()
            with self.assertRaises(ReconciliationRequired): self.call()
            with self.assertRaises(ReconciliationRequired): self.call(work_id="another")
        self.assertEqual(send.call_count, 1)
        self.assertIsNone(self.journal.summary()["total_cost_usd"])
        self.assertEqual(self.journal.summary()["unresolved_attempts"], 1)
        self.assertEqual(self.journal.summary()["reserved_usd"], .01)

    def test_timeout_and_process_restart_require_reconciliation(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError) as send:
            with self.assertRaises(ReconciliationRequired): self.call()
        self.assertEqual(send.call_count, 1)
        self.journal.close()
        self.journal = AttemptJournal(self.path, "scope")
        self.options["journal"] = self.journal
        with patch("urllib.request.urlopen") as send:
            with self.assertRaises(ReconciliationRequired): self.call()
            self.journal.reconcile("w", 1, cost_usd=.02, evidence_reference="provider-generation-billing-record",
                                   completed_payload=completion(cost=.02))
            result = self.call()
        send.assert_not_called()
        self.assertEqual(result.usage["provider_cost_usd"], .02)

    def test_interrupt_leaves_durable_inflight_reservation(self):
        with patch("urllib.request.urlopen", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.call()
        self.assertEqual(self.journal.attempts_for("w")[0]["state"], "IN_FLIGHT")
        with patch("urllib.request.urlopen") as send:
            with self.assertRaises(ReconciliationRequired): self.call()
        send.assert_not_called()
        with self.assertRaises(ReconciliationRequired):
            self.journal.release_interrupted_claim("w", worker_stopped=True)

    def test_interrupted_claim_between_attempts_can_be_released(self):
        with patch("urllib.request.urlopen", return_value=Response(completion(""))), patch("time.sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.call()
        self.journal.release_interrupted_claim("w", worker_stopped=True)
        with patch("urllib.request.urlopen", return_value=Response(completion())) as send:
            result = self.call()
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result.attempts, 2)

    def test_known_zero_http_rejection_is_distinct_from_unknown_cost(self):
        rejected = urllib.error.HTTPError("https://openrouter.ai", 429, "rate limit", {},
                                         io.BytesIO(json.dumps({"usage": {"cost": 0}}).encode()))
        with patch("urllib.request.urlopen", side_effect=[rejected, Response(completion())]), patch("time.sleep"):
            self.call()
        self.assertEqual(self.journal.attempts_for("w")[0]["cost"], 0.)
        self.assertEqual(self.journal.summary()["known_cost_usd"], .01)

    def test_unknown_429_and_terminal_failures_retain_accounting(self):
        rejected = urllib.error.HTTPError("https://openrouter.ai", 429, "rate limit", {}, io.BytesIO(b'{"error":{}}'))
        with patch("urllib.request.urlopen", side_effect=rejected) as send:
            with self.assertRaises(ReconciliationRequired): self.call()
        self.assertEqual(send.call_count, 1)
        self.assertIsNone(self.journal.attempts_for("w")[0]["cost"])

    def test_cross_connection_claim_prevents_duplicate_remote_work(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        def send():
            entered.set()
            if not release.wait(5): raise AssertionError("test worker timed out")
            return completion()
        def worker():
            try:
                self.journal.execute("w", {"prompt": "same"}, send, lambda p: None,
                    budget_usd=1, not_after=self.options["not_after"], reservation_usd=.01)
            except BaseException as exc: errors.append(exc)
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(entered.wait(5))
        other = AttemptJournal(self.path, "scope")
        try:
            with self.assertRaises(ReconciliationRequired):
                other.execute("w", {"prompt": "same"}, lambda: self.fail("duplicate send"), lambda p: None,
                    budget_usd=1, not_after=self.options["not_after"], reservation_usd=.01)
        finally:
            release.set()
            thread.join(5)
            other.close()
        self.assertFalse(errors)
        self.assertEqual(self.journal.summary()["attempts"], 1)

    def test_scope_request_conflicts_and_bad_costs(self):
        for value in (None, float("nan"), float("inf"), -1, True, "0"):
            self.assertIsNone(reported_cost({"usage": {"cost": value}}))
        with self.assertRaises(ExecutionError): AttemptJournal(self.path, "different-scope")
        with patch("urllib.request.urlopen", return_value=Response(completion())):
            self.call()
            with self.assertRaises(ExecutionError):
                self.provider.complete_managed("changed", "user", **self.options)

    def test_exhausted_errors_keep_every_billed_attempt(self):
        with patch("urllib.request.urlopen", return_value=Response(completion(""))) as send, patch("time.sleep"):
            with self.assertRaises(ExecutionError): self.call()
            with self.assertRaises(ExecutionError): self.call()
        self.assertEqual(send.call_count, 3)
        self.assertAlmostEqual(self.journal.summary()["known_cost_usd"], .03)


class RevisedRunnerTests(unittest.TestCase):
    def test_v2_freeze_and_reanalysis_preserve_source_results(self):
        stage = _small_stage()
        stage[1][0]["survey_text"] = "Q1:\nPrivacy\nOptions:\n" + "\n".join(f"  {i} - Rating {i}" for i in range(1, 8))
        response = json.dumps({"Q1": {"Answers": {"SelectedByPosition": 4}}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, marker = root / "v2.jsonl", root / "freeze.json"
            with patch("rival.mega_study_v2.runner.load_prediction_stage", return_value=stage), patch("urllib.request.urlopen", return_value=Response(completion(response))):
                run_benchmark(root, results, budget_usd=1., not_after=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), api_key="secret")
            original_hash = file_hash(results)
            with patch("rival.mega_study_v2.freeze.load_prediction_stage", return_value=stage), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(["freeze", "--stage-root", str(root), "--results", str(results), "--freeze-marker", str(marker)]), 0)
            freeze = json.loads(marker.read_text())
            outcome_path = root / "sealed" / "outcomes.jsonl"
            write_jsonl(outcome_path, [{"case_id": stage[1][0]["case_id"], "cells": [{"outcome_id": "PPV", "value": 3., "minimum": 1., "maximum": 7.}]}])
            outcome_manifest = {"protocol_sha256": load_manifest()["manifest_sha256"], "stage_sha256": stage[0]["stage_sha256"],
                "prediction_freeze_sha256": freeze["freeze_sha256"], "materialized_after_prediction_freeze": True,
                "outcomes_path": outcome_path.name, "outcomes_sha256": file_hash(outcome_path)}
            outcome_manifest["outcome_manifest_sha256"] = canonical_hash(outcome_manifest)
            atomic_json(root / "sealed" / "outcome_manifest.json", outcome_manifest)
            report, markdown = root / "report.json", root / "report.md"
            with patch("rival.mega_study_v2.evaluation.load_prediction_stage", return_value=stage), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(["reanalyze", "--stage-root", str(root), "--results", str(results), "--freeze-marker", str(marker),
                    "--json-report", str(report), "--markdown-report", str(markdown)]), 0)
            self.assertEqual(file_hash(results), original_hash)
            content = json.loads(report.read_text())
            self.assertEqual(content["status"], "COMPLETE_REANALYSIS_REPORT")
            self.assertFalse(content["confirmation_claim_allowed"])
            self.assertAlmostEqual(content["overall"]["generic"]["normalized_accuracy"], 5 / 6)
            self.assertIn("interval unavailable", markdown.read_text())

    def test_run_resume_and_lost_export_do_not_repeat_calls(self):
        stage = _small_stage()
        stage[1][0]["survey_text"] = "Q1:\nPrivacy\nOptions:\n" + "\n".join(f"  {i} - Rating {i}" for i in range(1, 8))
        response = json.dumps({"Q1": {"Answers": {"SelectedByPosition": 4}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.jsonl"
            options = dict(budget_usd=1., not_after=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), api_key="secret")
            with patch("rival.mega_study_v2.runner.load_prediction_stage", return_value=stage), patch("urllib.request.urlopen", return_value=Response(completion(response))) as send:
                first = run_benchmark(directory, path, **options)
                second = run_benchmark(directory, path, **options)
                # Simulate a crash after the durable response, before JSONL export.
                path.write_text("", encoding="utf-8")
                third = run_benchmark(directory, path, **options)
            self.assertEqual(send.call_count, 4)
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(second["new_attempts"], 0)
            self.assertEqual(third["new_attempts"], 0)
            self.assertEqual(len(read_jsonl(path)), 4)
            self.assertAlmostEqual(third["known_cost_usd"], .04)
            self.assertNotIn("secret", path.read_text())

    def test_same_output_file_has_one_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writer.lock"
            with exclusive_run_lock(path):
                with self.assertRaises(ExecutionError):
                    with exclusive_run_lock(path): pass


if __name__ == "__main__": unittest.main()
