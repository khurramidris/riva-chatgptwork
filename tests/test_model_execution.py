from datetime import timedelta
from email.message import Message
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np

from rival.commands import main
from rival.elicitation import SemanticSimilarityRater, SSRScale
from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, strict_json
from rival.evidence_commands import example_files
from rival.execution import ReconciliationRequired, TerminalResponseError
from rival.integrity import IntegrityError
from rival.mathx import canonical_hash
from rival.model_providers import model_provider
from rival.schemas import utc_now
from rival.study_contract import StudyRequestV3, parse_study_request
from rival.study_evidence import bind_evidence
from rival.study_execution_audit import audit_study_execution, compare_studies
from rival.study_workflow import prepare_study, run_study, export_study, study_status
from rival.vendor.semantic_similarity_rating.compute import response_embeddings_to_pmf, scale_pmf
from test_execution_journal import Response


class MatrixEmbedder:
    identity = "test-fixed-matrix"

    def __init__(self, vectors):
        self.vectors, self.calls = vectors, []

    def encode(self, texts):
        self.calls.append(tuple(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


class ModelExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "study"
        files = example_files()
        self.catalog = EvidenceCatalog(self.root / "catalog")
        source = self.root / "people.csv"
        source.write_bytes(files["people.csv"])
        bundle = self.catalog.import_file(source, EvidenceImportSpec.model_validate(strict_json(files["import.json"])))
        brief = strict_json(files["brief.json"])
        brief["schema_version"] = "rival.study-request.v3"
        brief["brief"]["sample_size"] = 40
        brief["execution"] = {"mode": "managed", "model": "test-model",
            "base_url": "http://127.0.0.1:12345/chat/completions", "budget_usd": 1.0,
            "reservation_usd": 0.05, "max_attempts": 20, "max_retries": 2,
            "not_after": (utc_now() + timedelta(hours=1)).isoformat(), "generation_seed": 42,
            "model_pin": {"kind": "checkpoint", "revision": "a" * 40,
                          "reference": "generated test weights", "expected_response_model": "served-test-model",
                          "expected_system_fingerprint": "test-runtime"}}
        self.request = bind_evidence(brief, self.catalog.root, [bundle.bundle_sha256], strict_json(files["support.json"]))
        self.ids = itertools.count()

    def completion(self, text=None, **changes):
        return {"id": "test-request-" + str(next(self.ids)), "model": "served-test-model",
            "system_fingerprint": "test-runtime",
            "choices": [{"finish_reason": "stop", "message": {"content": text or
                '{"standard":0.6,"flexible":0.3,"neither":0.1}'}}],
            "usage": {"cost": 0.01, "prompt_tokens": 100, "completion_tokens": 10}, **changes}

    def changed(self, **settings):
        payload = self.request.model_dump(mode="json")
        payload["execution"].update(settings)
        return parse_study_request(payload)

    def prepare(self, request=None, root=None):
        return prepare_study(root or self.workspace, request or self.request, catalog_root=self.catalog.root)

    def run_model(self, result=None, root=None):
        with patch("urllib.request.urlopen", side_effect=lambda *a, **k: Response(result or self.completion())) as send:
            status = run_study(root or self.workspace, api_key="private-test-secret")
        return status, send

    def test_v3_schema_binds_model_inputs_without_changing_v2_defaults(self):
        self.assertIsInstance(self.request, StudyRequestV3)
        encoded = self.request.model_dump(mode="json")
        self.assertEqual(canonical_hash(self.request), canonical_hash(parse_study_request(encoded)))
        encoded["schema_version"], encoded["execution"] = "rival.study-request.v2", {"mode": "offline"}
        legacy = parse_study_request(encoded).model_dump(mode="json")
        self.assertNotIn("model_pin", legacy["execution"])
        self.assertNotIn("generation_seed", legacy["execution"])

    def test_mutable_model_and_embedding_revisions_are_rejected(self):
        for revision in ["main", "latest", "abc", " a" * 20]:
            payload = self.request.model_dump(mode="json")
            payload["execution"]["model_pin"]["revision"] = revision
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                parse_study_request(payload)
        with self.assertRaises(ValueError):
            self.changed(elicitation={"method": "ssr", "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2", "revision": "main"}})

    def test_openrouter_route_seed_and_prompt_settings_are_bound(self):
        pin = self.request.execution.model_pin.model_dump(mode="json")
        pin.update(provider_route="example-route", expected_response_provider="Example Provider")
        request = self.changed(base_url="https://openrouter.ai/api/v1/chat/completions", model_pin=pin)
        provider = model_provider(request.execution, api_key="secret")
        payload = provider._request_payload(request.audience.records[0], request.scenario())
        self.assertEqual(payload["provider"], {"only": ["example-route"], "allow_fallbacks": False, "require_parameters": True})
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("calibrated behavioral", json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "route"):
            model_provider(self.changed(base_url="https://openrouter.ai/api/v1/chat/completions").execution, api_key="secret")

    def test_direct_study_has_pinned_identity_accounting_and_cached_resume(self):
        prepared = self.prepare()
        status, send = self.run_model()
        self.assertTrue(status["complete"])
        self.assertEqual(send.call_count, prepared["planned_unique_seeds"])
        with patch("urllib.request.urlopen", side_effect=AssertionError("must reuse journal")):
            self.assertEqual(status, run_study(self.workspace))
        audit = audit_study_execution(self.workspace)
        self.assertEqual(audit["accepted_seed_requests"], 6)
        self.assertAlmostEqual(audit["accounting"]["total_cost_usd"], 0.06)
        self.assertEqual(audit["tokens"]["total_tokens"]["reported"], 660)
        self.assertEqual(audit["transport_latency_ms"]["measured_attempts"], 6)
        export_study(self.workspace, self.root / "report")
        text = (self.root / "report/report.json").read_text()
        self.assertNotIn("private-test-secret", text)
        report = json.loads(text)
        self.assertEqual(report["schema_version"], "rival.study-report.v3")
        self.assertTrue(report["confidence"]["abstain"])
        self.assertEqual(report["execution"]["measurements"], audit)

    def test_wrong_model_is_terminal_and_keeps_cost_and_planned_people(self):
        self.prepare()
        with patch("urllib.request.urlopen", return_value=Response(self.completion(model="other-model"))) as send:
            with self.assertRaises(TerminalResponseError):
                run_study(self.workspace, api_key="secret")
        self.assertEqual(send.call_count, 1)
        audit = audit_study_execution(self.workspace)
        self.assertEqual(audit["unaccepted_seed_requests"], 6)
        self.assertEqual(audit["accounting"]["total_cost_usd"], 0.01)
        self.assertEqual(audit["attempt_outcomes"], {"TerminalResponseError": 1})
        self.assertFalse(audit["complete"])
        self.assertEqual(study_status(self.workspace)["completed_draws"], 0)

    def test_missing_fingerprint_truncated_and_refused_outputs_fail(self):
        provider = model_provider(self.request.execution, api_key="secret")
        cases = [self.completion(model=None), self.completion(system_fingerprint="changed"), self.completion(id=None),
                 self.completion(choices=[{"finish_reason": "length", "message": {"content": "partial"}}]),
                 self.completion(choices=[{"finish_reason": "stop", "message": {"content": "no", "refusal": "refused"}}])]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(TerminalResponseError):
                provider.validate_response(payload)

    def test_billed_invalid_json_retries_and_cost_survives(self):
        self.prepare()
        replies = [self.completion('{"standard":0.6,"standard":0.6,"flexible":0.3,"neither":0.1}')]
        with patch("time.sleep"), patch("urllib.request.urlopen", side_effect=lambda *a, **k: Response(replies.pop(0) if replies else self.completion())) as send:
            run_study(self.workspace, api_key="secret")
        audit = audit_study_execution(self.workspace)
        self.assertEqual(send.call_count, 7)
        self.assertAlmostEqual(audit["accounting"]["total_cost_usd"], 0.07)
        self.assertEqual(audit["attempt_outcomes"]["ValueError"], 1)

    def test_missing_billing_is_quarantined_and_never_replayed(self):
        self.prepare()
        with patch("urllib.request.urlopen", return_value=Response(self.completion(usage={}))) as send:
            for _ in range(2):
                with self.assertRaises(ReconciliationRequired):
                    run_study(self.workspace, api_key="secret")
        self.assertEqual(send.call_count, 1)
        audit = audit_study_execution(self.workspace)
        self.assertEqual(audit["accounting"]["unresolved_attempts"], 1)
        self.assertIsNone(audit["accounting"]["total_cost_usd"])
        self.assertEqual(audit["tokens"]["total_tokens"]["reporting_attempts"], 0)

    def test_changed_runtime_stops_before_network(self):
        self.prepare()
        with patch("rival.model_providers.runtime_identity", return_value={"changed": True}), patch("urllib.request.urlopen") as send:
            with self.assertRaises(IntegrityError):
                run_study(self.workspace, api_key="secret")
        send.assert_not_called()

    def test_ssr_workflow_uses_same_journal_slot_and_explicit_limits(self):
        request = self.changed(history_limit=0, max_output_tokens=77,
            elicitation={"method": "ssr", "embedding": {"kind": "hashing", "dimensions": 64}})
        provider = model_provider(request.execution, api_key="secret")
        payload = provider.generator._payload(request.audience.records[0], request.scenario())
        self.assertEqual(payload["max_tokens"], 77)
        self.assertEqual(json.loads(payload["messages"][1]["content"])["person"]["relevant_history"], [])
        self.assertEqual(len(json.loads(payload["messages"][1]["content"])["scenario"]["choices"]), 3)
        self.assertNotIn("response_format", payload)
        self.prepare(request)
        status, send = self.run_model(self.completion("I prefer standard delivery."))
        self.assertTrue(status["complete"])
        self.assertEqual(send.call_count, 6)

    def test_embedding_load_failure_spends_nothing(self):
        request = self.changed(elicitation={"method": "ssr", "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2", "revision": "a" * 40}})
        self.prepare(request)
        with patch("rival.elicitation.SentenceTransformerEmbedder._load", side_effect=RuntimeError("unavailable")), patch("urllib.request.urlopen") as send:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                run_study(self.workspace, api_key="secret")
        send.assert_not_called()
        self.assertEqual(study_status(self.workspace)["accounting"]["attempts"], 0)

    def test_fresh_comparison_is_separate_from_replay_and_checks_inputs(self):
        self.prepare()
        self.run_model()
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_studies(self.workspace, self.workspace)
        payload = self.request.model_dump(mode="json")
        payload["brief"]["study_id"] += "-replicate"
        second = self.root / "replicate"
        self.prepare(parse_study_request(payload), second)
        self.run_model(root=second)
        compared = compare_studies(self.workspace, second)
        self.assertEqual(compared["matched_seed_requests"], 6)
        self.assertEqual(compared["aggregate_tvd"], 0)
        self.assertEqual(compared["physical_attempts"], [6, 6])
        payload["brief"]["question"] = "A different question"
        third = self.root / "different"
        self.prepare(parse_study_request(payload), third)
        self.run_model(root=third)
        with self.assertRaisesRegex(ValueError, "inputs"):
            compare_studies(self.workspace, third)

    def test_cli_exposes_schema_and_no_call_execution_audit(self):
        self.prepare()
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["study", "execution-audit", "--workspace", str(self.workspace)]), 0)
        self.assertEqual(json.loads(output.getvalue())["unaccepted_seed_requests"], 6)
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["study", "schema", "--output", str(self.root / "schema.json")]), 0)
        self.assertIn("StudyModelExecution", (self.root / "schema.json").read_text())

    def test_ssr_rejects_structured_answers_instead_of_rating_them_as_text(self):
        request = self.changed(elicitation={"method": "ssr", "embedding": {"kind": "hashing"}})
        provider = model_provider(request.execution, api_key="secret")
        for text in ['{"choice":"standard"}', '4', '```json\n{}\n```']:
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                provider.generator.validate_response(self.completion(text))

    def test_recovery_after_saved_calls_never_reissues_inference(self):
        self.prepare()
        with patch("rival.store.EvidenceStore.save_run", side_effect=RuntimeError("local save interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_model()
        with patch("urllib.request.urlopen", side_effect=AssertionError("must recover")):
            self.assertTrue(run_study(self.workspace, api_key="secret")["complete"])
        self.assertEqual(audit_study_execution(self.workspace)["accounting"]["attempts"], 6)

    def test_billed_http_429_is_counted_before_retry(self):
        self.prepare()
        headers = Message()
        headers['Retry-After'] = '0'
        error = urllib.error.HTTPError('http://127.0.0.1/test', 429, 'rate limit', headers,
            io.BytesIO(json.dumps({'usage': {'cost': 0.01}, 'error': {'message': 'secret'}}).encode()))
        attempts = [error]
        def reply(*args, **kwargs):
            if attempts:
                raise attempts.pop()
            return Response(self.completion())
        with patch('time.sleep'), patch('urllib.request.urlopen', side_effect=reply):
            self.assertTrue(run_study(self.workspace, api_key='secret')['complete'])
        audit = audit_study_execution(self.workspace)
        self.assertAlmostEqual(audit['accounting']['total_cost_usd'], .07)
        self.assertEqual(audit['attempt_outcomes']['http_429'], 1)
        self.assertNotIn('secret', json.dumps(audit))

    def test_declared_provider_mismatch_is_terminal(self):
        pin = self.request.execution.model_pin.model_dump(mode='json')
        pin.update(provider_route='chosen', expected_response_provider='Chosen Provider')
        provider = model_provider(self.changed(model_pin=pin,
            base_url='https://openrouter.ai/api/v1/chat/completions').execution, api_key='secret')
        with self.assertRaises(TerminalResponseError):
            provider.validate_response(self.completion(provider='Other Provider'))


class SSRTieTests(unittest.TestCase):
    def rater(self, anchors, vectors, **options):
        return SemanticSimilarityRater(SSRScale(tuple(anchors), tuple(anchors)), MatrixEmbedder(vectors), **options)

    def test_default_epsilon_flat_similarities_are_uniform_and_exposed(self):
        rater = self.rater(["a", "b", "c"], {"a": [1, 0], "b": [1, 0], "c": [1, 0], "response": [1, 0]})
        probabilities, diagnostics = rater.rate_with_diagnostics("response")
        np.testing.assert_allclose(list(probabilities.values()), [1/3] * 3)
        self.assertEqual(diagnostics["ssr_degenerate"], 1)

    def test_partial_minimum_ties_share_epsilon_under_all_permutations(self):
        vectors = {"a": [1, 0], "b": [0, 1], "c": [0, -1], "response": [1, 0]}
        for order in itertools.permutations(["a", "b", "c"]):
            result = self.rater(order, vectors, epsilon=0.2).rate("response")
            self.assertAlmostEqual(result["b"], 1/7)
            self.assertAlmostEqual(result["c"], 1/7)
            self.assertAlmostEqual(result["a"], 5/7)

    def test_zero_temperature_splits_maxima_without_position_bias(self):
        vectors = {"a": [1, 0], "b": [1, 0], "c": [-1, 0], "response": [1, 0]}
        result = self.rater(["a", "b", "c"], vectors, temperature=0).rate("response")
        self.assertEqual(result, {"a": .5, "b": .5, "c": 0.0})

    def test_unique_extrema_match_upstream_and_tiny_temperature_is_stable(self):
        vectors = {"a": [1, 0], "b": [0, 1], "c": [-1, 0], "response": [1, .2]}
        rater = self.rater(["a", "b", "c"], vectors, epsilon=.1, temperature=.7)
        expected = scale_pmf(response_embeddings_to_pmf(np.array([vectors['response']]),
            np.array([vectors[key] for key in ['a', 'b', 'c']]).T, epsilon=.1)[0], .7)
        np.testing.assert_allclose(rater.rate_array("response"), expected, atol=1e-14)
        sharp = self.rater(["a", "b", "c"], vectors, temperature=1e-300).rate("response")
        self.assertEqual(sharp, {"a": 1.0, "b": 0.0, "c": 0.0})

    def test_zero_vectors_flagged_nonfinite_and_blank_text_fail(self):
        vectors = {"a": [1, 0], "b": [0, 1], "response": [0, 0]}
        rater = self.rater(["a", "b"], vectors)
        self.assertEqual(rater.rate_with_diagnostics("response")[1]["ssr_degenerate"], 1)
        for bad in [[float('nan'), 0], [float('inf'), 0]]:
            vectors['response'] = bad
            with self.assertRaises(ValueError):
                self.rater(["a", "b"], vectors).rate("response")
        with self.assertRaises(ValueError):
            rater.rate(" ")

    def test_anchor_and_identical_response_embeddings_are_reused(self):
        rater = self.rater(["a", "b"], {"a": [1, 0], "b": [0, 1], "response": [1, 0]})
        for _ in range(20):
            rater.rate("response")
        self.assertEqual(len(rater.embedder.calls), 2)
