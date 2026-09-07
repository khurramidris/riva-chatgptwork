import io
import json
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from rival.commands import main
from rival.confidence import ConfidenceModel
from rival.confidence_evidence import ConfidenceEvidenceRegistry
from rival.engine import RivalEngine
from rival.execution import AuthorizationStopped, ExecutionError, ReconciliationRequired
from rival.integrity import IntegrityError, ManifestSigner, ProspectiveStudyManager
from rival.managed_execution import ExecutionSession
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.providers import OpenAICompatibleProvider, BehavioralModelProvider, ProviderError
from rival.reporting import evidence_card, markdown_report
from rival.release import build_release_manifest, verify_release_manifest, sha256_file
from rival.version import __version__
from rival.elicitation import OpenAICompatibleTextGenerator, SSRElicitationProvider
from rival.evaluation import evaluate_distribution
import zipfile
from rival.schemas import PopulationRecord, PreregistrationSpec, utc_now
from test_foundation_regressions import scenario
from test_execution_journal import Response, completion


class ConfidenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.engine = RivalEngine(store_path=Path(self.directory.name) / "evidence.db")
        self.manager = ProspectiveStudyManager(self.engine.store, ManifestSigner("test-manifest-key-material-long-enough"))
        self.registry = ConfidenceEvidenceRegistry(self.manager)
        self.vault = OutcomeVault(Path(self.directory.name) / "vault.db")

    def tearDown(self):
        self.vault.close()
        self.engine.store.close()
        self.directory.cleanup()

    def protected_study(self, index, role):
        sim = self.engine.simulate([PopulationRecord(person_id="p")], scenario(novelty=index / 10))
        study = sim.scenario.scenario_id
        self.registry.assign(study, group_id=f"independent-study-{index}", role=role, source_reference=f"public-source-{index}")
        sealed = self.manager.lock_prediction(sim, PreregistrationSpec())
        self.vault.deposit(study, canonical_hash(sealed), {"a": .2, "b": .8}, "test-vault-key-material-long-enough", utc_now())
        observed, _ = self.manager.reveal_outcomes(study, self.vault, "test-vault-key-material-long-enough")
        evaluation = self.engine.evaluate(sim, observed, canonical_hash(sealed.manifest.preregistration))
        self.manager.record_evaluation(study, evaluation)
        self.registry.admit(study)
        return study

    def test_distinct_protected_training_excludes_heldout_roles_and_repeat_admission(self):
        studies = [self.protected_study(i, "training") for i in range(5)]
        self.protected_study(5, "calibration")
        self.protected_study(6, "evaluation")
        self.registry.admit(studies[0])
        model = self.registry.fit_research_model()
        self.assertEqual(model.training_examples, 5)
        result = model.assess({"scenario_novelty": .1})
        self.assertEqual((result.label, result.abstain, result.lower_tvd, result.upper_tvd), ("unqualified", True, 0., 1.))

    def test_duplicate_group_late_assignment_and_relabeling_are_rejected(self):
        study = self.protected_study(0, "evaluation")
        with self.assertRaises(sqlite3.IntegrityError):
            self.registry.assign("another-study", group_id="independent-study-0", role="training", source_reference="source")
        with self.assertRaises(IntegrityError):
            self.registry.assign(study, group_id="late", role="training", source_reference="source")
        self.engine.store.connection.execute("UPDATE confidence_assignments SET role='training' WHERE study_id=?", (study,))
        with self.assertRaises(IntegrityError):
            self.registry.training_rows()

    def test_unprotected_self_evaluation_cannot_be_admitted(self):
        sim = self.engine.simulate([PopulationRecord(person_id="p")], scenario())
        self.registry.assign(sim.scenario.scenario_id, group_id="self-eval", role="training", source_reference="source")
        self.engine.evaluate(sim, sim.distribution)
        with self.assertRaises(IntegrityError):
            self.registry.admit(sim.scenario.scenario_id)

    def test_unqualified_features_and_legacy_report_cannot_gain_confidence(self):
        model = ConfidenceModel()
        a = model.assess({"human_anchor_rate": 0})
        b = model.assess({"human_anchor_rate": 100})
        self.assertEqual(a.expected_tvd, b.expected_tvd)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.assess({"scenario_novelty": value})
            with self.assertRaises(ValueError):
                model.fit([{}] * 5, [value] * 5)
        sim = self.engine.simulate([PopulationRecord(person_id="p")], scenario())
        sim.confidence = sim.confidence.model_copy(update={"label": "high", "abstain": False})
        evaluation = self.engine.evaluate(sim, sim.distribution)
        card = evidence_card(sim, evaluation=evaluation)
        self.assertTrue(card["confidence"]["abstain"])
        self.assertFalse(card["release_claims"]["confidence_qualified"])
        report = markdown_report(sim, evaluation=evaluation)
        self.assertNotIn("Protected-outcome TVD", report)
        self.assertIn("Decision support withheld", report)


class ManagedProviderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "attempts.db"
        self.options = dict(scope_id="study-batch", budget_usd=1., reservation_usd=.01,
                            not_after=utc_now() + timedelta(hours=1), max_total_attempts=10)
        self.execution = ExecutionSession(self.path, **self.options)
        self.person = PopulationRecord(person_id="p")
        self.scenario = scenario()

    def tearDown(self):
        self.execution.close()
        self.directory.cleanup()

    def provider(self, behavioral=False):
        if behavioral:
            return BehavioralModelProvider("pinned-test-model", "https://provider.example/v1/chat/completions",
                api_key="secret-test-key", model_revision="revision-1", training_corpus="test", model_license="test",
                execution=self.execution)
        return OpenAICompatibleProvider("pinned-test-model", api_key="secret-test-key", execution=self.execution)

    def test_current_network_adapters_require_session_before_network(self):
        provider = OpenAICompatibleProvider("test", api_key="secret")
        with patch("urllib.request.urlopen") as send, self.assertRaises(ProviderError):
            provider.predict(self.person, self.scenario)
        send.assert_not_called()
        generator = OpenAICompatibleTextGenerator("test", api_key="secret")
        with patch("urllib.request.urlopen") as send, self.assertRaises(ProviderError):
            generator.generate(self.person, self.scenario)
        send.assert_not_called()
        with self.assertRaises(ProviderError):
            BehavioralModelProvider("test", "http://remote.example/api", model_revision="r", training_corpus="c", model_license="l", api_key="secret")

    def test_ssr_text_generation_uses_same_retry_billing_and_cache(self):
        provider = SSRElicitationProvider(OpenAICompatibleTextGenerator("test", api_key="secret", execution=self.execution))
        with patch("urllib.request.urlopen", side_effect=[Response(completion("")), Response(completion("I choose A"))]) as send, patch("time.sleep"):
            output = provider.predict(self.person, self.scenario)
            cached = provider.predict(self.person, self.scenario)
        self.assertEqual(send.call_count, 2)
        self.assertAlmostEqual(output.diagnostics["provider_cost_usd"], .02)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(cached.diagnostics["provider_cost_usd"], 0.)
        self.assertAlmostEqual(sum(output.probabilities.values()), 1.)

    def test_probability_retries_accumulate_billing_and_resume_without_replay(self):
        provider = self.provider()
        responses = [completion('{"a":0,"a":0.7,"b":0.3}'), completion('{"a":7,"b":-6}'), completion('{"a":0.7,"b":0.3}')]
        with patch("urllib.request.urlopen", side_effect=list(map(Response, responses))) as send, patch("time.sleep"):
            output = provider.predict(self.person, self.scenario)
        self.assertEqual(send.call_count, 3)
        self.assertAlmostEqual(output.diagnostics["provider_cost_usd"], .03)
        self.assertEqual(output.diagnostics["prompt_tokens"], 300)
        self.execution.close()
        self.execution = ExecutionSession(self.path, **self.options)
        provider = self.provider()
        with patch("urllib.request.urlopen") as send:
            cached = provider.predict(self.person, self.scenario)
        send.assert_not_called()
        self.assertTrue(cached.cache_hit)
        self.assertEqual(cached.diagnostics["provider_cost_usd"], 0.)
        self.assertAlmostEqual(cached.diagnostics["request_cost_usd"], .03)

    def test_behavioral_provider_budget_blocks_billed_retry(self):
        self.execution.budget_usd = .015
        with patch("urllib.request.urlopen", return_value=Response(completion(""))) as send, patch("time.sleep"):
            with self.assertRaises(AuthorizationStopped):
                self.provider(True).predict(self.person, self.scenario)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(self.execution.journal.summary()["known_cost_usd"], .01)

    def test_unknown_cost_quarantines_all_providers_in_shared_scope(self):
        with patch("urllib.request.urlopen", return_value=Response(completion('{"a":0.7,"b":0.3}', None))) as send:
            with self.assertRaises(ReconciliationRequired):
                self.provider().predict(self.person, self.scenario)
            with self.assertRaises(ReconciliationRequired):
                self.provider(True).predict(self.person, self.scenario)
        self.assertEqual(send.call_count, 1)
        self.assertIsNone(self.execution.journal.summary()["total_cost_usd"])

    def test_global_attempt_limit_and_request_changes_do_not_trigger_hidden_calls(self):
        self.execution.max_total_attempts = 1
        provider = self.provider()
        with patch("urllib.request.urlopen", return_value=Response(completion('{"a":0.7,"b":0.3}'))) as send:
            provider.predict(self.person, self.scenario)
            with self.assertRaises(AuthorizationStopped):
                provider.predict(PopulationRecord(person_id="p2"), self.scenario)
            provider.model = "changed-model"
            with self.assertRaisesRegex(ExecutionError, "different request"):
                provider.predict(self.person, self.scenario)
        self.assertEqual(send.call_count, 1)

    def test_resampled_seed_shares_model_response_and_retains_all_draws(self):
        engine = RivalEngine()
        self.addCleanup(engine.store.close)
        engine.register_provider("managed", self.provider())
        with patch("urllib.request.urlopen", return_value=Response(completion('{"a":0.7,"b":0.3}'))) as send:
            simulation = engine.simulate([self.person], scenario(model_family="managed").model_copy(update={"sample_size": 40}))
        self.assertEqual(send.call_count, 1)
        self.assertEqual(len(simulation.predictions), 40)
        self.assertEqual(sum(p.provider_call.cache_hit for p in simulation.predictions), 39)


class CurrentReleaseTests(unittest.TestCase):
    def test_uniform_outcome_has_json_safe_undefined_variance(self):
        evaluation = evaluate_distribution("test", {"a": .7, "b": .3}, {"a": .5, "b": .5})
        self.assertIsNone(evaluation.metrics["variance_ratio"])
        json.dumps(evaluation.model_dump(mode="json"), allow_nan=False)

    def test_wheel_and_inventory_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("rival_sim.dist-info/METADATA", f"Name: rival-sim\nVersion: {__version__}\n")
                archive.writestr("rival/identity.txt", "original")
            manifest_path = build_release_manifest(wheel, root / "release")
            self.assertEqual(verify_release_manifest(manifest_path)["status"], "PASS")
            inventory = root / "release/wheel_inventory.json"
            payload = json.loads(inventory.read_text())
            payload["wheel_files"]["rival/identity.txt"] = "forged-hash"
            inventory.write_text(json.dumps(payload))
            self.assertEqual(verify_release_manifest(manifest_path)["status"], "FAIL")
            manifest = json.loads(manifest_path.read_text())
            manifest["source_inventory_sha256"] = sha256_file(inventory)
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(verify_release_manifest(manifest_path)["status"], "FAIL")

    def test_managed_cli_completes_and_recovers_export_without_new_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "study.json"
            inputs.write_text(json.dumps({"scenario": scenario().model_dump(mode="json"),
                                          "records": [{"person_id": "p"}]}))
            args = ["simulate-managed", "--input", str(inputs), "--output", str(root / "result.json"),
                    "--database", str(root / "evidence.db"), "--journal", str(root / "attempts.db"),
                    "--scope-id", "cli-recovery", "--model", "pinned-test-model", "--budget-usd", "1",
                    "--reservation-usd", ".01", "--max-attempts", "1", "--not-after", (utc_now() + timedelta(hours=1)).isoformat()]
            with patch.dict("os.environ", {"RIVAL_API_KEY": "test-key"}), patch("sys.stdout", new_callable=io.StringIO), patch("urllib.request.urlopen", return_value=Response(completion('{"a":0.7,"b":0.3}'))) as send:
                self.assertEqual(main(args), 0)
                (root / "result.json").unlink()
                self.assertEqual(main(args), 0)
            self.assertEqual(send.call_count, 1)
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["accounting"]["attempts"], 1)
            self.assertTrue(result["simulation"]["confidence"]["abstain"])
            (root / "result.json").unlink()
            (root / "attempts.db").unlink()
            with patch("urllib.request.urlopen") as send, patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(main(args), 2)
            send.assert_not_called()

    def test_qualification_failure_returns_nonzero_and_historical_claims_are_separate(self):
        with patch("rival.research.qualification.run_all", return_value={"status": "FAIL"}), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["qualify-all", "--output-dir", "unused"]), 1)
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(["status"]), 0)
        self.assertFalse(json.loads(out.getvalue())["customer_launch_ready"])


if __name__ == "__main__":
    unittest.main()
