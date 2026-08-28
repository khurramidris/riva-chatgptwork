import tempfile
import unittest
import json
import hashlib
from datetime import timedelta
from pathlib import Path

from rival.demo import demo_population, demo_scenario, demo_targets
from rival.engine import RivalEngine
from rival.integrity import (
    IntegrityError,
    LockedContextMismatch,
    ManifestSigner,
    OutcomeFirewallError,
    ProspectiveStudyManager,
    prediction_context_digest,
)
from rival.mathx import canonical_hash
from rival.outcome_vault import (
    OutcomeNotAvailable,
    OutcomeVault,
    OutcomeVaultAccessError,
    OutcomeVaultError,
)
from rival.providers import (
    OpenAICompatibleProvider,
    PredictionProvider,
    ProviderPrediction,
)
from rival.research.integrity_qualification import run_integrity_qualification
from rival.release import verify_release_manifest
from rival.schemas import OutcomeRevealReceipt, PopulationRecord, PreregistrationSpec, utc_now
from rival.server import RivalApplication
from rival.version import __version__


class CountingProvider(PredictionProvider):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def predict(self, person, scenario):
        self.calls += 1
        probability = 1.0 / len(scenario.choices)
        return ProviderPrediction(
            probabilities={choice.choice_id: probability for choice in scenario.choices}
        )


class PredictionContextTests(unittest.TestCase):
    def setUp(self):
        self.engine = RivalEngine()
        self.records = demo_population(50)
        self.scenario = demo_scenario(30, 0).model_copy(
            update={"information_cutoff": "2026-01-01T00:00:00+00:00"}
        )

    def tearDown(self):
        self.engine.store.close()

    def test_context_is_deterministic_and_self_consistent(self):
        first, _ = self.engine.prepare_prediction_context(
            self.records, self.scenario, demo_targets()
        )
        second, _ = self.engine.prepare_prediction_context(
            self.records, self.scenario, demo_targets()
        )
        self.assertEqual(first.context_sha256, second.context_sha256)
        self.assertEqual(first.context_sha256, prediction_context_digest(first))

    def test_history_after_cutoff_is_excluded_and_audited(self):
        changed = self.records[0].model_copy(
            update={
                "history": [
                    {"date": "2025-12-01", "event": "eligible"},
                    {"date": "2026-02-01", "event": "future"},
                ]
            }
        )
        _, audit = self.engine.prepare_prediction_context(
            [changed, *self.records[1:]], self.scenario, demo_targets()
        )
        entry = next(item for item in audit.entries if item.person_id == changed.person_id)
        self.assertEqual(entry.included_history_count, 1)
        self.assertEqual(entry.excluded_history_count, 1)
        self.assertEqual(entry.exclusion_reasons["after_information_cutoff"], 1)

    def test_outcome_field_fails_closed(self):
        changed = self.records[0].model_copy(
            update={"attributes": {**self.records[0].attributes, "observed_outcome": 1}}
        )
        with self.assertRaises(OutcomeFirewallError):
            self.engine.prepare_prediction_context(
                [changed, *self.records[1:]], self.scenario, demo_targets()
            )

    def test_locked_mismatch_occurs_before_provider_call(self):
        provider = CountingProvider()
        self.engine.register_provider("counting", provider)
        scenario = self.scenario.model_copy(update={"model_family": "counting"})
        context, _ = self.engine.prepare_prediction_context(
            self.records, scenario, demo_targets()
        )
        changed = self.records[0].model_copy(update={"weight": 1.5})
        with self.assertRaises(LockedContextMismatch):
            self.engine.simulate(
                [changed, *self.records[1:]],
                scenario,
                demo_targets(),
                locked_context=context,
            )
        self.assertEqual(provider.calls, 0)

    def test_simulation_records_provider_call_identity(self):
        result = self.engine.simulate(
            self.records, self.scenario, demo_targets()
        )
        call = result.predictions[0].provider_call
        self.assertIsNotNone(result.prediction_context)
        self.assertIsNotNone(call)
        self.assertEqual(len(call.request_sha256), 64)
        self.assertEqual(len(call.cache_key), 64)

    def test_openai_identity_never_contains_api_key(self):
        secret = "not-a-real-secret-key"
        provider = OpenAICompatibleProvider(model="example", api_key=secret)
        self.assertNotIn(secret, str(provider.identity().model_dump()))


class ManifestAndVaultTests(unittest.TestCase):
    def _locked(self):
        engine = RivalEngine()
        simulation = engine.simulate(
            demo_population(50), demo_scenario(30, 0), demo_targets()
        )
        signer = ManifestSigner(b"manifest-test-key-material-00000001")
        manager = ProspectiveStudyManager(engine.store, signer)
        sealed = manager.lock_prediction(simulation, PreregistrationSpec())
        return engine, signer, manager, simulation, sealed

    def test_manifest_tampering_is_detected(self):
        engine, signer, _, _, sealed = self._locked()
        tampered = sealed.model_copy(
            update={
                "manifest": sealed.manifest.model_copy(
                    update={"predictions_sha256": "0" * 64}
                )
            }
        )
        self.assertTrue(signer.verify(sealed))
        self.assertFalse(signer.verify(tampered))
        engine.store.close()

    def test_manifest_seal_metadata_is_authenticated(self):
        engine, signer, _, _, sealed = self._locked()
        changed_seal = sealed.seal.model_copy(
            update={"sealed_at": sealed.seal.sealed_at + timedelta(seconds=1)}
        )
        self.assertFalse(signer.verify(sealed.model_copy(update={"seal": changed_seal})))
        engine.store.close()

    def test_manifest_key_minimum_is_enforced(self):
        with self.assertRaises(ValueError):
            ManifestSigner("short")

    def test_phase_transition_cannot_skip_reveal(self):
        engine, _, manager, simulation, _ = self._locked()
        evaluation = engine.evaluate(simulation, simulation.distribution)
        with self.assertRaises(IntegrityError):
            manager.record_evaluation(simulation.scenario.scenario_id, evaluation)
        self.assertTrue(engine.store.verify_phase_chain(simulation.scenario.scenario_id))
        engine.store.close()

    def test_outcome_receipt_must_match_sealed_manifest(self):
        engine, _, manager, simulation, _ = self._locked()
        receipt = OutcomeRevealReceipt(
            study_id=simulation.scenario.scenario_id,
            manifest_sha256="0" * 64,
            outcome_sha256="1" * 64,
        )
        with self.assertRaises(IntegrityError):
            manager.record_outcome_reveal(simulation.scenario.scenario_id, receipt)
        self.assertEqual(
            engine.store.last_phase_event(simulation.scenario.scenario_id)["to_phase"],
            "prediction_locked",
        )
        engine.store.close()

    def test_vault_is_encrypted_time_gated_and_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vault.sqlite3"
            vault = OutcomeVault(path)
            marker = "UNIQUE-PLAINTEXT-OUTCOME-MARKER"
            available = utc_now() + timedelta(minutes=10)
            key = b"outcome-test-key-material-000001"
            vault.deposit("study", "a" * 64, {"marker": marker}, key, available)
            vault.connection.execute("PRAGMA wal_checkpoint(FULL)")
            self.assertNotIn(marker.encode(), path.read_bytes())
            with self.assertRaises(OutcomeNotAvailable):
                vault.reveal("study", "a" * 64, key, now=available - timedelta(seconds=1))
            with self.assertRaises(OutcomeVaultAccessError):
                vault.reveal(
                    "study",
                    "a" * 64,
                    b"wrong-key-material-but-long-enough",
                    now=available + timedelta(seconds=1),
                )
            outcome, receipt = vault.reveal(
                "study", "a" * 64, key, now=available + timedelta(seconds=1)
            )
            self.assertEqual(outcome["marker"], marker)
            self.assertEqual(receipt.outcome_sha256, canonical_hash(outcome))
            vault.close()

    def test_vault_outcome_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = OutcomeVault(Path(directory) / "vault.sqlite3")
            key = b"outcome-test-key-material-000001"
            vault.deposit("study", "a" * 64, {"value": 1}, key, utc_now())
            with self.assertRaises(OutcomeVaultError):
                vault.deposit("study", "a" * 64, {"value": 2}, key, utc_now())
            vault.close()


class IntegritySurfaceTests(unittest.TestCase):
    def test_server_prepares_and_locks_without_reveal_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            app = RivalApplication(
                str(Path(directory) / "ledger.sqlite3"),
                manifest_secret="server-manifest-test-key-material-0001",
            )
            records = demo_population(40)
            scenario = demo_scenario(25, 0)
            payload = {
                "records": [row.model_dump(mode="json") for row in records],
                "scenario": scenario.model_dump(mode="json"),
                "targets": demo_targets().model_dump(mode="json"),
            }
            prepared = app.prediction_context(payload)
            payload["locked_context"] = prepared["prediction_context"]
            simulation = app.simulate(payload)
            sealed = app.lock_study(
                {"simulation": simulation, "preregistration": {"primary_metrics": ["tvd"]}}
            )
            self.assertEqual(sealed["manifest"]["phase"], "prediction_locked")
            self.assertTrue(app.health()["prospective_locking_configured"])
            self.assertFalse(app.health()["outcome_reveal_exposed"])
            self.assertFalse(hasattr(app, "reveal_outcome"))
            app.engine.store.close()

    def test_integrity_qualification_passes(self):
        report = run_integrity_qualification()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["checks"]), 10)

    def test_release_manifest_verifier_detects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"verified")
            digest = hashlib.sha256(b"verified").hexdigest()
            manifest = {
                "release": __version__,
                "wheel": {"path": "artifact.bin", "sha256": digest},
                "qualification_artifacts": {},
            }
            manifest_path = root / "RELEASE_MANIFEST.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_release_manifest(manifest_path)["status"], "PASS")
            artifact.write_bytes(b"tampered")
            self.assertEqual(verify_release_manifest(manifest_path)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
