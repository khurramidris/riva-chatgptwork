import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from rival.engine import RivalEngine
from rival.evaluation import spearman_rank
from rival.experiments.srct import PairedPrediction, estimate_paired_srct, simulate_paired_srct
from rival.hybrid import HybridEstimator
from rival.integrity import IntegrityError, ManifestSigner, ProspectiveStudyManager, OutcomeFirewallError
from rival.mathx import canonical_hash, validate_probabilities
from rival.mega_study.utils import ResponseParseError, text_hash
from rival.mega_study_v2.evaluation import _metric_block, _cell_frame
from rival.mega_study_v2.outcomes import extract_outcome_cells, parse_model_json, validate_complete_response
from rival.outcome_vault import OutcomeVault, OutcomeVaultError
from rival.providers import PredictionProvider, ProviderPrediction
from rival.schemas import ChoiceSpec, HumanObservation, OutcomeRevealReceipt, PopulationRecord, PreregistrationSpec, ScenarioSpec, utc_now
from rival.store import ImmutableConflict


def scenario(**kwargs):
    return ScenarioSpec(name="Regression", question="Choose", context="", sample_size=20,
                        choices=[ChoiceSpec(choice_id="a", label="A"), ChoiceSpec(choice_id="b", label="B")],
                        **kwargs)


class CountingProvider(PredictionProvider):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def predict(self, person, scenario):
        self.calls += 1
        return ProviderPrediction({"a": .7, "b": .3})


def instrument(questions):
    return "\n".join(
        f"Q{i}:\n{text}\nQuestion Type: Single Choice\nOptions:\n" +
        "\n".join(f"  {j} - Option {j}" for j in range(1, maximum + 1)) + "\nAnswer: [Masked]"
        for i, (text, maximum) in enumerate(questions, 1)
    )


def response(values):
    return {f"Q{i}": {"Answers": {"SelectedByPosition": value}} for i, value in enumerate(values, 1)}


class OutcomeValidationTests(unittest.TestCase):
    def setUp(self):
        self.survey = instrument(
            [("Which of the following do you think best represents what a fee is assessed for?", 4)] * 6
            + [("How fair do you think it is to charge for a fee?", 7)] * 6
            + [("Should pricing practices be regulated by the government?", 7),
               ("Support government regulation that bans firms from separating out mandatory fees", 7)]
        )

    def test_primitive_errors_cannot_average_into_valid_composites(self):
        for values in ([999] * 6 + [4] * 8, [1] * 6 + [0, 8] * 4):
            with self.subTest(values=values), self.assertRaises(ResponseParseError):
                extract_outcome_cells("junk_fees", self.survey, response(values))

    def test_valid_fractional_composites_keep_official_formula(self):
        cells = extract_outcome_cells("junk_fees", self.survey, response([1] * 5 + [2] + [1, 2] * 4))
        self.assertAlmostEqual(cells[0].value, 100 * 5 / 6)
        self.assertEqual(cells[1].value, 1.5)
        self.assertEqual(cells[2].value, 1.5)
        self.assertEqual({cell.outcome_type for cell in cells}, {"composite"})

    def test_fractional_scoring_uses_declared_type(self):
        for kind, values in (("composite", [1.5, 2.5, 3.5, 4.5]), ("composite", [1., 2., 3., 4.]),
                             ("continuous", [1.5, 2.5, 3.5, 4.5])):
            result = _metric_block(pd.DataFrame({"human": values, "predicted": values,
                "range": [6.] * 4, "valid": [True] * 4, "outcome_type": [kind] * 4}))
            self.assertEqual(result["mae"], 0)
            self.assertEqual(result["normalized_accuracy"], 1)
            self.assertIsNone(result["balanced_accuracy"])
        ordinal = _metric_block(pd.DataFrame({"human": [1., 1., 2., 2.], "predicted": [1., 1., 1., 2.],
            "range": [6.] * 4, "valid": [True] * 4, "outcome_type": ["ordinal"] * 4}))
        self.assertEqual(ordinal["balanced_accuracy"], .75)

    def test_strict_json_and_exact_question_set(self):
        for raw in ('{"Q1":{},"Q1":{}}', '{"Q1":{"Answers":{"SelectedByPosition":NaN}}}'):
            with self.assertRaises(ResponseParseError):
                parse_model_json(raw)
        for value in (True, 1.5, 0, 999):
            with self.assertRaises(ResponseParseError):
                validate_complete_response(instrument([("Choose", 7)]), response([value]))
        with self.assertRaises(ResponseParseError):
            validate_complete_response(instrument([("Choose", 7)]), response([1, 1]))
        with self.assertRaises(ResponseParseError):
            validate_complete_response("Q1:\nNo codebook", response([1]))

    def test_matrix_cardinality_and_range(self):
        survey = "Q1:\nQuestion Type: Matrix\nOptions:\n  1 = Low\n  2 = High\n1. First\n2. Second"
        validate_complete_response(survey, response([[1, 2]]))
        for values in ([1], [1, 3], [1, 2, 1]):
            with self.assertRaises(ResponseParseError):
                validate_complete_response(survey, response([values]))

    def test_historical_success_is_revalidated_without_changing_ledger(self):
        case = {"case_id": "x", "study_id": "privacy", "pid": "p", "survey_text": instrument([("Privacy", 7)])}
        raw = json.dumps(response([999]))
        row = {"status": "SUCCESS", "raw_response": raw, "raw_response_sha256": text_hash(raw),
               "predicted_cells": [{"outcome_id": "PPV", "value": 4.}]}
        before = canonical_hash(row)
        frame = _cell_frame([case], [{"case_id": "x", "cells": [{"outcome_id": "PPV", "value": 4., "minimum": 1., "maximum": 7.}]}], {"x::generic": row})
        self.assertEqual(len(frame), 4)
        self.assertFalse(frame["valid"].any())
        self.assertEqual(frame.iloc[0]["result_status"], "REVALIDATION_FAILURE")
        self.assertEqual(canonical_hash(row), before)


class ConfidenceAndBoundaryTests(unittest.TestCase):
    def test_self_evaluation_does_not_train_and_planned_anchors_get_no_credit(self):
        engine = RivalEngine()
        engine.register_provider("counting", CountingProvider())
        s = scenario(model_family="counting")
        sim = engine.simulate([PopulationRecord(person_id="p")], s)
        with_anchor = engine.simulate([PopulationRecord(person_id="p")], scenario(human_anchor_size=20, model_family="counting"))
        self.assertEqual(sim.confidence.expected_tvd, with_anchor.confidence.expected_tvd)
        for _ in range(6):
            engine.evaluate(sim, sim.distribution)
        self.assertEqual(engine.confidence_model.training_examples, 0)
        with self.assertRaisesRegex(ValueError, "automatic confidence"):
            engine.evaluate(sim, sim.distribution, learn_confidence=True)
        engine.store.close()

    def test_conflicting_study_rejected_before_provider_request(self):
        engine = RivalEngine()
        provider = CountingProvider()
        engine.register_provider("counting", provider)
        s = scenario(model_family="counting")
        engine.simulate([PopulationRecord(person_id="p")], s)
        before = provider.calls
        with self.assertRaises(ImmutableConflict):
            engine.simulate([PopulationRecord(person_id="p")], s.model_copy(update={"question": "Changed"}))
        self.assertEqual(provider.calls, before)
        engine.store.close()

    def test_single_anchor_is_uninformative_and_duplicates_rejected(self):
        anchor = HumanObservation(person_id="p", observed_choice="a", synthetic_probabilities={"a": .9, "b": .1})
        result = HybridEstimator().correct({"a": .9, "b": .1}, [anchor])
        for interval in result.intervals.values():
            self.assertEqual((interval.lower, interval.upper, interval.standard_error), (0., 1., None))
        with self.assertRaises(ValueError):
            HybridEstimator().correct({"a": .9, "b": .1}, [anchor, anchor])
        with self.assertRaises(ValueError):
            estimate_paired_srct([PairedPrediction("p", .2, .7)])

    def test_invalid_pmf_and_tied_spearman(self):
        for values in ([-5, 0], [0, 0], [float("nan"), 1], [True, False], [.2, .2], [2, -1]):
            with self.assertRaises(ValueError):
                validate_probabilities(values)
        self.assertAlmostEqual(spearman_rank({"a": .4, "b": .4, "c": .1, "d": .1},
                                            {"a": .5, "b": .2, "c": .2, "d": .1}), 2 ** -.5)

    def test_undated_history_is_excluded_and_srct_checks_both_arms(self):
        engine = RivalEngine()
        _, audit = engine.prepare_prediction_context([PopulationRecord(person_id="p", history=[{"text": "undated"}])],
                                                     scenario(information_cutoff="2020-01-01"))
        self.assertEqual(audit.entries[0].included_history_count, 0)
        provider = CountingProvider()
        with self.assertRaises(OutcomeFirewallError):
            simulate_paired_srct(provider, [PopulationRecord(person_id="p"), PopulationRecord(person_id="q")], scenario(),
                                 scenario(metadata={"ground_truth": "a"}), "a")
        self.assertEqual(provider.calls, 0)
        engine.store.close()


class BoundOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.engine = RivalEngine(store_path=Path(self.directory.name) / "ledger.db")
        self.sim = self.engine.simulate([PopulationRecord(person_id="p")], scenario())
        self.signer = ManifestSigner("x" * 32)
        self.manager = ProspectiveStudyManager(self.engine.store, self.signer)
        self.study = self.sim.scenario.scenario_id
        self.vault = OutcomeVault(Path(self.directory.name) / "vault.db")
        self.key = "vault-test-key-material-long-enough"

    def tearDown(self):
        self.vault.close()
        self.engine.store.close()
        self.directory.cleanup()

    def lock(self, future=False):
        prereg = PreregistrationSpec(outcome_not_before=utc_now() + timedelta(days=30) if future else None)
        sealed = self.manager.lock_prediction(self.sim, prereg)
        self.vault.deposit(self.study, canonical_hash(sealed), {"a": .2, "b": .8}, self.key, utc_now())
        return sealed

    def test_fabricated_receipt_and_early_vault_reveal_are_rejected(self):
        sealed = self.lock(future=True)
        fake = OutcomeRevealReceipt(study_id=self.study, manifest_sha256=canonical_hash(sealed), outcome_sha256=canonical_hash({"a": 1., "b": 0.}))
        with self.assertRaises(IntegrityError):
            self.manager.record_outcome_reveal(self.study, fake)
        with self.assertRaisesRegex(IntegrityError, "availability"):
            self.manager.reveal_outcomes(self.study, self.vault, self.key)
        self.assertEqual(self.engine.store.last_phase_event(self.study)["to_phase"], "prediction_locked")

    def test_exact_outcomes_and_recomputed_metrics_are_required(self):
        sealed = self.lock()
        outcome, _ = self.manager.reveal_outcomes(self.study, self.vault, self.key)
        prereg_hash = canonical_hash(sealed.manifest.preregistration)
        wrong = self.engine.evaluate(self.sim, {"a": 1., "b": 0.}, prereg_hash)
        with self.assertRaisesRegex(IntegrityError, "outcomes differ"):
            self.manager.record_evaluation(self.study, wrong)
        correct = self.engine.evaluate(self.sim, outcome, prereg_hash)
        tampered = correct.model_copy(update={"metrics": {**correct.metrics, "tvd": 0.}})
        with self.assertRaisesRegex(IntegrityError, "recomputed"):
            self.manager.record_evaluation(self.study, tampered)
        self.manager.record_evaluation(self.study, correct)
        # New manager reads persisted evidence, without vault access.
        self.assertTrue(ProspectiveStudyManager(self.engine.store, self.signer).verify(sealed))
        self.engine.store.connection.execute("DELETE FROM phase_payloads WHERE id = ?", (self.engine.store.last_phase_event(self.study)["payload_sha256"],))
        self.assertFalse(self.manager.verify(sealed))

    def test_redeposit_cannot_change_release_time(self):
        self.lock()
        with self.assertRaises(OutcomeVaultError):
            self.vault.deposit(self.study, canonical_hash(self.manager._stored_manifest(self.study)),
                               {"a": .2, "b": .8}, self.key, utc_now() + timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
