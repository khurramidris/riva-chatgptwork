from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest

from rival.commands import main
from rival.integrity import IntegrityError
from rival.runtime_uncertainty import (UncertaintySettings, assess_error, binomial_bounds,
    conformal_bound, evaluate_policy, expected_errors, fit_error_model)
from rival.schemas import utc_now
from rival.study_contract import parse_study_request
from rival.study_workflow import (_Workspace, _workspace, check_study, export_study,
    prepare_study, run_study, study_status)
from rival.uncertainty_catalog import UncertaintyCatalog
import test_runtime_calibration as calibration_helpers


SETTINGS = {"max_tvd": 0.4, "required_coverage": 0.8, "max_bad_acceptance_rate": 0.1,
            "max_failure_rate": 0.1, "nominal_coverage": 0.9}


class UncertaintyNumericsTests(unittest.TestCase):
    def model(self):
        return fit_error_model([[0.5] * 4] * 5, [0.2] * 5, SETTINGS)

    def test_exact_rank_has_finite_sample_coverage_and_small_samples_abstain(self):
        model = self.model()
        population = np.linspace(0.21, 0.9, 10)
        covered = 0
        for i in range(10):
            calibration = np.delete(population, i)
            bound = conformal_bound(model, [[0.5] * 4] * 9, calibration, 0.9)
            self.assertEqual(bound["rank"], 9)
            upper = assess_error(model, bound, [0.5] * 4, SETTINGS)["upper_tvd"]
            covered += population[i] <= upper + 1e-14
        self.assertEqual(covered, 9)
        bound = conformal_bound(model, [[0.5] * 4] * 8, [0.2] * 8, 0.9)
        decision = assess_error(model, bound, [0.5] * 4, SETTINGS)
        self.assertEqual(decision["upper_tvd"], 1)
        self.assertFalse(decision["candidate_accept"])

    def test_clopper_pearson_matches_independent_scipy_binomtest(self):
        for n in (1, 10, 100):
            for k in {0, n // 2, n}:
                bounds = binomial_bounds(k, n, 0.01)
                lower = binomtest(k, n, alternative="greater").proportion_ci(0.99).low
                upper = binomtest(k, n, alternative="less").proportion_ci(0.99).high
                self.assertAlmostEqual(bounds["lower"], lower, places=12)
                self.assertAlmostEqual(bounds["upper"], upper, places=12)
        self.assertEqual(binomial_bounds(0, 0, 0.01), {"lower": 0, "upper": 1})

    def test_finite_inputs_and_thresholds_are_required(self):
        for bad in ({**SETTINGS, "max_tvd": True}, {**SETTINGS, "ridge": float("nan")},
                    {**SETTINGS, "required_coverage": 0.99}, {**SETTINGS, "max_failure_rate": 0.0}):
            with self.assertRaises(ValueError):
                UncertaintySettings.model_validate(bad)
        for rows, errors in (([[float("nan")] * 4] * 5, [0.1] * 5),
                             ([[0.5] * 4] * 5, [1.1] * 5), ([[0.5] * 4] * 4, [0.1] * 4)):
            with self.assertRaises(ValueError):
                fit_error_model(rows, errors, SETTINGS)

    def test_held_out_labels_do_not_change_fit_and_shift_fails_the_gate(self):
        model = self.model()
        before = json.dumps(model, sort_keys=True)
        bound = conformal_bound(model, [[0.5] * 4] * 19, [0.22] * 19, 0.9)
        good = evaluate_policy(model, bound, [{"features": [0.5] * 4, "observed_tvd": 0.21}] * 60, SETTINGS)
        bad = evaluate_policy(model, bound, [{"features": [0.5] * 4, "observed_tvd": 0.8}] * 60, SETTINGS)
        self.assertTrue(good["statistical_gates_passed"])
        self.assertFalse(bad["statistical_gates_passed"])
        self.assertEqual(bad["bad_accepted_groups"], 60)
        self.assertEqual(json.dumps(model, sort_keys=True), before)

    def test_missing_studies_and_empty_acceptance_keep_their_denominators(self):
        model = self.model()
        bound = conformal_bound(model, [[0.5] * 4] * 19, [0.22] * 19, 0.9)
        rows = [{"features": [0.5] * 4, "observed_tvd": 0.21}] * 10 + [None] * 20
        result = evaluate_policy(model, bound, rows, SETTINGS)
        self.assertEqual((result["planned_groups"], result["scored_groups"], result["failed_groups"]), (30, 10, 20))
        self.assertFalse(result["gates"]["failure_rate"])
        empty = evaluate_policy(model, bound, [None], SETTINGS)
        self.assertIsNone(empty["accepted_mean_tvd"])
        self.assertEqual(empty["bad_acceptance_rate_upper"], 1)
        self.assertFalse(empty["statistical_gates_passed"])

    def test_selective_policy_reduces_risk_and_rejects_feature_extrapolation(self):
        rng = np.random.default_rng(733)
        x = np.linspace(0, 1, 30)
        rows = np.column_stack((x, np.full((30, 3), 0.5)))
        model = fit_error_model(rows, 0.05 + 0.7 * x, SETTINGS)
        xcal = rng.uniform(size=199)
        cal = np.column_stack((xcal, np.full((199, 3), 0.5)))
        bound = conformal_bound(model, cal, 0.05 + 0.7 * xcal + rng.uniform(0, 0.01, 199), 0.9)
        xtest = rng.uniform(size=500)
        result = evaluate_policy(model, bound, [{"features": [x, 0.5, 0.5, 0.5],
            "observed_tvd": 0.05 + 0.7 * x} for x in xtest], SETTINGS)
        self.assertGreater(result["accepted_groups"], 100)
        self.assertLess(result["accepted_mean_tvd"], result["mean_tvd"] - 0.1)
        outside = assess_error(model, bound, [0.5, 0.6, 0.5, 0.5], SETTINGS)
        self.assertEqual(outside["upper_tvd"], 1)
        self.assertIn("features_outside_training_range", outside["reasons"])


class UncertaintyWorkflowTests(unittest.TestCase):
    setUp = calibration_helpers.RuntimeCalibrationTests.setUp
    changed = calibration_helpers.RuntimeCalibrationTests.changed
    execute = calibration_helpers.RuntimeCalibrationTests.execute
    evaluate = calibration_helpers.RuntimeCalibrationTests.evaluate
    training = calibration_helpers.RuntimeCalibrationTests.training

    def test_integer_outcomes_verify_without_weakening_exact_outcome_binding(self):
        request = self.changed("integer-outcomes", "evaluation")
        root = self.root / "integer-outcomes"
        prepare_study(root, request, catalog_root=self.catalog.root)
        self.execute(root)
        observed = {"standard": 0, "flexible": 0, "neither": 1}
        self.evaluate(root, observed)
        with _workspace(root) as w, w.execution() as session:
            result, sealed = w.require_complete(w.journal_snapshot(session))
            from rival.evaluation import evaluate_distribution
            from rival.mathx import canonical_hash
            evaluation = evaluate_distribution(result.run_id, result.distribution, observed, preregistration_hash=canonical_hash(request.preregistration))
            w.manager._validate_evaluation(sealed, evaluation, observed)
            for invalid in ({"standard": False, "flexible": 0, "neither": True},
                            {"standard": "0", "flexible": 0, "neither": "1"}):
                with self.assertRaises(ValueError):
                    w.manager._validate_evaluation(sealed, evaluation, invalid)
            changed = evaluation.model_copy(update={"observed_distribution": {"standard": 1e-12, "flexible": 0., "neither": 1 - 1e-12}})
            with self.assertRaisesRegex(IntegrityError, "outcomes differ"):
                w.manager._validate_evaluation(sealed, changed, observed)
            reveal = next(e for e in w.store.phase_events(request.brief.study_id) if e["to_phase"] == "outcomes_revealed")
            original = w.store.phase_evidence(reveal["payload_sha256"])["payload"]["outcome"]["distribution"]
            self.assertTrue(all(type(value) is int for value in original.values()))

    def cohort(self, calibrated=False):
        self.uq = UncertaintyCatalog(self.root / "uncertainty")
        if calibrated:
            bank = self.calibration.create_bank(self.training())
            adapter = self.calibration.fit(bank["bank_sha256"])
            payload = self.request.model_dump(mode="json")
            payload.update(schema_version="rival.study-request.v4", calibration={"adapter_sha256": adapter["adapter_sha256"]})
            payload["brief"]["information_cutoff"] = utc_now().isoformat()
            self.request = parse_study_request(payload)
        partitions = {}
        for role, count in (("training", 5), ("calibration", 1), ("evaluation", 1)):
            partitions[role] = []
            for i in range(count):
                request = self.changed(f"uq-{role}-{i}", role)
                root = self.root / request.brief.study_id
                prepare_study(root, request, catalog_root=self.catalog.root,
                              calibration_catalog_root=self.calibration.root)
                partitions[role].append(root)
        self.partitions = partitions
        self.all_roots = [root for roots in partitions.values() for root in roots]
        self.plan = self.uq.plan(self.all_roots, SETTINGS)
        return self.plan

    def finish_stage(self, role):
        for root in self.partitions[role]:
            self.execute(root)
            self.evaluate(root)

    def fit_uq(self):
        self.finish_stage("training")
        self.fit_record = self.uq.fit(self.plan["plan_sha256"], self.partitions["training"])
        return self.fit_record

    def calibrate_uq(self):
        self.finish_stage("calibration")
        self.bound = self.uq.calibrate(self.fit_record["fit_sha256"], self.partitions["calibration"])
        return self.bound

    def assessment(self, calibrated=False):
        self.cohort(calibrated)
        self.fit_uq()
        self.calibrate_uq()
        self.finish_stage("evaluation")
        self.assessed = self.uq.evaluate(self.bound["bound_sha256"], self.partitions["evaluation"])
        return self.assessed

    def target(self, identifier="uq-new"):
        payload = self.changed(identifier, "evaluation").model_dump(mode="json")
        payload.update(schema_version="rival.study-request.v5", uncertainty=self.assessed)
        payload["brief"]["information_cutoff"] = utc_now().isoformat()
        return parse_study_request(payload)

    def prepare_target(self, request=None):
        request = request or self.target()
        root = self.root / request.brief.study_id
        prepare_study(root, request, catalog_root=self.catalog.root, calibration_catalog_root=self.calibration.root,
                      uncertainty_catalog_root=self.uq.root)
        return root

    def test_raw_workflow_seals_reports_recovers_and_retains_customer_abstention(self):
        self.assessment()
        root = self.prepare_target()
        self.execute(root)
        export_study(root, self.root / "before")
        before = json.loads((self.root / "before/report.json").read_text())
        self.assertEqual(before["schema_version"], "rival.study-report.v5")
        self.assertTrue(before["confidence"]["abstain"])
        self.assertIn("too_few_calibration_groups", before["confidence"]["reason_codes"])
        self.evaluate(root)
        export_study(root, self.root / "after")
        after = json.loads((self.root / "after/report.json").read_text())
        self.assertEqual(before["uncertainty"]["prediction"], after["uncertainty"]["prediction"])
        self.assertFalse(after["uncertainty"]["comparison"]["refitted"])
        self.assertNotIn("coefficients", json.dumps(after))
        with patch("urllib.request.urlopen", side_effect=AssertionError("no replay")):
            self.assertTrue(run_study(root)["complete"])
            export_study(root, self.root / "after")

    def test_calibrated_distribution_is_the_error_target(self):
        self.assessment(calibrated=True)
        root = self.prepare_target()
        self.execute(root)
        self.evaluate(root)
        export_study(root, self.root / "report")
        report = json.loads((self.root / "report/report.json").read_text())
        self.assertEqual(report["uncertainty"]["prediction"]["prediction_kind"], "calibrated")
        self.assertAlmostEqual(report["uncertainty"]["comparison"]["observed_tvd"],
                               report["calibration"]["comparison"]["metrics"]["calibrated"]["tvd"])

    def test_cohort_cannot_be_selected_after_execution_or_contain_duplicate_groups(self):
        self.cohort()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.uq.plan(self.all_roots + [self.all_roots[0]], SETTINGS)
        self.execute(self.all_roots[0])
        with self.assertRaisesRegex(ValueError, "before any model execution"):
            self.uq.plan(self.all_roots, SETTINGS)

    def test_partitions_cannot_be_omitted_swapped_or_run_before_fit(self):
        self.cohort()
        self.finish_stage("calibration")
        self.fit_uq()
        with self.assertRaisesRegex(ValueError, "after the preceding"):
            self.uq.calibrate(self.fit_record["fit_sha256"], self.partitions["calibration"])
        with self.assertRaisesRegex(ValueError, "roster/role"):
            self.uq.calibrate(self.fit_record["fit_sha256"], self.partitions["training"])
        with self.assertRaisesRegex(ValueError, "every workspace"):
            self.uq.calibrate(self.fit_record["fit_sha256"], [])

    def test_cohort_cannot_try_different_settings_even_in_another_catalog(self):
        self.cohort()
        self.assertEqual(self.plan, self.uq.plan(self.all_roots[::-1], SETTINGS))
        alternative = UncertaintyCatalog(self.root / "alternate-catalog")
        with self.assertRaisesRegex(IntegrityError, "immutable"):
            alternative.plan(self.all_roots, {**SETTINGS, "max_tvd": 0.5})

    def test_evaluation_must_be_locked_after_calibration_and_targets_cannot_reuse_responses(self):
        self.cohort()
        self.fit_uq()
        self.finish_stage("evaluation")
        self.calibrate_uq()
        with self.assertRaisesRegex(ValueError, "after the preceding"):
            self.uq.evaluate(self.bound["bound_sha256"], self.partitions["evaluation"])

    def test_target_cannot_reuse_uncertainty_model_responses(self):
        self.assessment()
        root = self.prepare_target()
        self.counter = iter(range(100))
        with self.assertRaisesRegex(IntegrityError, "reused"):
            self.execute(root)
        self.assertFalse(study_status(root)["complete"])

    def test_failed_evaluation_is_counted_and_cannot_be_rerolled(self):
        self.cohort()
        self.fit_uq()
        self.calibrate_uq()
        assessed = self.uq.evaluate(self.bound["bound_sha256"], self.partitions["evaluation"])
        artifact = self.uq.load_assessment(assessed["assessment_sha256"])
        self.assertEqual(artifact["assessment"]["metrics"]["failed_groups"], 1)
        self.assertFalse(artifact["assessment"]["metrics"]["statistical_gates_passed"])
        self.finish_stage("evaluation")
        self.assertEqual(assessed, self.uq.evaluate(self.bound["bound_sha256"], self.partitions["evaluation"]))

    def test_identity_overlap_and_future_evidence_rejected_before_calls(self):
        self.assessment()
        for mutate in (lambda p: p["brief"].update(sample_size=41),
                       lambda p: p["brief"].update(information_cutoff="2026-09-01T00:00:00Z"),
                       lambda p: p["evidence"].update(group_id="uq-evaluation-0"),
                       lambda p: p["execution"].update(generation_seed=43),
                       lambda p: p["brief"].update(question="Which delivery plan fits uq-training-0?")):
            payload = self.target().model_dump(mode="json")
            mutate(payload)
            with self.assertRaises(ValueError), patch("urllib.request.urlopen", side_effect=AssertionError("no calls")):
                self.prepare_target(parse_study_request(payload))

    def test_resume_needs_no_catalog_and_interruption_reuses_model_responses(self):
        self.assessment()
        root = self.prepare_target()
        shutil.rmtree(self.uq.root)
        original = _Workspace.put
        def interrupt(w, name, value):
            original(w, name, value)
            if name == "uncertainty_prediction":
                raise RuntimeError("interrupted after uncertainty save")
        with patch.object(_Workspace, "put", interrupt), self.assertRaises(RuntimeError):
            self.execute(root)
        with patch("urllib.request.urlopen", side_effect=AssertionError("no replay")):
            self.assertTrue(run_study(root)["complete"])

    def test_resigned_wrong_prediction_and_artifact_tampering_fail(self):
        self.assessment()
        root = self.prepare_target()
        self.execute(root)
        with _workspace(root) as w:
            value = w.read("uncertainty_prediction")
            value["research_assessment"]["upper_tvd"] = 0
            envelope = w.signer.attest({"name": "uncertainty_prediction", "value": value})
            with w.store.connection:
                w.store.connection.execute("UPDATE workflow_records SET envelope=? WHERE name='uncertainty_prediction'", (json.dumps(envelope),))
        with self.assertRaises(IntegrityError):
            export_study(root, self.root / "forged")
        path = self.uq.root / ("assessment-" + self.assessed["assessment_sha256"] + ".json")
        value = json.loads(path.read_text())
        value["payload"]["customer_qualified"] = True
        path.write_text(json.dumps(value))
        with self.assertRaises(IntegrityError):
            self.prepare_target(self.target("another"))

    def test_model_response_reuse_across_partitions_is_rejected(self):
        self.cohort()
        self.fit_uq()
        self.counter = iter(range(100))
        self.finish_stage("calibration")
        with self.assertRaisesRegex(IntegrityError, "reused"):
            self.uq.calibrate(self.fit_record["fit_sha256"], self.partitions["calibration"])

    def test_cli_inspects_aggregate_evidence_and_writes_v5_schema(self):
        self.assessment()
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["uncertainty", "inspect", "--catalog", str(self.uq.root),
                                   "--assessment", self.assessed["assessment_sha256"]]), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["customer_qualified"])
        self.assertNotIn("rows", result["metrics"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["study", "schema", "--version", "v5", "--output", str(self.root / "schema.json")]), 0)
        self.assertIn("UncertaintyPin", (self.root / "schema.json").read_text())
