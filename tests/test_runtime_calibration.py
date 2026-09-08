import ast
from contextlib import redirect_stdout
from datetime import timedelta
import hashlib
import io
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize

from rival.calibration_catalog import CalibrationCatalog
from rival.commands import main
from rival.evidence_catalog import EvidenceCatalog, EvidenceImportSpec, strict_json
from rival.evidence_commands import example_files
from rival.integrity import IntegrityError
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.runtime_calibration import (CalibrationSettings, apply_ensemble, comparison_metrics,
    fit_ensemble, objective_gradient, one_hot_answers, probability_array)
from rival.schemas import utc_now
from rival.study_contract import parse_study_request
from rival.study_evidence import bind_evidence
from rival.study_workflow import (_Workspace, _workspace, check_study, evaluate_study,
    export_study, prepare_study, run_study, study_status)
from test_execution_journal import Response


class EnsembleNumericsTests(unittest.TestCase):
    def test_archived_upstream_objective_and_gradient_match_on_one_hot_interior(self):
        path = Path(__file__).resolve().parents[1] / "vendor/syn_digits/distribution_calibration.py"
        source = path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         "c371e103c7a733d8d060d3f1c7ba0330f73486eae94eea88fb8789cea802aed2")
        # Execute the exact archived class; omit only optional plotting imports.
        tree = ast.parse(source)
        selected = ast.Module(body=[ast.parse("from __future__ import annotations").body[0],
            next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DistributionCalibration")], type_ignores=[])
        namespace = {"np": np}
        exec(compile(selected, str(path), "exec"), namespace)
        rng = np.random.default_rng(61)
        observed = rng.dirichlet([2, 3, 4], size=7)
        answers = rng.integers(0, 3, size=(7, 4))
        settings = CalibrationSettings(reg_persona=0.003, reg_base=0.007)
        with redirect_stdout(io.StringIO()):
            upstream = namespace["DistributionCalibration"](observed, answers, None, {}, list(range(4)),
                np.arange(3), reg_w=settings.reg_persona, reg_v=settings.reg_base,
                reg_mse=0, train_test_ratio=1.0, adaptive_lr=False)
        responses = one_hot_answers(answers[upstream.train_indices], 3)
        for _ in range(5):
            theta = rng.dirichlet(np.full(7, 2))
            objective, gradient = objective_gradient(responses, observed[upstream.train_indices], theta, settings)
            self.assertAlmostEqual(objective, upstream.compute_train_objective(theta), places=13)
            np.testing.assert_allclose(gradient, upstream.compute_gradient(theta), rtol=1e-12, atol=1e-12)

    def test_soft_gradient_matches_finite_differences_and_independent_solver(self):
        rng = np.random.default_rng(28)
        responses = rng.dirichlet([2, 2, 2], size=(8, 3))
        theta = np.array([0.12, 0.08, 0.35, 0.15, 0.10, 0.20])
        observed = np.einsum("mnk,n->mk", responses, theta[:3]) + theta[3:]
        settings = CalibrationSettings(max_iter=4000, tolerance=1e-8, reg_persona=0.001, reg_base=0.001)
        _, gradient = objective_gradient(responses, observed, theta, settings)
        for index in range(6):
            delta = np.eye(6)[index] * 1e-6
            numeric = (objective_gradient(responses, observed, theta + delta, settings)[0]
                       - objective_gradient(responses, observed, theta - delta, settings)[0]) / 2e-6
            self.assertAlmostEqual(numeric, gradient[index], places=7)
        fitted = fit_ensemble(responses, observed, settings)
        optimum = minimize(lambda x: objective_gradient(responses, observed, x, settings)[0],
            np.ones(6) / 6, method="SLSQP", bounds=[(0, 1)] * 6,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"ftol": 1e-12})
        self.assertTrue(optimum.success)
        self.assertLess(abs(fitted["diagnostics"]["final_objective"] - optimum.fun), 2e-6)
        self.assertTrue(fitted["diagnostics"]["monotone_objective"])

    def test_invalid_values_and_fractional_answers_are_not_repaired(self):
        for value in [[[0.6, 0.6]], [[-0.1, 1.1]], [[float("nan"), 1]], [[True, False]], [["0.2", "0.8"]], []]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                probability_array(value, 2)
        for values in ([[0.2, 1]], [[True, False]], [[0, -1]], [[0, 2]], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                one_hot_answers(values, 2)
        for settings in ({"max_iter": True}, {"learning_rate": float("inf")}, {"reg_base": -1}):
            with self.assertRaises(ValueError):
                CalibrationSettings.model_validate(settings)

    def test_flat_signals_zero_observed_mass_and_permutation_are_finite(self):
        responses = np.full((3, 4, 3), 1 / 3)
        observed = np.tile([0.0, 0.3, 0.7], (3, 1))
        fit = fit_ensemble(responses, observed)
        probabilities = apply_ensemble(responses, fit)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertLess(comparison_metrics(probabilities, observed)["tvd"][0], 0.02)
        reverse = fit_ensemble(responses[:, ::-1, ::-1], observed[:, ::-1])
        np.testing.assert_allclose(probabilities, apply_ensemble(responses[:, ::-1, ::-1], reverse)[:, ::-1], atol=1e-14)
        broken = {**fit, "coefficients": [1, 1]}
        with self.assertRaises(ValueError):
            apply_ensemble(responses, broken)

    def test_harmful_adjustment_is_reported_as_a_loss(self):
        responses = np.tile([[[0.8, 0.2], [0.6, 0.4]]], (2, 1, 1))
        fit = fit_ensemble(responses, [[0.1, 0.9], [0.1, 0.9]])
        target = [[0.7, 0.3]]
        self.assertGreater(comparison_metrics(apply_ensemble(responses[:1], fit), target)["tvd"][0], 0.5)


class RuntimeCalibrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = EvidenceCatalog(self.root / "evidence")
        self.calibration = CalibrationCatalog(self.root / "calibration")
        files = example_files()
        source = self.root / "people.csv"
        source.write_bytes(files["people.csv"])
        bundle = self.catalog.import_file(source, EvidenceImportSpec.model_validate(strict_json(files["import.json"])))
        brief = strict_json(files["brief.json"])
        brief["schema_version"] = "rival.study-request.v3"
        brief["brief"]["sample_size"] = 40
        brief["execution"] = {"mode": "managed", "model": "fixture-model", "generation_seed": 42,
            "base_url": "http://127.0.0.1:12345/chat/completions", "budget_usd": 1,
            "reservation_usd": 0.01, "max_attempts": 20, "max_retries": 1,
            "not_after": (utc_now() + timedelta(hours=1)).isoformat(),
            "model_pin": {"kind": "hosted_endpoint", "revision": "fixture-v1",
                          "reference": "Generated responses", "expected_response_model": "fixture-model"}}
        self.request = bind_evidence(brief, self.catalog.root, [bundle.bundle_sha256], strict_json(files["support.json"]))
        self.counter = itertools.count()
        self.key = "generated-outcome-custodian-key"

    def changed(self, identifier, role="training", **extra):
        payload = self.request.model_dump(mode="json")
        payload["brief"].update(study_id=identifier, question="Which delivery plan fits " + identifier + "?")
        payload["evidence"].update(role=role, group_id=identifier)
        payload.update(extra)
        return parse_study_request(payload)

    def execute(self, root):
        def respond(request, **kwargs):
            payload = json.loads(request.data)
            text = json.loads(payload["messages"][1]["content"])
            probabilities = ([0.65, 0.30, 0.05] if text["person"]["attributes"]["segment"] == "value" else [0.4, 0.5, 0.1])
            content = dict(zip(["standard", "flexible", "neither"], probabilities))
            return Response({"id": "calibration-request-" + str(next(self.counter)), "model": "fixture-model",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)
                    if "response_format" in payload else "I choose standard delivery."}}], "usage": {"cost": 0}})
        with patch("urllib.request.urlopen", side_effect=respond) as send:
            status = run_study(root, api_key="generated-private-key")
        return status, send.call_count

    def evaluate(self, root, observed=None):
        with _workspace(root) as workspace, workspace.execution() as session:
            _, sealed = workspace.require_complete(workspace.journal_snapshot(session))
            study_id = workspace.request.brief.study_id
        path = self.root / (study_id + "-vault.sqlite3")
        exists = path.exists()
        vault = OutcomeVault(path)
        try:
            if not exists:
                vault.deposit(study_id, canonical_hash(sealed), {"distribution": observed or
                    {"standard": 0.25, "flexible": 0.50, "neither": 0.25}, "source": "generated test outcomes"},
                    self.key, utc_now() - timedelta(seconds=1))
        finally:
            vault.close()
        return evaluate_study(root, vault.path, key_material=self.key)

    def training(self, identifiers=("reference-a", "reference-b"), role="training", evaluate=True):
        roots = []
        for identifier in identifiers:
            root = self.root / identifier
            prepare_study(root, self.changed(identifier, role), catalog_root=self.catalog.root)
            self.execute(root)
            if evaluate:
                self.evaluate(root)
            roots.append(root)
        return roots

    def fit(self):
        roots = self.training()
        self.bank = self.calibration.create_bank(roots)
        return self.calibration.fit(self.bank["bank_sha256"])

    def target(self, adapter, identifier="held-out"):
        payload = self.changed(identifier, "evaluation").model_dump(mode="json")
        payload.update(schema_version="rival.study-request.v4", calibration={"adapter_sha256": adapter["adapter_sha256"]})
        payload["brief"]["information_cutoff"] = utc_now().isoformat()
        return parse_study_request(payload)

    def prepare_target(self, request):
        return prepare_study(self.root / request.brief.study_id, request, catalog_root=self.catalog.root,
                             calibration_catalog_root=self.calibration.root)

    def test_end_to_end_fit_seal_evaluate_and_preserve_raw_results(self):
        adapter = self.fit()
        request = self.target(adapter)
        self.prepare_target(request)
        root = self.root / "held-out"
        status, calls = self.execute(root)
        self.assertTrue(status["complete"])
        self.assertEqual(calls, 6)
        export_study(root, self.root / "before")
        before = strict_json((self.root / "before/report.json").read_bytes())
        self.assertIsNone(before["calibration"]["comparison"])
        self.assertNotEqual(before["calibration"]["prediction"]["raw_distribution"],
                            before["calibration"]["prediction"]["calibrated_distribution"])
        self.evaluate(root)
        export_study(root, self.root / "after")
        after = strict_json((self.root / "after/report.json").read_bytes())
        self.assertEqual(after["schema_version"], "rival.study-report.v4")
        self.assertEqual(before["calibration"]["prediction"], after["calibration"]["prediction"])
        comparison = after["calibration"]["comparison"]
        self.assertGreater(comparison["tvd_reduction_vs_raw"], 0.2)
        self.assertEqual(after["confidence"]["label"], "unqualified")
        self.assertIn("historical_mean", comparison["metrics"])
        self.assertNotIn("coefficients", json.dumps(after))
        self.assertNotIn("generated-private-key", json.dumps(after))
        with patch("urllib.request.urlopen", side_effect=AssertionError("no replay")):
            self.assertTrue(run_study(root)["complete"])
            self.evaluate(root)
            export_study(root, self.root / "after")

    def test_nontraining_partitions_and_unrevealed_studies_cannot_be_fitted(self):
        for role in ("development", "calibration", "evaluation"):
            roots = self.training((role + "-a", role + "-b"), role)
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "training"):
                self.calibration.create_bank(roots)
        with self.assertRaisesRegex(IntegrityError, "protected evaluation"):
            self.calibration.create_bank(self.training(evaluate=False))

    def test_repeated_and_copied_groups_do_not_add_training_evidence(self):
        roots = self.training()
        copy = self.root / "copy"
        shutil.copytree(roots[0], copy)
        for members in ([roots[0]], [roots[0], roots[0]], [roots[0], copy]):
            with self.assertRaises(ValueError):
                self.calibration.create_bank(members)
        bank = self.calibration.create_bank(roots)
        self.assertEqual(bank, self.calibration.create_bank(roots[::-1]))

    def test_model_choice_audience_and_cutoff_drift_fail_before_calls(self):
        adapter = self.fit()
        original = self.target(adapter).model_dump(mode="json")
        cases = []
        for mutate in (lambda p: p["execution"].update(generation_seed=43),
                       lambda p: p["brief"]["choices"][0].update(label="Changed option"),
                       lambda p: p["audience"].update(description="Different population"),
                       lambda p: p["brief"].update(information_cutoff="2026-09-01T00:00:00Z"),
                       lambda p: p["evidence"].update(group_id="reference-a"),
                       lambda p: p["brief"].update(question="  WHICH delivery plan fits reference-a?  ")):
            payload = json.loads(json.dumps(original))
            mutate(payload)
            cases.append(parse_study_request(payload))
        with patch("urllib.request.urlopen", side_effect=AssertionError("no call")):
            for request in cases:
                with self.subTest(request=request), self.assertRaises(ValueError):
                    self.prepare_target(request)
        self.assertFalse((self.root / "held-out").exists())

    def test_calibration_is_pinned_at_preparation_and_catalog_is_not_needed_to_resume(self):
        adapter = self.fit()
        request = self.target(adapter)
        with patch("urllib.request.urlopen", side_effect=AssertionError("no preparation call")):
            check_study(request, catalog_root=self.catalog.root, calibration_catalog_root=self.calibration.root)
            self.prepare_target(request)
        shutil.rmtree(self.calibration.root)
        self.assertTrue(self.execute(self.root / "held-out")[0]["complete"])
        self.assertTrue(run_study(self.root / "held-out")["complete"])

    def test_hash_and_signature_tampering_are_rejected(self):
        adapter = self.fit()
        path = self.calibration.root / ("adapter-" + adapter["adapter_sha256"] + ".json")
        payload = strict_json(path.read_bytes())
        payload["payload"]["fit"]["coefficients"][0] += 0.1
        path.write_text(json.dumps(payload))
        with self.assertRaises(IntegrityError):
            self.prepare_target(self.target(adapter))

    def test_missing_key_and_path_traversal_fail_closed(self):
        adapter = self.fit()
        with self.assertRaises(ValueError):
            self.calibration.load_adapter("../../anything")
        (self.calibration.root / "catalog.key").unlink()
        with self.assertRaises(IntegrityError):
            self.calibration.load_adapter(adapter["adapter_sha256"])

    def test_changed_saved_prediction_cannot_be_exported_even_if_resigned(self):
        adapter = self.fit()
        self.prepare_target(self.target(adapter))
        root = self.root / "held-out"
        self.execute(root)
        with _workspace(root) as workspace:
            value = workspace.read("calibrated_prediction")
            value["calibrated_distribution"] = {"standard": 0.5, "flexible": 0.25, "neither": 0.25}
            envelope = workspace.signer.attest({"name": "calibrated_prediction", "value": value})
            with workspace.store.connection:
                workspace.store.connection.execute("UPDATE workflow_records SET envelope=? WHERE name='calibrated_prediction'", (json.dumps(envelope),))
        with self.assertRaises(IntegrityError):
            export_study(root, self.root / "forged")

    def test_interruption_after_prediction_save_recovers_without_inference(self):
        adapter = self.fit()
        self.prepare_target(self.target(adapter))
        root = self.root / "held-out"
        original = _Workspace.put
        def interrupt(workspace, name, value):
            original(workspace, name, value)
            if name == "calibrated_prediction":
                raise RuntimeError("simulated interruption")
        with patch.object(_Workspace, "put", interrupt), self.assertRaises(RuntimeError):
            self.execute(root)
        with patch("urllib.request.urlopen", side_effect=AssertionError("no replay")):
            self.assertTrue(run_study(root)["complete"])

    def test_repeated_fit_is_identical_and_cli_exposes_the_workflow(self):
        adapter = self.fit()
        self.assertEqual(adapter, self.calibration.fit(self.bank["bank_sha256"]))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["calibration", "fit", "--catalog", str(self.calibration.root), "--bank", self.bank["bank_sha256"]]), 0)
        self.assertEqual(strict_json(output.getvalue()), adapter)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["study", "schema", "--version", "v4", "--output", str(self.root / "schema.json")]), 0)
        self.assertIn("CalibrationPin", (self.root / "schema.json").read_text())

    def test_pinned_ssr_probability_vectors_pass_through_calibration(self):
        payload = self.request.model_dump(mode="json")
        payload["execution"]["elicitation"] = {"method": "ssr", "embedding": {
            "kind": "sentence_transformer", "model": "sentence-transformers/test-fixture", "revision": "a" * 40}}
        self.request = parse_study_request(payload)
        def encode(_, texts):
            return np.asarray([[1, 0, 0] if "standard" in text.casefold() else
                               [0, 1, 0] if "flexible" in text.casefold() else [0, 0, 1] for text in texts], dtype=float)
        with patch("rival.elicitation.SentenceTransformerEmbedder.encode", encode):
            adapter = self.fit()
            self.prepare_target(self.target(adapter))
            self.assertEqual(self.execute(self.root / "held-out")[1], 6)
            self.evaluate(self.root / "held-out")
        export_study(self.root / "held-out", self.root / "ssr-report")
        report = strict_json((self.root / "ssr-report/report.json").read_bytes())
        self.assertEqual(report["execution"]["measurements"]["accepted_seed_requests"], 6)
        self.assertLess(report["calibration"]["comparison"]["metrics"]["calibrated"]["tvd"], 0.02)

    def test_missing_panel_seeds_are_rejected_at_preparation(self):
        from rival.study_workflow import _make_plan
        adapter = self.fit()
        payload = self.target(adapter).model_dump(mode="json")
        payload["brief"]["sample_size"] = 20
        for seed in range(100):
            payload["brief"]["seed"] = seed
            request = parse_study_request(payload)
            if _make_plan(request)["planned_unique_seeds"] < 6:
                break
        else:
            self.fail("fixture did not omit a seed")
        with self.assertRaisesRegex(ValueError, "every eligible seed"):
            self.prepare_target(request)

    def test_target_cannot_reuse_reference_model_response_ids(self):
        adapter = self.fit()
        self.prepare_target(self.target(adapter))
        self.counter = itertools.count()  # Provider returns already-used IDs.
        with self.assertRaisesRegex(IntegrityError, "reuse reference"):
            self.execute(self.root / "held-out")
        self.assertFalse(study_status(self.root / "held-out")["complete"])

    def test_held_out_outcomes_change_metrics_without_refitting(self):
        adapter = self.fit()
        self.prepare_target(self.target(adapter))
        root = self.root / "held-out"
        self.execute(root)
        with _workspace(root) as workspace:
            raw = workspace.simulation().distribution
            before = workspace.read("calibrated_prediction")
        self.evaluate(root, raw)
        with _workspace(root) as workspace:
            self.assertEqual(before, workspace.read("calibrated_prediction"))
            self.assertLess(workspace.read("calibration_comparison")["tvd_reduction_vs_raw"], 0)
        self.assertEqual(adapter, self.calibration.fit(self.bank["bank_sha256"]))
