import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rival.mega_study.checkpoint import audit_checkpoint
from rival.mega_study.constants import VARIANTS
from rival.mega_study.utils import LeakageError, ProtocolError


def _fixture():
    provider = {"provider_name": "test", "model": "frozen"}
    manifest = {
        "study_id": "test-mega",
        "manifest_sha256": "protocol-hash",
        "development_studies": ["study_a"],
        "provider_identity": provider,
        "prompt_template_sha256": {
            variant: f"template-{variant}" for variant in VARIANTS
        },
        "model_context_policy": {"max_chars": 10000},
    }
    cases = [
        {
            "case_id": f"case-{index:03d}",
            "study_id": "study_a",
            "pid": f"pid_{index}",
            "variant_order": list(VARIANTS),
            "expected_outcome_ids": ["target"],
        }
        for index in range(300)
    ]
    return manifest, {"stage_sha256": "stage-hash"}, cases, provider


def _row(case, variant, provider, *, status="SUCCESS"):
    return {
        "protocol_sha256": "protocol-hash",
        "stage_sha256": "stage-hash",
        "work_id": f"{case['case_id']}::{variant}",
        "case_id": case["case_id"],
        "participant_study": case["study_id"],
        "pid": case["pid"],
        "variant": variant,
        "provider": provider,
        "prompt_template_sha256": f"template-{variant}",
        "context_chars": 100,
        "status": status,
        "latency_ms": 12.0,
        "attempts": 1,
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        "cost_usd": 0.001,
        "predicted_cells": (
            [
                {
                    "outcome_id": "target",
                    "value": 3,
                    "natural_min": 1,
                    "natural_max": 5,
                }
            ]
            if status == "SUCCESS"
            else []
        ),
    }


class MegaCheckpointAuditTests(unittest.TestCase):
    def _patches(self, manifest, stage, cases):
        return (
            patch("rival.mega_study.checkpoint.load_manifest", return_value=manifest),
            patch(
                "rival.mega_study.checkpoint.load_prediction_stage",
                return_value=(stage, cases, {}),
            ),
        )

    def test_report_is_blind_and_verifies_canonical_prefix(self):
        manifest, stage, cases, provider = _fixture()
        rows = [_row(cases[0], variant, provider) for variant in VARIANTS]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.jsonl"
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            first, second = self._patches(manifest, stage, cases)
            with first, second:
                report = audit_checkpoint(
                    root,
                    results,
                    expected_terminal=4,
                    budget_usd=0.10,
                )

        rendered = json.dumps(report, sort_keys=True)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["ledger"]["successful_work_items"], 4)
        self.assertEqual(
            report["coverage"]["by_variant"],
            {variant: 1 for variant in VARIANTS},
        )
        self.assertEqual(
            report["scientific_accuracy_status"],
            "NOT_EVALUATED_TO_PRESERVE_BLIND",
        )
        self.assertNotIn("pid_0", rendered)
        self.assertNotIn('"value": 3', rendered)
        self.assertFalse(
            report["leakage_firewall"]["prediction_values_summarized"]
        )

    def test_rejects_noncanonical_order(self):
        manifest, stage, cases, provider = _fixture()
        rows = [
            _row(cases[0], VARIANTS[0], provider),
            _row(cases[0], VARIANTS[2], provider),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.jsonl"
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            first, second = self._patches(manifest, stage, cases)
            with first, second, self.assertRaises(ProtocolError):
                audit_checkpoint(root, results)

    def test_rejects_protected_keys_failures_and_opened_outcomes(self):
        manifest, stage, cases, provider = _fixture()
        leaked = _row(cases[0], VARIANTS[0], provider)
        leaked["human_outcome"] = 3
        failure = _row(cases[0], VARIANTS[0], provider, status="API_FAILURE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.jsonl"
            first, second = self._patches(manifest, stage, cases)
            with first, second:
                results.write_text(json.dumps(leaked) + "\n", encoding="utf-8")
                with self.assertRaises(LeakageError):
                    audit_checkpoint(root, results)

                results.write_text(json.dumps(failure) + "\n", encoding="utf-8")
                with self.assertRaises(ProtocolError):
                    audit_checkpoint(root, results, max_failures=0)

                (root / "sealed").mkdir()
                with self.assertRaises(LeakageError):
                    audit_checkpoint(root, results, max_failures=1)


if __name__ == "__main__":
    unittest.main()
