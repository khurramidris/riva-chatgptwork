import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from rival.mega_study.evaluation import _metric_block, _paired_lift
from rival.mega_study.provider import OpenRouterSurveyProvider, SurveyCompletion
from rival.mega_study.runner import (
    balanced_variant_order,
    freeze_predictions,
    run_benchmark,
)
from rival.mega_study.stage import _verified_freeze_marker
from rival.mega_study.utils import ProtocolError, read_jsonl



def synthetic_persona_json():
    demographics = [
        {
            "QuestionID": f"D{index}",
            "QuestionText": f"Demographic field {index}?",
            "Answers": {"SelectedText": [f"Value {index}"]},
        }
        for index in range(1, 15)
    ]
    history = [
        {
            "QuestionID": f"H{index}",
            "QuestionText": f"Earlier privacy trust fairness attitude {index}?",
            "Answers": {"SelectedText": [f"Historical answer {index}"]},
        }
        for index in range(1, 101)
    ]
    return json.dumps(
        [
            {"BlockName": "Demographics", "Questions": demographics},
            {"BlockName": "Earlier Surveys", "Questions": history},
        ]
    )


class _FakeProvider:
    def __init__(self):
        self.prompts = []

    def identity(self):
        return OpenRouterSurveyProvider.identity()

    def complete(self, system, user):
        self.prompts.append((system, user))
        return SurveyCompletion(
            content=json.dumps(
                {"Q1": {"Answers": {"SelectedByPosition": 4}}}
            ),
            response_id=f"fake-{len(self.prompts)}",
            attempts=1,
            latency_ms=1.0,
            usage={"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0},
        )


def _small_stage():
    persona_json = synthetic_persona_json()
    case = {
        "case_id": "mega-runtime-case",
        "study_id": "privacy",
        "pid": "pid_1",
        "selection_rank": 1,
        "survey_text": "Q1:\nChoose one privacy rating.\n### Format Instructions:\nReturn JSON",
        "survey_text_sha256": "safe",
        "expected_outcome_ids": ["PPV"],
        "answerable_question_count": 1,
        "variant_order": list(balanced_variant_order("privacy", 1)),
    }
    persona = {
        "pid": "pid_1",
        "persona_text": "A complete earlier persona with stable attitudes.",
        "persona_json": persona_json,
        "demographics": "Fourteen frozen demographic answers.",
        "demographics_sha256": "safe",
        "historical_evidence_count": 114,
    }
    return {"stage_sha256": "safe-stage"}, [case], {"pid_1": persona}


class MegaRuntimeTests(unittest.TestCase):
    def test_preflight_calls_same_provider_four_times_and_is_resumable(self):
        fake = _FakeProvider()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.jsonl"
            future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            with (
                patch("rival.mega_study.runner.load_prediction_stage", return_value=_small_stage()),
                patch("rival.mega_study.runner.OpenRouterSurveyProvider", return_value=fake),
            ):
                first = run_benchmark(
                    root,
                    results,
                    api_key="not-persisted",
                    phase="preflight",
                    budget_usd=1.0,
                    not_after=future,
                    max_new_calls=4,
                )
                second = run_benchmark(
                    root,
                    results,
                    api_key="not-persisted",
                    phase="preflight",
                    budget_usd=1.0,
                    not_after=future,
                    max_new_calls=4,
                )
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(first["successful_work_items"], 4)
            self.assertEqual(second["new_calls"], 0)
            self.assertEqual(len(fake.prompts), 4)
            rows = read_jsonl(results)
            self.assertEqual({row["variant"] for row in rows}, {
                "generic", "demographics", "full_persona", "rival_retrieval"
            })
            self.assertNotIn("not-persisted", results.read_text(encoding="utf-8"))
            self.assertTrue(all(row["status"] == "SUCCESS" for row in rows))

    def test_freeze_binds_results_and_detects_later_change(self):
        fake = _FakeProvider()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.jsonl"
            marker_path = root / "freeze.json"
            future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            with (
                patch("rival.mega_study.runner.load_prediction_stage", return_value=_small_stage()),
                patch("rival.mega_study.runner.OpenRouterSurveyProvider", return_value=fake),
            ):
                run_benchmark(
                    root,
                    results,
                    api_key="secret",
                    phase="preflight",
                    budget_usd=1.0,
                    not_after=future,
                )
                marker = freeze_predictions(root, results, marker_path)
            self.assertFalse(marker["outcomes_opened"])
            _verified_freeze_marker(_small_stage()[0], results, marker_path)
            results.write_text(results.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ProtocolError):
                _verified_freeze_marker(_small_stage()[0], results, marker_path)

    def test_metric_and_paired_lift(self):
        metric_frame = pd.DataFrame(
            {
                "valid": [True, True, True, True],
                "human": [1.0, 2.0, 3.0, 4.0],
                "predicted": [1.0, 2.0, 2.0, 4.0],
                "range": [4.0, 4.0, 4.0, 4.0],
            }
        )
        metrics = _metric_block(metric_frame)
        self.assertEqual(metrics["valid_coverage"], 1.0)
        self.assertAlmostEqual(metrics["normalized_accuracy"], 0.9375)
        rows = []
        for pid, human in enumerate((1.0, 2.0, 3.0, 4.0), 1):
            for variant, predicted in (
                ("generic", 1.0),
                ("rival_retrieval", human),
            ):
                rows.append(
                    {
                        "case_id": f"case-{pid}",
                        "study_id": "privacy",
                        "pid": f"pid_{pid}",
                        "outcome_id": "PPV",
                        "variant": variant,
                        "valid": True,
                        "human": human,
                        "predicted": predicted,
                        "range": 6.0,
                    }
                )
        lift = _paired_lift(
            pd.DataFrame(rows),
            "rival_retrieval",
            "generic",
            bootstrap_samples=200,
        )
        self.assertGreater(lift["mean_lift"], 0)
        self.assertEqual(lift["paired_cells"], 4)
        self.assertEqual(lift["bootstrap_unit"], "participant")


if __name__ == "__main__":
    unittest.main()
