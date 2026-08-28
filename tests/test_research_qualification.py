import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from rival.research.calibration import VectorizedPersonaCalibrator
from rival.research.datasets import OpinionQADataset, load_opinionqa, load_twin2k
from rival.research.firewall import opinionqa_family_split, twin_family_split
from rival.research.opinionqa import benchmark_opinionqa
from rival.research.qualification import load_bundled_summary
from rival.research.twin2k import benchmark_twin2k


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.opinion = load_opinionqa()
        cls.twin = load_twin2k()

    def test_released_datasets_are_aligned(self):
        self.assertEqual(self.opinion.shape, (489, 2058, 5))
        self.assertEqual(len(self.twin.twin_ids), 2058)
        self.assertEqual(len(self.twin.questions), 126)
        self.assertEqual(
            list(self.twin.human_history.index), list(self.twin.human_outcomes.index)
        )

    def test_family_firewalls_do_not_cross_folds(self):
        opinion_split = opinionqa_family_split(
            self.opinion.question_ids, self.opinion.question_texts
        )
        opinion_split.assert_no_leakage()
        twin_split = twin_family_split(self.twin.questions)
        twin_split.assert_no_leakage()
        family_by_qid = {
            question.question_id: family
            for question, family in zip(
                self.twin.questions, twin_split.family_ids, strict=True
            )
        }
        self.assertEqual(family_by_qid["QID183"], family_by_qid["QID184"])
        self.assertEqual(family_by_qid["QID194"], family_by_qid["QID195"])

    def test_incorporated_files_match_provenance_lock(self):
        root = Path(__file__).parents[1]
        lock = json.loads((root / "upstreams.lock.json").read_text(encoding="utf-8"))
        for upstream in lock["incorporated"]:
            for item in upstream["files"]:
                path = root / item["rival_path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, item["sha256"], str(path))


class CalibrationTests(unittest.TestCase):
    def test_vectorized_calibration_recovers_better_mixture(self):
        rng = np.random.default_rng(7)
        answers = rng.integers(0, 3, size=(60, 40))
        truth = np.zeros(40)
        truth[:5] = [0.35, 0.25, 0.18, 0.12, 0.10]
        human = np.zeros((len(answers), 3))
        for row in range(len(answers)):
            np.add.at(human[row], answers[row], truth)
        raw = VectorizedPersonaCalibrator.raw_predict(answers, 3)
        calibrated = VectorizedPersonaCalibrator(max_iter=100).fit(
            human, answers
        ).predict(answers)
        raw_tvd = np.mean(0.5 * np.abs(raw - human).sum(axis=1))
        calibrated_tvd = np.mean(0.5 * np.abs(calibrated - human).sum(axis=1))
        self.assertLess(calibrated_tvd, raw_tvd * 0.7)

    def test_benchmark_smoke_uses_family_holdout(self):
        source = load_opinionqa()
        subset = OpinionQADataset(
            question_ids=source.question_ids[:30],
            question_texts=source.question_texts[:30],
            choices=source.choices[:30],
            human_distributions=source.human_distributions[:30],
            human_sample_sizes=source.human_sample_sizes[:30],
            persona_answers=source.persona_answers[:30, :120],
            persona_ids=source.persona_ids[:120],
            source_hashes=source.source_hashes,
        )
        report = benchmark_opinionqa(
            subset, folds=3, max_iter=8, include_question_results=False
        )
        self.assertEqual(report["dataset"]["questions"], 30)
        self.assertEqual(report["protocol"]["folds"], 3)
        self.assertEqual(len(report["folds"]), 3)


class QualificationReportTests(unittest.TestCase):
    def test_twin_smoke_and_bundled_summary(self):
        report = benchmark_twin2k(
            limit_questions=2, include_question_results=False, anchor_size=30
        )
        self.assertEqual(report["dataset"]["evaluated_questions"], 2)
        self.assertFalse(report["protocol"]["wave4_used_for_training"])
        summary = load_bundled_summary()
        self.assertEqual(summary["release"], "0.5.0")
        self.assertEqual(summary["prospective_integrity"]["status"], "PASS")
        self.assertEqual(
            summary["release_decision"]["individual_novel-question_prediction"],
            "research only; baseline not beaten",
        )


if __name__ == "__main__":
    unittest.main()
