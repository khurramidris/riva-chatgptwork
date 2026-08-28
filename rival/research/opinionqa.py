from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import VectorizedPersonaCalibrator
from .datasets import OpinionQADataset, load_opinionqa
from .firewall import FamilySplit, opinionqa_family_split
from .provenance import json_safe, stable_hash


def _tvd(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(predicted - observed).sum(axis=1)


def _js(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    epsilon = 1e-12
    predicted = np.clip(predicted, epsilon, 1.0)
    observed = np.clip(observed, epsilon, 1.0)
    midpoint = 0.5 * (predicted + observed)
    return 0.5 * (
        np.sum(observed * np.log(observed / midpoint), axis=1)
        + np.sum(predicted * np.log(predicted / midpoint), axis=1)
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def benchmark_opinionqa(
    dataset: OpinionQADataset | None = None,
    *,
    folds: int = 5,
    max_iter: int = 150,
    learning_rate: float = 1.0,
    include_question_results: bool = True,
) -> dict[str, Any]:
    data = dataset or load_opinionqa()
    if folds < 2 or folds > len(data.question_ids):
        raise ValueError("folds must be between 2 and the number of questions")
    split = opinionqa_family_split(
        data.question_ids, data.question_texts, folds=folds
    )
    if split.family_count < folds:
        raise ValueError("folds cannot exceed the number of semantic families")
    choices = data.human_distributions.shape[1]
    raw = VectorizedPersonaCalibrator.raw_predict(data.persona_answers, choices)
    calibrated = np.zeros_like(raw)
    global_history = np.zeros_like(raw)
    fold_rows: list[dict[str, Any]] = []
    effective_personas: list[float] = []
    base_mass: list[float] = []

    for fold in range(folds):
        test_mask = split.folds == fold
        train_mask = ~test_mask
        calibrator = VectorizedPersonaCalibrator(
            max_iter=max_iter,
            learning_rate=learning_rate,
        ).fit(
            data.human_distributions[train_mask],
            data.persona_answers[train_mask],
        )
        calibrated[test_mask] = calibrator.predict(data.persona_answers[test_mask])
        global_history[test_mask] = data.human_distributions[train_mask].mean(axis=0)
        raw_error = _tvd(raw[test_mask], data.human_distributions[test_mask])
        calibrated_error = _tvd(
            calibrated[test_mask], data.human_distributions[test_mask]
        )
        global_error = _tvd(
            global_history[test_mask], data.human_distributions[test_mask]
        )
        effective_personas.append(calibrator.effective_personas)
        assert calibrator.base_weights_ is not None
        base_mass.append(float(calibrator.base_weights_.sum()))
        fold_rows.append(
            {
                "fold": fold,
                "train_questions": int(train_mask.sum()),
                "test_questions": int(test_mask.sum()),
                "raw_mean_tvd": float(raw_error.mean()),
                "global_history_mean_tvd": float(global_error.mean()),
                "calibrated_mean_tvd": float(calibrated_error.mean()),
                "win_rate": float(np.mean(calibrated_error < raw_error)),
                "win_rate_vs_global_history": float(
                    np.mean(calibrated_error < global_error)
                ),
                "effective_personas": effective_personas[-1],
                "base_mass": base_mass[-1],
                "final_training_objective": calibrator.objective_history_[-1],
            }
        )

    raw_tvd = _tvd(raw, data.human_distributions)
    global_tvd = _tvd(global_history, data.human_distributions)
    calibrated_tvd = _tvd(calibrated, data.human_distributions)
    raw_js = _js(raw, data.human_distributions)
    calibrated_js = _js(calibrated, data.human_distributions)
    improvement = (raw_tvd - calibrated_tvd) / np.clip(raw_tvd, 1e-12, None)
    report: dict[str, Any] = {
        "schema_version": "rival.qualification.v1",
        "benchmark": "OpinionQA distribution calibration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "real-data, family-held-out",
        "claim_scope": (
            "Population response distributions for five-choice Pew OpinionQA items; "
            "not individual prediction and not a universal behavior claim."
        ),
        "dataset": {
            "questions": len(data.question_ids),
            "personas": len(data.persona_ids),
            "human_responses_total": int(data.human_sample_sizes.sum()),
            "human_responses_per_question": {
                "minimum": int(data.human_sample_sizes.min()),
                "median": int(np.median(data.human_sample_sizes)),
                "maximum": int(data.human_sample_sizes.max()),
            },
            "license": "MIT for SYN-DIGITS code; source survey/data terms preserved upstream",
            "source_hashes": data.source_hashes,
        },
        "protocol": {
            "folds": folds,
            "split_unit": "canonical/TF-IDF question family",
            "family_count": split.family_count,
            "split_manifest_sha256": split.manifest_hash,
            "calibration": "KL mirror descent over persona and shared base weights",
            "iterations": max_iter,
            "learning_rate": learning_rate,
            "outcome_access": "human distributions used only in training families",
        },
        "metrics": {
            "raw_tvd": _summary(raw_tvd),
            "global_history_tvd": _summary(global_tvd),
            "calibrated_tvd": _summary(calibrated_tvd),
            "raw_jensen_shannon": _summary(raw_js),
            "calibrated_jensen_shannon": _summary(calibrated_js),
            "relative_mean_tvd_reduction": float(
                (raw_tvd.mean() - calibrated_tvd.mean()) / raw_tvd.mean()
            ),
            "relative_mean_tvd_reduction_vs_global_history": float(
                (global_tvd.mean() - calibrated_tvd.mean()) / global_tvd.mean()
            ),
            "median_question_relative_tvd_reduction": float(np.median(improvement)),
            "question_win_rate": float(np.mean(calibrated_tvd < raw_tvd)),
            "question_win_rate_vs_global_history": float(
                np.mean(calibrated_tvd < global_tvd)
            ),
            "mean_effective_personas": float(np.mean(effective_personas)),
            "mean_base_weight_mass": float(np.mean(base_mass)),
        },
        "folds": fold_rows,
        "limitations": [
            "Persona answers are released model outputs, not live Rival provider calls.",
            "All items have five response choices; transfer to other response formats is untested.",
            "The benchmark evaluates aggregate distributions, not person-level fidelity.",
            "Question-family grouping reduces lexical leakage but cannot prove semantic independence.",
        ],
    }
    if include_question_results:
        report["questions"] = [
            {
                "question_id": question_id,
                "family_id": family_id,
                "fold": int(fold),
                "human_n": int(human_n),
                "raw_tvd": float(raw_error),
                "global_history_tvd": float(global_error),
                "calibrated_tvd": float(calibrated_error),
            }
            for (
                question_id,
                family_id,
                fold,
                human_n,
                raw_error,
                global_error,
                calibrated_error,
            ) in zip(
                data.question_ids,
                split.family_ids,
                split.folds,
                data.human_sample_sizes,
                raw_tvd,
                global_tvd,
                calibrated_tvd,
                strict=True,
            )
        ]
    report = json_safe(report)
    report["report_sha256"] = stable_hash(report)
    return report


def write_opinionqa_report(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    report = benchmark_opinionqa(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
