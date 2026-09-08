"""Retrospective OpinionQA comparison; these released data are already development data.

No inference, frozen experiments, hyperparameter selection, or qualification.
Every question gets one prediction fitted using only the other question families.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rival.mathx import canonical_hash
from rival.research.calibration import VectorizedPersonaCalibrator
from rival.research.datasets import load_opinionqa
from rival.research.firewall import opinionqa_family_split
from rival.runtime_calibration import (CalibrationSettings, apply_ensemble,
    comparison_metrics, fit_ensemble, one_hot_answers)


def compare():
    started = time.perf_counter()
    data = load_opinionqa()
    split = opinionqa_family_split(data.question_ids, data.question_texts, folds=5)
    split.assert_no_leakage()
    settings = CalibrationSettings()  # One declared default, no selection on results.
    responses = one_hot_answers(data.persona_answers, data.human_distributions.shape[1])
    raw = responses.mean(axis=1)
    current, old, historical = np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw)
    fits = []
    for fold in range(5):
        start = time.perf_counter()
        test, train = split.folds == fold, split.folds != fold
        fitted = fit_ensemble(responses[train], data.human_distributions[train], settings)
        current[test] = apply_ensemble(responses[test], fitted)
        prior = VectorizedPersonaCalibrator(max_iter=150).fit(data.human_distributions[train], data.persona_answers[train])
        old[test] = prior.predict(data.persona_answers[test])
        historical[test] = data.human_distributions[train].mean(axis=0)
        fits.append({"fold": fold, "training_questions": int(train.sum()), "test_questions": int(test.sum()),
                     "fit_sha256": canonical_hash(fitted), "diagnostics": fitted["diagnostics"],
                     "elapsed_seconds": time.perf_counter() - start})
        print(json.dumps({"completed_fold": fold, "seconds": fits[-1]["elapsed_seconds"]}), flush=True)
    predictions = {"uniform_persona": raw, "historical_mean": historical, "previous_research_adapter": old,
                   "runtime_adapter": current}
    metrics = {name: comparison_metrics(values, data.human_distributions) for name, values in predictions.items()}
    summary = {name: {metric: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                               "p90": float(np.quantile(values, 0.9))} for metric, values in scores.items()}
               for name, scores in metrics.items()}
    report = {"schema_version": "rival.runtime-calibration-development.v1",
        "scope": "retrospective development comparison on previously used public OpinionQA data; not untouched qualification",
        "qualified": False, "new_model_calls": 0, "live_provider_binding_tested": False,
        "sources": data.source_hashes, "questions": len(data.question_ids), "personas": len(data.persona_ids),
        "human_responses_total": int(data.human_sample_sizes.sum()), "all_planned_questions_scored": True,
        "missing_or_dropped_questions": 0,
        "split": {"folds": 5, "unit": "canonical/TF-IDF question family", "family_count": split.family_count,
                  "manifest_sha256": split.manifest_hash},
        "runtime_settings": settings.model_dump(mode="json"), "previous_adapter_iterations": 150,
        "comparison_scope": "different solver defaults; measures current candidates, not a controlled causal attribution to one change",
        "metrics": summary, "fits": fits,
        "questions_scored": [{"question_id": question, "family_id": family, "fold": int(fold),
            "human_n": int(n), "tvd": {name: scores["tvd"][i] for name, scores in metrics.items()}}
            for i, (question, family, fold, n) in enumerate(zip(data.question_ids, split.family_ids, split.folds,
                                                               data.human_sample_sizes, strict=True))],
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["The model responses were released upstream; their live provider identity cannot be retrospectively attested.",
            "Only one-hot, five-choice aggregate responses are measured; soft SSR probabilities and customer domains are not qualified.",
            "Question-family separation is lexical and cannot prove semantic independence.",
            "The data and baseline results were already used in development. No frozen Wave4 or Mega outcomes were used.",
            "No hyperparameter or model selection was made using these results. Calibration can worsen individual questions."]}
    report["report_sha256"] = canonical_hash(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("comparison output already exists")
    report = compare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"], "qualified": False}), flush=True)


if __name__ == "__main__":
    main()
