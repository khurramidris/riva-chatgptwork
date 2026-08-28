from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import Twin2KDataset, TwinQuestion, load_twin2k
from .firewall import twin_family_split
from .provenance import json_safe, stable_hash


def _valid_pair(predicted: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(predicted) & np.isfinite(observed)
    return predicted[mask], observed[mask]


def _nearest(values: np.ndarray, categories: tuple[float, ...]) -> np.ndarray:
    options = np.asarray(categories, dtype=float)
    return options[np.abs(values[:, None] - options[None, :]).argmin(axis=1)]


def _distribution_tvd(
    predicted: np.ndarray, observed: np.ndarray, categories: tuple[float, ...]
) -> float:
    predicted, observed = _valid_pair(predicted, observed)
    if not len(observed):
        return math.nan
    options = np.asarray(categories, dtype=float)
    p = np.array([np.mean(predicted == option) for option in options])
    q = np.array([np.mean(observed == option) for option in options])
    return float(0.5 * np.abs(p - q).sum())


def _categorical_metrics(
    predicted: np.ndarray, observed: np.ndarray, categories: tuple[float, ...]
) -> dict[str, float | int]:
    predicted, observed = _valid_pair(predicted, observed)
    return {
        "n": int(len(observed)),
        "accuracy": float(np.mean(predicted == observed)) if len(observed) else math.nan,
        "distribution_tvd": _distribution_tvd(predicted, observed, categories),
    }


def _continuous_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    predicted, observed = _valid_pair(predicted, observed)
    if not len(observed):
        return {"n": 0, "normalized_mae": math.nan, "spearman": math.nan}
    q05, q95 = np.quantile(observed, [0.05, 0.95])
    scale = max(float(q95 - q05), float(np.std(observed)), 1e-8)
    correlation = (
        float(spearmanr(predicted, observed).statistic)
        if len(observed) > 2 and np.std(predicted) > 0 and np.std(observed) > 0
        else math.nan
    )
    return {
        "n": int(len(observed)),
        "normalized_mae": float(np.mean(np.abs(predicted - observed)) / scale),
        "spearman": correlation,
    }


def _population_prediction(history: np.ndarray, question: TwinQuestion) -> np.ndarray:
    valid = history[np.isfinite(history)]
    if not len(valid):
        return np.full_like(history, np.nan, dtype=float)
    if question.kind == "categorical":
        values, counts = np.unique(valid, return_counts=True)
        value = float(values[np.argmax(counts)])
    else:
        value = float(np.median(valid))
    return np.full(len(history), value, dtype=float)


def _transfer_prediction(
    data: Twin2KDataset,
    target_index: int,
    family_ids: tuple[str, ...],
    alpha: float,
) -> tuple[np.ndarray, int]:
    target = data.questions[target_index]
    target_family = family_ids[target_index]
    candidate_donors = [
        question.column
        for index, question in enumerate(data.questions)
        if index != target_index
        and family_ids[index] != target_family
        and data.llm_predictions[question.column].notna().any()
        and data.human_history[question.column].notna().any()
    ]
    llm_y = data.llm_predictions[target.column].to_numpy(dtype=float)
    training_mask = np.isfinite(llm_y)
    candidate_x = data.llm_predictions[candidate_donors].to_numpy(dtype=float)
    keep = np.isfinite(candidate_x[training_mask]).any(axis=0)
    donors = [
        column for column, include in zip(candidate_donors, keep, strict=True) if include
    ]
    if training_mask.sum() < 30 or not donors:
        return np.full(len(data.twin_ids), np.nan), len(donors)
    llm_x = data.llm_predictions[donors].to_numpy(dtype=float)
    human_x = data.human_history[donors].to_numpy(dtype=float)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr"),
    )
    model.fit(llm_x[training_mask], llm_y[training_mask])
    prediction = model.predict(human_x)
    if target.kind == "categorical":
        prediction = _nearest(prediction, target.categories)
    return prediction, len(donors)


def _mean_metric(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    values = [
        float(row[method][metric])
        for row in rows
        if math.isfinite(float(row[method][metric]))
    ]
    return float(np.mean(values)) if values else math.nan


def _distribution(values: np.ndarray, categories: tuple[float, ...]) -> np.ndarray:
    options = np.asarray(categories, dtype=float)
    result = np.array([np.mean(values == option) for option in options], dtype=float)
    return result / result.sum()


def _anchor_benchmark(
    data: Twin2KDataset, anchor_size: int, seed: int = 20260827
) -> dict[str, Any]:
    categorical_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    for index, question in enumerate(data.questions):
        outcome = data.human_outcomes[question.column].to_numpy(dtype=float)
        direct = data.llm_predictions[question.column].to_numpy(dtype=float)
        history = data.human_history[question.column].to_numpy(dtype=float)
        valid = np.flatnonzero(
            np.isfinite(outcome) & np.isfinite(direct) & np.isfinite(history)
        )
        if len(valid) <= anchor_size + 20:
            continue
        anchor = np.random.default_rng(seed + index).choice(
            valid, size=anchor_size, replace=False
        )
        anchor_mask = np.zeros(len(outcome), dtype=bool)
        anchor_mask[anchor] = True
        test = valid[~anchor_mask[valid]]
        if question.kind == "categorical":
            observed = _distribution(outcome[test], question.categories)
            raw = _distribution(direct[test], question.categories)
            human_anchor = _distribution(outcome[anchor], question.categories)
            synthetic_anchor = _distribution(direct[anchor], question.categories)
            historical = _distribution(history[test], question.categories)
            corrected = np.clip(raw + human_anchor - synthetic_anchor, 0.0, None)
            corrected /= corrected.sum()

            def tvd(left: np.ndarray) -> float:
                return float(0.5 * np.abs(left - observed).sum())

            categorical_rows.append(
                {
                    "question": question.column,
                    "raw_tvd": tvd(raw),
                    "hybrid_tvd": tvd(corrected),
                    "human_anchor_only_tvd": tvd(human_anchor),
                    "historical_distribution_tvd": tvd(historical),
                }
            )
        else:
            observed_mean = float(np.mean(outcome[test]))
            raw_mean = float(np.mean(direct[test]))
            anchor_mean = float(np.mean(outcome[anchor]))
            corrected_mean = raw_mean + float(
                np.mean(outcome[anchor] - direct[anchor])
            )
            historical_mean = float(np.mean(history[test]))
            q05, q95 = np.quantile(outcome[test], [0.05, 0.95])
            scale = max(float(q95 - q05), float(np.std(outcome[test])), 1e-8)
            continuous_rows.append(
                {
                    "question": question.column,
                    "raw_normalized_mean_error": abs(raw_mean - observed_mean) / scale,
                    "hybrid_normalized_mean_error": abs(corrected_mean - observed_mean)
                    / scale,
                    "human_anchor_only_normalized_mean_error": abs(
                        anchor_mean - observed_mean
                    )
                    / scale,
                    "historical_normalized_mean_error": abs(
                        historical_mean - observed_mean
                    )
                    / scale,
                }
            )

    def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }

    raw = np.asarray([row["raw_tvd"] for row in categorical_rows])
    hybrid = np.asarray([row["hybrid_tvd"] for row in categorical_rows])
    anchor = np.asarray([row["human_anchor_only_tvd"] for row in categorical_rows])
    categorical = {
        "questions": len(categorical_rows),
        "raw_released_llm_tvd": summarize(categorical_rows, "raw_tvd"),
        "prediction_powered_hybrid_tvd": summarize(categorical_rows, "hybrid_tvd"),
        "human_anchor_only_tvd": summarize(categorical_rows, "human_anchor_only_tvd"),
        "historical_distribution_tvd": summarize(
            categorical_rows, "historical_distribution_tvd"
        ),
        "hybrid_relative_mean_reduction_vs_raw": float(
            (raw.mean() - hybrid.mean()) / raw.mean()
        ),
        "hybrid_win_rate_vs_raw": float(np.mean(hybrid < raw)),
        "hybrid_win_rate_vs_human_anchor_only": float(np.mean(hybrid < anchor)),
    }
    continuous = {
        "questions": len(continuous_rows),
        "raw_released_llm_normalized_mean_error": summarize(
            continuous_rows, "raw_normalized_mean_error"
        ),
        "prediction_powered_hybrid_normalized_mean_error": summarize(
            continuous_rows, "hybrid_normalized_mean_error"
        ),
        "human_anchor_only_normalized_mean_error": summarize(
            continuous_rows, "human_anchor_only_normalized_mean_error"
        ),
        "historical_normalized_mean_error": summarize(
            continuous_rows, "historical_normalized_mean_error"
        ),
    }
    return {
        "anchor_size": anchor_size,
        "test_people_per_question": f"all valid minus {anchor_size}",
        "seed": seed,
        "categorical": categorical,
        "continuous": continuous,
        "interpretation": (
            "The hybrid removes most released-model distribution bias, but in this "
            "repeated-question panel the human-only anchor and full historical "
            "distribution remain stronger baselines."
        ),
    }


def benchmark_twin2k(
    dataset: Twin2KDataset | None = None,
    *,
    ridge_alpha: float = 10.0,
    anchor_size: int = 80,
    include_question_results: bool = True,
    limit_questions: int | None = None,
) -> dict[str, Any]:
    data = dataset or load_twin2k()
    if anchor_size < 1:
        raise ValueError("anchor_size must be positive")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be nonnegative")
    if limit_questions is not None and limit_questions < 1:
        raise ValueError("limit_questions must be positive when supplied")
    split = twin_family_split(data.questions)
    question_indices = list(range(len(data.questions)))
    if limit_questions is not None:
        question_indices = question_indices[:limit_questions]
    categorical_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []

    for index in question_indices:
        question = data.questions[index]
        outcome = data.human_outcomes[question.column].to_numpy(dtype=float)
        direct = data.llm_predictions[question.column].to_numpy(dtype=float)
        history = data.human_history[question.column].to_numpy(dtype=float)
        population = _population_prediction(history, question)
        transfer, donor_count = _transfer_prediction(
            data, index, split.family_ids, ridge_alpha
        )
        metric_function = (
            (lambda left, right: _categorical_metrics(left, right, question.categories))
            if question.kind == "categorical"
            else _continuous_metrics
        )
        row = {
            "column": question.column,
            "question_id": question.question_id,
            "block": question.block,
            "family_id": split.family_ids[index],
            "family_fold": int(split.folds[index]),
            "donor_questions": donor_count,
            "direct_released_llm": metric_function(direct, outcome),
            "human_test_retest": metric_function(history, outcome),
            "population_baseline": metric_function(population, outcome),
            "leakage_safe_transfer": metric_function(transfer, outcome),
        }
        if question.kind == "categorical":
            # A distribution-only historical baseline preserves prevalence instead of
            # collapsing every individual to the mode.
            row["historical_distribution_tvd"] = _distribution_tvd(
                history, outcome, question.categories
            )
            categorical_rows.append(row)
        else:
            continuous_rows.append(row)

    methods = [
        "direct_released_llm",
        "human_test_retest",
        "population_baseline",
        "leakage_safe_transfer",
    ]
    categorical_summary = {
        method: {
            "mean_accuracy": _mean_metric(categorical_rows, method, "accuracy"),
            "mean_distribution_tvd": _mean_metric(
                categorical_rows, method, "distribution_tvd"
            ),
        }
        for method in methods
    }
    continuous_summary = {
        method: {
            "mean_normalized_mae": _mean_metric(
                continuous_rows, method, "normalized_mae"
            ),
            "mean_spearman": _mean_metric(continuous_rows, method, "spearman"),
        }
        for method in methods
    }
    report: dict[str, Any] = {
        "schema_version": "rival.qualification.v1",
        "benchmark": "Twin-2K longitudinal individual response prediction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "real-data, target-family-excluded transfer",
        "claim_scope": (
            "Wave-4 response prediction for the released Twin-2K panel and mapped "
            "behavioral questions; not a population-representative universal twin claim."
        ),
        "dataset": {
            "twins": int(len(data.twin_ids)),
            "mapped_questions": len(data.questions),
            "evaluated_questions": len(question_indices),
            "license": "CC BY 4.0",
            "source_hashes": data.source_hashes,
        },
        "protocol": {
            "outcome": "human wave-4 response",
            "direct_released_llm": (
                "released GPT-4.1-mini imputation; may include target-history context"
            ),
            "human_test_retest": "same person's wave-1/3 response to mapped item",
            "population_baseline": "wave-1/3 modal category or median value",
            "leakage_safe_transfer": (
                "fixed-alpha ridge trained only on released LLM cross-person response "
                "relationships, then applied to human history donors"
            ),
            "target_family_exclusion": (
                "target item, same QuestionID, normalized block variants, and semantic "
                "siblings are excluded from every transfer donor matrix"
            ),
            "ridge_alpha": ridge_alpha,
            "family_count": split.family_count,
            "family_manifest_sha256": split.manifest_hash,
            "wave4_used_for_training": False,
        },
        "metrics": {
            "categorical_questions": len(categorical_rows),
            "continuous_questions": len(continuous_rows),
            "categorical": categorical_summary,
            "continuous": continuous_summary,
            "mean_historical_distribution_tvd": float(
                np.mean([row["historical_distribution_tvd"] for row in categorical_rows])
            )
            if categorical_rows
            else math.nan,
            "heldout_human_anchor": _anchor_benchmark(data, anchor_size),
        },
        "limitations": [
            "The direct released-LLM benchmark may be retest-grounded and is labeled accordingly.",
            "The transfer model is a research baseline, not yet a production personalization model.",
            "Family grouping is lexical/metadata-based and cannot guarantee causal independence.",
            "Some continuous responses are sparse or heavy-tailed; normalized MAE uses each outcome's 5th–95th percentile span.",
            "Twin-2K is a longitudinal research panel, not a representative customer sample.",
        ],
    }
    if include_question_results:
        report["questions"] = categorical_rows + continuous_rows
    report = json_safe(report)
    report["report_sha256"] = stable_hash(report)
    return report


def write_twin2k_report(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    report = benchmark_twin2k(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
