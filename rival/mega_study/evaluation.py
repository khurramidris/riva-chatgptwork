"""Outcome-gated evaluation for the frozen Mega-Study A/B/C/D pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score

from .constants import EXPECTED_OUTCOMES, SCHEMA_VERSION, VARIANT_LABELS, VARIANTS
from .protocol import load_manifest
from .stage import load_prediction_stage
from .utils import ProtocolError, canonical_hash, file_hash, read_jsonl


def _latest_ledger(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row["work_id"])] = row
    return latest


def _fisher_mean(values: list[float]) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return None
    finite = np.clip(finite, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(finite))))


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    if len(left) < 4:
        return None
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = spearmanr(left, right).statistic if rank else pearsonr(left, right).statistic
    return float(value) if math.isfinite(float(value)) else None


def _distribution_tvd(human: np.ndarray, predicted: np.ndarray) -> float | None:
    values = sorted(set(human.tolist()) | set(predicted.tolist()))
    if not values:
        return None
    h = np.asarray([(human == value).mean() for value in values])
    p = np.asarray([(predicted == value).mean() for value in values])
    return float(np.abs(h - p).sum() / 2.0)


def _metric_block(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = len(frame)
    valid = frame[frame["valid"]].copy()
    result: dict[str, Any] = {
        "eligible_cells": eligible,
        "valid_cells": len(valid),
        "valid_coverage": len(valid) / eligible if eligible else 0.0,
        "normalized_accuracy": None,
        "exact_accuracy": None,
        "balanced_accuracy": None,
        "mae": None,
        "rmse": None,
        "pearson": None,
        "spearman": None,
        "population_mean_absolute_error": None,
        "population_mean_normalized_error": None,
        "absolute_glass_delta": None,
        "sd_ratio": None,
        "distribution_tvd": None,
    }
    if valid.empty:
        return result
    human = valid["human"].to_numpy(dtype=float)
    predicted = valid["predicted"].to_numpy(dtype=float)
    ranges = valid["range"].to_numpy(dtype=float)
    errors = predicted - human
    result.update(
        {
            "normalized_accuracy": float(np.mean(1.0 - np.abs(errors) / ranges)),
            "exact_accuracy": float(np.mean(predicted == human)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "pearson": _safe_correlation(human, predicted),
            "spearman": _safe_correlation(human, predicted, rank=True),
            "population_mean_absolute_error": float(abs(predicted.mean() - human.mean())),
            "population_mean_normalized_error": float(
                abs(predicted.mean() - human.mean()) / np.mean(ranges)
            ),
            "absolute_glass_delta": (
                float(abs(predicted.mean() - human.mean()) / np.std(human, ddof=1))
                if len(human) > 1 and np.std(human, ddof=1) > 0
                else None
            ),
            "sd_ratio": (
                float(np.std(predicted, ddof=1) / np.std(human, ddof=1))
                if len(human) > 1 and np.std(human, ddof=1) > 0
                else None
            ),
            "distribution_tvd": _distribution_tvd(human, predicted),
        }
    )
    classes = np.unique(human)
    if len(classes) >= 2:
        result["balanced_accuracy"] = float(
            balanced_accuracy_score(human, predicted)
        )
    return result


def _macro_metric(blocks: list[dict[str, Any]], name: str) -> float | None:
    values = [float(block[name]) for block in blocks if block.get(name) is not None]
    return float(np.mean(values)) if values else None


def _aggregate(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sum(int(block["eligible_cells"]) for block in blocks)
    valid = sum(int(block["valid_cells"]) for block in blocks)
    metrics = [
        "normalized_accuracy",
        "exact_accuracy",
        "balanced_accuracy",
        "mae",
        "rmse",
        "spearman",
        "population_mean_absolute_error",
        "population_mean_normalized_error",
        "absolute_glass_delta",
        "sd_ratio",
        "distribution_tvd",
    ]
    result = {
        "eligible_cells": eligible,
        "valid_cells": valid,
        "valid_coverage": valid / eligible if eligible else 0.0,
        **{name: _macro_metric(blocks, name) for name in metrics},
        "pearson": _fisher_mean(
            [float(block["pearson"]) for block in blocks if block.get("pearson") is not None]
        ),
    }
    return result


def _macro_lift(frame: pd.DataFrame) -> float:
    by_outcome = (
        frame.groupby(["study_id", "outcome_id"], sort=False)["lift"].mean().reset_index()
    )
    by_study = by_outcome.groupby("study_id", sort=False)["lift"].mean()
    return float(by_study.mean())


def _paired_lift(
    cells: pd.DataFrame,
    treatment: str,
    baseline: str,
    *,
    study: str | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 20260829,
) -> dict[str, Any]:
    subset = cells if study is None else cells[cells["study_id"] == study]
    treated = subset[(subset["variant"] == treatment) & subset["valid"]].copy()
    base = subset[(subset["variant"] == baseline) & subset["valid"]].copy()
    keys = ["case_id", "study_id", "pid", "outcome_id"]
    treated["treatment_score"] = 1.0 - np.abs(
        treated["predicted"] - treated["human"]
    ) / treated["range"]
    base["baseline_score"] = 1.0 - np.abs(base["predicted"] - base["human"]) / base["range"]
    paired = treated[keys + ["treatment_score"]].merge(
        base[keys + ["baseline_score"]], on=keys, how="inner", validate="one_to_one"
    )
    if paired.empty:
        return {
            "treatment": treatment,
            "baseline": baseline,
            "study_id": study or "macro",
            "paired_cells": 0,
            "paired_participants": 0,
            "mean_lift": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    paired["lift"] = paired["treatment_score"] - paired["baseline_score"]
    point = _macro_lift(paired)
    pids = np.asarray(sorted(paired["pid"].unique()), dtype=object)
    rng = np.random.default_rng(
        seed + sum(map(ord, treatment + baseline + (study or "macro")))
    )
    boot = np.empty(bootstrap_samples, dtype=float)
    by_pid = {pid: paired[paired["pid"] == pid] for pid in pids}
    for index in range(bootstrap_samples):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        draw = pd.concat([by_pid[pid] for pid in sampled], ignore_index=True)
        boot[index] = _macro_lift(draw)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "treatment": treatment,
        "baseline": baseline,
        "study_id": study or "macro",
        "paired_cells": len(paired),
        "paired_participants": len(pids),
        "mean_lift": point,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_unit": "participant",
        "bootstrap_samples": bootstrap_samples,
    }


def _cell_frame(
    cases: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    truth = {
        str(row["case_id"]): {
            str(cell["outcome_id"]): cell for cell in row["cells"]
        }
        for row in outcomes
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id, study, pid = (
            str(case["case_id"]),
            str(case["study_id"]),
            str(case["pid"]),
        )
        if case_id not in truth:
            raise ProtocolError(f"sealed outcomes missing for {case_id}")
        for variant in VARIANTS:
            result = latest.get(f"{case_id}::{variant}", {})
            predicted = {
                str(cell["outcome_id"]): cell
                for cell in result.get("predicted_cells", [])
            }
            for outcome_id in EXPECTED_OUTCOMES[study]:
                actual = truth[case_id].get(outcome_id)
                if actual is None:
                    raise ProtocolError(f"sealed outcome {case_id}/{outcome_id} is missing")
                candidate = predicted.get(outcome_id)
                rows.append(
                    {
                        "case_id": case_id,
                        "study_id": study,
                        "pid": pid,
                        "variant": variant,
                        "outcome_id": outcome_id,
                        "human": float(actual["value"]),
                        "predicted": (
                            float(candidate["value"]) if candidate is not None else np.nan
                        ),
                        "minimum": float(actual["minimum"]),
                        "maximum": float(actual["maximum"]),
                        "range": float(actual["maximum"]) - float(actual["minimum"]),
                        "valid": bool(result.get("status") == "SUCCESS" and candidate),
                        "result_status": str(result.get("status", "MISSING")),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_benchmark(
    stage_root: str | Path,
    results_path: str | Path,
    freeze_marker: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest()
    stage, cases, _ = load_prediction_stage(stage_root)
    results = Path(results_path)
    marker_path = Path(freeze_marker)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("prediction freeze marker is missing or invalid") from exc
    marker_payload = dict(marker)
    digest = marker_payload.pop("freeze_sha256", None)
    if digest != canonical_hash(marker_payload):
        raise ProtocolError("prediction freeze marker hash does not verify")
    if marker.get("protocol_sha256") != manifest["manifest_sha256"]:
        raise ProtocolError("prediction freeze marker belongs to another protocol")
    if marker.get("stage_sha256") != stage["stage_sha256"]:
        raise ProtocolError("prediction freeze marker belongs to another stage")
    if marker.get("results_sha256") != file_hash(results):
        raise ProtocolError("prediction ledger changed after freezing")
    sealed_dir = Path(stage_root) / "sealed"
    outcome_manifest_path = sealed_dir / "outcome_manifest.json"
    try:
        outcome_manifest = json.loads(
            outcome_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "sealed outcomes have not been materialized after prediction freeze"
        ) from exc
    outcome_payload = dict(outcome_manifest)
    outcome_digest = outcome_payload.pop("outcome_manifest_sha256", None)
    if outcome_digest != canonical_hash(outcome_payload):
        raise ProtocolError("sealed outcome manifest hash does not verify")
    if outcome_manifest.get("protocol_sha256") != manifest["manifest_sha256"]:
        raise ProtocolError("sealed outcomes belong to another protocol")
    if outcome_manifest.get("stage_sha256") != stage["stage_sha256"]:
        raise ProtocolError("sealed outcomes belong to another prediction stage")
    if outcome_manifest.get("prediction_freeze_sha256") != marker["freeze_sha256"]:
        raise ProtocolError("sealed outcomes were opened for another prediction freeze")
    if outcome_manifest.get("materialized_after_prediction_freeze") is not True:
        raise ProtocolError("sealed outcome manifest lacks post-freeze attestation")
    outcomes_path = sealed_dir / str(outcome_manifest["outcomes_path"])
    if file_hash(outcomes_path) != outcome_manifest["outcomes_sha256"]:
        raise ProtocolError("sealed outcome file hash does not verify")
    outcomes = read_jsonl(outcomes_path)
    latest = _latest_ledger(results)
    cells = _cell_frame(cases, outcomes, latest)

    per_outcome: dict[str, dict[str, dict[str, Any]]] = {}
    per_study: dict[str, dict[str, dict[str, Any]]] = {}
    overall: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_frame = cells[cells["variant"] == variant]
        per_study[variant], per_outcome[variant] = {}, {}
        study_blocks: list[dict[str, Any]] = []
        for study in manifest["development_studies"]:
            outcome_blocks: list[dict[str, Any]] = []
            for outcome_id in EXPECTED_OUTCOMES[study]:
                block = _metric_block(
                    variant_frame[
                        (variant_frame["study_id"] == study)
                        & (variant_frame["outcome_id"] == outcome_id)
                    ]
                )
                per_outcome[variant][f"{study}:{outcome_id}"] = block
                outcome_blocks.append(block)
            study_block = _aggregate(outcome_blocks)
            per_study[variant][study] = study_block
            study_blocks.append(study_block)
        overall[variant] = _aggregate(study_blocks)

    contrasts = (
        ("demographics", "generic"),
        ("full_persona", "generic"),
        ("rival_retrieval", "generic"),
        ("full_persona", "demographics"),
        ("rival_retrieval", "demographics"),
        ("rival_retrieval", "full_persona"),
    )
    paired: dict[str, Any] = {}
    for treatment, baseline in contrasts:
        label = f"{treatment}_minus_{baseline}"
        paired[label] = {
            "macro": _paired_lift(cells, treatment, baseline),
            "by_study": {
                study: _paired_lift(cells, treatment, baseline, study=study)
                for study in manifest["development_studies"]
            },
        }
    failure_counts = (
        cells[["case_id", "study_id", "variant", "result_status"]]
        .drop_duplicates()
        .groupby(["study_id", "variant", "result_status"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    table = [
        {
            "variant": VARIANT_LABELS[variant],
            "individual_prediction": overall[variant]["normalized_accuracy"],
            "person_level_correlation": overall[variant]["pearson"],
            "population_error": overall[variant]["population_mean_normalized_error"],
            "valid_coverage": overall[variant]["valid_coverage"],
        }
        for variant in VARIANTS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "protocol_sha256": manifest["manifest_sha256"],
        "prediction_freeze_sha256": marker["freeze_sha256"],
        "status": "COMPLETE_BASELINE_REPORT",
        "development_only": True,
        "confirmation_claim_allowed": False,
        "primary_metric": "1 - MAE / preregistered natural range",
        "correlation_aggregation": "Pearson per outcome across participants; Fisher-z macro mean",
        "probabilistic_score": "NOT_COLLECTED_IN_PHASE_1",
        "summary_table": table,
        "overall": overall,
        "per_study": per_study,
        "per_outcome": per_outcome,
        "paired_lifts": paired,
        "primary_rival_contrasts": [
            "rival_retrieval_minus_generic",
            "rival_retrieval_minus_demographics",
            "rival_retrieval_minus_full_persona",
        ],
        "failure_counts": failure_counts,
        "denominator_policy": "every frozen participant-study-variant-outcome cell",
        "matched_comparison_policy": "intersection of valid paired cells; coverage reported separately",
    }


def markdown_report(report: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    def lift_fmt(value: dict[str, Any]) -> str:
        if value["mean_lift"] is None:
            return "—"
        return (
            f"{value['mean_lift']:.4f} "
            f"[{value['ci95_lower']:.4f}, {value['ci95_upper']:.4f}]"
        )

    lines = [
        "# Rival Mega-Study Development Baseline",
        "",
        "> Development studies only. This report is not confirmation evidence.",
        "",
        "| Variant | Individual prediction | Person-level correlation | Population error | Valid coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["summary_table"]:
        lines.append(
            f"| {row['variant']} | {fmt(row['individual_prediction'])} | "
            f"{fmt(row['person_level_correlation'])} | {fmt(row['population_error'])} | "
            f"{fmt(row['valid_coverage'])} |"
        )
    for study in ("junk_fees", "hiring_algorithms", "privacy"):
        lines.extend(
            [
                "",
                f"## Study: `{study}`",
                "",
                "| Variant | Individual prediction | Person-level correlation | Population error | Valid coverage |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            block = report["per_study"][variant][study]
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {fmt(block['normalized_accuracy'])} | "
                f"{fmt(block['pearson'])} | "
                f"{fmt(block['population_mean_normalized_error'])} | "
                f"{fmt(block['valid_coverage'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "Values are lift in normalized accuracy with 95% participant-bootstrap CI.",
            "",
            "| Contrast | Macro | Junk Fees | Hiring Algorithms | Privacy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, payload in report["paired_lifts"].items():
        lines.append(
            f"| {name} | {lift_fmt(payload['macro'])} | "
            f"{lift_fmt(payload['by_study']['junk_fees'])} | "
            f"{lift_fmt(payload['by_study']['hiring_algorithms'])} | "
            f"{lift_fmt(payload['by_study']['privacy'])} |"
        )
    lines.extend(
        [
            "",
            "Outcome-level metrics, matched-cell counts, and every failure count are retained in the JSON report.",
            "No architecture or prompt should be tuned until this complete baseline is frozen.",
            "",
        ]
    )
    return "\n".join(lines)
