from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from .opinionqa import benchmark_opinionqa
from .integrity_qualification import run_integrity_qualification
from .provenance import stable_hash
from .twin2k import benchmark_twin2k


def build_summary(
    opinionqa: dict[str, Any], twin2k: dict[str, Any], integrity: dict[str, Any]
) -> dict[str, Any]:
    opinion_metrics = opinionqa["metrics"]
    twin_metrics = twin2k["metrics"]
    summary = {
        "schema_version": "rival.qualification.summary.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release": "0.4.0",
        "headline": "Two real-data qualification tracks; one strong aggregate result and one bounded individual result.",
        "opinionqa": {
            "status": opinionqa["status"],
            "questions": opinionqa["dataset"]["questions"],
            "personas": opinionqa["dataset"]["personas"],
            "raw_mean_tvd": opinion_metrics["raw_tvd"]["mean"],
            "global_history_mean_tvd": opinion_metrics["global_history_tvd"][
                "mean"
            ],
            "calibrated_mean_tvd": opinion_metrics["calibrated_tvd"]["mean"],
            "relative_mean_tvd_reduction": opinion_metrics[
                "relative_mean_tvd_reduction"
            ],
            "relative_mean_tvd_reduction_vs_global_history": opinion_metrics[
                "relative_mean_tvd_reduction_vs_global_history"
            ],
            "question_win_rate": opinion_metrics["question_win_rate"],
            "question_win_rate_vs_global_history": opinion_metrics[
                "question_win_rate_vs_global_history"
            ],
            "split_manifest_sha256": opinionqa["protocol"][
                "split_manifest_sha256"
            ],
            "report_sha256": opinionqa["report_sha256"],
        },
        "twin2k": {
            "status": twin2k["status"],
            "twins": twin2k["dataset"]["twins"],
            "questions": twin2k["dataset"]["evaluated_questions"],
            "direct_llm_categorical_accuracy": twin_metrics["categorical"][
                "direct_released_llm"
            ]["mean_accuracy"],
            "human_test_retest_categorical_accuracy": twin_metrics["categorical"][
                "human_test_retest"
            ]["mean_accuracy"],
            "population_mode_categorical_accuracy": twin_metrics["categorical"][
                "population_baseline"
            ]["mean_accuracy"],
            "transfer_categorical_accuracy": twin_metrics["categorical"][
                "leakage_safe_transfer"
            ]["mean_accuracy"],
            "hybrid_mean_tvd": twin_metrics["heldout_human_anchor"]["categorical"][
                "prediction_powered_hybrid_tvd"
            ]["mean"],
            "raw_mean_tvd": twin_metrics["heldout_human_anchor"]["categorical"][
                "raw_released_llm_tvd"
            ]["mean"],
            "report_sha256": twin2k["report_sha256"],
        },
        "prospective_integrity": {
            "status": integrity["status"],
            "checks_passed": sum(
                item["status"] == "PASS" for item in integrity["checks"]
            ),
            "checks_total": len(integrity["checks"]),
            "scope": integrity["scope"],
            "report_sha256": integrity["report_sha256"],
        },
        "release_decision": {
            "population_distribution_calibration": "qualified for bounded pilots",
            "individual_novel-question_prediction": "research only; baseline not beaten",
            "universal_human_simulation": "not claimed",
        },
    }
    summary["summary_sha256"] = stable_hash(summary)
    return summary


def run_all(
    output_dir: str | Path,
    *,
    compact: bool = False,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    opinionqa = benchmark_opinionqa(include_question_results=not compact)
    twin2k = benchmark_twin2k(include_question_results=not compact)
    integrity = run_integrity_qualification()
    summary = build_summary(opinionqa, twin2k, integrity)
    for name, value in (
        ("opinionqa_qualification.json", opinionqa),
        ("twin2k_qualification.json", twin2k),
        ("integrity_qualification.json", integrity),
        ("qualification_summary.json", summary),
    ):
        (destination / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return summary


def load_bundled_summary() -> dict[str, Any]:
    path = files("rival").joinpath("qualification/summary.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "release": "0.4.0",
            "status": "qualification artifact not bundled",
        }
