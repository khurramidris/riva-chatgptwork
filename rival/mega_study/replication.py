"""No-API compatibility test on a non-target official Mega-Study result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .utils import ProtocolError


def _load_values(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=0, skiprows=[1, 2], compression="gzip")


def _digital_certification_values(
    human_path: Path, twin_path: Path
) -> pd.DataFrame:
    """Follow the authors' digital-certification evaluation preprocessing."""

    human, twin = _load_values(human_path), _load_values(twin_path)
    required = {"TWIN_ID", "WTP", "DV1_1", "DV1_2", "DV1_3"}
    if not required <= set(human) or not required <= set(twin):
        raise ProtocolError("official digital-certification columns are missing")
    invalid = set(
        twin.loc[pd.to_numeric(twin["WTP"], errors="coerce").isna(), "TWIN_ID"]
    )
    human = human.loc[~human["TWIN_ID"].isin(invalid)].copy()
    twin = twin.loc[~twin["TWIN_ID"].isin(invalid)].copy()
    for frame in (human, twin):
        frame["WTP"] = frame["WTP"].astype(float)
        frame["DV1"] = frame[["DV1_1", "DV1_2", "DV1_3"]].mean(axis=1)
        frame["log_WTP"] = np.log(1e-10 + frame["WTP"])
    return human[["TWIN_ID", "DV1", "log_WTP"]].merge(
        twin[["TWIN_ID", "DV1", "log_WTP"]],
        on="TWIN_ID",
        suffixes=("_human", "_twin"),
        validate="one_to_one",
    )


def run_official_digital_certification_replication(
    reference_root: str | Path,
) -> dict[str, Any]:
    """Reproduce authors' metrics without touching any target-study outcome."""

    root = Path(reference_root)
    merged = _digital_certification_values(
        root / "consolidated_original_answers_values.csv.gz",
        root / "consolidated_llm_values.csv.gz",
    )
    if len(merged) != 600:
        raise ProtocolError(
            f"official digital-certification replication mapped {len(merged)} "
            "rather than 600 people"
        )
    published = pd.read_csv(root / "meta analysis.csv.gz", compression="gzip")
    published = published.set_index("variable name")
    observations: dict[str, dict[str, float | int | None]] = {}
    expected: dict[str, dict[str, float | int | None]] = {}
    checks: dict[str, bool] = {}
    for variable in ("DV1", "log_WTP"):
        human = merged[f"{variable}_human"]
        twin = merged[f"{variable}_twin"]
        observations[variable] = {
            "sample_size": len(merged),
            "correlation": float(pearsonr(human, twin).statistic),
            "accuracy": (
                float(1.0 - np.mean(np.abs(human - twin)) / 18.0)
                if variable == "DV1"
                else None
            ),
            "mean_human": float(human.mean()),
            "mean_twin": float(twin.mean()),
            "std_human": float(human.std(ddof=1)),
            "std_twin": float(twin.std(ddof=1)),
        }
        row = published.loc[variable]
        accuracy = row["accuracy between humans vs. their twins"]
        expected[variable] = {
            "sample_size": int(row["sample size"]),
            "correlation": float(
                row["correlation between the responses from humans vs. their twins"]
            ),
            "accuracy": float(accuracy) if pd.notna(accuracy) else None,
            "mean_human": float(row["mean_human"]),
            "mean_twin": float(row["mean_twin"]),
            "std_human": float(row["std_human"]),
            "std_twin": float(row["std_twin"]),
        }
        for name, value in observations[variable].items():
            wanted = expected[variable][name]
            key = f"{variable}.{name}"
            checks[key] = bool(
                (value is None and wanted is None)
                or (
                    value is not None
                    and wanted is not None
                    and abs(float(value) - float(wanted)) <= 1e-12
                )
            )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "reference": (
            "official digital_certification/"
            "full_persona_without_reasoning_2025-06-23"
        ),
        "target_study_overlap": False,
        "confirmation_study_overlap": False,
        "checks": checks,
        "observed": observations,
        "published": expected,
        "validates": [
            "official result file parsing",
            "TWIN_ID pairing",
            "invalid-WTP matched exclusion",
            "DV1 three-item construction",
            "log(WTP + 1e-10) construction",
            "1-MAD/range accuracy",
            "Pearson correlation",
            "population mean and standard deviation",
        ],
    }
