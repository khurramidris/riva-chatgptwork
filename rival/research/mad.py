from __future__ import annotations

from typing import Any


TWIN2K_MEGA_COMMIT = "afe2bb933fce377ed196f441a4c12962cb55a53a"


def _upstream():
    from ..vendor.twin2k_mega import mad_accuracy_evaluation

    return mad_accuracy_evaluation


def summary_mad(values) -> tuple[float, float, float, float]:
    return _upstream().summary_mad(values)


def compute_column_mad(*args: Any, **kwargs: Any):
    return _upstream().compute_column_mad(*args, **kwargs)


def compute_task_mad(*args: Any, **kwargs: Any):
    return _upstream().compute_task_mad(*args, **kwargs)


def source_identity() -> dict[str, str]:
    return {
        "component": "Twin-2K-500 Mega Study MAD evaluation",
        "upstream_commit": TWIN2K_MEGA_COMMIT,
        "integration": "vendored-wrapper",
    }

