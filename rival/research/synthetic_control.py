from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


CompletionMethod = Literal["hard_svd", "soft_svd", "als"]
SYN_DIGITS_COMMIT = "db891b6f821c914455b11763a96679864bf4fc48"


def _upstream_class():
    # Keep the research dependency lazy: importing Rival's API should not create
    # diagnostic directories or import plotting libraries.
    from ..vendor.syn_digits.synthetic_control import SyntheticControl

    return SyntheticControl


@dataclass(slots=True)
class ResearchSyntheticControl:
    """Production boundary around the complete licensed SYN-DIGITS component."""

    real_matrix: np.ndarray
    synthetic_matrix: np.ndarray
    dataset_name: str = "rival"
    additional_baseline_matrix: np.ndarray | None = None
    imputation_rank: int | None = None
    _model: Any = None

    def __post_init__(self) -> None:
        self.real_matrix = np.asarray(self.real_matrix, dtype=float)
        self.synthetic_matrix = np.asarray(self.synthetic_matrix, dtype=float)
        if self.real_matrix.shape != self.synthetic_matrix.shape:
            raise ValueError("real_matrix and synthetic_matrix must have the same shape")
        if self.real_matrix.ndim != 2 or min(self.real_matrix.shape) < 2:
            raise ValueError("synthetic-control matrices must be two-dimensional")
        if self.additional_baseline_matrix is not None:
            self.additional_baseline_matrix = np.asarray(
                self.additional_baseline_matrix, dtype=float
            )

    @property
    def source(self) -> dict[str, str]:
        return {
            "component": "SYN-DIGITS SyntheticControl",
            "upstream_commit": SYN_DIGITS_COMMIT,
            "integration": "vendored-with-portability-patches",
        }

    @property
    def model(self):
        if self._model is None:
            self._model = _upstream_class()(
                real_matrix=self.real_matrix,
                synthetic_matrix=self.synthetic_matrix,
                dataset_name=self.dataset_name,
                additional_baseline_matrix=self.additional_baseline_matrix,
                imputation_rank=self.imputation_rank,
            )
        return self._model

    @staticmethod
    def complete(
        matrix: np.ndarray,
        method: CompletionMethod = "hard_svd",
        rank: int = 2,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        regularization: float = 0.1,
        random_state: int = 42,
    ) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or min(values.shape) < 2:
            raise ValueError("matrix must be two-dimensional with at least two rows and columns")
        if rank < 1 or rank > min(values.shape):
            raise ValueError("rank must be between one and the smallest matrix dimension")
        if max_iter < 1 or tolerance <= 0 or regularization < 0:
            raise ValueError("invalid matrix-completion parameters")
        component = _upstream_class()
        if method == "hard_svd":
            result = component.hard_impute_svd(
                values, rank=rank, max_iter=max_iter, tol=tolerance
            )
        elif method == "soft_svd":
            result = component.soft_impute_svd(
                values,
                rank=rank,
                lambda_=regularization,
                max_iter=max_iter,
                tol=tolerance,
            )
        elif method == "als":
            result = component.als_complete(
                values,
                rank=rank,
                lambda_=regularization,
                max_iter=max_iter,
                tol=tolerance,
                random_state=random_state,
            )
        else:
            raise ValueError("method must be hard_svd, soft_svd, or als")
        completed = np.asarray(result, dtype=float)
        if completed.shape != values.shape or not np.all(np.isfinite(completed)):
            raise RuntimeError("SYN-DIGITS returned an invalid completed matrix")
        return completed

    def evaluate_column(self, target_col_index: int, **kwargs: Any) -> dict[str, Any]:
        result = dict(self.model.evaluate_column(target_col_index, **kwargs))
        result["rival_source"] = self.source
        return result

    def evaluate_row(self, target_row_index: int, **kwargs: Any) -> dict[str, Any]:
        result = dict(self.model.evaluate_row(target_row_index, **kwargs))
        result["rival_source"] = self.source
        return result

