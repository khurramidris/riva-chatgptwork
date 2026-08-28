from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _validate_inputs(
    human_distributions: np.ndarray, persona_answers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    human = np.asarray(human_distributions, dtype=np.float64)
    answers = np.asarray(persona_answers, dtype=np.int64)
    if human.ndim != 2 or answers.ndim != 2:
        raise ValueError("human distributions and persona answers must be matrices")
    if human.shape[0] != answers.shape[0]:
        raise ValueError("human distributions and persona answers must share rows")
    if human.shape[0] == 0 or answers.shape[1] == 0 or human.shape[1] < 2:
        raise ValueError("calibration matrices must be non-empty")
    if not np.all(np.isfinite(human)) or np.any(human < 0):
        raise ValueError("human distributions must be finite and nonnegative")
    row_sums = human.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("every human distribution must have positive mass")
    human = human / row_sums[:, None]
    if answers.min() < 0 or answers.max() >= human.shape[1]:
        raise ValueError("persona answer indices are outside the choice range")
    return human, answers


@dataclass
class VectorizedPersonaCalibrator:
    """Memory-safe mirror descent for SYN-DIGITS distribution calibration.

    This is an independently vectorized implementation of the released objective:
    learn nonnegative persona weights plus a shared base-choice distribution on the
    simplex by minimizing KL divergence to observed question distributions.
    """

    learning_rate: float = 1.0
    max_iter: int = 150
    reg_persona: float = 1e-6
    reg_base: float = 1e-6
    gradient_clip: float = 10.0
    epsilon: float = 1e-12
    persona_weights_: np.ndarray | None = field(default=None, init=False)
    base_weights_: np.ndarray | None = field(default=None, init=False)
    objective_history_: list[float] = field(default_factory=list, init=False)

    def _predict_with(
        self, answers: np.ndarray, persona: np.ndarray, base: np.ndarray
    ) -> np.ndarray:
        rows = answers.shape[0]
        choices = len(base)
        predicted = np.broadcast_to(base, (rows, choices)).copy()
        row_index = np.repeat(np.arange(rows), answers.shape[1])
        np.add.at(predicted, (row_index, answers.ravel()), np.tile(persona, rows))
        return predicted / np.clip(predicted.sum(axis=1, keepdims=True), self.epsilon, None)

    def fit(
        self, human_distributions: np.ndarray, persona_answers: np.ndarray
    ) -> "VectorizedPersonaCalibrator":
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least one")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.reg_persona < 0 or self.reg_base < 0:
            raise ValueError("regularization values must be nonnegative")
        human, answers = _validate_inputs(human_distributions, persona_answers)
        rows, personas = answers.shape
        choices = human.shape[1]
        theta = np.full(personas + choices, 1.0 / (personas + choices))
        self.objective_history_ = []

        for _ in range(self.max_iter):
            persona, base = theta[:personas], theta[personas:]
            predicted = self._predict_with(answers, persona, base)
            clipped = np.clip(predicted, self.epsilon, None)
            objective = float(
                np.sum(human * (np.log(np.clip(human, self.epsilon, None)) - np.log(clipped)))
                / rows
                + self.reg_persona * np.dot(persona, persona)
                + self.reg_base * np.dot(base, base)
            )
            self.objective_history_.append(objective)
            gradient_q = -human / clipped / rows
            gradient_persona = gradient_q[
                np.arange(rows)[:, None], answers
            ].sum(axis=0) + 2.0 * self.reg_persona * persona
            gradient_base = gradient_q.sum(axis=0) + 2.0 * self.reg_base * base
            gradient = np.concatenate((gradient_persona, gradient_base))
            norm = float(np.linalg.norm(gradient))
            if norm > self.gradient_clip:
                gradient *= self.gradient_clip / norm
            log_theta = np.log(np.clip(theta, self.epsilon, None))
            log_theta -= self.learning_rate * gradient
            log_theta -= float(log_theta.max())
            theta = np.exp(np.clip(log_theta, -700, 0))
            theta /= theta.sum()

        self.persona_weights_ = theta[:personas].copy()
        self.base_weights_ = theta[personas:].copy()
        return self

    def predict(self, persona_answers: np.ndarray) -> np.ndarray:
        if self.persona_weights_ is None or self.base_weights_ is None:
            raise RuntimeError("calibrator must be fitted before prediction")
        answers = np.asarray(persona_answers, dtype=np.int64)
        if answers.ndim != 2 or answers.shape[1] != len(self.persona_weights_):
            raise ValueError("persona answer matrix has the wrong shape")
        if answers.min() < 0 or answers.max() >= len(self.base_weights_):
            raise ValueError("persona answer indices are outside the choice range")
        return self._predict_with(answers, self.persona_weights_, self.base_weights_)

    @staticmethod
    def raw_predict(persona_answers: np.ndarray, choices: int) -> np.ndarray:
        answers = np.asarray(persona_answers, dtype=np.int64)
        rows, personas = answers.shape
        predicted = np.zeros((rows, choices), dtype=float)
        row_index = np.repeat(np.arange(rows), personas)
        np.add.at(
            predicted,
            (row_index, answers.ravel()),
            np.full(rows * personas, 1.0 / personas),
        )
        return predicted

    @property
    def effective_personas(self) -> float:
        if self.persona_weights_ is None:
            raise RuntimeError("calibrator has not been fitted")
        normalized = self.persona_weights_ / self.persona_weights_.sum()
        return float(1.0 / np.square(normalized).sum())
