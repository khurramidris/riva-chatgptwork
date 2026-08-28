from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .provenance import sha256_file


def _dataset_root() -> Path:
    return Path(str(files("rival").joinpath("datasets")))


@dataclass(frozen=True)
class OpinionQADataset:
    question_ids: tuple[str, ...]
    question_texts: tuple[str, ...]
    choices: tuple[tuple[str, ...], ...]
    human_distributions: np.ndarray
    human_sample_sizes: np.ndarray
    persona_answers: np.ndarray
    persona_ids: tuple[str, ...]
    source_hashes: dict[str, str]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            len(self.question_ids),
            len(self.persona_ids),
            self.human_distributions.shape[1],
        )


@dataclass(frozen=True)
class TwinQuestion:
    column: str
    question_id: str
    catalog_column: str
    text: str
    block: str
    kind: Literal["categorical", "continuous"]
    categories: tuple[float, ...]


@dataclass(frozen=True)
class Twin2KDataset:
    twin_ids: np.ndarray
    human_history: pd.DataFrame
    human_outcomes: pd.DataFrame
    llm_predictions: pd.DataFrame
    questions: tuple[TwinQuestion, ...]
    source_hashes: dict[str, str]


def load_opinionqa(root: str | Path | None = None) -> OpinionQADataset:
    base = Path(root) if root is not None else _dataset_root() / "opinionqa"
    question_path = base / "questions.json"
    persona_path = base / "persona_answers.csv"
    question_payload = json.loads(question_path.read_text(encoding="utf-8"))
    response_frame = pd.read_csv(persona_path)
    id_column = response_frame.columns[0]
    response_frame[id_column] = response_frame[id_column].astype(str)
    response_frame = response_frame.set_index(id_column)

    question_ids = tuple(
        question_id
        for question_id in question_payload
        if question_id in response_frame.index
    )
    if len(question_ids) != len(question_payload):
        missing = sorted(set(question_payload) - set(response_frame.index))
        raise ValueError(f"OpinionQA persona responses are missing questions: {missing[:5]}")

    distributions: list[np.ndarray] = []
    sample_sizes: list[int] = []
    choices: list[tuple[str, ...]] = []
    for question_id in question_ids:
        item = question_payload[question_id]
        item_choices = tuple(str(choice) for choice in item["choices"])
        if len(item_choices) != 5:
            raise ValueError(f"{question_id} does not have the required five choices")
        counts = np.array(
            [float(item["choice_counts"][str(index)]) for index in range(1, 6)],
            dtype=float,
        )
        if counts.sum() <= 0:
            raise ValueError(f"{question_id} has no human observations")
        choices.append(item_choices)
        sample_sizes.append(int(counts.sum()))
        distributions.append(counts / counts.sum())

    answer_frame = response_frame.loc[list(question_ids)].apply(
        pd.to_numeric, errors="raise"
    )
    answers = answer_frame.to_numpy(dtype=np.int16) - 1
    if answers.min() < 0 or answers.max() >= 5:
        raise ValueError("OpinionQA persona answers must use the integer scale 1..5")
    return OpinionQADataset(
        question_ids=question_ids,
        question_texts=tuple(
            str(question_payload[question_id]["question"])
            for question_id in question_ids
        ),
        choices=tuple(choices),
        human_distributions=np.vstack(distributions),
        human_sample_sizes=np.asarray(sample_sizes, dtype=np.int64),
        persona_answers=answers,
        persona_ids=tuple(str(column) for column in answer_frame.columns),
        source_hashes={
            "questions.json": sha256_file(question_path),
            "persona_answers.csv": sha256_file(persona_path),
        },
    )


def _read_twin_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=[1], low_memory=False)
    frame["TWIN_ID"] = pd.to_numeric(frame["TWIN_ID"], errors="coerce")
    frame = frame.dropna(subset=["TWIN_ID"])
    frame["TWIN_ID"] = frame["TWIN_ID"].astype(np.int64)
    if frame["TWIN_ID"].duplicated().any():
        raise ValueError(f"duplicate TWIN_ID values in {path.name}")
    return frame.set_index("TWIN_ID")


def _question_kind(frames: tuple[pd.DataFrame, ...], column: str) -> tuple[str, tuple[float, ...]]:
    values = pd.concat(
        [pd.to_numeric(frame[column], errors="coerce") for frame in frames],
        ignore_index=True,
    ).dropna()
    unique = np.sort(values.unique().astype(float))
    integral = bool(len(unique)) and bool(np.allclose(unique, np.round(unique)))
    if integral and len(unique) <= 12:
        return "categorical", tuple(float(value) for value in unique)
    return "continuous", tuple()


def load_twin2k(root: str | Path | None = None) -> Twin2KDataset:
    base = Path(root) if root is not None else _dataset_root() / "twin2k"
    paths = {
        "human_history.csv": base / "human_history.csv",
        "human_outcomes.csv": base / "human_outcomes.csv",
        "llm_predictions.csv": base / "llm_predictions.csv",
        "wave4_mapping.json": base / "wave4_mapping.json",
        "question_catalog.json": base / "question_catalog.json",
    }
    history = _read_twin_frame(paths["human_history.csv"])
    outcomes = _read_twin_frame(paths["human_outcomes.csv"])
    predictions = _read_twin_frame(paths["llm_predictions.csv"])
    twin_ids = np.array(
        sorted(set(history.index) & set(outcomes.index) & set(predictions.index)),
        dtype=np.int64,
    )
    if not len(twin_ids):
        raise ValueError("Twin-2K sources have no common TWIN_ID values")

    mapping = json.loads(paths["wave4_mapping.json"].read_text(encoding="utf-8"))
    catalog_payload = json.loads(
        paths["question_catalog.json"].read_text(encoding="utf-8")
    )
    catalog = {str(item["QuestionID"]): item for item in catalog_payload}
    columns = [str(item["formatted_column"]) for item in mapping]
    for label, frame in (
        ("history", history),
        ("outcomes", outcomes),
        ("predictions", predictions),
    ):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Twin-2K {label} is missing mapped columns: {missing[:5]}")

    selected: list[pd.DataFrame] = []
    for frame in (history, outcomes, predictions):
        aligned = frame.loc[twin_ids, columns].copy()
        for column in columns:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
        selected.append(aligned)

    questions: list[TwinQuestion] = []
    for item in mapping:
        column = str(item["formatted_column"])
        qid = str(item["QuestionID"])
        metadata = catalog.get(qid, {})
        kind, categories = _question_kind(tuple(selected), column)
        questions.append(
            TwinQuestion(
                column=column,
                question_id=qid,
                catalog_column=str(item["catalog_csv_column"]),
                text=str(metadata.get("QuestionText", column)),
                block=str(metadata.get("BlockName", qid)).strip(),
                kind=kind,  # type: ignore[arg-type]
                categories=categories,
            )
        )
    return Twin2KDataset(
        twin_ids=twin_ids,
        human_history=selected[0],
        human_outcomes=selected[1],
        llm_predictions=selected[2],
        questions=tuple(questions),
        source_hashes={name: sha256_file(path) for name, path in paths.items()},
    )
