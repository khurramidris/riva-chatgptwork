"""Validate primitive answers before applying the preserved official outcome map.

The codebook is read only from Options and Matrix rows in the answer-free survey.
Missing or unsupported instrument structure fails closed. No respondent outcomes
are consulted and none of the v1 instrument, formula, or range files are changed.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from ..mega_study import outcomes as frozen
from ..mega_study.constants import EXPECTED_OUTCOMES
from ..mega_study.utils import ResponseParseError

OutcomeCell = frozen.OutcomeCell
survey_question_chunks = frozen.survey_question_chunks


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponseParseError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ResponseParseError(f"nonstandard JSON constant {value!r}")


def parse_model_json(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines[0].strip().casefold() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ResponseParseError("model response contains no JSON object")
    try:
        result = json.loads(value[start:end + 1], object_pairs_hook=_unique_object,
                            parse_constant=_invalid_constant)
    except json.JSONDecodeError as exc:
        raise ResponseParseError("model response is not strict JSON") from exc
    if not isinstance(result, dict):
        raise ResponseParseError("response must be an object")
    return result


def question_codebook(chunk: str) -> tuple[tuple[int, ...], int]:
    """Return allowed option positions and answer count from the public instrument."""
    if "Options:" not in chunk:
        raise ResponseParseError("question has no answer-free option codebook")
    options = chunk.split("Options:", 1)[1]
    positions = tuple(int(x) for x in re.findall(r"(?m)^\s+(\d+)\s+[-=]\s+", options))
    if not positions or positions != tuple(range(1, len(positions) + 1)):
        raise ResponseParseError("option positions must form a contiguous unique sequence")
    kind = re.search(r"(?m)^Question Type:\s*(.+)$", chunk)
    kind = kind.group(1).strip() if kind else "Single Choice"
    if kind == "Single Choice":
        return positions, 1
    if kind != "Matrix":
        raise ResponseParseError(f"unsupported question type {kind!r}")
    rows = tuple(int(x) for x in re.findall(r"(?m)^(\d+)\.\s+", options))
    if not rows or rows != tuple(range(1, len(rows) + 1)):
        raise ResponseParseError("matrix must contain contiguous numbered rows")
    return positions, len(rows)


def validate_complete_response(survey_text: str, response: dict[str, Any]) -> None:
    chunks = survey_question_chunks(survey_text)
    if set(response) != set(chunks):
        raise ResponseParseError("response question set differs from the instrument")
    for key, chunk in chunks.items():
        positions, length = question_codebook(chunk)
        if length == 1:
            values = [frozen._single_position(response, key)]
        else:
            values = frozen._matrix_positions(response, key, length=length)
        if any(value not in positions for value in values):
            raise ResponseParseError(f"{key} contains an out-of-codebook position")


def outcome_type(study: str, outcome_id: str) -> str:
    if outcome_id not in EXPECTED_OUTCOMES.get(study, ()):
        raise ResponseParseError(f"unknown outcome {study}.{outcome_id}")
    return "composite" if study == "junk_fees" else "ordinal"


def extract_outcome_cells(study: str, survey_text: str,
                          response: dict[str, Any]) -> list[OutcomeCell]:
    validate_complete_response(survey_text, response)
    return [replace(cell, outcome_type=outcome_type(study, cell.outcome_id))
            for cell in frozen.extract_outcome_cells(study, survey_text, response)]
