"""Strict response parsing and official Mega-Study outcome construction.

The formulas and variable definitions follow the Apache-2.0 official
Twin-2K-500 Mega-Study repository at commit
``afe2bb933fce377ed196f441a4c12962cb55a53a``.  This implementation is kept
small so prediction code never needs to import an answer-bearing survey.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .constants import EXPECTED_OUTCOMES, OUTCOME_RANGES
from .utils import ResponseParseError


_QUESTION_RE = re.compile(r"(?m)^Q(\d+):\s*$")


@dataclass(frozen=True, slots=True)
class OutcomeCell:
    outcome_id: str
    value: float
    minimum: float
    maximum: float
    outcome_type: str = "ordinal_or_continuous"

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "outcome_type": self.outcome_type,
        }


def survey_question_chunks(survey_text: str) -> dict[str, str]:
    """Return the answerable Q1..Qn blocks from an answer-free survey prompt."""

    body = survey_text.split("### Format Instructions:", 1)[0]
    matches = list(_QUESTION_RE.finditer(body))
    chunks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        key = f"Q{int(match.group(1))}"
        if key in chunks:
            raise ResponseParseError(f"duplicate question label {key}")
        chunks[key] = body[match.end() : end].strip()
    if not chunks:
        raise ResponseParseError("survey text contains no answerable Q labels")
    expected = [f"Q{index}" for index in range(1, len(chunks) + 1)]
    if list(chunks) != expected:
        raise ResponseParseError("survey Q labels are not a contiguous Q1..Qn sequence")
    return chunks


def parse_model_json(content: str) -> dict[str, Any]:
    """Apply the frozen strict parser: fence removal, object extraction, JSON."""

    value = content.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().casefold() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ResponseParseError("model response contains no JSON object")
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ResponseParseError("model response is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise ResponseParseError("model response root must be an object")
    return parsed


def validate_complete_response(
    survey_text: str, response: dict[str, Any]
) -> None:
    """Require an answer object for every question in the rendered survey."""

    expected = tuple(survey_question_chunks(survey_text))
    missing = [question for question in expected if question not in response]
    if missing:
        raise ResponseParseError(
            f"model response omitted {len(missing)} survey questions; first is {missing[0]}"
        )
    for question in expected:
        _answers(response, question)


def _answers(response: dict[str, Any], question: str) -> dict[str, Any]:
    item = response.get(question)
    if not isinstance(item, dict):
        raise ResponseParseError(f"{question} is missing or is not an object")
    answers = item.get("Answers")
    if not isinstance(answers, dict):
        raise ResponseParseError(f"{question}.Answers is missing or is not an object")
    return answers


def _number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseParseError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not result.is_integer():
        raise ResponseParseError(f"{location} must be a finite integer position")
    return result


def _single_position(response: dict[str, Any], question: str) -> float:
    value = _answers(response, question).get("SelectedByPosition")
    if isinstance(value, list):
        if len(value) != 1:
            raise ResponseParseError(f"{question} must contain exactly one selected position")
        value = value[0]
    return _number(value, location=f"{question}.SelectedByPosition")


def _matrix_positions(
    response: dict[str, Any], question: str, *, length: int
) -> list[float]:
    value = _answers(response, question).get("SelectedByPosition")
    if not isinstance(value, list) or len(value) != length:
        raise ResponseParseError(
            f"{question}.SelectedByPosition must contain exactly {length} values"
        )
    return [
        _number(item, location=f"{question}.SelectedByPosition[{index}]")
        for index, item in enumerate(value)
    ]


def _validate_range(study: str, outcome_id: str, value: float) -> OutcomeCell:
    minimum, maximum = OUTCOME_RANGES[study][outcome_id]
    if value < minimum or value > maximum:
        raise ResponseParseError(
            f"{study}.{outcome_id}={value:g} lies outside [{minimum:g}, {maximum:g}]"
        )
    return OutcomeCell(outcome_id, float(value), minimum, maximum)


def _privacy(response: dict[str, Any]) -> list[OutcomeCell]:
    return [_validate_range("privacy", "PPV", _single_position(response, "Q1"))]


def _hiring(response: dict[str, Any]) -> list[OutcomeCell]:
    direct = {
        "Q6": "Q2",
        "Q8": "Q3",
        "Q9": "Q4",
        "Q10": "Q5",
        "Q13": "Q6",
        "Q14": "Q7",
        "Q15": "Q8",
        "Q16": "Q9",
    }
    cells: list[OutcomeCell] = []
    recode_q9 = {3.0: 1.0, 4.0: 2.0, 2.0: 3.0, 1.0: 4.0}
    recode_q14 = {1.0: 1.0, 3.0: 2.0, 4.0: 3.0, 2.0: 4.0}
    for outcome_id, prompt_id in direct.items():
        value = _single_position(response, prompt_id)
        if outcome_id == "Q9":
            if value not in recode_q9:
                raise ResponseParseError("Q9 response is not a valid four-position answer")
            value = recode_q9[value]
        elif outcome_id == "Q14":
            if value not in recode_q14:
                raise ResponseParseError("Q14 response is not a valid four-position answer")
            value = recode_q14[value]
        cells.append(_validate_range("hiring_algorithms", outcome_id, value))
    for job, prompt_id in enumerate(("Q19", "Q20", "Q21", "Q22"), 1):
        for item, value in enumerate(
            _matrix_positions(response, prompt_id, length=8), 1
        ):
            outcome_id = f"job{job}_item{item}"
            cells.append(
                _validate_range("hiring_algorithms", outcome_id, value)
            )
    return cells


def _junk_fees(survey_text: str, response: dict[str, Any]) -> list[OutcomeCell]:
    chunks = survey_question_chunks(survey_text)
    mcq = [
        key
        for key, chunk in chunks.items()
        if "which of the following do you think best represents what" in chunk.casefold()
        and "is assessed for" in chunk.casefold()
    ]
    fairness = [
        key
        for key, chunk in chunks.items()
        if "how fair do you think it is" in chunk.casefold()
        and "to charge for" in chunk.casefold()
    ]
    regulation = [
        key
        for key, chunk in chunks.items()
        if "should pricing practices be regulated by the government" in chunk.casefold()
    ]
    support = [
        key
        for key, chunk in chunks.items()
        if "support government regulation that bans firms from separating out mandatory fees"
        in chunk.casefold()
    ]
    if not (len(mcq) == len(fairness) == 6 and len(regulation) == len(support) == 1):
        raise ResponseParseError(
            "junk_fees answer-free survey does not expose the frozen 6/6/1/1 outcome map"
        )
    correct = sum(_single_position(response, key) == 1.0 for key in mcq)
    fairness_average = sum(_single_position(response, key) for key in fairness) / 6.0
    reg_support = (
        _single_position(response, regulation[0])
        + _single_position(response, support[0])
    ) / 2.0
    return [
        _validate_range("junk_fees", "percent_correct", correct / 6.0 * 100.0),
        _validate_range("junk_fees", "fairness_average", fairness_average),
        _validate_range("junk_fees", "reg_support", reg_support),
    ]


def extract_outcome_cells(
    study: str, survey_text: str, response: dict[str, Any]
) -> list[OutcomeCell]:
    if study == "privacy":
        cells = _privacy(response)
    elif study == "hiring_algorithms":
        cells = _hiring(response)
    elif study == "junk_fees":
        cells = _junk_fees(survey_text, response)
    else:
        raise ResponseParseError(f"unsupported development study {study!r}")
    ids = tuple(cell.outcome_id for cell in cells)
    if ids != EXPECTED_OUTCOMES[study]:
        raise ResponseParseError(f"{study} outcome order does not match preregistration")
    return cells


def answer_bearing_questions(survey: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect displayed questions containing an Answers member in flow order."""

    questions: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "QuestionID" in value and "Answers" in value:
                questions.append(value)
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(survey.get("Elements", []))
    return questions


def human_response_object(survey: dict[str, Any]) -> dict[str, Any]:
    """Project a sealed answer-bearing survey into the same Q1..Qn output schema."""

    response: dict[str, Any] = {}
    for index, question in enumerate(answer_bearing_questions(survey), 1):
        response[f"Q{index}"] = {
            "Question Type": (
                "Matrix" if question.get("QuestionType") == "Matrix" else "Single Choice"
            ),
            "Answers": question.get("Answers", {}),
        }
    return response
