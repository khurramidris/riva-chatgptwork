from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field, field_validator, model_validator

from .mathx import canonical_hash
from .schemas import PopulationRecord, StrictModel


_PROTECTED_KEYS = {
    "actual_outcome",
    "actual_result",
    "ground_truth",
    "human_outcome",
    "observed_choice",
    "observed_outcome",
    "outcome",
    "outcomes",
    "post_treatment_outcome",
    "protected_outcome",
    "wave4_outcome",
}


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _protected_paths(value: Any, path: str = "transcript") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if normalized in _PROTECTED_KEYS or normalized.endswith("_ground_truth") or normalized.startswith("outcome_"):
                found.append(child_path)
            found.extend(_protected_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_protected_paths(child, f"{path}[{index}]"))
    return found


class InterviewTurn(StrictModel):
    speaker: Literal["interviewer", "participant", "system"]
    text: str = Field(min_length=1)
    occurred_at: str | None = None
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("interview text cannot be blank")
        return cleaned


class InterviewTranscript(StrictModel):
    person_id: str
    turns: list[InterviewTurn] = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, float] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_protected_outcomes(self):
        payload = self.model_dump(mode="json")
        matches = _protected_paths(payload)
        if matches:
            raise ValueError(
                "protected outcome fields are not allowed in persona inputs: "
                + ", ".join(matches[:5])
            )
        return self


class InterviewPersonaBuilder:
    """Convert rich pre-outcome interviews into auditable Rival person state."""

    def __init__(self, maximum_turns: int = 80, participant_only: bool = False):
        if maximum_turns < 1:
            raise ValueError("maximum_turns must be positive")
        self.maximum_turns = int(maximum_turns)
        self.participant_only = bool(participant_only)

    def build(self, transcript: InterviewTranscript) -> PopulationRecord:
        selected = transcript.turns[-self.maximum_turns :]
        if self.participant_only:
            selected = [turn for turn in selected if turn.speaker == "participant"]
        history = [
            {
                "source": "interview",
                "speaker": turn.speaker,
                "text": turn.text,
                **({"date": turn.occurred_at} if turn.occurred_at else {}),
                **({"topic": turn.topic} if turn.topic else {}),
            }
            for turn in selected
        ]
        transcript_hash = canonical_hash(transcript.model_dump(mode="json"))
        attributes = dict(transcript.attributes)
        attributes["interview_transcript_sha256"] = transcript_hash
        attributes["interview_turn_count"] = len(selected)
        return PopulationRecord(
            person_id=transcript.person_id,
            attributes=attributes,
            preferences=transcript.preferences,
            history=history,
            evidence_ids=transcript.evidence_ids,
        )

    def build_many(self, transcripts: Sequence[InterviewTranscript]) -> list[PopulationRecord]:
        records = [self.build(item) for item in transcripts]
        identifiers = [item.person_id for item in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("interview person_id values must be unique")
        return records


def load_interview_jsonl(path: str | Path) -> list[InterviewTranscript]:
    source = Path(path)
    transcripts = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            transcripts.append(InterviewTranscript.model_validate(json.loads(raw)))
        except Exception as exc:
            raise ValueError(f"invalid interview JSONL at line {line_number}: {exc}") from exc
    return transcripts


def load_interview_csv(path: str | Path) -> list[InterviewTranscript]:
    """Load long-form CSV columns: person_id, speaker, text, occurred_at, topic."""

    grouped: dict[str, list[InterviewTurn]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"person_id", "speaker", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("interview CSV requires person_id, speaker, and text columns")
        for row_number, row in enumerate(reader, 2):
            try:
                grouped[str(row["person_id"])].append(
                    InterviewTurn(
                        speaker=row["speaker"],
                        text=row["text"],
                        occurred_at=row.get("occurred_at") or None,
                        topic=row.get("topic") or None,
                    )
                )
            except Exception as exc:
                raise ValueError(f"invalid interview CSV at row {row_number}: {exc}") from exc
    return [
        InterviewTranscript(person_id=person_id, turns=turns)
        for person_id, turns in sorted(grouped.items())
    ]

