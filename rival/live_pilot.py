from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .integrity import prepare_prediction_context
from .mathx import canonical_hash, normalize, stable_unit_interval
from .providers import (
    OpenAICompatibleProvider,
    PredictionProvider,
    ProviderPrediction,
)
from .research.datasets import TwinQuestion, load_twin2k
from .research.firewall import twin_family_split
from .research.provenance import sha256_file
from .schemas import ChoiceSpec, PopulationRecord, ProviderIdentity, ScenarioSpec


SCHEMA_VERSION = "rival.live-twin2k-pilot.v1"
DEFAULT_STUDY_ID = "rival-twin2k-live-provider-v2"
TEXT_HASH_POLICY = "sha256-utf8-lf-normalized-v1"


class PilotProtocolError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(str(files("rival").joinpath("datasets", "twin2k")))


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=[1], low_memory=False)
    frame["TWIN_ID"] = pd.to_numeric(frame["TWIN_ID"], errors="coerce")
    frame = frame.dropna(subset=["TWIN_ID"])
    frame["TWIN_ID"] = frame["TWIN_ID"].astype(np.int64)
    if frame["TWIN_ID"].duplicated().any():
        raise PilotProtocolError(f"duplicate TWIN_ID values in {path.name}")
    return frame.set_index("TWIN_ID")


def _stable_key(seed: int, *parts: Any) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_portable_text(path: Path) -> str:
    """Hash UTF-8 text with line endings normalized to LF.

    Git may check text files out as CRLF on Windows. JSONL semantics do not
    depend on the line-ending convention, so the integrity hash binds the
    normalized text while all other byte changes remain detectable.
    """

    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8", newline=None) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), ""):
            digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PilotProtocolError(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
    return rows


def _catalog_question(metadata: dict[str, Any], catalog_column: str) -> str:
    prompt = str(metadata.get("QuestionText", catalog_column)).strip()
    if str(metadata.get("QuestionType", "")).casefold() == "matrix":
        columns = [str(value) for value in metadata.get("csv_columns", [])]
        rows = [str(value) for value in metadata.get("Rows", [])]
        if catalog_column in columns:
            index = columns.index(catalog_column)
            if index < len(rows):
                prompt = f"{prompt} {rows[index]}".strip()
    return prompt


def _choice_labels(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("Options") or metadata.get("Columns") or []
    return [str(value).strip() for value in values]


@dataclass(frozen=True)
class _PredictionInputs:
    root: Path
    history: pd.DataFrame
    questions: tuple[TwinQuestion, ...]
    metadata: dict[str, dict[str, Any]]
    source_hashes: dict[str, str]
    protected_outcome_sha256: str
    information_cutoff: str
    information_cutoff_basis: str


def _load_prediction_inputs(root: str | Path | None = None) -> _PredictionInputs:
    base = _dataset_root(root)
    paths = {
        "human_history.csv": base / "human_history.csv",
        "wave4_mapping.json": base / "wave4_mapping.json",
        "question_catalog.json": base / "question_catalog.json",
    }
    outcome_path = base / "human_outcomes.csv"
    history = _read_frame(paths["human_history.csv"])
    recorded = pd.to_datetime(history.get("RecordedDate"), errors="coerce", utc=True)
    if recorded is None or not recorded.notna().any():
        recorded = pd.to_datetime(history.get("EndDate"), errors="coerce", utc=True)
    if recorded is None or not recorded.notna().any():
        # The released subset redacts collection timestamps. The paper's public
        # release month is therefore used only as an administrative upper bound;
        # wave identity, not this date, determines input/outcome eligibility.
        information_cutoff = "2025-05-31T23:59:59+00:00"
        information_cutoff_basis = (
            "public-release-month upper bound; source timestamps are redacted; "
            "waves 1-3 are inputs and wave 4 is protected"
        )
    else:
        information_cutoff = recorded.max().to_pydatetime().astimezone(timezone.utc).isoformat()
        information_cutoff_basis = "maximum released pre-wave RecordedDate/EndDate"
    mapping = json.loads(paths["wave4_mapping.json"].read_text(encoding="utf-8"))
    catalog_payload = json.loads(
        paths["question_catalog.json"].read_text(encoding="utf-8")
    )
    metadata = {str(item["QuestionID"]): item for item in catalog_payload}
    columns = [str(item["formatted_column"]) for item in mapping]
    missing = sorted(set(columns) - set(history.columns))
    if missing:
        raise PilotProtocolError(f"Twin-2K history is missing columns: {missing[:5]}")
    history = history.loc[:, columns].copy()
    for column in columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    questions: list[TwinQuestion] = []
    for item in mapping:
        column = str(item["formatted_column"])
        qid = str(item["QuestionID"])
        catalog_column = str(item["catalog_csv_column"])
        item_metadata = metadata.get(qid, {})
        values = history[column].dropna().to_numpy(dtype=float)
        unique = np.sort(np.unique(values))
        integral = bool(len(unique)) and bool(np.allclose(unique, np.round(unique)))
        kind = "categorical" if integral and len(unique) <= 12 else "continuous"
        questions.append(
            TwinQuestion(
                column=column,
                question_id=qid,
                catalog_column=catalog_column,
                text=_catalog_question(item_metadata, catalog_column),
                block=str(item_metadata.get("BlockName", qid)).strip(),
                kind=kind,  # type: ignore[arg-type]
                categories=tuple(float(value) for value in unique) if integral else tuple(),
            )
        )
    return _PredictionInputs(
        root=base,
        history=history,
        questions=tuple(questions),
        metadata=metadata,
        source_hashes={
            name: _sha256_portable_text(path) for name, path in paths.items()
        },
        protected_outcome_sha256=_sha256_portable_text(outcome_path),
        information_cutoff=information_cutoff,
        information_cutoff_basis=information_cutoff_basis,
    )


def _labels_for(question: TwinQuestion, metadata: dict[str, dict[str, Any]]) -> list[str]:
    item = metadata.get(question.question_id, {})
    labels = _choice_labels(item)
    categories = [int(value) for value in question.categories]
    if (
        labels
        and categories
        and min(categories) >= 1
        and max(categories) <= len(labels)
    ):
        return [labels[value - 1] for value in categories]
    return [f"Scale position {value:g}" for value in question.categories]


def _scenario_payload(
    study_id: str,
    question: TwinQuestion,
    metadata: dict[str, dict[str, Any]],
    cohort_size: int,
    anchor_size: int,
    seed: int,
    information_cutoff: str,
) -> dict[str, Any]:
    labels = _labels_for(question, metadata)
    choices = [
        ChoiceSpec(
            choice_id=f"value_{value:g}",
            label=label,
            description=f"Recorded survey code {value:g}",
        )
        for value, label in zip(question.categories, labels, strict=True)
    ]
    scenario = ScenarioSpec(
        scenario_id=f"{study_id}-{hashlib.sha256(question.column.encode()).hexdigest()[:12]}",
        name=f"Twin-2K held-out response: {question.column}",
        question=question.text,
        context=(
            "Predict this participant's answer in the next survey wave using only "
            "the supplied pre-wave behavioral history."
        ),
        choices=choices,
        task_type="survey",
        information_cutoff=information_cutoff,
        horizon="next survey wave",
        model_family="provider-bound-at-run-time",
        sample_size=max(20, cohort_size),
        human_anchor_size=anchor_size,
        novelty=0.5,
        seed=seed,
        intended_use="retrospective live-provider qualification",
        prohibited_inferences=[
            "clinical diagnosis",
            "employment decision",
            "credit decision",
            "individual adverse action",
        ],
        metadata={
            "dataset": "Twin-2K-500",
            "target_column": question.column,
            "target_family_excluded_from_history": True,
        },
    )
    return scenario.model_dump(mode="json")


def _history_entry(
    question: TwinQuestion,
    value: float,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels = _labels_for(question, metadata)
    values = list(question.categories)
    label = f"Scale position {value:g}"
    if value in values:
        label = labels[values.index(value)]
    return {
        "question_id": question.question_id,
        "question_column": question.column,
        "question": question.text,
        "answer_code": float(value),
        "answer_label": label,
        "source_wave": "pre-wave-4",
    }


def _person_payload(
    participant_id: int,
    variant: str,
    donor_columns: list[str],
    inputs: _PredictionInputs,
) -> dict[str, Any]:
    by_column = {question.column: question for question in inputs.questions}
    history: list[dict[str, Any]] = []
    if variant == "twin":
        for column in donor_columns:
            value = float(inputs.history.loc[participant_id, column])
            history.append(_history_entry(by_column[column], value, inputs.metadata))
    person = PopulationRecord(
        person_id=f"twin_{participant_id}",
        attributes={
            "panel": "Twin-2K-500",
            "evidence_period": "waves 1-3",
            "persona_variant": variant,
        },
        history=history,
        evidence_ids=["twin2k-pre-wave4-history"],
    )
    return person.model_dump(mode="json")


def _case_input(
    case: dict[str, Any],
    protocol: dict[str, Any],
    inputs: _PredictionInputs,
) -> tuple[PopulationRecord, ScenarioSpec]:
    target = next(
        item for item in protocol["targets"] if item["column"] == case["target_column"]
    )
    question = next(
        item for item in inputs.questions if item.column == case["target_column"]
    )
    person_payload = _person_payload(
        int(case["participant_id"]),
        str(case["variant"]),
        [str(value) for value in case["history_columns"]],
        inputs,
    )
    scenario_payload = _scenario_payload(
        str(protocol["study_id"]),
        question,
        inputs.metadata,
        int(protocol["selection"]["cohort_size"]),
        int(protocol["selection"]["anchor_size"]),
        int(protocol["selection"]["seed"]),
        inputs.information_cutoff,
    )
    if scenario_payload != target["scenario"]:
        raise PilotProtocolError(f"scenario reconstruction drift for {question.column}")
    digest = canonical_hash({"person": person_payload, "scenario": scenario_payload})
    if digest != case["input_sha256"]:
        raise PilotProtocolError(f"case reconstruction drift for {case['case_id']}")
    return (
        PopulationRecord.model_validate(person_payload),
        ScenarioSpec.model_validate(scenario_payload),
    )


def prepare_twin2k_live_pilot(
    output_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
    cohort_size: int = 50,
    target_count: int = 15,
    anchor_size: int = 10,
    history_items: int = 16,
    minimum_history_items: int = 8,
    seed: int = 20260828,
    study_id: str = DEFAULT_STUDY_ID,
) -> dict[str, Any]:
    if cohort_size < 20:
        raise ValueError("cohort_size must be at least 20")
    if target_count < 3:
        raise ValueError("target_count must be at least 3")
    if not 1 <= anchor_size < cohort_size:
        raise ValueError("anchor_size must be between 1 and cohort_size - 1")
    if not 1 <= minimum_history_items <= history_items:
        raise ValueError("minimum_history_items must be within history_items")

    inputs = _load_prediction_inputs(dataset_root)
    split = twin_family_split(inputs.questions)
    family_by_column = dict(zip(split.item_ids, split.family_ids, strict=True))
    question_by_column = {question.column: question for question in inputs.questions}

    candidates: list[TwinQuestion] = []
    for question in inputs.questions:
        labels = _labels_for(question, inputs.metadata)
        categories = list(question.categories)
        if question.kind != "categorical" or not 2 <= len(categories) <= 7:
            continue
        if len(labels) != len(categories):
            continue
        candidates.append(question)
    candidates.sort(key=lambda question: _stable_key(seed, "target", question.column))
    targets: list[TwinQuestion] = []
    used_families: set[str] = set()
    for question in candidates:
        family = family_by_column[question.column]
        if family in used_families:
            continue
        targets.append(question)
        used_families.add(family)
        if len(targets) == target_count:
            break
    if len(targets) != target_count:
        raise PilotProtocolError(
            f"only {len(targets)} independent categorical target families are eligible"
        )

    participant_ids = sorted(
        (int(value) for value in inputs.history.index),
        key=lambda value: _stable_key(seed, "participant", value),
    )[:cohort_size]
    anchors = sorted(
        participant_ids,
        key=lambda value: _stable_key(seed, "anchor", value),
    )[:anchor_size]

    documents = [question.text for question in inputs.questions]
    document_matrix = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    ).fit_transform(documents)
    similarities = cosine_similarity(document_matrix, dense_output=True)
    column_index = {
        question.column: index for index, question in enumerate(inputs.questions)
    }
    request_template_provider = OpenAICompatibleProvider(
        model="__MODEL_ID__",
        api_key="non-secret-protocol-placeholder",
        temperature=0.0,
        max_retries=1,
        history_limit=history_items,
        max_output_tokens=300,
        use_response_format=True,
    )

    target_payloads: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for target in targets:
        target_index = column_index[target.column]
        donor_indices = [
            index
            for index, question in enumerate(inputs.questions)
            if family_by_column[question.column] != family_by_column[target.column]
        ]
        donor_indices.sort(
            key=lambda index: (
                -float(similarities[target_index, index]),
                _stable_key(seed, "donor", target.column, inputs.questions[index].column),
            )
        )
        scenario = _scenario_payload(
            study_id,
            target,
            inputs.metadata,
            cohort_size,
            anchor_size,
            seed,
            inputs.information_cutoff,
        )
        labels = _labels_for(target, inputs.metadata)
        target_payloads.append(
            {
                "column": target.column,
                "question_id": target.question_id,
                "family_id": family_by_column[target.column],
                "question": target.text,
                "choice_values": list(target.categories),
                "choice_labels": labels,
                "scenario": scenario,
            }
        )
        for participant_id in participant_ids:
            donor_columns = [
                inputs.questions[index].column
                for index in donor_indices
                if math.isfinite(
                    float(inputs.history.loc[participant_id, inputs.questions[index].column])
                )
            ][:history_items]
            if len(donor_columns) < minimum_history_items:
                continue
            for variant in ("generic", "twin"):
                history_columns = donor_columns if variant == "twin" else []
                person = _person_payload(
                    participant_id, variant, history_columns, inputs
                )
                input_sha256 = canonical_hash({"person": person, "scenario": scenario})
                request_template = request_template_provider._request_payload(
                    PopulationRecord.model_validate(person),
                    ScenarioSpec.model_validate(scenario),
                )
                request_text = json.dumps(
                    request_template, sort_keys=True, separators=(",", ":")
                )
                case_id = "case_" + canonical_hash(
                    {
                        "study_id": study_id,
                        "participant_id": participant_id,
                        "target_column": target.column,
                        "variant": variant,
                    }
                )[:20]
                cases.append(
                    {
                        "case_id": case_id,
                        "participant_id": participant_id,
                        "target_column": target.column,
                        "variant": variant,
                        "history_columns": history_columns,
                        "input_sha256": input_sha256,
                        "prompt_messages_sha256": canonical_hash(
                            request_template["messages"]
                        ),
                        "estimated_input_tokens": max(1, math.ceil(len(request_text) / 3)),
                    }
                )

    selected_target_columns = [target.column for target in targets[:3]]
    selected_preflight_people = participant_ids[:5]
    for case in cases:
        case["preflight"] = bool(
            case["target_column"] in selected_target_columns
            and case["participant_id"] in selected_preflight_people
        )
    cases.sort(
        key=lambda case: (
            not case["preflight"],
            case["target_column"],
            int(case["participant_id"]),
            case["variant"],
        )
    )

    def envelope(selected_cases: list[dict[str, Any]]) -> dict[str, Any]:
        values = np.asarray(
            [int(case["estimated_input_tokens"]) for case in selected_cases],
            dtype=int,
        )
        return {
            "calls": len(selected_cases),
            "estimated_input_tokens": int(values.sum()),
            "mean_input_tokens_per_call": float(values.mean()),
            "p95_input_tokens_per_call": float(np.quantile(values, 0.95)),
            "max_input_tokens_per_call": int(values.max()),
            "maximum_output_tokens": 300 * len(selected_cases),
            "local_reservation_safety_factor": 1.25,
        }

    request_envelopes = {
        "preflight": envelope([case for case in cases if case["preflight"]]),
        "pilot": envelope(cases),
        "pilot_by_variant": {
            variant: envelope([case for case in cases if case["variant"] == variant])
            for variant in ("generic", "twin")
        },
    }

    destination = Path(output_dir)
    cases_path = destination / "cases.jsonl"
    _write_jsonl(cases_path, cases)
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "created_at": _utc_now(),
        "status": "FROZEN_BEFORE_LIVE_PROVIDER_CALLS",
        "claim_scope": (
            "Retrospective live-provider qualification on public Twin-2K data; "
            "not an independent prospective customer result."
        ),
        "dataset": {
            "name": "Twin-2K-500",
            "license": "CC BY 4.0",
            "integrity_hash_policy": TEXT_HASH_POLICY,
            "prediction_input_hashes": inputs.source_hashes,
            "protected_outcome_sha256": inputs.protected_outcome_sha256,
            "information_cutoff": inputs.information_cutoff,
            "information_cutoff_basis": inputs.information_cutoff_basis,
        },
        "selection": {
            "seed": seed,
            "cohort_size": cohort_size,
            "target_count": target_count,
            "anchor_size": anchor_size,
            "history_items": history_items,
            "minimum_history_items": minimum_history_items,
            "variants": ["generic", "twin"],
            "participant_ids": participant_ids,
            "anchor_participant_ids": anchors,
            "question_family_manifest_sha256": split.manifest_hash,
            "target_family_exclusion": True,
        },
        "targets": target_payloads,
        "cases": {
            "path": "cases.jsonl",
            "sha256": _sha256_portable_text(cases_path),
            "total": len(cases),
            "preflight": sum(bool(case["preflight"]) for case in cases),
        },
        "provider_request_policy": {
            "adapter": "OpenAICompatibleProvider",
            "temperature": 0.0,
            "history_limit": history_items,
            "max_output_tokens": 300,
            "use_response_format": True,
            "response_format_type": "json_schema",
            "reasoning_effort": "none",
            "exclude_reasoning": True,
            "require_parameters": True,
            "provider_host": "openrouter.ai",
            "prompt_messages_bound_per_case": True,
        },
        "revision": {
            "supersedes_study_id": "rival-twin2k-live-provider-v1",
            "reason": (
                "V1 compatibility calls produced zero parseable predictions. "
                "Before outcome evaluation or any successful study prediction, V2 "
                "replaced generic JSON mode with strict per-choice JSON Schema and "
                "disabled reasoning for the probability-only response."
            ),
            "integrity_hash_revision": (
                "Before any successful study prediction or outcome evaluation, "
                "text-file integrity hashes were made invariant to LF/CRLF "
                "checkout conversion. Dataset content, cohort, cases, prompts, "
                "outcomes, comparators and gates did not change."
            ),
        },
        "request_envelopes": request_envelopes,
        "preregistration": {
            "primary_metrics": [
                "individual_accuracy",
                "multiclass_brier",
                "population_tvd",
            ],
            "comparators": [
                "generic_same_model",
                "pre-wave population mode",
                "same-person test-retest",
                "released Twin-2K model",
                "equal-size human anchor",
            ],
            "gates": [
                "twin individual accuracy exceeds generic",
                "twin population TVD is lower than generic",
                "anchor-corrected twin TVD is lower than raw twin TVD",
            ],
            "all_eligible_cases_retained": True,
        },
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    _atomic_json(destination / "protocol.json", protocol)
    return protocol


def load_and_verify_protocol(
    protocol_path: str | Path,
    cases_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], _PredictionInputs]:
    path = Path(protocol_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected_protocol_hash = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    if expected_protocol_hash != canonical_hash(unsigned):
        raise PilotProtocolError("protocol hash does not verify")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise PilotProtocolError("unsupported live-pilot schema")
    if protocol.get("dataset", {}).get("integrity_hash_policy") != TEXT_HASH_POLICY:
        raise PilotProtocolError("unsupported dataset integrity hash policy")
    resolved_cases_path = Path(cases_path) if cases_path else path.parent / "cases.jsonl"
    if _sha256_portable_text(resolved_cases_path) != protocol["cases"]["sha256"]:
        raise PilotProtocolError("cases file hash does not verify")
    cases = _read_jsonl(resolved_cases_path)
    if len(cases) != int(protocol["cases"]["total"]):
        raise PilotProtocolError("cases file count does not match protocol")
    inputs = _load_prediction_inputs(dataset_root)
    if inputs.source_hashes != protocol["dataset"]["prediction_input_hashes"]:
        raise PilotProtocolError("prediction input dataset hashes drifted")
    if inputs.protected_outcome_sha256 != protocol["dataset"]["protected_outcome_sha256"]:
        raise PilotProtocolError("protected outcome file hash drifted")
    return protocol, cases, inputs


@dataclass
class BudgetGuard:
    budget_usd: float
    input_cost_per_million: float
    output_cost_per_million: float
    max_calls: int
    not_after: datetime | None = None
    safety_factor: float = 1.25
    spent_usd: float = 0.0
    successful_calls: int = 0

    def __post_init__(self) -> None:
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("model prices must be nonnegative")
        if self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        if self.not_after and self.not_after.tzinfo is None:
            self.not_after = self.not_after.replace(tzinfo=timezone.utc)

    def restore(self, successful_rows: Iterable[dict[str, Any]]) -> None:
        rows = list(successful_rows)
        self.successful_calls = len(rows)
        self.spent_usd = float(sum(float(row.get("billed_cost_usd", 0.0)) for row in rows))

    def estimate(self, provider: PredictionProvider, person: PopulationRecord, scenario: ScenarioSpec) -> tuple[int, int, float]:
        payload_builder = getattr(provider, "_request_payload", None)
        if callable(payload_builder):
            payload = payload_builder(person, scenario)
            characters = len(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            characters = len(
                json.dumps(
                    {
                        "person": person.model_dump(mode="json"),
                        "scenario": scenario.model_dump(mode="json"),
                    },
                    sort_keys=True,
                )
            )
        input_tokens = max(1, math.ceil(characters / 3))
        output_tokens = int(getattr(provider, "max_output_tokens", 300))
        estimate = self.safety_factor * (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000
        return input_tokens, output_tokens, float(estimate)

    def authorize(self, estimate_usd: float) -> None:
        now = datetime.now(timezone.utc)
        if self.not_after and now >= self.not_after.astimezone(timezone.utc):
            raise BudgetExceeded("local run authorization has expired")
        if self.successful_calls >= self.max_calls:
            raise BudgetExceeded("maximum successful call count reached")
        if self.spent_usd + estimate_usd > self.budget_usd + 1e-12:
            raise BudgetExceeded("next call would exceed the local USD budget")

    def debit(
        self,
        output: ProviderPrediction,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        estimate_usd: float,
    ) -> float:
        diagnostics = output.diagnostics
        provider_cost = diagnostics.get("provider_cost_usd")
        prompt_tokens = diagnostics.get("prompt_tokens")
        completion_tokens = diagnostics.get("completion_tokens")
        if provider_cost is not None:
            billed = float(provider_cost)
        elif prompt_tokens is not None and completion_tokens is not None:
            billed = (
                float(prompt_tokens) * self.input_cost_per_million
                + float(completion_tokens) * self.output_cost_per_million
            ) / 1_000_000
        else:
            billed = estimate_usd
        if billed < 0:
            billed = estimate_usd
        self.spent_usd += billed
        self.successful_calls += 1
        return float(billed)


class DeterministicRehearsalProvider(PredictionProvider):
    name = "deterministic-rehearsal"

    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider_name=self.name,
            provider_version="1",
            model="no-network-no-scientific-claim",
            configuration_sha256=canonical_hash({"algorithm": "stable-hash-soft-probabilities-v1"}),
        )

    def predict(self, person: PopulationRecord, scenario: ScenarioSpec) -> ProviderPrediction:
        scores = np.asarray(
            [
                0.25
                + stable_unit_interval(
                    person.person_id,
                    scenario.scenario_id,
                    choice.choice_id,
                    len(person.history),
                )
                for choice in scenario.choices
            ],
            dtype=float,
        )
        probabilities = normalize(scores)
        return ProviderPrediction(
            probabilities={
                choice.choice_id: float(value)
                for choice, value in zip(scenario.choices, probabilities, strict=True)
            },
            diagnostics={"rehearsal": 1.0},
        )


def _successful_by_case(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    successful: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "SUCCESS":
            successful[str(row["case_id"])] = row
    return successful


def run_live_pilot(
    protocol_path: str | Path,
    results_path: str | Path,
    provider: PredictionProvider,
    budget: BudgetGuard,
    *,
    phase: str = "preflight",
    cases_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    max_errors: int = 3,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    if phase not in {"preflight", "pilot"}:
        raise ValueError("phase must be 'preflight' or 'pilot'")
    protocol, cases, inputs = load_and_verify_protocol(
        protocol_path, cases_path=cases_path, dataset_root=dataset_root
    )
    selected = [case for case in cases if phase == "pilot" or case["preflight"]]
    destination = Path(results_path)
    prior_rows = _read_jsonl(destination)
    successful = _successful_by_case(prior_rows)
    provider_identity = provider.identity().model_dump(mode="json")
    policy = protocol["provider_request_policy"]
    if isinstance(provider, OpenAICompatibleProvider):
        observed_policy = provider.request_policy()
        expected_policy = {
            key: policy[key]
            for key in (
                "temperature",
                "history_limit",
                "max_output_tokens",
                "use_response_format",
                "response_format_type",
                "reasoning_effort",
                "exclude_reasoning",
                "require_parameters",
                "provider_host",
            )
        }
        if observed_policy != expected_policy:
            raise PilotProtocolError(
                f"provider request policy drift: expected {expected_policy}, got {observed_policy}"
            )
    for row in successful.values():
        if row.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise PilotProtocolError("existing result belongs to a different protocol")
        if row.get("provider") != provider_identity:
            raise PilotProtocolError("existing result belongs to a different provider identity")
    budget.restore(successful.values())

    errors = 0
    stop_reason: str | None = None
    new_successes = 0
    for case in selected:
        if case["case_id"] in successful:
            continue
        person, scenario = _case_input(case, protocol, inputs)
        if isinstance(provider, OpenAICompatibleProvider):
            messages = provider._request_payload(person, scenario)["messages"]
            if canonical_hash(messages) != case["prompt_messages_sha256"]:
                raise PilotProtocolError(
                    f"provider-visible prompt drift for {case['case_id']}"
                )
        prepared = prepare_prediction_context(
            [person], scenario, None, provider.identity()
        )
        estimated_input, estimated_output, estimated_cost = budget.estimate(
            provider, person, scenario
        )
        try:
            budget.authorize(estimated_cost)
        except BudgetExceeded as exc:
            stop_reason = str(exc)
            break
        try:
            output = provider.predict(person, scenario)
            billed_cost = budget.debit(
                output, estimated_input, estimated_output, estimated_cost
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "SUCCESS",
                "created_at": _utc_now(),
                "protocol_sha256": protocol["protocol_sha256"],
                "case_id": case["case_id"],
                "participant_id": case["participant_id"],
                "target_column": case["target_column"],
                "variant": case["variant"],
                "preflight": case["preflight"],
                "provider": provider_identity,
                "prediction_context_sha256": prepared.context.context_sha256,
                "request_sha256": provider.request_sha256(person, scenario),
                "probabilities": output.probabilities,
                "diagnostics": output.diagnostics,
                "provider_request_id": output.provider_request_id,
                "attempts": output.attempts,
                "latency_ms": output.latency_ms,
                "estimated_input_tokens": estimated_input,
                "estimated_output_tokens": estimated_output,
                "reserved_cost_usd": estimated_cost,
                "billed_cost_usd": billed_cost,
            }
            result["result_sha256"] = canonical_hash(result)
            _append_jsonl(destination, result)
            successful[str(case["case_id"])] = result
            new_successes += 1
        except Exception as exc:  # retain failures without hiding eligible cases
            errors += 1
            failure = {
                "schema_version": SCHEMA_VERSION,
                "status": "ERROR",
                "created_at": _utc_now(),
                "protocol_sha256": protocol["protocol_sha256"],
                "case_id": case["case_id"],
                "participant_id": case["participant_id"],
                "target_column": case["target_column"],
                "variant": case["variant"],
                "preflight": case["preflight"],
                "provider": provider_identity,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failure["result_sha256"] = canonical_hash(failure)
            _append_jsonl(destination, failure)
            if errors >= max_errors:
                stop_reason = f"stopped after {errors} provider errors"
                break

    complete = sum(case["case_id"] in successful for case in selected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": protocol["study_id"],
        "phase": phase,
        "status": "COMPLETE" if complete == len(selected) else "STOPPED",
        "protocol_sha256": protocol["protocol_sha256"],
        "provider": provider_identity,
        "selected_cases": len(selected),
        "successful_cases": complete,
        "new_successes": new_successes,
        "errors_this_run": errors,
        "spent_usd": budget.spent_usd,
        "budget_usd": budget.budget_usd,
        "max_calls": budget.max_calls,
        "stop_reason": stop_reason,
        "results_sha256": sha256_file(destination) if destination.exists() else None,
        "finished_at": _utc_now(),
    }
    if summary_path:
        _atomic_json(Path(summary_path), summary)
    return summary


def _distribution(values: np.ndarray, categories: list[float]) -> np.ndarray:
    result = np.asarray([np.mean(values == category) for category in categories])
    total = float(result.sum())
    return result / total if total > 0 else np.full(len(categories), np.nan)


def _tvd(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _interval(values: list[float], seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "mean": math.nan, "lower": math.nan, "upper": math.nan}
    if len(array) == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "lower": value, "upper": value}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def evaluate_live_pilot(
    protocol_path: str | Path,
    results_path: str | Path,
    *,
    cases_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    protocol, cases, _ = load_and_verify_protocol(
        protocol_path, cases_path=cases_path, dataset_root=dataset_root
    )
    rows = _read_jsonl(Path(results_path))
    successful = _successful_by_case(rows)
    case_by_id = {str(case["case_id"]): case for case in cases}
    for case_id, row in successful.items():
        if case_id not in case_by_id:
            raise PilotProtocolError(f"unknown result case {case_id}")
        if row.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise PilotProtocolError("result protocol hash mismatch")
        unsigned = dict(row)
        expected = unsigned.pop("result_sha256", None)
        if expected != canonical_hash(unsigned):
            raise PilotProtocolError(f"result hash mismatch for {case_id}")
    if not successful:
        raise PilotProtocolError("no successful predictions to evaluate")

    data = load_twin2k(dataset_root)
    anchors = set(int(value) for value in protocol["selection"]["anchor_participant_ids"])
    target_by_column = {item["column"]: item for item in protocol["targets"]}
    provider_identity = next(iter(successful.values()))["provider"]
    if any(row["provider"] != provider_identity for row in successful.values()):
        raise PilotProtocolError("results mix multiple provider identities")

    question_rows: list[dict[str, Any]] = []
    for target_column, target in target_by_column.items():
        categories = [float(value) for value in target["choice_values"]]
        choice_ids = [f"value_{value:g}" for value in categories]
        for variant in protocol["selection"]["variants"]:
            predictions: list[tuple[int, np.ndarray, float]] = []
            for case in cases:
                if case["target_column"] != target_column or case["variant"] != variant:
                    continue
                row = successful.get(str(case["case_id"]))
                if row is None:
                    continue
                participant = int(case["participant_id"])
                actual = float(data.human_outcomes.loc[participant, target_column])
                if not math.isfinite(actual) or actual not in categories:
                    continue
                probabilities = np.asarray(
                    [float(row["probabilities"][choice_id]) for choice_id in choice_ids],
                    dtype=float,
                )
                probabilities = probabilities / probabilities.sum()
                predictions.append((participant, probabilities, actual))
            anchor_rows = [item for item in predictions if item[0] in anchors]
            test_rows = [item for item in predictions if item[0] not in anchors]
            if len(anchor_rows) < 3 or len(test_rows) < 10:
                continue

            test_probabilities = np.vstack([item[1] for item in test_rows])
            test_actual = np.asarray([item[2] for item in test_rows], dtype=float)
            test_onehot = np.vstack(
                [np.asarray([float(value == actual) for value in categories]) for actual in test_actual]
            )
            raw_distribution = test_probabilities.mean(axis=0)
            observed_distribution = test_onehot.mean(axis=0)
            anchor_probabilities = np.vstack([item[1] for item in anchor_rows]).mean(axis=0)
            anchor_actual_values = np.asarray([item[2] for item in anchor_rows], dtype=float)
            human_anchor = _distribution(anchor_actual_values, categories)
            corrected = np.clip(raw_distribution + human_anchor - anchor_probabilities, 0, None)
            corrected = corrected / corrected.sum()

            history_values = pd.to_numeric(
                data.human_history.loc[[item[0] for item in test_rows], target_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            released_values = pd.to_numeric(
                data.llm_predictions.loc[[item[0] for item in test_rows], target_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            history_valid = np.isfinite(history_values) & np.isin(history_values, categories)
            released_valid = np.isfinite(released_values) & np.isin(released_values, categories)
            history_distribution = _distribution(history_values[history_valid], categories)
            released_distribution = _distribution(released_values[released_valid], categories)
            full_history = pd.to_numeric(
                data.human_history.loc[:, target_column], errors="coerce"
            ).to_numpy(dtype=float)
            full_history = full_history[np.isfinite(full_history) & np.isin(full_history, categories)]
            population_mode = float(
                categories[int(np.argmax([np.mean(full_history == value) for value in categories]))]
            )
            predicted_values = np.asarray(
                [categories[index] for index in np.argmax(test_probabilities, axis=1)]
            )
            epsilon = 1e-12
            actual_indices = np.asarray(
                [categories.index(value) for value in test_actual], dtype=int
            )
            row = {
                "target_column": target_column,
                "question_id": target["question_id"],
                "variant": variant,
                "anchor_n": len(anchor_rows),
                "test_n": len(test_rows),
                "individual_accuracy": float(np.mean(predicted_values == test_actual)),
                "multiclass_brier": float(np.mean(np.sum((test_probabilities - test_onehot) ** 2, axis=1))),
                "negative_log_likelihood": float(
                    -np.mean(np.log(np.clip(test_probabilities[np.arange(len(test_rows)), actual_indices], epsilon, 1)))
                ),
                "raw_provider_tvd": _tvd(raw_distribution, observed_distribution),
                "anchor_corrected_provider_tvd": _tvd(corrected, observed_distribution),
                "human_anchor_only_tvd": _tvd(human_anchor, observed_distribution),
                "history_distribution_tvd": _tvd(history_distribution, observed_distribution)
                if np.isfinite(history_distribution).all()
                else math.nan,
                "released_model_distribution_tvd": _tvd(released_distribution, observed_distribution)
                if np.isfinite(released_distribution).all()
                else math.nan,
                "history_test_retest_accuracy": float(
                    np.mean(history_values[history_valid] == test_actual[history_valid])
                )
                if history_valid.any()
                else math.nan,
                "released_model_accuracy": float(
                    np.mean(released_values[released_valid] == test_actual[released_valid])
                )
                if released_valid.any()
                else math.nan,
                "population_mode_accuracy": float(np.mean(test_actual == population_mode)),
                "observed_distribution": {
                    choice_id: float(value)
                    for choice_id, value in zip(choice_ids, observed_distribution, strict=True)
                },
                "raw_provider_distribution": {
                    choice_id: float(value)
                    for choice_id, value in zip(choice_ids, raw_distribution, strict=True)
                },
                "anchor_corrected_distribution": {
                    choice_id: float(value)
                    for choice_id, value in zip(choice_ids, corrected, strict=True)
                },
            }
            question_rows.append(row)

    metrics = [
        "individual_accuracy",
        "multiclass_brier",
        "negative_log_likelihood",
        "raw_provider_tvd",
        "anchor_corrected_provider_tvd",
        "human_anchor_only_tvd",
        "history_distribution_tvd",
        "released_model_distribution_tvd",
        "history_test_retest_accuracy",
        "released_model_accuracy",
        "population_mode_accuracy",
    ]
    variants: dict[str, Any] = {}
    seed = int(protocol["selection"]["seed"])
    for variant in protocol["selection"]["variants"]:
        selected_rows = [row for row in question_rows if row["variant"] == variant]
        variants[variant] = {
            metric: _interval(
                [float(row[metric]) for row in selected_rows if math.isfinite(float(row[metric]))],
                seed + index,
            )
            for index, metric in enumerate(metrics)
        }
        variants[variant]["evaluated_targets"] = len(selected_rows)

    generic = variants.get("generic", {})
    twin = variants.get("twin", {})
    complete = len(successful) == int(protocol["cases"]["total"])
    gates = {
        "twin_accuracy_exceeds_generic": (
            twin.get("individual_accuracy", {}).get("mean", math.nan)
            > generic.get("individual_accuracy", {}).get("mean", math.nan)
        ),
        "twin_population_tvd_below_generic": (
            twin.get("raw_provider_tvd", {}).get("mean", math.inf)
            < generic.get("raw_provider_tvd", {}).get("mean", math.inf)
        ),
        "anchor_corrected_twin_tvd_below_raw": (
            twin.get("anchor_corrected_provider_tvd", {}).get("mean", math.inf)
            < twin.get("raw_provider_tvd", {}).get("mean", math.inf)
        ),
    }
    rehearsal = provider_identity.get("provider_name") == DeterministicRehearsalProvider.name
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": protocol["study_id"],
        "status": (
            "REHEARSAL_ONLY"
            if rehearsal
            else "RETROSPECTIVE_COMPLETE" if complete else "PARTIAL_UNEVALUABLE"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "results_sha256": sha256_file(Path(results_path)),
        "provider": provider_identity,
        "successful_cases": len(successful),
        "expected_cases": int(protocol["cases"]["total"]),
        "variants": variants,
        "gates": gates,
        "gate_status": "PASS" if complete and all(gates.values()) else "FAIL" if complete else "UNEVALUABLE",
        "questions": question_rows,
        "limitations": [
            "This public-data benchmark is retrospective and is not customer-domain proof.",
            "The development team has previously accessed Twin-2K outcomes; external blinding is not claimed.",
            "The same-item human history baseline is unusually strong in this repeated-question panel.",
            "Question-level bootstrap intervals do not represent all possible customer domains.",
            "Rehearsal-provider results test machinery only and carry no predictive claim.",
        ],
        "generated_at": _utc_now(),
    }
    report["report_sha256"] = canonical_hash(report)
    return report


def parse_not_after(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def make_openai_provider(
    *,
    model: str,
    base_url: str | None,
    timeout_seconds: int,
    temperature: float,
    max_retries: int,
    history_limit: int,
    max_output_tokens: int,
    use_response_format: bool,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_retries=max_retries,
        history_limit=history_limit,
        max_output_tokens=max_output_tokens,
        use_response_format=use_response_format,
    )
