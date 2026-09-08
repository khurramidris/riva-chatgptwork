"""Versioned inputs for an independent choice-distribution study."""

from datetime import datetime
import json
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from .schemas import (ChoiceSpec, EvidenceSource, PopulationRecord, PopulationTargets,
                      PreregistrationSpec, ScenarioSpec, StrictModel)
from .evidence_catalog import EvidenceImportManifest, records_digest
from .mathx import canonical_hash
from .study_support import StudySupportPolicy, validate_filters


class StudyBrief(StrictModel):
    study_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    question: str = Field(min_length=1)
    context: str = ""
    choices: list[ChoiceSpec] = Field(min_length=2, max_length=20)
    task_type: Literal["survey", "choice", "message"] = "choice"
    information_cutoff: datetime
    sample_size: StrictInt = Field(default=1000, ge=20, le=100_000)
    seed: StrictInt = 20260827

    @field_validator("information_cutoff")
    @classmethod
    def aware_cutoff(cls, value):
        if value.tzinfo is None:
            raise ValueError("information_cutoff must include a timezone")
        return value

    @field_validator("study_id", "title", "decision", "question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("study brief fields cannot be blank")
        return value


class StudyAudience(StrictModel):
    description: str = Field(min_length=1)
    geography: list[str] = Field(min_length=1)
    records: list[PopulationRecord] = Field(min_length=1)
    sources: list[EvidenceSource] = Field(min_length=1)
    filters: dict = Field(default_factory=dict)
    targets: PopulationTargets | None = None

    @field_validator("sources", mode="before")
    @classmethod
    def explicit_sources(cls, value):
        for source in value:
            if isinstance(source, dict) and not {"source_id", "collected_at"} <= source.keys():
                raise ValueError("sources need stable source_id and collected_at values")
        return value

    @model_validator(mode="after")
    def source_links(self):
        if not self.description.strip() or any(not item.strip() for item in self.geography):
            raise ValueError("audience description and geography cannot be blank")
        ids = [record.person_id for record in self.records]
        if any(not identifier.strip() for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("audience person IDs must be nonempty and unique")
        sources = {source.source_id: source for source in self.sources}
        if len(sources) != len(self.sources):
            raise ValueError("source IDs must be unique")
        for source in self.sources:
            if not source.source_id.strip() or not source.rights_reference.strip():
                raise ValueError("sources need an ID and a rights reference")
            if source.source_type == "outcome":
                raise ValueError("outcome sources cannot be prediction inputs")
            if source.collected_at.tzinfo is None:
                raise ValueError("source collection dates must include a timezone")
            if "simulation" not in source.permitted_uses or "simulation" in source.prohibited_uses:
                raise ValueError("source declarations must permit simulation")
        for record in self.records:
            if not record.evidence_ids or not set(record.evidence_ids) <= sources.keys():
                raise ValueError("each audience record must reference declared evidence sources")
        for key, condition in self.filters.items():
            if isinstance(condition, dict) and (not condition or not set(condition) <= {"min", "max"}):
                raise ValueError(f"unsupported audience filter for {key}")
        return self


class StudyEvidenceRole(StrictModel):
    role: Literal["development", "training", "calibration", "evaluation"] = "development"
    group_id: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)

    @field_validator("group_id", "source_reference")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("evidence declarations cannot be blank")
        return value


class StudyExecution(StrictModel):
    mode: Literal["offline", "managed"] = "offline"
    model: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)
    max_retries: StrictInt = Field(default=3, ge=1, le=10)
    timeout_seconds: StrictInt = Field(default=60, ge=1, le=600)
    history_limit: StrictInt = Field(default=16, ge=0)
    max_output_tokens: StrictInt = Field(default=300, ge=1)
    budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    reservation_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_attempts: StrictInt | None = Field(default=None, ge=1)
    not_after: datetime | None = None

    @model_validator(mode="after")
    def explicit_limits(self):
        managed = (self.model, self.base_url, self.budget_usd, self.reservation_usd,
                   self.max_attempts, self.not_after)
        if self.mode == "managed":
            if any(value is None for value in managed) or not self.model.strip():
                raise ValueError("managed execution needs model, endpoint, budget, reservation, attempt limit and expiry")
            if self.not_after.tzinfo is None:
                raise ValueError("execution expiry must include a timezone")
            if self.reservation_usd > self.budget_usd:
                raise ValueError("the attempt reservation exceeds the entire budget")
        elif any(value is not None for value in managed):
            raise ValueError("offline studies cannot declare managed execution settings")
        return self


class StudyRequest(StrictModel):
    schema_version: Literal["rival.study-request.v1"] = "rival.study-request.v1"
    brief: StudyBrief
    audience: StudyAudience
    evidence: StudyEvidenceRole
    execution: StudyExecution = Field(default_factory=StudyExecution)
    preregistration: PreregistrationSpec = Field(default_factory=PreregistrationSpec)

    @model_validator(mode="after")
    def validate_boundary(self):
        # Includes arbitrary nested attributes/history, where NaN could otherwise
        # survive validation and be silently converted during JSON serialization.
        json.dumps(self.model_dump(mode="python"), default=str, allow_nan=False)
        if any(not choice.choice_id.strip() or not choice.label.strip() for choice in self.brief.choices):
            raise ValueError("choices need nonempty IDs and labels")
        if any(source.collected_at > self.brief.information_cutoff for source in self.audience.sources):
            raise ValueError("source collection is later than the declared information cutoff")
        if self.preregistration.evaluation_protocol != "rival.distribution-evaluation.v2":
            raise ValueError("unsupported evaluation protocol")
        if self.preregistration.outcome_not_before and self.preregistration.outcome_not_before.tzinfo is None:
            raise ValueError("outcome availability must include a timezone")
        self.scenario()  # Reuse the core choice validation before preparing work.
        return self

    def scenario(self) -> ScenarioSpec:
        return ScenarioSpec(
            scenario_id=self.brief.study_id, name=self.brief.title,
            question=self.brief.question, context=self.brief.context,
            choices=self.brief.choices, task_type=self.brief.task_type,
            population_filter=self.audience.filters, geography=self.audience.geography,
            information_cutoff=self.brief.information_cutoff.isoformat(),
            model_family="managed" if self.execution.mode == "managed" else "heuristic",
            sample_size=self.brief.sample_size, seed=self.brief.seed, human_anchor_size=0,
            intended_use="research",
        )


class StudyRequestV2(StudyRequest):
    schema_version: Literal["rival.study-request.v2"] = "rival.study-request.v2"
    imports: list[EvidenceImportManifest] = Field(min_length=1)
    support: StudySupportPolicy

    @model_validator(mode="after")
    def bound_imports(self):
        sources = {source.source_id: source for source in self.audience.sources}
        declared = [bundle.spec.source.source_id for bundle in self.imports]
        if len(set(declared)) != len(declared) or set(declared) != set(sources):
            raise ValueError("each audience source must have exactly one pinned import")
        for bundle in self.imports:
            source_id = bundle.spec.source.source_id
            if canonical_hash(sources[source_id]) != canonical_hash(bundle.source()):
                raise ValueError("source declaration differs from its import manifest")
            records = [record for record in self.audience.records if source_id in record.evidence_ids]
            if any(record.evidence_ids != [source_id] for record in records):
                raise ValueError("imported records must belong to exactly one source")
            if len(records) != bundle.spec.expected_records or records_digest(records) != bundle.records_sha256:
                raise ValueError("audience records differ from the normalized import; select subsets using filters")
        if self.support.geography_attribute in self.audience.filters:
            raise ValueError("declare geography in audience.geography; its attribute filter is generated automatically")
        validate_filters(self.audience.filters, self.audience.records)
        return self

    def scenario(self):
        scenario = super().scenario()
        return scenario.model_copy(update={
            "population_filter": {**scenario.population_filter, self.support.geography_attribute: self.audience.geography},
            "metadata": {"support_policy_sha256": canonical_hash(self.support),
                         "evidence_imports_sha256": canonical_hash(self.imports)},
        })


def parse_study_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("study input must be a JSON object")
    model = StudyRequestV2 if payload.get("schema_version") == "rival.study-request.v2" else StudyRequest
    return model.model_validate(payload)


def load_study_request(path) -> StudyRequest | StudyRequestV2:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate study input key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite study input: {value}")

    return parse_study_request(json.loads(path.read_text(encoding="utf-8"),
        object_pairs_hook=unique, parse_constant=invalid_constant))
