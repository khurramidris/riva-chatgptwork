"""Outcome-free contracts for a fixed, aggregate qualification experiment."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from .schemas import StrictModel


Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Number = Annotated[StrictFloat, Field(allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]
Rate = Annotated[Number, Field(gt=0, lt=1)]
Text = Annotated[str, Field(min_length=1, max_length=4000, pattern=r"\S")]


class BaselineSettings(StrictModel):
    feature_names: list[Text] = Field(min_length=1, max_length=32)
    feature_provenance: Text
    relevance: Text
    ridge: Positive = 1.0
    max_iter: StrictInt = Field(default=1000, ge=1, le=10000)
    tolerance: Positive = 1e-8

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("baseline feature names must be unique")
        return self


class TrainingDesign(StrictModel):
    study_id: Text
    features: list[Number] = Field(min_length=1, max_length=32)
    weight: Positive


class QualificationMember(StrictModel):
    study_id: Text
    features: list[Number] = Field(min_length=1, max_length=32)
    source_reference: Text
    instrument_sha256: Hash
    rights_reference: Text
    evidence_origin: Literal["human", "development", "generated"]
    outcome_access: Literal["uninspected", "inspected"]
    timing: Literal["historical", "prospective"]
    fieldwork_starts_at: datetime
    outcomes_available_at: datetime
    independence_rationale: Text
    anchor_units: list[Hash] = Field(min_length=1, max_length=10000)
    evaluation_units: list[Hash] = Field(min_length=1, max_length=100000)
    unit_weights: dict[Hash, Positive]

    @model_validator(mode="after")
    def valid(self):
        for value in (self.fieldwork_starts_at, self.outcomes_available_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("qualification dates require an explicit timezone")
        if self.outcomes_available_at < self.fieldwork_starts_at:
            raise ValueError("outcomes cannot predate fieldwork")
        units = self.anchor_units + self.evaluation_units
        if len(set(units)) != len(units):
            raise ValueError("anchor and evaluation units must be unique and disjoint")
        if set(self.unit_weights) != set(units):
            raise ValueError("freeze the weight of every anchor and evaluation unit")
        return self


class QualificationProtocol(StrictModel):
    schema_version: Literal["rival.qualification-protocol.v1"] = "rival.qualification-protocol.v1"
    study_family: Text
    audience: Text
    geography: list[Text] = Field(min_length=1)
    intended_decision: Text
    threshold_rationale: Text
    subgroup_scope: Text
    # This increment qualifies aggregate errors only. No implicit subgroup claim.
    priority_subgroups: list[Text] = Field(default_factory=list)
    max_tvd: Rate
    minimum_baseline_improvement: Rate
    required_coverage: Rate
    minimum_acceptance_rate: Rate
    max_bad_acceptance_rate: Rate
    max_failure_rate: Rate
    max_missing_rate: Rate
    family_error_rate: Rate = 0.05
    budget_per_study_usd: Positive
    human_unit_cost_usd: Positive
    max_turnaround_seconds: Positive
    minimum_historical_groups: StrictInt = Field(default=3, ge=3)
    minimum_prospective_groups: StrictInt = Field(default=1, ge=1)
    members: list[QualificationMember] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def roster(self):
        from decimal import Decimal
        quota = int(Decimal(str(self.budget_per_study_usd)) / Decimal(str(self.human_unit_cost_usd)))
        if quota < 1:
            raise ValueError("comparison budget cannot fund a human response")
        if any(len(m.anchor_units) != quota for m in self.members):
            raise ValueError("human-only roster must use the full prespecified response quota at the same budget cap")
        ids = [m.study_id for m in self.members]
        units = [u for m in self.members for u in m.anchor_units + m.evaluation_units]
        if len(set(ids)) != len(ids) or len(set(units)) != len(units):
            raise ValueError("qualification studies and human units cannot repeat across independent groups")
        return self


class HumanResponse(StrictModel):
    unit_sha256: Hash
    choice_id: str | None
    weight: Positive


class HumanSample(StrictModel):
    source_reference: Text
    rows: list[HumanResponse] = Field(min_length=1, max_length=100000)


class OperationEvidence(StrictModel):
    # API charges are read from the journal, never supplied by the caller.
    candidate_other_cost_usd: Nonnegative
    human_only_total_cost_usd: Nonnegative
    human_only_turnaround_seconds: Nonnegative
    receipt_reference: Text
