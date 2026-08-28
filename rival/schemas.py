from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceSource(StrictModel):
    source_id: str = Field(default_factory=lambda: f"src_{uuid4().hex[:16]}")
    name: str
    source_type: Literal[
        "public", "licensed", "customer", "participant", "synthetic", "outcome"
    ]
    collected_at: datetime = Field(default_factory=utc_now)
    period_start: str | None = None
    period_end: str | None = None
    geography: list[str] = Field(default_factory=list)
    sample_frame: str | None = None
    rights_reference: str
    permitted_uses: list[str] = Field(default_factory=list)
    prohibited_uses: list[str] = Field(default_factory=list)
    contains_personal_data: bool = False
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PopulationRecord(StrictModel):
    person_id: str
    weight: float = Field(default=1.0, gt=0)
    attributes: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, float] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class PopulationTargets(StrictModel):
    controls: dict[str, dict[str, float]]
    label: str = "target population"

    @field_validator("controls")
    @classmethod
    def validate_controls(cls, controls: dict[str, dict[str, float]]):
        if not controls:
            raise ValueError("at least one population control is required")
        for feature, categories in controls.items():
            if not categories or sum(categories.values()) <= 0:
                raise ValueError(f"control {feature!r} must have positive mass")
            if any(value < 0 for value in categories.values()):
                raise ValueError(f"control {feature!r} contains a negative target")
        return controls


class PopulationDiagnostics(StrictModel):
    converged: bool
    iterations: int
    max_absolute_margin_error: float
    effective_sample_size: float
    effective_sample_ratio: float
    marginal_errors: dict[str, dict[str, float]]
    unsupported_categories: dict[str, list[str]] = Field(default_factory=dict)


class ChoiceSpec(StrictModel):
    choice_id: str
    label: str
    description: str = ""
    features: dict[str, float] = Field(default_factory=dict)


class ScenarioSpec(StrictModel):
    scenario_id: str = Field(default_factory=lambda: f"scn_{uuid4().hex[:16]}")
    name: str
    question: str
    context: str
    choices: list[ChoiceSpec] = Field(min_length=2)
    task_type: Literal["survey", "choice", "message", "pricing"] = "choice"
    population_filter: dict[str, Any] = Field(default_factory=dict)
    geography: list[str] = Field(default_factory=list)
    information_cutoff: str | None = None
    horizon: str = "immediate"
    model_family: str = "heuristic"
    interaction_mode: Literal["independent"] = "independent"
    sample_size: int = Field(default=1000, ge=20, le=100_000)
    human_anchor_size: int = Field(default=0, ge=0, le=10_000)
    novelty: float = Field(default=0.35, ge=0, le=1)
    seed: int = 20260827
    intended_use: str = "research"
    prohibited_inferences: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_choices(self):
        choice_ids = [choice.choice_id for choice in self.choices]
        if len(choice_ids) != len(set(choice_ids)):
            raise ValueError("choice_id values must be unique")
        return self


class ProviderIdentity(StrictModel):
    provider_name: str
    provider_version: str
    model: str
    endpoint_sha256: str | None = None
    configuration_sha256: str


class ProviderCallIdentity(StrictModel):
    provider: ProviderIdentity
    request_sha256: str
    cache_key: str
    provider_request_id: str | None = None
    attempts: int = Field(default=1, ge=1)
    latency_ms: float = Field(default=0.0, ge=0)
    cache_hit: bool = False


class RetrievalAuditEntry(StrictModel):
    person_id: str
    included_history_count: int = Field(ge=0)
    excluded_history_count: int = Field(ge=0)
    provider_visible_sha256: str
    excluded_sha256: str
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)


class RetrievalAudit(StrictModel):
    policy_version: str = "rival.outcome-firewall.v1"
    information_cutoff: str | None = None
    entries: list[RetrievalAuditEntry]
    audit_sha256: str
    outcome_keys_detected: list[str] = Field(default_factory=list)
    passed: bool = True


class PredictionContext(StrictModel):
    schema_version: str = "rival.prediction-context.v1"
    context_id: str = Field(default_factory=lambda: f"ctx_{uuid4().hex[:16]}")
    created_at: datetime = Field(default_factory=utc_now)
    scenario_id: str
    scenario_sha256: str
    population_sha256: str
    targets_sha256: str | None = None
    retrieval_audit_sha256: str
    provider: ProviderIdentity
    code_version: str
    information_cutoff: str | None = None
    outcome_free: bool = True
    context_sha256: str


class AgentPrediction(StrictModel):
    person_id: str
    probabilities: dict[str, float]
    sampled_choice: str
    weight: float
    provider: str
    diagnostics: dict[str, float] = Field(default_factory=dict)
    provider_call: ProviderCallIdentity | None = None


class ConfidenceAssessment(StrictModel):
    label: Literal["high", "medium", "low"]
    expected_tvd: float
    lower_tvd: float
    upper_tvd: float
    abstain: bool
    reason: str
    training_examples: int = 0
    features: dict[str, float] = Field(default_factory=dict)


class SimulationResult(StrictModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:16]}")
    created_at: datetime = Field(default_factory=utc_now)
    scenario: ScenarioSpec
    distribution: dict[str, float]
    sample_counts: dict[str, int]
    population_size: int
    effective_sample_size: float
    average_entropy: float
    provider: str
    population_diagnostics: PopulationDiagnostics | None = None
    predictions: list[AgentPrediction] = Field(default_factory=list)
    confidence: ConfidenceAssessment | None = None
    prediction_context: PredictionContext | None = None
    retrieval_audit: RetrievalAudit | None = None
    lineage_hash: str
    warnings: list[str] = Field(default_factory=list)


class HumanObservation(StrictModel):
    person_id: str
    observed_choice: str
    synthetic_probabilities: dict[str, float]
    weight: float = Field(default=1.0, gt=0)
    subgroup: str | None = None


class EstimateInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    standard_error: float


class HybridResult(StrictModel):
    synthetic_distribution: dict[str, float]
    corrected_distribution: dict[str, float]
    residual_adjustment: dict[str, float]
    intervals: dict[str, EstimateInterval]
    human_sample_size: int
    effective_human_sample_size: float
    warnings: list[str] = Field(default_factory=list)


class EvaluationResult(StrictModel):
    evaluation_id: str = Field(default_factory=lambda: f"eval_{uuid4().hex[:16]}")
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    observed_distribution: dict[str, float]
    metrics: dict[str, float]
    subgroup_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    preregistration_hash: str | None = None
    outcome_available_at: datetime | None = None


StudyPhase = Literal[
    "draft", "prediction_locked", "outcomes_revealed", "evaluated"
]


class PreregistrationSpec(StrictModel):
    schema_version: str = "rival.preregistration.v1"
    primary_metrics: list[str] = Field(default_factory=lambda: ["tvd"])
    acceptance_thresholds: dict[str, float] = Field(default_factory=dict)
    subgroup_keys: list[str] = Field(default_factory=list)
    evaluation_protocol: str = "rival.distribution-evaluation.v1"
    outcome_not_before: datetime | None = None
    notes: str = ""


class StudyManifest(StrictModel):
    schema_version: str = "rival.study-manifest.v1"
    manifest_id: str = Field(default_factory=lambda: f"mft_{uuid4().hex[:16]}")
    study_id: str
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    phase: Literal["prediction_locked"] = "prediction_locked"
    prediction_context: PredictionContext
    preregistration: PreregistrationSpec
    predictions_sha256: str
    simulation_sha256: str


class ManifestSeal(StrictModel):
    algorithm: Literal["HMAC-SHA256"] = "HMAC-SHA256"
    key_id: str
    digest: str
    sealed_at: datetime = Field(default_factory=utc_now)


class SealedStudyManifest(StrictModel):
    manifest: StudyManifest
    seal: ManifestSeal


class PhaseEvent(StrictModel):
    schema_version: str = "rival.phase-event.v1"
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:16]}")
    study_id: str
    from_phase: StudyPhase
    to_phase: StudyPhase
    created_at: datetime = Field(default_factory=utc_now)
    payload_sha256: str
    previous_event_sha256: str | None = None
    event_sha256: str


class OutcomeRevealReceipt(StrictModel):
    schema_version: str = "rival.outcome-reveal.v1"
    study_id: str
    manifest_sha256: str
    outcome_sha256: str
    revealed_at: datetime = Field(default_factory=utc_now)
