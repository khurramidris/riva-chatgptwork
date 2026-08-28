from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .mathx import canonical_hash
from .schemas import (
    EvaluationResult,
    ManifestSeal,
    PhaseEvent,
    OutcomeRevealReceipt,
    PopulationRecord,
    PopulationTargets,
    PredictionContext,
    PreregistrationSpec,
    ProviderIdentity,
    RetrievalAudit,
    RetrievalAuditEntry,
    ScenarioSpec,
    SealedStudyManifest,
    SimulationResult,
    StudyManifest,
    StudyPhase,
    utc_now,
)
from .store import EvidenceStore
from .version import __version__


class IntegrityError(RuntimeError):
    """Base class for prospective-integrity failures."""


class OutcomeFirewallError(IntegrityError):
    """Raised when provider-visible inputs contain protected outcomes."""


class LockedContextMismatch(IntegrityError):
    """Raised before provider calls when a locked context no longer matches."""


class ManifestVerificationError(IntegrityError):
    """Raised when a sealed study manifest does not verify."""


_FORBIDDEN_KEYS = {
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
_TIMESTAMP_KEYS = ("occurred_at", "recorded_at", "collected_at", "timestamp", "date")


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _forbidden_paths(value: Any, path: str) -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _normalize_key(raw_key)
            child_path = f"{path}.{raw_key}"
            if (
                key in _FORBIDDEN_KEYS
                or key.endswith("_ground_truth")
                or key.startswith("outcome_")
            ):
                matches.append(child_path)
            matches.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return matches


def _parse_instant(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    instant = datetime.fromisoformat(normalized)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def _history_instant(item: dict[str, Any]) -> datetime | None:
    for key in _TIMESTAMP_KEYS:
        raw = item.get(key)
        if raw is not None:
            try:
                return _parse_instant(str(raw))
            except ValueError as exc:
                raise OutcomeFirewallError(
                    f"history timestamp {key!r} is not valid ISO-8601"
                ) from exc
    return None


@dataclass(frozen=True, slots=True)
class PreparedPredictionContext:
    records: list[PopulationRecord]
    context: PredictionContext
    audit: RetrievalAudit


def _context_payload(context: PredictionContext) -> dict[str, Any]:
    return {
        "schema_version": context.schema_version,
        "scenario_id": context.scenario_id,
        "scenario_sha256": context.scenario_sha256,
        "population_sha256": context.population_sha256,
        "targets_sha256": context.targets_sha256,
        "retrieval_audit_sha256": context.retrieval_audit_sha256,
        "provider": context.provider.model_dump(mode="json"),
        "code_version": context.code_version,
        "information_cutoff": context.information_cutoff,
        "outcome_free": context.outcome_free,
    }


def prediction_context_digest(context: PredictionContext) -> str:
    return canonical_hash(_context_payload(context))


def prepare_prediction_context(
    records: list[PopulationRecord],
    scenario: ScenarioSpec,
    targets: PopulationTargets | None,
    provider: ProviderIdentity,
) -> PreparedPredictionContext:
    """Create the exact outcome-free input boundary used for prediction."""

    protected: list[str] = []
    protected.extend(_forbidden_paths(scenario.metadata, "scenario.metadata"))
    sanitized: list[PopulationRecord] = []
    entries: list[RetrievalAuditEntry] = []
    cutoff = _parse_instant(scenario.information_cutoff) if scenario.information_cutoff else None

    for record in records:
        protected.extend(_forbidden_paths(record.attributes, f"records.{record.person_id}.attributes"))
        protected.extend(_forbidden_paths(record.preferences, f"records.{record.person_id}.preferences"))
        protected.extend(_forbidden_paths(record.history, f"records.{record.person_id}.history"))
        included: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        reasons: dict[str, int] = {}
        for item in record.history:
            instant = _history_instant(item)
            if cutoff is not None and instant is not None and instant > cutoff:
                excluded.append(item)
                reasons["after_information_cutoff"] = (
                    reasons.get("after_information_cutoff", 0) + 1
                )
            else:
                included.append(item)
        clean = record.model_copy(update={"history": included}, deep=True)
        sanitized.append(clean)
        provider_visible = {
            "attributes": clean.attributes,
            "preferences": clean.preferences,
            "history": clean.history,
        }
        entries.append(
            RetrievalAuditEntry(
                person_id=record.person_id,
                included_history_count=len(included),
                excluded_history_count=len(excluded),
                provider_visible_sha256=canonical_hash(provider_visible),
                excluded_sha256=canonical_hash(excluded),
                exclusion_reasons=reasons,
            )
        )

    if protected:
        locations = sorted(set(protected))
        preview = ", ".join(locations[:5])
        suffix = "" if len(locations) <= 5 else f" (+{len(locations) - 5} more)"
        raise OutcomeFirewallError(
            f"protected outcome fields are provider-visible: {preview}{suffix}"
        )

    audit_payload = {
        "policy_version": "rival.outcome-firewall.v1",
        "information_cutoff": scenario.information_cutoff,
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "outcome_keys_detected": [],
        "passed": True,
    }
    audit = RetrievalAudit(
        information_cutoff=scenario.information_cutoff,
        entries=entries,
        audit_sha256=canonical_hash(audit_payload),
    )
    context_stub = PredictionContext(
        context_id="ctx_pending",
        scenario_id=scenario.scenario_id,
        scenario_sha256=canonical_hash(scenario),
        population_sha256=canonical_hash(
            [record.model_dump(mode="json") for record in sanitized]
        ),
        targets_sha256=canonical_hash(targets) if targets is not None else None,
        retrieval_audit_sha256=audit.audit_sha256,
        provider=provider,
        code_version=__version__,
        information_cutoff=scenario.information_cutoff,
        context_sha256="pending",
    )
    digest = prediction_context_digest(context_stub)
    context = context_stub.model_copy(
        update={"context_id": f"ctx_{digest[:16]}", "context_sha256": digest}
    )
    return PreparedPredictionContext(sanitized, context, audit)


def verify_locked_context(
    locked: PredictionContext, expected: PredictionContext
) -> None:
    locked_digest = prediction_context_digest(locked)
    if locked.context_sha256 != locked_digest:
        raise LockedContextMismatch("locked prediction context is internally inconsistent")
    if locked.context_id != f"ctx_{locked_digest[:16]}":
        raise LockedContextMismatch("locked prediction context ID does not match its digest")
    if locked.context_sha256 != expected.context_sha256:
        raise LockedContextMismatch(
            "current inputs, retrieval policy, provider, or code do not match the locked context"
        )


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


class ManifestSigner:
    """Deployment-local symmetric seal for immutable study manifests."""

    def __init__(self, secret: bytes | str, key_id: str = "rival-manifest-v1"):
        self.secret = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(self.secret) < 32:
            raise ValueError("manifest secret must contain at least 32 bytes")
        self.key_id = key_id

    @staticmethod
    def _seal_payload(
        manifest: StudyManifest, key_id: str, sealed_at: datetime
    ) -> dict[str, Any]:
        return {
            "manifest": manifest.model_dump(mode="json"),
            "seal_metadata": {
                "algorithm": "HMAC-SHA256",
                "key_id": key_id,
                "sealed_at": sealed_at.isoformat(),
            },
        }

    def seal(self, manifest: StudyManifest) -> SealedStudyManifest:
        sealed_at = utc_now()
        digest = hmac.new(
            self.secret,
            _canonical_bytes(self._seal_payload(manifest, self.key_id, sealed_at)),
            hashlib.sha256,
        ).hexdigest()
        return SealedStudyManifest(
            manifest=manifest,
            seal=ManifestSeal(
                key_id=self.key_id, digest=digest, sealed_at=sealed_at
            ),
        )

    def verify(self, sealed: SealedStudyManifest) -> bool:
        if sealed.seal.key_id != self.key_id:
            return False
        expected = hmac.new(
            self.secret,
            _canonical_bytes(
                self._seal_payload(
                    sealed.manifest, sealed.seal.key_id, sealed.seal.sealed_at
                )
            ),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, sealed.seal.digest)

    def require_valid(self, sealed: SealedStudyManifest) -> None:
        if not self.verify(sealed):
            raise ManifestVerificationError("study manifest seal is invalid")


def _phase_event(
    study_id: str,
    from_phase: StudyPhase,
    to_phase: StudyPhase,
    payload_sha256: str,
    previous_event_sha256: str | None,
) -> PhaseEvent:
    event = PhaseEvent(
        study_id=study_id,
        from_phase=from_phase,
        to_phase=to_phase,
        payload_sha256=payload_sha256,
        previous_event_sha256=previous_event_sha256,
        event_sha256="pending",
    )
    payload = event.model_dump(mode="json", exclude={"event_sha256"})
    return event.model_copy(update={"event_sha256": canonical_hash(payload)})


class ProspectiveStudyManager:
    """Locks predictions and advances the append-only prospective phase chain."""

    def __init__(self, store: EvidenceStore, signer: ManifestSigner):
        self.store = store
        self.signer = signer

    def lock_prediction(
        self,
        simulation: SimulationResult,
        preregistration: PreregistrationSpec,
    ) -> SealedStudyManifest:
        context = simulation.prediction_context
        if context is None or not context.outcome_free:
            raise IntegrityError("simulation has no outcome-free prediction context")
        if prediction_context_digest(context) != context.context_sha256:
            raise IntegrityError("simulation prediction context is inconsistent")
        predictions_sha256 = canonical_hash(
            {
                "distribution": simulation.distribution,
                "predictions": simulation.predictions,
            }
        )
        manifest = StudyManifest(
            study_id=simulation.scenario.scenario_id,
            run_id=simulation.run_id,
            prediction_context=context,
            preregistration=preregistration,
            predictions_sha256=predictions_sha256,
            simulation_sha256=canonical_hash(simulation),
        )
        sealed = self.signer.seal(manifest)
        self.store.save_manifest(sealed)
        previous = self.store.last_phase_event(manifest.study_id)
        from_phase: StudyPhase = previous["to_phase"] if previous else "draft"
        if from_phase != "draft":
            raise IntegrityError(f"study is already in phase {from_phase!r}")
        event = _phase_event(
            manifest.study_id,
            "draft",
            "prediction_locked",
            canonical_hash(sealed),
            previous["event_sha256"] if previous else None,
        )
        self.store.append_phase_event(event)
        return sealed

    def _stored_manifest(self, study_id: str) -> SealedStudyManifest:
        payload = self.store.manifest_for_study(study_id)
        if payload is None:
            raise IntegrityError("study has no sealed manifest")
        sealed = SealedStudyManifest.model_validate(payload)
        self.signer.require_valid(sealed)
        return sealed

    def record_outcome_reveal(
        self, study_id: str, receipt: OutcomeRevealReceipt
    ) -> PhaseEvent:
        previous = self.store.last_phase_event(study_id)
        if not previous or previous["to_phase"] != "prediction_locked":
            raise IntegrityError("outcomes can only be revealed after prediction lock")
        sealed = self._stored_manifest(study_id)
        if receipt.study_id != study_id:
            raise IntegrityError("outcome receipt belongs to a different study")
        if receipt.manifest_sha256 != canonical_hash(sealed):
            raise IntegrityError("outcome receipt is not bound to the sealed manifest")
        event = _phase_event(
            study_id,
            "prediction_locked",
            "outcomes_revealed",
            canonical_hash(receipt),
            previous["event_sha256"],
        )
        self.store.append_phase_event(event)
        return event

    def record_evaluation(
        self, study_id: str, evaluation: EvaluationResult
    ) -> PhaseEvent:
        previous = self.store.last_phase_event(study_id)
        if not previous or previous["to_phase"] != "outcomes_revealed":
            raise IntegrityError("evaluation can only follow outcome reveal")
        sealed = self._stored_manifest(study_id)
        if evaluation.run_id != sealed.manifest.run_id:
            raise IntegrityError("evaluation run does not match the sealed manifest")
        expected_preregistration = canonical_hash(sealed.manifest.preregistration)
        if evaluation.preregistration_hash != expected_preregistration:
            raise IntegrityError("evaluation preregistration hash does not match")
        event = _phase_event(
            study_id,
            "outcomes_revealed",
            "evaluated",
            canonical_hash(evaluation),
            previous["event_sha256"],
        )
        self.store.append_phase_event(event)
        return event

    def verify(self, sealed: SealedStudyManifest) -> bool:
        return self.signer.verify(sealed) and self.store.verify_phase_chain(
            sealed.manifest.study_id
        )
