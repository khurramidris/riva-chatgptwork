"""Locally attested reference banks harvested from protected training studies."""

from datetime import datetime
import os
from pathlib import Path
import re
import secrets
import tempfile

import numpy as np

from .evidence_catalog import MAX_BYTES, json_bytes, read_bounded, strict_json
from .integrity import IntegrityError, ManifestSigner
from .mathx import canonical_hash
from .mega_study_v2.runner import exclusive_run_lock
from .runtime_calibration import (ADAPTER_VERSION, apply_ensemble, comparison_metrics,
                                  fit_ensemble, probability_array)
from .schemas import utc_now


def question_fingerprint(request):
    return canonical_hash({"question": " ".join(request.brief.question.casefold().split()),
                           "context": " ".join(request.brief.context.casefold().split()),
                           "choices": request.brief.choices})


def calibration_identity(request, prepared):
    if prepared["planned_unique_seeds"] != prepared["eligible_seed_records"]:
        raise ValueError("fixed-panel calibration requires every eligible seed to be sampled; increase sample_size")
    entries = sorted((entry["person_id"], entry["provider_visible_sha256"])
                     for entry in prepared["retrieval_audit"]["entries"])
    return {"provider": prepared["prediction_context"]["provider"],
            "choices_sha256": canonical_hash(request.brief.choices),
            "choice_ids": [choice.choice_id for choice in request.brief.choices],
            "task_type": request.brief.task_type,
            "audience_sha256": canonical_hash(request.audience),
            "imports_sha256": canonical_hash(request.imports),
            "support_sha256": canonical_hash(request.support),
            "visible_panel_sha256": canonical_hash(entries),
            "seed_ids": [entry[0] for entry in entries]}


def seed_responses(result, identity):
    choices = identity["choice_ids"]
    by_seed = {}
    for prediction in result.predictions:
        seed = prediction.person_id.rsplit("__draw_", 1)[0]
        if set(prediction.probabilities) != set(choices):
            raise IntegrityError("seed response choices differ from the calibration panel")
        values = [prediction.probabilities[choice] for choice in choices]
        if seed in by_seed and by_seed[seed] != values:
            raise IntegrityError("repeated draws disagree for a calibration seed")
        by_seed[seed] = values
    if set(by_seed) != set(identity["seed_ids"]):
        raise IntegrityError("calibration seed panel is incomplete or changed")
    return probability_array([by_seed[seed] for seed in identity["seed_ids"]], 2)


def protected_observation(workspace, sealed):
    study_id = workspace.request.brief.study_id
    if (not workspace.manager.verify(sealed)
            or workspace.store.last_phase_event(study_id)["to_phase"] != "evaluated"):
        raise IntegrityError("calibration evidence requires a protected evaluation")
    reveals = [event for event in workspace.store.phase_events(study_id)
               if event["to_phase"] == "outcomes_revealed"]
    if len(reveals) != 1:
        raise IntegrityError("calibration needs exactly one authenticated outcome reveal")
    evidence = workspace.store.phase_evidence(reveals[0]["payload_sha256"])
    observed = workspace.manager._validate_reveal(sealed, evidence)
    choices = [choice.choice_id for choice in workspace.request.brief.choices]
    if set(observed) != set(choices):
        raise IntegrityError("calibration outcomes must contain the exact declared choice set")
    values = probability_array([[observed[choice] for choice in choices]], 2)[0]
    return values, evidence["payload"]["receipt"]["revealed_at"], reveals[0]["payload_sha256"]


def _harvest(root):
    from .study_workflow import _workspace
    with _workspace(root) as workspace, workspace.execution() as session:
        request = workspace.request
        if request.schema_version != "rival.study-request.v3":
            raise ValueError("reference banks require uncalibrated pinned v3 studies")
        if request.evidence.role != "training":
            raise ValueError("only prospectively assigned training studies may enter a reference bank")
        snapshot = workspace.journal_snapshot(session)
        result, sealed = workspace.require_complete(snapshot)
        from .confidence_evidence import ConfidenceEvidenceRegistry
        _, assignment = ConfidenceEvidenceRegistry(workspace.manager)._assignment(request.brief.study_id)
        if datetime.fromisoformat(assignment["payload"]["assigned_at"]) > sealed.seal.sealed_at:
            raise IntegrityError("training role was assigned after prediction lock")
        identity = calibration_identity(request, workspace.prepared)
        responses = seed_responses(result, identity)
        observed, available, reveal_hash = protected_observation(workspace, sealed)
        request_ids = sorted({prediction.provider_call.provider_request_id for prediction in result.predictions
                              if prediction.provider_call and prediction.provider_call.provider_request_id})
        if len(request_ids) != len(identity["seed_ids"]):
            raise IntegrityError("reference study needs distinct returned model request IDs for every seed")
        return identity, {"study_id": request.brief.study_id, "group_id": request.evidence.group_id,
            "role": request.evidence.role, "question_sha256": question_fingerprint(request),
            "request_sha256": workspace.prepared["request_sha256"], "run_id": result.run_id,
            "manifest_sha256": canonical_hash(sealed), "outcome_evidence_sha256": reveal_hash,
            "outcome_revealed_at": available, "responses": responses.tolist(), "observed": observed.tolist(),
            "model_request_sha256": [canonical_hash(identifier) for identifier in request_ids],
            "attempts": snapshot["accounting"]["attempts"],
            "known_api_cost_usd": snapshot["accounting"]["total_cost_usd"]}


class CalibrationCatalog:
    schema_prefix = "rival.calibration"

    def __init__(self, root):
        self.root = Path(root).resolve()

    def _signer(self, *, create=False):
        if not self.root.exists() and create:
            self.root.mkdir(parents=True, mode=0o700)
            with os.fdopen(os.open(self.root / "catalog.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as file:
                file.write(secrets.token_bytes(64))
                file.flush()
                os.fsync(file.fileno())
        key = self.root / "catalog.key"
        if not key.is_file():
            raise IntegrityError("calibration catalog key is missing; restore the original catalog")
        return ManifestSigner(key.read_bytes())

    def _read(self, kind, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise ValueError("calibration identifiers must be SHA-256 hashes")
        envelope = strict_json(read_bounded(self.root / f"{kind}-{identifier}.json"))
        if (not self._signer().verify_attestation(envelope)
                or canonical_hash(envelope["payload"]) != identifier
                or envelope["payload"].get("schema_version") != f"{self.schema_prefix}-{kind}.v1"):
            raise IntegrityError("calibration artifact hash or local signature is invalid")
        return envelope["payload"]

    def _write(self, kind, payload):
        identifier = canonical_hash(payload)
        with exclusive_run_lock(self.root.with_name(self.root.name + ".lock")):
            signer = self._signer(create=True)
            target = self.root / f"{kind}-{identifier}.json"
            if target.exists():
                self._read(kind, identifier)
                return identifier
            content = json_bytes(signer.attest(payload))
            if len(content) > MAX_BYTES:
                raise ValueError("calibration artifact exceeds the 50 MiB limit")
            descriptor, temporary = tempfile.mkstemp(prefix="calibration-", dir=self.root)
            try:
                with os.fdopen(descriptor, "wb") as file:
                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return identifier

    def create_bank(self, workspaces):
        if len(workspaces) < 2:
            raise ValueError("a reference bank needs at least two distinct training study groups")
        # Sequential reads release workspace locks before acquiring another.
        harvested = [_harvest(root) for root in workspaces]
        identity = harvested[0][0]
        if any(canonical_hash(item[0]) != canonical_hash(identity) for item in harvested):
            raise ValueError("reference studies have different model, choices, evidence or visible seed panels")
        members = sorted([item[1] for item in harvested], key=lambda item: item["study_id"])
        for key in ("study_id", "group_id", "question_sha256", "run_id", "manifest_sha256"):
            if len({member[key] for member in members}) != len(members):
                raise ValueError(f"duplicate reference {key}; repeated studies cannot add evidence")
        ids = [identifier for member in members for identifier in member["model_request_sha256"]]
        if len(ids) != len(set(ids)):
            raise ValueError("reference studies reused model response IDs")
        payload = {"schema_version": "rival.calibration-bank.v1", "identity": identity, "members": members,
                   "verification_scope": "harvested from locally verified training ledgers; origin and group independence remain declarations"}
        identifier = self._write("bank", payload)
        return {"bank_sha256": identifier, "training_groups": len(members), "seed_records": len(identity["seed_ids"])}

    def fit(self, bank_id, settings=None):
        bank = self._read("bank", bank_id)
        members = bank["members"]
        responses = probability_array([item["responses"] for item in members], 3)
        observed = probability_array([item["observed"] for item in members], 2)
        fit = fit_ensemble(responses, observed, settings)
        calibrated = apply_ensemble(responses, fit)
        payload = {"schema_version": "rival.calibration-adapter.v1", "bank_sha256": bank_id,
            "identity": bank["identity"], "fit": fit,
            "members": [{key: value for key, value in member.items()
                         if key not in {"responses", "observed", "model_request_sha256"}} for member in members],
            "reference_request_sha256": sorted(identifier for member in members for identifier in member["model_request_sha256"]),
            "historical_mean": observed.mean(axis=0).tolist(),
            "training_metrics": {"uniform_panel": comparison_metrics(responses.mean(axis=1), observed),
                                 "calibrated": comparison_metrics(calibrated, observed)},
            "qualification": "unqualified; training fit is not held-out evidence"}
        identifier = self._write("adapter", payload)
        return {"adapter_sha256": identifier, "bank_sha256": bank_id, "training_groups": len(members),
                "diagnostics": fit["diagnostics"], "qualification": payload["qualification"]}

    def load_adapter(self, identifier):
        adapter = self._read("adapter", identifier)
        bank = self._read("bank", adapter["bank_sha256"])
        if canonical_hash(bank["identity"]) != canonical_hash(adapter["identity"]):
            raise IntegrityError("calibration adapter identity differs from its reference bank")
        if adapter["fit"]["adapter_version"] != ADAPTER_VERSION:
            raise ValueError("unsupported calibration adapter version")
        return adapter


def bind_calibration(request, prepared, catalog_root):
    if catalog_root is None:
        raise ValueError("v4 studies require the calibration catalog at preparation")
    identifier = request.calibration.adapter_sha256
    adapter = CalibrationCatalog(catalog_root).load_adapter(identifier)
    identity = calibration_identity(request, prepared)
    if canonical_hash(identity) != canonical_hash(adapter["identity"]):
        raise ValueError("calibration model, choice schema, audience evidence or visible seed panel changed")
    question = question_fingerprint(request)
    for member in adapter["members"]:
        if (member["study_id"] == request.brief.study_id or member["group_id"] == request.evidence.group_id
                or member["question_sha256"] == question):
            raise ValueError("target study overlaps a reference training study or group")
        if datetime.fromisoformat(member["outcome_revealed_at"]) > request.brief.information_cutoff:
            raise ValueError("reference outcomes were revealed after the target information cutoff")
    return {"adapter_sha256": identifier, "artifact": adapter, "bound_at": utc_now().isoformat()}
