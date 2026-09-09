"""Prospective study cohorts and immutable train/calibrate/evaluate stages.

Local attestations bind declarations, not the truth of outcome data or the
statistical independence of study groups. L12 customer qualification is separate.
"""

from datetime import datetime
import os
from pathlib import Path
import tempfile

from .calibration_catalog import (CalibrationCatalog, calibration_identity,
    protected_observation, question_fingerprint, seed_responses)
from .confidence_evidence import ConfidenceEvidenceRegistry
from .evidence_catalog import json_bytes, read_bounded, strict_json
from .integrity import IntegrityError
from .mathx import canonical_hash
from .mega_study_v2.runner import exclusive_run_lock
from .runtime_calibration import comparison_metrics
from .runtime_uncertainty import (UncertaintySettings, conformal_bound, evaluate_policy,
                                  features, fit_error_model)
from .schemas import utc_now


def uncertainty_identity(request, prepared):
    calibration = getattr(request, "calibration", None)
    return {**calibration_identity(request, prepared),
            "prediction": "calibrated" if calibration else "raw",
            "calibration_adapter_sha256": calibration.adapter_sha256 if calibration else None,
            "context_sha256": canonical_hash(" ".join(request.brief.context.casefold().split())),
            "sample_size": request.brief.sample_size, "sampling_seed": request.brief.seed}


def member_identity(workspace):
    request = workspace.request
    return {"study_id": request.brief.study_id, "group_id": request.evidence.group_id,
            "role": request.evidence.role, "request_sha256": workspace.prepared["request_sha256"],
            "question_sha256": question_fingerprint(request)}


def prediction_values(workspace, result):
    identity = uncertainty_identity(workspace.request, workspace.prepared)
    panel = seed_responses(result, identity)
    if getattr(workspace.request, "calibration", None):
        from .study_calibration import calibrated_prediction
        distribution = calibrated_prediction(workspace, result)["calibrated_distribution"]
    else:
        distribution = result.distribution
    values = [distribution[choice] for choice in identity["choice_ids"]]
    ids = {canonical_hash(p.provider_call.provider_request_id) for p in result.predictions
           if p.provider_call and p.provider_call.provider_request_id}
    if len(ids) != len(identity["seed_ids"]):
        raise IntegrityError("uncertainty evidence requires distinct model response IDs for each seed")
    return values, features(values, panel), sorted(ids)


def reject_overlap(members):
    for key in ("study_id", "group_id", "question_sha256", "request_sha256"):
        if len({m[key] for m in members}) != len(members):
            raise ValueError(f"duplicate uncertainty {key}; studies cannot cross partitions")


class UncertaintyCatalog(CalibrationCatalog):
    schema_prefix = "rival.uncertainty"

    def plan(self, workspaces, settings):
        from .study_contract import StudyRequestV3, StudyRequestV5
        from .study_workflow import _workspace
        settings = UncertaintySettings.model_validate(settings).model_dump(mode="json")
        members, identities, references, reference_ids = [], [], [], []
        for root in workspaces:
            with _workspace(root) as w, w.execution() as session:
                if not isinstance(w.request, StudyRequestV3) or isinstance(w.request, StudyRequestV5):
                    raise ValueError("uncertainty cohorts require pinned v3/v4 studies")
                if w.request.evidence.role not in {"training", "calibration", "evaluation"}:
                    raise ValueError("assign training, calibration or evaluation roles before planning")
                if (w.simulation() is not None or w.journal_snapshot(session)["accounting"]["attempts"]
                        or w.store.last_phase_event(w.request.brief.study_id)):
                    raise ValueError("freeze the whole uncertainty cohort before any model execution or outcome reveal")
                members.append(member_identity(w))
                identities.append(uncertainty_identity(w.request, w.prepared))
                if getattr(w.request, "calibration", None):
                    adapter = w.prepared["calibration"]["artifact"]
                    references = adapter["members"]
                    reference_ids = adapter["reference_request_sha256"]
        counts = {role: sum(m["role"] == role for m in members)
                  for role in ("training", "calibration", "evaluation")}
        if counts["training"] < 5 or min(counts.values()) < 1:
            raise ValueError("cohort needs at least five training groups and separate calibration/evaluation groups")
        if len({canonical_hash(i) for i in identities}) != 1:
            raise ValueError("uncertainty cohort model, panel, choices, sampling or calibration identity differs")
        reject_overlap(members)
        for m in members:
            if any(m[key] == ref[key] for ref in references for key in ("study_id", "group_id", "question_sha256")):
                raise ValueError("uncertainty cohort overlaps distribution-calibration training evidence")
        payload = {"identity": identities[0], "settings": settings,
                   "members": sorted(members, key=lambda m: m["study_id"]),
                   "distribution_training_members": references, "distribution_request_sha256": reference_ids,
                   "qualification": "research; local custody, outcome truth and group independence remain assumptions"}
        cohort_id = canonical_hash(payload)
        # Reserve the cohort in each source workspace, so a second catalog
        # cannot try a different policy on the same studies. Interrupted
        # reservations can be resumed with the identical roster/settings.
        for root in workspaces:
            with _workspace(root) as w, w.execution() as session:
                if w.simulation() is not None or w.journal_snapshot(session)["accounting"]["attempts"]:
                    raise ValueError("cohort execution started during planning")
                w.put("uncertainty_cohort", {"cohort_sha256": cohort_id})
        result = self._stage("plan", cohort_id, lambda: {**payload, "cohort_sha256": cohort_id})
        return {**result, "groups": counts}

    def _rows(self, plan, roots, role, after, *, allow_incomplete=False):
        from .study_workflow import _workspace
        expected = {m["study_id"]: m for m in plan["members"] if m["role"] == role}
        rows = {}
        for root in roots:
            with _workspace(root) as w, w.execution() as session:
                member = member_identity(w)
                identifier = member["study_id"]
                if identifier in rows or member != expected.get(identifier):
                    raise ValueError("workspace is duplicated or differs from the frozen uncertainty roster/role")
                if w.read("uncertainty_cohort", required=True) != {"cohort_sha256": plan["cohort_sha256"]}:
                    raise IntegrityError("workspace belongs to another uncertainty cohort")
                if canonical_hash(uncertainty_identity(w.request, w.prepared)) != canonical_hash(plan["identity"]):
                    raise IntegrityError("uncertainty cohort prediction identity changed")
                if w.read("completed") is None:
                    if not allow_incomplete:
                        raise IntegrityError("training and calibration require every planned study to complete")
                    rows[identifier] = {**member, "failure": "incomplete_execution"}
                    continue
                result, sealed = w.require_complete(w.journal_snapshot(session))
                if sealed.seal.sealed_at <= datetime.fromisoformat(after):
                    raise ValueError("stage predictions must be locked after the preceding uncertainty stage was frozen")
                if w.store.last_phase_event(identifier)["to_phase"] != "evaluated":
                    if not allow_incomplete:
                        raise IntegrityError("uncertainty evidence requires protected evaluated outcomes")
                    rows[identifier] = {**member, "failure": "missing_protected_outcome"}
                    continue
                # Admission rechecks signed pre-lock role, outcomes and raw metrics.
                ConfidenceEvidenceRegistry(w.manager).admit(identifier)
                observed, available, reveal_hash = protected_observation(w, sealed)
                distribution, x, response_ids = prediction_values(w, result)
                error = comparison_metrics([distribution], [observed])["tvd"][0]
                rows[identifier] = {**member, "features": x, "observed_tvd": error,
                    "outcome_revealed_at": available, "outcome_evidence_sha256": reveal_hash,
                    "manifest_sha256": canonical_hash(sealed), "run_id": result.run_id,
                    "distribution_sha256": canonical_hash(distribution), "model_request_sha256": response_ids}
        if set(rows) != set(expected):
            raise ValueError("supply every workspace in the frozen partition; omitted studies cannot be dropped")
        return [rows[key] for key in sorted(rows)]

    @staticmethod
    def _distinct_responses(plan, *partitions):
        ids = list(plan["distribution_request_sha256"])
        runs, manifests = [], []
        for rows in partitions:
            for row in rows:
                ids.extend(row.get("model_request_sha256", []))
                if "run_id" in row:
                    runs.append(row["run_id"])
                    manifests.append(row["manifest_sha256"])
        if any(len(v) != len(set(v)) for v in (ids, runs, manifests)):
            raise IntegrityError("uncertainty partitions reused model responses, runs or manifests")

    def _stage(self, kind, parent, build):
        # One immutable successor per parent prevents rerolling a failed test in
        # this catalog. A crash before the pointer write can leave an orphan blob.
        with exclusive_run_lock(self.root.with_name(self.root.name + ".stages.lock")):
            pointer = self.root / f"stage-{kind}-{parent}.json"
            if pointer.exists():
                envelope = strict_json(read_bounded(pointer))
                if not self._signer().verify_attestation(envelope):
                    raise IntegrityError("uncertainty stage pointer signature changed")
                item = envelope["payload"]
                if item["kind"] != kind or item["parent"] != parent:
                    raise IntegrityError("uncertainty stage pointer binding changed")
                self._read(kind, item["sha256"])
                return {kind + "_sha256": item["sha256"]}
            payload = {"schema_version": f"{self.schema_prefix}-{kind}.v1",
                       **build(), "created_at": utc_now().isoformat()}
            identifier = self._write(kind, payload)
            descriptor, temporary = tempfile.mkstemp(dir=self.root, prefix="stage-")
            try:
                with os.fdopen(descriptor, "wb") as f:
                    f.write(json_bytes(self._signer().attest({"kind": kind, "parent": parent, "sha256": identifier})))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, pointer)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return {kind + "_sha256": identifier}

    def fit(self, plan_id, workspaces):
        plan = self._read("plan", plan_id)
        def build():
            rows = self._rows(plan, workspaces, "training", plan["created_at"])
            self._distinct_responses(plan, rows)
            return {"plan_sha256": plan_id, "rows": rows,
                    "model": fit_error_model([r["features"] for r in rows],
                        [r["observed_tvd"] for r in rows], plan["settings"])}
        return self._stage("fit", plan_id, build)

    def calibrate(self, fit_id, workspaces):
        fit = self._read("fit", fit_id)
        plan = self._read("plan", fit["plan_sha256"])
        def build():
            rows = self._rows(plan, workspaces, "calibration", fit["created_at"])
            self._distinct_responses(plan, fit["rows"], rows)
            bound = conformal_bound(fit["model"], [r["features"] for r in rows],
                                    [r["observed_tvd"] for r in rows], plan["settings"]["nominal_coverage"])
            return {"fit_sha256": fit_id, "rows": rows, "bound": bound}
        return self._stage("bound", fit_id, build)

    def evaluate(self, bound_id, workspaces):
        bound = self._read("bound", bound_id)
        fit = self._read("fit", bound["fit_sha256"])
        plan = self._read("plan", fit["plan_sha256"])
        def build():
            rows = self._rows(plan, workspaces, "evaluation", bound["created_at"], allow_incomplete=True)
            self._distinct_responses(plan, fit["rows"], bound["rows"], rows)
            metrics = evaluate_policy(fit["model"], bound["bound"],
                                      [None if "failure" in r else r for r in rows], plan["settings"])
            return {"bound_sha256": bound_id, "rows": rows, "metrics": metrics,
                    "customer_qualified": False,
                    "qualification": "L11 statistical assessment only; L12 untouched domain qualification remains required"}
        return self._stage("assessment", bound_id, build)

    def load_assessment(self, identifier):
        assessment = self._read("assessment", identifier)
        bound = self._read("bound", assessment["bound_sha256"])
        fit = self._read("fit", bound["fit_sha256"])
        plan = self._read("plan", fit["plan_sha256"])
        self._distinct_responses(plan, fit["rows"], bound["rows"], assessment["rows"])
        return {"assessment": assessment, "bound": bound, "fit": fit, "plan": plan}


def bind_uncertainty(request, prepared, catalog_root):
    if catalog_root is None:
        raise ValueError("v5 studies require an uncertainty catalog at preparation")
    identifier = request.uncertainty.assessment_sha256
    artifact = UncertaintyCatalog(catalog_root).load_assessment(identifier)
    plan = artifact["plan"]
    if canonical_hash(uncertainty_identity(request, prepared)) != canonical_hash(plan["identity"]):
        raise ValueError("uncertainty model, panel, choice, sampling or calibration identity changed")
    question = question_fingerprint(request)
    for member in plan["members"] + plan["distribution_training_members"]:
        if (member["study_id"] == request.brief.study_id or member["group_id"] == request.evidence.group_id
                or member["question_sha256"] == question):
            raise ValueError("target overlaps uncertainty or distribution-calibration evidence")
    if datetime.fromisoformat(artifact["assessment"]["created_at"]) > request.brief.information_cutoff:
        raise ValueError("uncertainty assessment was created after the target information cutoff")
    return {"assessment_sha256": identifier, "artifact_sha256": canonical_hash(artifact), "artifact": artifact}
