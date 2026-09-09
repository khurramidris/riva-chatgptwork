"""Frozen qualification linked to Rival's protected study workspaces.

Source truth, costs outside the API journal, the host clock and semantic group
independence remain custodian declarations. Local signatures do not attest them.
"""

from datetime import datetime

from .calibration_catalog import _harvest, calibration_identity, protected_observation
from .confidence_evidence import ConfidenceEvidenceRegistry
from .integrity import IntegrityError
from .mathx import canonical_hash
from .qualification_baselines import fit_baselines, predict_baselines, summarize_sample
from .qualification_contract import (BaselineSettings, HumanSample, OperationEvidence,
    QualificationProtocol, TrainingDesign)
from .schemas import utc_now
from .uncertainty_catalog import UncertaintyCatalog, member_identity, prediction_values, reject_overlap


def _same_roster(rows, expected):
    if len(rows) != len({r["study_id"] for r in rows}) or {r["study_id"] for r in rows} != set(expected):
        raise ValueError("provide each frozen study exactly once")


def _reserved(workspace, plan):
    if workspace.read("qualification_cohort", required=True) != {"cohort_sha256": plan["cohort_sha256"]}:
        raise IntegrityError("workspace belongs to a different qualification protocol")
    member = member_identity(workspace)
    expected = next((m for m in plan["members"] if m["study_id"] == member["study_id"]), None)
    if expected != member:
        raise IntegrityError("workspace differs from the frozen qualification roster")


class QualificationCatalog(UncertaintyCatalog):
    schema_prefix = "rival.qualification"

    def fit(self, workspaces, design, settings):
        settings = BaselineSettings.model_validate(settings).model_dump(mode="json")
        design = [TrainingDesign.model_validate(d).model_dump(mode="json") for d in design]
        harvested = [_harvest(root) for root in workspaces]
        if len(harvested) < 3:
            raise ValueError("classical baselines require at least three protected training study groups")
        identities, members = zip(*harvested, strict=True)
        if len({canonical_hash(i) for i in identities}) != 1:
            raise ValueError("baseline training model, audience, choices or evidence differ")
        reject_overlap(members)
        ids = [i for m in members for i in m["model_request_sha256"]]
        if len(set(ids)) != len(ids):
            raise ValueError("baseline training studies reused model responses")
        _same_roster(design, [m["study_id"] for m in members])
        members = sorted(members, key=lambda m: m["study_id"])
        by_id = {d["study_id"]: d for d in design}
        payload = {"identity": identities[0], "members": members,
                   "design": [by_id[m["study_id"]] for m in members], "settings": settings}
        # The fit specification is content addressed before optimization. Never
        # refit these coefficients during target evaluation.
        return self._stage("baselines", canonical_hash(payload), lambda: {**payload,
            "model": fit_baselines([by_id[m["study_id"]]["features"] for m in members],
                [m["observed"] for m in members], [by_id[m["study_id"]]["weight"] for m in members], settings)})

    def plan(self, workspaces, protocol, baselines_id):
        from .study_workflow import _workspace
        from .qualification_report import design_feasibility
        protocol = QualificationProtocol.model_validate(protocol)
        baselines = self._read("baselines", baselines_id)
        declared = {m.study_id: m for m in protocol.members}
        members, pins = [], []
        for root in workspaces:
            with _workspace(root) as w, w.execution() as session:
                if w.request.schema_version != "rival.study-request.v5" or w.request.evidence.role != "evaluation":
                    raise ValueError("qualification requires prepared evaluation-role v5 studies")
                if not getattr(w.request, "calibration", None):
                    raise ValueError("qualification compares the calibrated Rival hybrid with raw synthetic predictions")
                if (w.simulation() is not None or w.journal_snapshot(session)["accounting"]["attempts"]
                        or w.store.last_phase_event(w.request.brief.study_id)):
                    raise ValueError("freeze qualification before any target execution or outcome reveal")
                member = member_identity(w)
                if member["study_id"] not in declared:
                    raise ValueError("workspace is outside the declared qualification roster")
                spec = declared[member["study_id"]]
                if (protocol.audience != w.request.audience.description
                        or protocol.geography != w.request.audience.geography):
                    raise ValueError("qualification audience and geography must match the prepared study audience")
                predict_baselines(baselines["model"], spec.features)
                if canonical_hash(calibration_identity(w.request, w.prepared)) != canonical_hash(baselines["identity"]):
                    raise ValueError("qualification targets differ from the classical training identity")
                if any(member[k] == r[k] for r in baselines["members"] for k in ("study_id", "group_id", "question_sha256")):
                    raise ValueError("qualification target overlaps classical baseline training evidence")
                uq = w.prepared["uncertainty"]["artifact"]
                if (protocol.max_tvd != uq["plan"]["settings"]["max_tvd"]
                        or protocol.required_coverage > uq["plan"]["settings"]["nominal_coverage"]):
                    raise ValueError("qualification must preserve the pinned error policy and nominal coverage limit")
                if spec.timing == "prospective" and spec.fieldwork_starts_at <= utc_now():
                    raise ValueError("prospective fieldwork must start after the qualification plan is frozen")
                members.append(member)
                pins.append({"calibration": w.request.calibration.adapter_sha256,
                             "uncertainty": w.request.uncertainty.assessment_sha256})
        _same_roster(members, declared)
        reject_overlap(members)
        if len({canonical_hash(p) for p in pins}) != 1:
            raise ValueError("qualification must use one frozen hybrid and uncertainty policy")
        payload = {"protocol": protocol.model_dump(mode="json"), "baselines_sha256": baselines_id,
                   "members": sorted(members, key=lambda m: m["study_id"]), "pins": pins[0]}
        cohort_id = canonical_hash(payload)
        # Reserve every source, also across separate catalogs. Identical partial
        # reservations can resume; coordinated execution starts after return.
        for root in workspaces:
            with _workspace(root) as w, w.execution() as session:
                if (w.simulation() is not None or w.journal_snapshot(session)["accounting"]["attempts"]
                        or w.store.last_phase_event(w.request.brief.study_id)):
                    raise ValueError("qualification execution started during planning")
                w.put("qualification_cohort", {"cohort_sha256": cohort_id})
        return {**self._stage("plan", cohort_id, lambda: {**payload, "cohort_sha256": cohort_id}),
                "design_feasibility": design_feasibility(protocol)}

    def seal(self, plan_id, root, anchors):
        from .study_workflow import _workspace
        plan = self._read("plan", plan_id)
        baseline = self._read("baselines", plan["baselines_sha256"])
        anchors = HumanSample.model_validate(anchors).model_dump(mode="json")
        with _workspace(root) as w, w.execution() as session:
            _reserved(w, plan)
            spec = next(m for m in plan["protocol"]["members"] if m["study_id"] == w.request.brief.study_id)
            if anchors["source_reference"] != spec["source_reference"]:
                raise ValueError("anchor sample source differs from the frozen study source")
            sample = summarize_sample(anchors, spec["anchor_units"], baseline["identity"]["choice_ids"], spec["unit_weights"])
            result, sealed = w.require_complete(w.journal_snapshot(session))
            if sealed.seal.sealed_at <= datetime.fromisoformat(plan["created_at"]):
                raise IntegrityError("qualification predictions predate the frozen plan")
            previous = w.read("qualification_prediction")
            if previous is not None:
                if previous["plan_sha256"] != plan_id or previous["anchor_sample_sha256"] != canonical_hash(anchors):
                    raise IntegrityError("cannot replace sealed qualification predictions or human anchors")
                self._verify_prediction(w, plan_id, plan, result, sealed, previous)
                return {"prediction_sha256": canonical_hash(previous), "reused": True}
            if w.store.last_phase_event(w.request.brief.study_id)["to_phase"] != "prediction_locked":
                raise ValueError("seal all qualification baselines before outcome reveal")
            choices = baseline["identity"]["choice_ids"]
            candidate, _, response_ids = prediction_values(w, result)
            references = {i for m in baseline["members"] for i in m["model_request_sha256"]}
            if references.intersection(response_ids):
                raise IntegrityError("qualification reused classical-training model responses")
            prediction = {"plan_sha256": plan_id, "study_id": w.request.brief.study_id,
                "created_at": utc_now().isoformat(), "manifest_sha256": canonical_hash(sealed),
                "candidate": candidate, "baselines": {**predict_baselines(baseline["model"], spec["features"]),
                    "synthetic_only": [result.distribution[c] for c in choices], "human_only": sample["distribution"]},
                "anchor_sample_sha256": canonical_hash(anchors), "anchor_summary": sample,
                "uncertainty": w.read("uncertainty_prediction", required=True)["research_assessment"],
                "model_request_sha256": response_ids}
            w.put("qualification_prediction", prediction)
            return {"prediction_sha256": canonical_hash(prediction), "reused": False}

    def _verify_prediction(self, w, plan_id, plan, result, sealed, prediction):
        baseline = self._read("baselines", plan["baselines_sha256"])
        spec = next(m for m in plan["protocol"]["members"] if m["study_id"] == w.request.brief.study_id)
        candidate, _, response_ids = prediction_values(w, result)
        expected = {**predict_baselines(baseline["model"], spec["features"]),
            "synthetic_only": [result.distribution[c] for c in baseline["identity"]["choice_ids"]],
            "human_only": prediction["anchor_summary"]["distribution"]}
        checks = {"plan_sha256": plan_id, "study_id": w.request.brief.study_id,
            "manifest_sha256": canonical_hash(sealed), "candidate": candidate, "baselines": expected,
            "uncertainty": w.read("uncertainty_prediction", required=True)["research_assessment"],
            "model_request_sha256": response_ids}
        if any(canonical_hash(prediction.get(k)) != canonical_hash(v) for k, v in checks.items()):
            raise IntegrityError("sealed qualification predictions differ from their protected derivation")
        created = datetime.fromisoformat(prediction["created_at"])
        if created < sealed.seal.sealed_at or sealed.seal.sealed_at <= datetime.fromisoformat(plan["created_at"]):
            raise IntegrityError("qualification prediction timing does not verify")
        return spec

    def assess(self, plan_id, workspaces, outcomes, operations):
        """Finalize once; every planned group remains, even when its workspace is omitted."""
        from .study_workflow import _workspace
        from .qualification_report import build_assessment
        plan = self._read("plan", plan_id)
        expected = {m["study_id"] for m in plan["members"]}
        roots = {}
        for root in workspaces:
            with _workspace(root) as w:
                identifier = w.request.brief.study_id
                if identifier in roots or identifier not in expected:
                    raise ValueError("duplicate or unplanned qualification workspace")
                roots[identifier] = root
        if not isinstance(outcomes, dict) or not isinstance(operations, dict) or (set(outcomes) | set(operations)) - expected:
            raise ValueError("outcomes and operations must be keyed only by planned study IDs")
        def build():
            rows = []
            used_responses = set()
            for identifier in sorted(expected):
                if identifier not in roots:
                    rows.append({"study_id": identifier, "failure": "missing_workspace"})
                    continue
                with _workspace(roots[identifier]) as w, w.execution() as session:
                    _reserved(w, plan)
                    snapshot = w.journal_snapshot(session)
                    row = {"study_id": identifier, "accounting": snapshot["accounting"]}
                    if w.read("completed") is None:
                        rows.append({**row, "failure": "incomplete_execution"})
                        continue
                    result, sealed = w.require_complete(snapshot)
                    prediction = w.read("qualification_prediction")
                    if prediction is None:
                        rows.append({**row, "failure": "missing_pre_outcome_baselines"})
                        continue
                    spec = self._verify_prediction(w, plan_id, plan, result, sealed, prediction)
                    row["prediction"] = prediction
                    if used_responses.intersection(prediction["model_request_sha256"]):
                        raise IntegrityError("qualification target studies reused model responses")
                    used_responses.update(prediction["model_request_sha256"])
                    if w.store.last_phase_event(identifier)["to_phase"] != "evaluated":
                        rows.append({**row, "failure": "missing_protected_outcome"})
                        continue
                    ConfidenceEvidenceRegistry(w.manager).admit(identifier)
                    observed, revealed_at, reveal_hash = protected_observation(w, sealed)
                    if datetime.fromisoformat(prediction["created_at"]) > datetime.fromisoformat(revealed_at):
                        raise IntegrityError("qualification baselines were sealed after outcomes")
                    if identifier not in outcomes:
                        rows.append({**row, "failure": "missing_human_denominator"})
                        continue
                    sample = HumanSample.model_validate(outcomes[identifier])
                    if sample.source_reference != spec["source_reference"]:
                        raise ValueError("evaluation sample source differs from the frozen source")
                    summary = summarize_sample(sample, spec["evaluation_units"],
                        [c.choice_id for c in w.request.brief.choices], spec["unit_weights"])
                    if summary["distribution"] is None:
                        rows.append({**row, "failure": "no_observed_human_responses", "human_sample": summary})
                        continue
                    if canonical_hash(observed.tolist()) != canonical_hash(summary["distribution"]):
                        raise IntegrityError("human sample does not reproduce the exact protected outcome distribution")
                    operation = (OperationEvidence.model_validate(operations[identifier]).model_dump(mode="json")
                                 if identifier in operations else None)
                    completes = [e for e in w.history() if e["state"] == "complete"]
                    end = datetime.fromisoformat(completes[0]["at"]) if completes else sealed.seal.sealed_at
                    rows.append({**row, "prediction": prediction, "human_sample": summary,
                        "observed": observed.tolist(), "operation": operation,
                        "candidate_turnaround_seconds": max(0.0, (end - datetime.fromisoformat(plan["created_at"])).total_seconds()),
                        "outcome_evidence_sha256": reveal_hash, "outcome_sample_sha256": canonical_hash(sample),
                        "prospective_timing_verified": (spec["timing"] == "prospective"
                            and sealed.seal.sealed_at < datetime.fromisoformat(spec["fieldwork_starts_at"])
                            and datetime.fromisoformat(revealed_at) >= datetime.fromisoformat(spec["outcomes_available_at"]))})
            baseline = self._read("baselines", plan["baselines_sha256"])
            return {"plan_sha256": plan_id, "rows": rows,
                    "report": build_assessment(plan, baseline, rows)}
        return self._stage("assessment", plan_id, build)

    def load_assessment(self, identifier):
        from .qualification_report import build_assessment
        assessment = self._read("assessment", identifier)
        plan = self._read("plan", assessment["plan_sha256"])
        baseline = self._read("baselines", plan["baselines_sha256"])
        if canonical_hash(assessment["report"]) != canonical_hash(build_assessment(plan, baseline, assessment["rows"])):
            raise IntegrityError("qualification report does not reproduce from its frozen evidence")
        return assessment
