"""Prospectively assigned study evidence for research confidence fitting.

Custodians declare independent study groups; signatures establish provenance,
not statistical independence. This register never qualifies a fitted model.
"""

import json

from .confidence import ConfidenceModel
from .integrity import IntegrityError, ProspectiveStudyManager
from .mathx import canonical_hash
from .schemas import utc_now


class ConfidenceEvidenceRegistry:
    def __init__(self, manager: ProspectiveStudyManager):
        self.manager = manager
        self.connection = manager.store.connection
        self.connection.execute("""CREATE TABLE IF NOT EXISTS confidence_assignments (
            study_id TEXT PRIMARY KEY, group_id TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL, assignment TEXT NOT NULL,
            evidence TEXT, run_id TEXT UNIQUE
        )""")
        self.connection.commit()

    def assign(self, study_id: str, *, group_id: str, role: str, source_reference: str) -> None:
        if role not in {"training", "calibration", "evaluation"}:
            raise ValueError("role must be training, calibration or evaluation")
        if not group_id.strip() or not source_reference.strip():
            raise ValueError("independent study group and source reference are required")
        if self.manager.store.last_phase_event(study_id):
            raise IntegrityError("assign evidence roles before prediction lock and outcome reveal")
        payload = {"study_id": study_id, "group_id": group_id, "role": role,
                   "source_reference": source_reference, "assigned_at": utc_now().isoformat()}
        envelope = self.manager.signer.attest(payload)
        # Immutable unique study/group IDs prevent repeat evaluations from
        # increasing the training count or crossing evidence partitions.
        with self.connection:
            self.connection.execute("INSERT INTO confidence_assignments VALUES (?, ?, ?, ?, NULL, NULL)",
                                    (study_id, group_id, role, json.dumps(envelope, sort_keys=True)))

    def _assignment(self, study_id: str) -> tuple[dict, dict]:
        row = self.connection.execute("SELECT * FROM confidence_assignments WHERE study_id=?", (study_id,)).fetchone()
        if row is None:
            raise IntegrityError("study has no prospective confidence evidence assignment")
        row = dict(row)
        envelope = json.loads(row["assignment"])
        if not self.manager.signer.verify_attestation(envelope):
            raise IntegrityError("confidence assignment signature is invalid")
        for key in ("study_id", "group_id", "role"):
            if envelope["payload"][key] != row[key]:
                raise IntegrityError("confidence assignment changed")
        return row, envelope

    def admit(self, study_id: str) -> dict:
        row, assignment = self._assignment(study_id)
        sealed = self.manager._stored_manifest(study_id)
        if not self.manager.verify(sealed):
            raise IntegrityError("protected study evidence does not verify")
        if assignment["payload"]["assigned_at"] > sealed.seal.sealed_at.isoformat():
            raise IntegrityError("confidence assignment was made after prediction lock")
        event = self.manager.store.last_phase_event(study_id)
        if event["to_phase"] != "evaluated":
            raise IntegrityError("confidence evidence requires a protected evaluation")
        evaluation = self.manager.store.phase_evidence(event["payload_sha256"])["payload"]
        simulation = self.manager.store.get("runs", sealed.manifest.run_id)
        if not simulation.get("confidence"):
            raise IntegrityError("locked confidence features are missing")
        evidence = {"study_id": study_id, "group_id": row["group_id"], "role": row["role"],
                    "manifest_sha256": canonical_hash(sealed), "evaluation_sha256": event["payload_sha256"],
                    "features": simulation["confidence"]["features"], "observed_tvd": evaluation["metrics"]["tvd"]}
        signed = self.manager.signer.attest(evidence)
        encoded = json.dumps(signed, sort_keys=True, allow_nan=False)
        if row["evidence"] and row["evidence"] != encoded:
            raise IntegrityError("admitted confidence evidence changed")
        with self.connection:
            self.connection.execute("UPDATE confidence_assignments SET evidence=?, run_id=? WHERE study_id=?",
                                    (encoded, sealed.manifest.run_id, study_id))
        return evidence

    def training_rows(self) -> list[dict]:
        rows = self.connection.execute("SELECT study_id FROM confidence_assignments WHERE evidence IS NOT NULL ORDER BY study_id").fetchall()
        # Verify every partition, including attempts to relabel held-out studies.
        evidence = [self.admit(row[0]) for row in rows]
        return [item for item in evidence if item["role"] == "training"]

    def fit_research_model(self) -> ConfidenceModel:
        rows = self.training_rows()
        model = ConfidenceModel()
        model.fit([item["features"] for item in rows], [item["observed_tvd"] for item in rows])
        return model
