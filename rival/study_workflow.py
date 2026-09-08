"""Durable orchestration of one declared audience and choice-distribution study.

The workspace owns its local signing key and independent accounting journal.
Source declarations and local signatures do not establish empirical validity.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import sqlite3

from .confidence_evidence import ConfidenceEvidenceRegistry
from .engine import RivalEngine
from .execution import ExecutionError
from .integrity import (IntegrityError, ManifestSigner, ProspectiveStudyManager,
                        _phase_event, verify_locked_context)
from .managed_execution import ExecutionSession
from .mathx import canonical_hash, effective_sample_size
from .mega_study_v2.runner import exclusive_run_lock
from .outcome_vault import OutcomeVault
from .providers import HeuristicChoiceProvider, OpenAICompatibleProvider
from .schemas import (EvaluationResult, PredictionContext, SealedStudyManifest,
                      SimulationResult, utc_now)
from .store import EvidenceStore
from .study_contract import StudyRequest, StudyRequestV2, StudyRequestV3, parse_study_request
from .study_evidence import verify_imports
from .study_support import require_support, validate_filters
from .version import __version__


WORKFLOW_VERSION = "rival.study-workflow.v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _provider(request, execution=None, api_key=None, *, preparing=False):
    options = request.execution
    if options.mode == "offline":
        return HeuristicChoiceProvider()
    if isinstance(request, StudyRequestV3):
        from .model_providers import model_provider
        return model_provider(options, execution,
            "preparation-only-no-network" if preparing else api_key)
    return OpenAICompatibleProvider(model=options.model, base_url=options.base_url,
        api_key="preparation-only-no-network" if preparing else api_key,
        execution=execution, temperature=options.temperature,
        max_retries=options.max_retries, timeout_seconds=options.timeout_seconds,
        history_limit=options.history_limit, max_output_tokens=options.max_output_tokens)


def _execution(root, request, scope_id):
    options = request.execution
    return ExecutionSession(root / "attempts.sqlite3", scope_id=scope_id,
        budget_usd=options.budget_usd, reservation_usd=options.reservation_usd,
        max_total_attempts=options.max_attempts, not_after=options.not_after)


def _make_plan(request):
    supported_contract = isinstance(request, StudyRequestV2)
    support = None
    engine = RivalEngine()
    try:
        provider = _provider(request, preparing=True)
        scenario = request.scenario()
        engine.register_provider(scenario.model_family, provider)
        if supported_contract:
            validate_filters(scenario.population_filter, request.audience.records)
        try:
            prepared, diagnostics, _ = engine._prepare(request.audience.records, scenario,
                                                       request.audience.targets)
        except ValueError:
            if supported_contract:
                require_support(request)  # Explain unsupported filters/controls.
            raise
        if supported_contract:
            support = require_support(request, prepared_records=prepared.records,
                                      population_diagnostics=diagnostics)
        if diagnostics and not diagnostics.converged:
            raise ValueError("audience controls did not converge; revise the audience before execution")
        sampled = engine.population.sample(prepared.records, scenario.sample_size, scenario.seed)
        work = {}
        for person in sampled:
            seed_id = person.person_id.rsplit("__draw_", 1)[0]
            identifier = canonical_hash({"study": scenario.scenario_id, "person": seed_id,
                                         "provider_slot": provider.name})
            work[identifier] = work.get(identifier, 0) + 1
        plan = {
            "prediction_context": prepared.context.model_dump(mode="json"),
            "retrieval_audit": prepared.audit.model_dump(mode="json"),
            "population_diagnostics": diagnostics.model_dump(mode="json") if diagnostics else None,
            "eligible_seed_records": len(prepared.records),
            "effective_seed_records": effective_sample_size(record.weight for record in prepared.records),
            "planned_unique_seeds": len(work), "planned_draws": len(sampled),
            "request_draw_counts": work,
            "sample_plan_sha256": canonical_hash([(person.person_id, person.weight) for person in sampled]),
        }
        if support is not None:
            plan["support_audit"] = support
        if isinstance(request, StudyRequestV3):
            from .model_providers import runtime_identity
            embedding = request.execution.elicitation.embedding
            plan["model_execution"] = {
                "schema_version": "rival.model-execution.v1",
                "settings": request.execution.model_dump(mode="json"),
                "runtime": runtime_identity(bool(embedding and embedding.kind == "sentence_transformer")),
                "identity_scope": "declared revision with checked response metadata; remote weights are not independently attested",
                "replication_scope": "journal replay reuses outputs; fresh repeatability requires a separately executed study",
            }
        return plan
    finally:
        engine.store.close()


class _Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / "study.sqlite3").is_file() or not (self.root / "manifest.key").is_file():
            raise IntegrityError("study workspace or signing key is missing; restore it before resuming")
        self.signer = ManifestSigner((self.root / "manifest.key").read_bytes())
        self.store = EvidenceStore(self.root / "study.sqlite3")
        try:
            self.store.connection.execute("PRAGMA synchronous=FULL")
            self.prepared = self.read("prepared", required=True)
            if self.prepared["workflow_version"] != WORKFLOW_VERSION:
                raise IntegrityError("unsupported study workflow version")
            self.request = parse_study_request(self.prepared["request"])
            if canonical_hash(self.request) != self.prepared["request_sha256"]:
                raise IntegrityError("study request fingerprint changed")
            self.manager = ProspectiveStudyManager(self.store, self.signer)
            if self.request.evidence.role != "development":
                _, assignment = ConfidenceEvidenceRegistry(self.manager)._assignment(self.request.brief.study_id)
                expected = {"study_id": self.request.brief.study_id, **self.request.evidence.model_dump(mode="json")}
                if any(assignment["payload"].get(key) != value for key, value in expected.items()):
                    raise IntegrityError("confidence evidence assignment differs from the study request")
            self.history()
        except BaseException:
            self.store.close()
            raise

    def close(self):
        self.store.close()

    def read(self, name, *, required=False):
        row = self.store.connection.execute("SELECT envelope FROM workflow_records WHERE name=?", (name,)).fetchone()
        if row is None:
            if required:
                raise IntegrityError(f"study checkpoint {name} is missing")
            return None
        envelope = json.loads(row[0])
        if not self.signer.verify_attestation(envelope) or envelope["payload"]["name"] != name:
            raise IntegrityError("study checkpoint signature does not verify")
        return envelope["payload"]["value"]

    def put(self, name, value):
        previous = self.read(name)
        if previous is not None:
            if canonical_hash(previous) != canonical_hash(value):
                raise IntegrityError("attempt to replace an immutable study checkpoint")
            return
        envelope = self.signer.attest({"name": name, "value": value})
        with self.store.connection:
            self.store.connection.execute("INSERT INTO workflow_records VALUES (?, ?)", (name, _json(envelope)))

    def history(self):
        rows = self.store.connection.execute("SELECT ordinal, envelope FROM workflow_events ORDER BY ordinal").fetchall()
        previous, result = None, []
        for ordinal, row in enumerate(rows):
            envelope = json.loads(row["envelope"])
            payload = envelope.get("payload", {})
            if (not self.signer.verify_attestation(envelope) or row["ordinal"] != ordinal
                    or payload.get("ordinal") != ordinal or payload.get("previous") != previous):
                raise IntegrityError("study execution history does not verify")
            previous = canonical_hash(envelope)
            result.append(payload)
        return result

    def event(self, state, detail=None):
        history = self.history()
        previous = None
        if history:
            row = self.store.connection.execute("SELECT envelope FROM workflow_events ORDER BY ordinal DESC LIMIT 1").fetchone()
            previous = canonical_hash(json.loads(row[0]))
        payload = {"ordinal": len(history), "previous": previous, "state": state,
                   "at": utc_now().isoformat(), "detail": detail}
        with self.store.connection:
            self.store.connection.execute("INSERT INTO workflow_events VALUES (?, ?)",
                                          (len(history), _json(self.signer.attest(payload))))

    @contextmanager
    def execution(self):
        if self.request.execution.mode == "offline":
            yield None
            return
        path = self.root / "attempts.sqlite3"
        if not path.is_file():
            raise IntegrityError("attempt journal is missing; restore accounting before resuming")
        # Refuse an empty or substituted database before ExecutionSession can
        # initialize it. This identity is created only with the original plan.
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT identity FROM workflow_journal_identity WHERE id=1").fetchone()
            if row is None or row[0] != self.prepared["execution_scope_id"]:
                raise IntegrityError("attempt journal identity changed")
        finally:
            connection.close()
        with _execution(self.root, self.request, self.prepared["execution_scope_id"]) as session:
            yield session

    @staticmethod
    def journal_snapshot(session):
        if session is None:
            return {"accounting": {"mode": "offline", "attempts": 0, "known_cost_usd": 0.0,
                    "total_cost_usd": 0.0, "reserved_usd": 0.0, "unresolved_attempts": 0},
                    "requests": [], "digest": None}
        journal = session.journal
        with journal.lock:
            requests = [dict(row) for row in journal.connection.execute("SELECT * FROM requests ORDER BY work_id")]
            attempts = [dict(row) for row in journal.connection.execute("SELECT * FROM attempts ORDER BY work_id, ordinal")]
            summary = {"mode": "managed", **journal.summary()}
        return {"accounting": summary, "requests": requests, "attempts": attempts,
                "digest": canonical_hash({"requests": requests, "attempts": attempts})}

    def simulation(self):
        rows = self.store.connection.execute("SELECT payload, sha256 FROM runs WHERE study_id=?",
                                             (self.request.brief.study_id,)).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise IntegrityError("study contains multiple runs; choose a new study for replicates")
        payload = json.loads(rows[0]["payload"])
        if canonical_hash(payload) != rows[0]["sha256"]:
            raise IntegrityError("stored simulation changed")
        result = SimulationResult.model_validate(payload)
        if canonical_hash(result.scenario) != canonical_hash(self.request.scenario()):
            raise IntegrityError("simulation belongs to a different study request")
        verify_locked_context(result.prediction_context,
                              PredictionContext.model_validate(self.prepared["prediction_context"]))
        if len(result.predictions) != self.prepared["planned_draws"]:
            raise IntegrityError("simulation does not contain every planned draw")
        return result

    def seal(self, result):
        stored = self.store.manifest_for_study(self.request.brief.study_id)
        if stored:
            sealed = SealedStudyManifest.model_validate(stored)
            self.signer.require_valid(sealed)
            if (sealed.manifest.simulation_sha256 != canonical_hash(result)
                    or canonical_hash(sealed.manifest.preregistration) != canonical_hash(self.request.preregistration)):
                raise IntegrityError("sealed study differs from the prepared request")
            if self.store.last_phase_event(self.request.brief.study_id) is None:
                # Recover a crash after save_manifest but before the original
                # manager appended its phase event, using the existing seal.
                event = _phase_event(self.request.brief.study_id, "draft", "prediction_locked", canonical_hash(sealed), None)
                self.store.append_phase_event(event, sealed.model_dump(mode="json"))
        else:
            sealed = self.manager.lock_prediction(result, self.request.preregistration)
        if not self.manager.verify(sealed):
            raise IntegrityError("sealed prediction does not verify")
        return sealed

    def require_complete(self, snapshot):
        completed = self.read("completed", required=True)
        result = self.simulation()
        if result is None or canonical_hash(result) != completed["simulation_sha256"]:
            raise IntegrityError("completed simulation is missing or changed")
        sealed = SealedStudyManifest.model_validate(self.store.manifest_for_study(self.request.brief.study_id))
        if (not self.manager.verify(sealed) or canonical_hash(sealed) != completed["manifest_sha256"]
                or snapshot["digest"] != completed["journal_sha256"]):
            raise IntegrityError("completed study or accounting no longer verifies")
        return result, sealed

    def status(self, snapshot):
        states = {}
        work = self.prepared["request_draw_counts"]
        for row in snapshot["requests"]:
            if row["work_id"] not in work:
                raise IntegrityError("journal contains a request outside the study plan")
            states[row["state"]] = states.get(row["state"], 0) + 1
        complete = self.read("completed") is not None
        result = self.require_complete(snapshot)[0] if complete else None
        history = self.history()
        return {"schema_version": "rival.study-status.v1", "study_id": self.request.brief.study_id,
                "state": "complete" if complete else (history[-1]["state"] if history else "prepared"),
                "complete": complete, "request_sha256": self.prepared["request_sha256"],
                "planned_draws": self.prepared["planned_draws"],
                "planned_unique_seeds": self.prepared["planned_unique_seeds"],
                "completed_draws": len(result.predictions) if result else 0,
                "request_states": states, "accounting": snapshot["accounting"],
                "last_execution_event": history[-1] if history else None,
                "prediction_phase": (self.store.last_phase_event(self.request.brief.study_id) or {}).get("to_phase", "draft")}


@contextmanager
def _workspace(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise IntegrityError("study workspace does not exist; prepare it first")
    with exclusive_run_lock(root.with_name(root.name + ".lock")):
        workspace = _Workspace(root)
        try:
            yield workspace
        finally:
            workspace.close()


def check_study(request, *, catalog_root=None):
    """Run the same no-call checks as preparation, without creating a workspace."""
    json.dumps(request.model_dump(mode="python"), default=str, allow_nan=False)
    request = parse_study_request(request.model_dump(mode="json"))
    if not isinstance(request, StudyRequestV2) and any(source.source_type != "synthetic" for source in request.audience.sources):
        raise ValueError("new studies with human/public/licensed evidence require a v2 request and verified catalog imports")
    imports = verify_imports(request, catalog_root)
    plan = _make_plan(request)
    return {"schema_version": "rival.study-check.v1", "study_id": request.brief.study_id,
            "passed": True, "planned_draws": plan["planned_draws"],
            "support_audit": plan.get("support_audit"), "import_verification": imports,
            "scope": "engineering checks; predictive accuracy remains unqualified"}


def prepare_study(root, request: StudyRequest, *, catalog_root=None):
    json.dumps(request.model_dump(mode="python"), default=str, allow_nan=False)
    request = parse_study_request(request.model_dump(mode="json"))
    root = Path(root).resolve()
    with exclusive_run_lock(root.with_name(root.name + ".lock")):
        if root.exists():
            workspace = _Workspace(root)
            try:
                if canonical_hash(request) != workspace.prepared["request_sha256"]:
                    raise IntegrityError("workspace already contains a different request; use a new study workspace")
                with workspace.execution() as session:
                    return workspace.status(workspace.journal_snapshot(session))
            finally:
                workspace.close()
        if not isinstance(request, StudyRequestV2) and any(source.source_type != "synthetic" for source in request.audience.sources):
            raise ValueError("new studies with human/public/licensed evidence require a v2 request and verified catalog imports")
        imported = verify_imports(request, catalog_root)
        plan = _make_plan(request)  # Validate all provider-visible inputs without calls.
        if imported is not None:
            plan["import_verification"] = imported
        root.mkdir(parents=True)
        secret = secrets.token_bytes(64)
        with os.fdopen(os.open(root / "manifest.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
            handle.write(secret)
            handle.flush()
            os.fsync(handle.fileno())
        store = EvidenceStore(root / "study.sqlite3")
        try:
            store.connection.execute("PRAGMA synchronous=FULL")
            store.connection.executescript("""
                CREATE TABLE workflow_records (name TEXT PRIMARY KEY, envelope TEXT NOT NULL);
                CREATE TABLE workflow_events (ordinal INTEGER PRIMARY KEY, envelope TEXT NOT NULL);
            """)
            scope_id = secrets.token_hex(32)
            if request.execution.mode == "managed":
                with _execution(root, request, scope_id) as session:
                    session.journal.connection.execute("CREATE TABLE workflow_journal_identity (id INTEGER PRIMARY KEY, identity TEXT NOT NULL)")
                    session.journal.connection.execute("INSERT INTO workflow_journal_identity VALUES (1, ?)", (scope_id,))
            for source in request.audience.sources:
                store.register_evidence(source)
            store.register_study(request.scenario())
            signer = ManifestSigner(secret)
            if request.evidence.role != "development":
                ConfidenceEvidenceRegistry(ProspectiveStudyManager(store, signer)).assign(
                    request.brief.study_id, group_id=request.evidence.group_id,
                    role=request.evidence.role, source_reference=request.evidence.source_reference)
            prepared = {"workflow_version": WORKFLOW_VERSION, "release": __version__,
                "created_at": utc_now().isoformat(), "request": request.model_dump(mode="json"),
                "request_sha256": canonical_hash(request), "execution_scope_id": scope_id, **plan}
            envelope = signer.attest({"name": "prepared", "value": prepared})
            with store.connection:
                store.connection.execute("INSERT INTO workflow_records VALUES ('prepared', ?)", (_json(envelope),))
        finally:
            store.close()
        workspace = _Workspace(root)
        try:
            workspace.event("prepared")
            with workspace.execution() as session:
                return workspace.status(workspace.journal_snapshot(session))
        finally:
            workspace.close()


def run_study(root, *, api_key=None):
    with _workspace(root) as workspace, workspace.execution() as session:
        snapshot = workspace.journal_snapshot(session)
        status = workspace.status(snapshot)
        if workspace.read("completed"):
            return status
        workspace.event("running")
        try:
            result = workspace.simulation()
            provider = _provider(workspace.request, session, api_key, preparing=result is not None)
            engine = RivalEngine(store=workspace.store)
            scenario = workspace.request.scenario()
            engine.register_provider(scenario.model_family, provider)
            locked = PredictionContext.model_validate(workspace.prepared["prediction_context"])
            expected, _ = engine.prepare_prediction_context(workspace.request.audience.records,
                scenario, workspace.request.audience.targets)
            verify_locked_context(locked, expected)
            if result is None:
                if hasattr(provider, "prepare_local"):
                    provider.prepare_local(scenario)
                result = engine.simulate(workspace.request.audience.records, scenario,
                    workspace.request.audience.targets, locked_context=locked)
            sealed = workspace.seal(result)
            snapshot = workspace.journal_snapshot(session)
            if session is not None and (snapshot["accounting"]["unresolved_attempts"]
                    or {row["work_id"] for row in snapshot["requests"]} != set(workspace.prepared["request_draw_counts"])
                    or any(row["state"] != "DONE" for row in snapshot["requests"])):
                raise IntegrityError("not every planned model request has resolved accounting")
            if isinstance(workspace.request, StudyRequestV3):
                from .study_execution_audit import execution_audit
                workspace.put("model_execution_evidence", execution_audit(workspace, snapshot, result))
            workspace.put("completed", {"simulation_sha256": canonical_hash(result),
                "manifest_sha256": canonical_hash(sealed), "journal_sha256": snapshot["digest"],
                "accounting": snapshot["accounting"]})
            workspace.event("complete")
            return workspace.status(snapshot)
        except BaseException as exc:
            # Provider exceptions can contain submitted text or credentials.
            # Persist the failure class, while the journal preserves accounting.
            workspace.event("blocked", {"error_type": type(exc).__name__})
            raise


def study_status(root):
    with _workspace(root) as workspace, workspace.execution() as session:
        return workspace.status(workspace.journal_snapshot(session))


def evaluate_study(root, vault_path, *, key_material):
    vault_path = Path(vault_path).resolve()
    if not vault_path.is_file():
        raise IntegrityError("outcome vault does not exist")
    with _workspace(root) as workspace, workspace.execution() as session:
        if vault_path.is_relative_to(workspace.root):
            raise ValueError("keep the outcome vault outside the prediction workspace")
        result, sealed = workspace.require_complete(workspace.journal_snapshot(session))
        study_id = workspace.request.brief.study_id
        phase = workspace.store.last_phase_event(study_id)["to_phase"]
        if phase == "prediction_locked":
            vault = OutcomeVault(vault_path)
            try:
                workspace.manager.reveal_outcomes(study_id, vault, key_material)
            finally:
                vault.close()
            phase = "outcomes_revealed"
        if phase == "outcomes_revealed":
            event = workspace.store.last_phase_event(study_id)
            evidence = workspace.store.phase_evidence(event["payload_sha256"])
            observed = workspace.manager._validate_reveal(sealed, evidence)
            rows = workspace.store.connection.execute("SELECT payload, sha256 FROM evaluations WHERE run_id=?", (result.run_id,)).fetchall()
            if len(rows) > 1:
                raise IntegrityError("study contains multiple evaluations")
            if rows:
                payload = json.loads(rows[0]["payload"])
                if canonical_hash(payload) != rows[0]["sha256"]:
                    raise IntegrityError("stored evaluation changed")
                evaluation = EvaluationResult.model_validate(payload)
            else:
                evaluation = RivalEngine(store=workspace.store).evaluate(result, observed,
                    preregistration_hash=canonical_hash(workspace.request.preregistration))
            workspace.manager.record_evaluation(study_id, evaluation)
        if workspace.request.evidence.role != "development":
            ConfidenceEvidenceRegistry(workspace.manager).admit(study_id)
        if not workspace.manager.verify(sealed):
            raise IntegrityError("evaluated study no longer verifies")
        return workspace.status(workspace.journal_snapshot(session))


def export_study(root, output):
    from .study_report import build_study_report, export_report
    with _workspace(root) as workspace, workspace.execution() as session:
        snapshot = workspace.journal_snapshot(session)
        result, sealed = workspace.require_complete(snapshot)
        evaluation = None
        event = workspace.store.last_phase_event(workspace.request.brief.study_id)
        if event["to_phase"] == "evaluated":
            evaluation = workspace.store.phase_evidence(event["payload_sha256"])["payload"]
        report = build_study_report(workspace.request, workspace.prepared, result, sealed,
                                    snapshot["accounting"], evaluation,
                                    model_execution=workspace.read("model_execution_evidence"))
        return export_report(report, output, workspace.root)
