from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from ..demo import demo_population, demo_scenario, demo_targets
from ..engine import RivalEngine
from ..integrity import (
    LockedContextMismatch,
    ManifestSigner,
    OutcomeFirewallError,
    ProspectiveStudyManager,
)
from ..mathx import canonical_hash
from ..outcome_vault import OutcomeNotAvailable, OutcomeVault
from ..schemas import PreregistrationSpec, utc_now
from .provenance import stable_hash


def run_integrity_qualification() -> dict[str, Any]:
    """Exercise prospective safeguards without claiming customer validation."""

    checks: list[dict[str, Any]] = []

    def check(name: str, operation: Callable[[], Any]) -> Any:
        try:
            detail = operation()
            checks.append({"name": name, "status": "PASS", "detail": str(detail)})
            return detail
        except Exception as exc:  # report every control in one qualification artifact
            checks.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            return None

    with TemporaryDirectory() as directory:
        root = Path(directory)
        engine = RivalEngine(store_path=root / "ledger.sqlite3")
        records = demo_population(80)
        scenario = demo_scenario(60, 0).model_copy(
            update={"information_cutoff": "2026-01-01T00:00:00+00:00"}
        )
        targets = demo_targets()

        context, audit = engine.prepare_prediction_context(records, scenario, targets)
        second_context, _ = engine.prepare_prediction_context(records, scenario, targets)
        check(
            "deterministic_prediction_context",
            lambda: (
                context.context_sha256
                if context.context_sha256 == second_context.context_sha256
                else (_ for _ in ()).throw(AssertionError("context digest changed"))
            ),
        )
        check(
            "retrieval_audit_bound",
            lambda: (
                audit.audit_sha256
                if context.retrieval_audit_sha256 == audit.audit_sha256
                else (_ for _ in ()).throw(AssertionError("audit is not bound"))
            ),
        )

        def firewall_check() -> str:
            contaminated = records[0].model_copy(
                update={"attributes": {**records[0].attributes, "ground_truth": "x"}}
            )
            try:
                engine.prepare_prediction_context(
                    [contaminated, *records[1:]], scenario, targets
                )
            except OutcomeFirewallError:
                return "protected field rejected"
            raise AssertionError("protected outcome reached the prediction boundary")

        check("outcome_firewall_fail_closed", firewall_check)

        def locked_context_check() -> str:
            changed = records[0].model_copy(
                update={"preferences": {**records[0].preferences, "quality": 99.0}}
            )
            try:
                engine.simulate(
                    [changed, *records[1:]],
                    scenario,
                    targets,
                    locked_context=context,
                )
            except LockedContextMismatch:
                return "changed inputs rejected before provider calls"
            raise AssertionError("changed inputs passed a locked context")

        check("locked_context_mismatch_rejected", locked_context_check)

        simulation = engine.simulate(
            records, scenario, targets, locked_context=context
        )
        signer = ManifestSigner(b"integrity-qualification-key-material-0001")
        manager = ProspectiveStudyManager(engine.store, signer)
        not_before = utc_now() + timedelta(hours=1)
        sealed = manager.lock_prediction(
            simulation,
            PreregistrationSpec(
                acceptance_thresholds={"tvd": 0.15},
                outcome_not_before=not_before,
            ),
        )
        check(
            "manifest_seal_verifies",
            lambda: sealed.seal.digest
            if signer.verify(sealed)
            else (_ for _ in ()).throw(AssertionError("seal did not verify")),
        )

        def tamper_check() -> str:
            tampered_manifest = sealed.manifest.model_copy(
                update={"predictions_sha256": "0" * 64}
            )
            tampered = sealed.model_copy(update={"manifest": tampered_manifest})
            if signer.verify(tampered):
                raise AssertionError("tampered manifest verified")
            return "tamper detected"

        check("manifest_tamper_detected", tamper_check)

        vault = OutcomeVault(root / "vault.sqlite3")
        manifest_sha256 = canonical_hash(sealed)
        marker = "RIVAL-PLAINTEXT-MUST-NOT-APPEAR"
        protected_outcome = {"distribution": {"a": 0.4, "b": 0.6}, "marker": marker}
        vault_key = b"outcome-qualification-key-material"
        vault.deposit(
            scenario.scenario_id,
            manifest_sha256,
            protected_outcome,
            vault_key,
            not_before,
        )

        def ciphertext_check() -> str:
            vault.connection.execute("PRAGMA wal_checkpoint(FULL)")
            for candidate in root.glob("vault.sqlite3*"):
                if marker.encode("utf-8") in candidate.read_bytes():
                    raise AssertionError(f"plaintext found in {candidate.name}")
            return "plaintext absent at rest"

        check("outcome_encrypted_at_rest", ciphertext_check)

        def early_reveal_check() -> str:
            try:
                vault.reveal(
                    scenario.scenario_id,
                    manifest_sha256,
                    vault_key,
                    now=not_before - timedelta(seconds=1),
                )
            except OutcomeNotAvailable:
                return "early reveal blocked"
            raise AssertionError("outcome was revealed before not_before")

        check("time_gate_blocks_early_reveal", early_reveal_check)
        revealed, receipt = vault.reveal(
            scenario.scenario_id,
            manifest_sha256,
            vault_key,
            now=not_before + timedelta(seconds=1),
        )
        check(
            "authenticated_outcome_reveal",
            lambda: receipt.outcome_sha256
            if revealed == protected_outcome
            else (_ for _ in ()).throw(AssertionError("revealed payload changed")),
        )
        manager.record_outcome_reveal(scenario.scenario_id, receipt)
        evaluation = engine.evaluate(
            simulation,
            simulation.distribution,
            preregistration_hash=canonical_hash(sealed.manifest.preregistration),
            learn_confidence=False,
        )
        manager.record_evaluation(scenario.scenario_id, evaluation)
        check(
            "append_only_phase_chain",
            lambda: "draft -> prediction_locked -> outcomes_revealed -> evaluated"
            if engine.store.verify_phase_chain(scenario.scenario_id)
            else (_ for _ in ()).throw(AssertionError("phase chain failed verification")),
        )
        vault.close()
        engine.store.close()

    report: dict[str, Any] = {
        "schema_version": "rival.integrity-qualification.v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "scope": "deterministic engineering controls; not a prospective customer validity result",
        "checks": checks,
    }
    report["report_sha256"] = stable_hash(report)
    return report
