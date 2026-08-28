# Prospective Study Operations

## Purpose

This runbook separates prediction creation from outcome custody. The goal is to make it technically difficult to tune Rival after seeing a customer outcome and then present the result as prospective.

## Roles

| Role | May access | Must not access before reveal |
|---|---|---|
| Model team | pre-cutoff evidence, scenario, prediction context, sealed manifest | outcome-vault key and protected outcome |
| Outcome custodian | encrypted outcome deposit, reveal key, availability rule | model tuning and replacement predictions |
| Evaluator | sealed manifest, reveal receipt, preregistered metrics, released outcome | unregistered exclusions or metric changes |

Use separate people or systems for these roles in a customer study. A single developer running every role is useful for testing the machinery but is not an independent prospective design.

## Lock sequence

1. Set `ScenarioSpec.information_cutoff` and populate only lawfully usable pre-cutoff evidence.
2. Call `RivalEngine.prepare_prediction_context`. Resolve any outcome-firewall error; do not rename or obscure protected fields to make them pass.
3. Save the returned context outside the mutable request-building system.
4. Run `RivalEngine.simulate(..., locked_context=context)`. The engine recomputes the context and fails before provider calls on a mismatch.
5. Define `PreregistrationSpec`, including primary metrics, thresholds, subgroups, and `outcome_not_before`.
6. Call `ProspectiveStudyManager.lock_prediction`. Export the complete `SealedStudyManifest` to the evaluator or custodian.
7. Deposit the outcome in a separate `OutcomeVault` database. Keep the vault key and database outside the simulation service.
8. After `outcome_not_before`, reveal in the custodian/evaluation environment. Append the `OutcomeRevealReceipt`, run the registered evaluation, and append the evaluation phase event.
9. Verify both the manifest seal and phase chain. Publish every eligible study, including misses.

## Minimal Python flow

```python
from datetime import datetime, timezone

from rival.engine import RivalEngine
from rival.integrity import ManifestSigner, ProspectiveStudyManager
from rival.mathx import canonical_hash
from rival.outcome_vault import OutcomeVault
from rival.schemas import PreregistrationSpec

engine = RivalEngine(store_path="prediction-ledger.sqlite3")
context, audit = engine.prepare_prediction_context(records, scenario, targets)
simulation = engine.simulate(records, scenario, targets, locked_context=context)

manager = ProspectiveStudyManager(
    engine.store,
    ManifestSigner(manifest_key, key_id="customer-pilot-1"),
)
sealed = manager.lock_prediction(
    simulation,
    PreregistrationSpec(
        primary_metrics=["tvd"],
        acceptance_thresholds={"tvd": 0.15},
        outcome_not_before=datetime(2026, 10, 1, tzinfo=timezone.utc),
    ),
)

vault = OutcomeVault("separate-outcome-vault.sqlite3")
vault.deposit(
    scenario.scenario_id,
    canonical_hash(sealed),
    protected_outcome,
    outcome_key,
    sealed.manifest.preregistration.outcome_not_before,
)
```

Do not put `manifest_key`, `outcome_key`, provider API keys, or plaintext outcomes in source control. `RIVAL_MANIFEST_KEY` configures API locking, but the API intentionally exposes no reveal route.

## What v0.5 proves

The deterministic qualification verifies that contexts reproduce, outcome fields fail closed, input changes are rejected before provider calls, manifest tampering is detected, plaintext is absent from vault files, early and unauthenticated reveal fail, and phase chains verify. A separate eight-check gate verifies the v0.5 research-component wiring and numerical parity.

It does not prove independent key custody, trusted external timestamping, customer-domain accuracy, interval coverage, subgroup safety, or superiority to classical and equal-cost human baselines. Those require the locked customer study described in `VALIDATION_PROTOCOL.md`.
