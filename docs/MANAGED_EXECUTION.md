# Managed research execution

The OpenAI-compatible probability adapter, behavioral adapters (including
Centauri/Socrates) and OpenAI-compatible text generator all require one explicit
`ExecutionSession`. Model hosting and model weights are not included. Supply the
real pinned model/revision and applicable data/model declarations.

```python
from datetime import datetime, timedelta, timezone
from rival.engine import RivalEngine
from rival.managed_execution import ExecutionSession
from rival.providers import OpenAICompatibleProvider

# records, scenario and targets are your validated study inputs. Keep the
# scenario_id, scope ID, provider configuration and journal stable on restart.
with ExecutionSession(
    "study.attempts.sqlite3", scope_id="concept-study-001",
    budget_usd=1.0, reservation_usd=0.05, max_total_attempts=20,
    not_after=datetime.now(timezone.utc) + timedelta(minutes=30),
) as execution:
    provider = OpenAICompatibleProvider(model="YOUR_PINNED_MODEL", execution=execution)
    engine = RivalEngine(store_path="study.evidence.sqlite3")
    try:
        engine.register_provider("managed", provider)
        scenario = scenario.model_copy(update={"model_family": "managed"})
        result = engine.simulate(records, scenario, targets)
        accounting = execution.journal.summary()
    finally:
        engine.store.close()
```

Choose the reservation from a conservative input/output token bound and the
provider's current prices; the example is illustrative. Provider-side limits are
needed for a remote hard spending ceiling. A response can cost more than its
reservation. The journal records the actual cost and refuses the next request
when the budget is exhausted; it cannot undo a provider's charge.

For JSON inputs containing `scenario` (with a stable `scenario_id`), `records` and
optional `targets`, the CLI exposes the same path:

```sh
python -m rival simulate-managed --input study.json --output result.json --database study.evidence.sqlite3 --journal study.attempts.sqlite3 --scope-id concept-study-001 --model YOUR_PINNED_MODEL --budget-usd 1 --reservation-usd 0.05 --max-attempts 20 --not-after YOUR_ISO_8601_UTC_EXPIRY
```

Use `RIVAL_API_KEY` or `OPENROUTER_API_KEY` for authentication. Never put a key in
the JSON input, endpoint URL or command arguments. Behavioral endpoints can use
`RIVAL_BEHAVIORAL_API_KEY`; local endpoints may use no authentication, but still
require managed accounting. A local server must report an explicit zero cost to
assert no provider charge. An omitted cost is unknown, including for free routes.

## Scope and accounting

Share the session across providers that share one spending authorization. The
budget and physical-attempt limit are cumulative over the journal's lifetime.
Use one stable provider name per model slot in a study; changes to its endpoint,
model, prompt or parameters are rejected for an existing work ID. If you need
multiple models from the same adapter, give them distinct stable provider names
before locking their identities. Model replicates require explicitly separate
study IDs; resampling a person is not an independent model replicate.

`provider_cost_usd` in a successful call describes newly incurred charges during
that call, including rejected attempts. A cached call adds zero. `request_cost_usd`
is the whole request's recorded charge across restarts. The journal summary is
authoritative for a batch, including failed requests that produced no simulation.
No remote exactly-once guarantee is made.

The current general CLI refuses an existing result export. If an export is lost,
rerun the same input, journal and evidence database to regenerate it from cached
responses. Do not discard the journal to escape an error or an exhausted limit.

## Recovery

A timeout, interrupted process, or missing cost preserves an unresolved attempt
and stops further requests in the shared journal. First stop the original worker.
Use `AttemptJournal.attempts_for(work_id)` and the provider's generation/billing
record to establish actual cost and whether a completion exists. Call
`reconcile(work_id, ordinal, cost_usd=..., evidence_reference=...,
completed_payload=...)` with that evidence. Without a recovered completion the
request stays terminal; no automatic replacement is made. `release_interrupted_claim`
only releases a stopped worker between attempts when no remote attempt is unknown.

The journal contains research inputs/outputs and usage records. Include it in
study backups with the result and evidence database. The future customer delivery
phase adds access controls, managed backup and operational recovery rehearsal.

## Explicit Mega v2 workflow

Use new v2 stage/result/report paths. `prepare` verifies and stages public
answer-free inputs, `run` requires budget/expiry/attempt limits, `freeze` binds the
result ledger and resolved journal, `materialize-outcomes` opens the frozen
outcomes, and `reanalyze` writes a separate corrected report. `--help` on each
command lists required paths. Existing v1 outcomes may be reanalyzed into v2
reports once already frozen and revealed; the original files remain unchanged.

## Confidence evidence

Create a `ConfidenceEvidenceRegistry(manager)` with the existing prospective
manager. Call `assign(study_id, group_id=..., role="training" | "calibration" |
"evaluation", source_reference=...)` before `manager.lock_prediction`. After
vault-mediated reveal and verified evaluation, call `admit(study_id)`.
`fit_research_model()` reads only distinct admitted training studies. Repeating
admission does not increase their count; roles cannot move after lock.

Custodians declare independent study groups. Signatures and uniqueness constraints
prove the recorded workflow, not population representativeness or statistical
independence. Those judgments, untouched coverage tests and model deployment
qualification belong to L11–L12. Research fits never unlock operational confidence.

## Rebuild the development release

From the checkout, after tests and wheel installation verification:

```sh
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/verify_installed_wheel.py --wheel dist/rival_sim-0.6.0.dev4-py3-none-any.whl --output reports/installed_wheel.json
python scripts/build_phase_one_release.py --wheel dist/rival_sim-0.6.0.dev4-py3-none-any.whl --output-dir dist/phase-one --evidence reports/installed_wheel.json
python -m rival verify-release --manifest dist/phase-one/RELEASE_MANIFEST.json
```

The new bundle contains the wheel, complete file inventory, selected verification
artifacts and their hashes. It verifies artifact integrity and version alignment;
it cannot qualify accuracy. The historical root release manifest stays unchanged.
