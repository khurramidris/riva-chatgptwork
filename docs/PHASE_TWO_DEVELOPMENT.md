# Phase two: a connected research pipeline

Phase one (L01–L06) completed at `c67da54`, with Linux and Windows CI passing.
Phase two develops L07–L10. L07 completed on `codex/phase-two-study-workflow` at
`50e7494`, with Linux and Windows CI passing on draft PR #6. L08 completed at
`07ec73e`, with Linux and Windows CI passing on draft PR #7. L09 builds on that
commit on `codex/phase-two-model-execution` (draft PR #8), completing at `cead206`
with Linux and Windows CI passing. L10 builds on it on `codex/phase-two-calibration`.
The current development version is `0.6.0.dev8`; all four phase-two engineering
tasks are complete. Empirical qualification and the full operator GUI remain later work.

| Task | Phase-two acceptance | Current position |
|---|---|---|
| L07 Integrated study workflow | Save and validate a versioned brief, audience, sources, evidence role, execution policy and preregistration; execute and recover through the request journal; seal predictions; export an aggregate report; optionally compare authenticated outcomes | Complete; 152 local tests and isolated installed-wheel verification pass |
| L08 Public evidence and population support | Import permitted public or licensed evidence with traceable versions and dates; enforce declared audience and condition support; expose missing support | Complete for local CSV/JSONL snapshots and explicit support policies; 172 tests and a 2,058-row public demographic import rehearsal pass |
| L09 Model execution and elicitation | Run a pinned real model through the complete workflow; verify elicitation, reproducibility, failures, cost and latency | Complete engineering milestone: pinned real local SSR runs, fresh repeat, failures/cost/latency and recovery verified; the small model failed direct-probability validation |
| L10 Runtime calibration | Bind the reference bank and calibration model to evidence and provider identities; keep fitting separate from held-out studies; integrate and compare calibration in the workflow | Complete engineering milestone: protected training banks, pinned v4 raw/calibrated outputs, held-out comparisons, numerical parity and 215 local tests |

L11–L12 establish qualified uncertainty and performance on untouched studies.
L13–L14 complete the operator interface and delivery rehearsal. Passing the new
workflow's engineering tests does not complete those later tasks.

## Fourth increment: runtime calibration

L10 integrates the SYN-DIGITS distributional objective into saved studies, with
an explicit extension for probability responses. References must be complete,
protected training studies from the same panel and model. New targets bind an
immutable adapter at preparation, save both predictions before reveal and report
when calibration helps or hurts. All 215 local tests pass, including 19 new
calibration tests and archived-source numerical checks.

The generated HTTP rehearsal completes two training and two evaluation studies,
including a harmful adjustment. A separate retrospective comparison on 489 public
OpinionQA questions finds mean TVD 0.166793, versus raw 0.331125, historical mean
0.226775 and prior research calibration 0.169495. These data were already used in
development. All five numerical fits reached their iteration limit; their gaps
are retained. No current provider or customer accuracy claim follows.

See [runtime calibration](RUNTIME_CALIBRATION.md), [the verification record](verification/phase_two_calibration.json)
and [the generated example](examples/calibration/held-out-benefit/report.md).
L11 is next: qualified uncertainty and abstention using separate evidence groups.

## Third increment: pinned model execution and elicitation

L09 adds the v3 study contract, selectable direct/SSR elicitation, checked returned
model identities, bound generation/runtime settings, execution measurements and
fresh-run comparisons. The SSR wrapper fixes order-dependent ties and underflow,
exposes degenerate signals, and reuses embeddings. The vendored source is unchanged.

The recorded CPU rehearsal made 14 real Qwen2.5-0.5B-Instruct calls: two direct
attempts failed because the returned probabilities totaled 1.1; both SSR studies
completed all six seed requests and 40 draws. The fresh SSR replicate matched
all six per-seed distributions exactly (aggregate TVD 0). Resume made no new calls.
The two complete runs took 48.39 and 47.26 seconds including local embedding work.
API fees were zero; compute/electricity and production hosted costs are unmeasured.

This completes L09's engineering acceptance through the real SSR route. It does
not qualify this small model for production. Direct elicitation still needs a
model/endpoint that reliably follows its response contract; domain accuracy and
production cost/latency require their own qualification. No human outcomes were
used. L10 now adds reference-bank identity and runtime calibration with fit/evaluation
separation as described above.

See [model execution](MODEL_EXECUTION.md), the [real-run receipt](examples/model-execution/rehearsal.json),
and [the L09 verification record](verification/phase_two_model_execution.json).
The local suite has 196 passing tests; installed-wheel verification exercises
both direct and SSR routes through loopback HTTP fixtures.

## Second increment: evidence and support

L08 adds `rival evidence` for pinned imports and `rival study bind-evidence` /
`check` for the v2 study contract. Source bytes and their conversion must verify
at preparation. Geography is applied to the actual audience; missing requested
regions, conditions, attributes or joint cells stop preparation, along with
unsupported controls and low effective seed counts under the declared policy.
The report includes the source, version, dates, hashes and aggregate support audit.

See [the evidence guide](EVIDENCE_IMPORTS.md) for the full workflow, its explicit
declaration/semantic limits, public-data recipe and compatibility behavior.
Existing completed v1 workspaces remain readable; incomplete studies still require
their pinned runtime. New nonsynthetic studies require the v2 evidence contract.
L09 and L10 add the execution and calibration evidence described above.

## First increment: one study from input to report

The new operator entry point is `python -m rival study`. One study explores one
question with 2–20 competing choices for a declared audience. It returns simulated
choice shares and differences from the first choice. These differences are not
causal intervention effects. Multiple conditions, longitudinal interactions and
customer-specific calibration are separate work.

| Command | Behavior |
|---|---|
| `example` | Write an explicitly generated offline study input |
| `schema` | Write the JSON schema for the versioned input contract |
| `prepare` | Validate and save the request, source declarations, filtered/calibrated audience plan, outcome firewall, provider identity and evidence role; makes no model requests |
| `run` | Execute the saved plan or recover it; completed runs retain their original result IDs and cost accounting |
| `status` | Inspect the planned denominator, progress, request states, billing and prediction phase |
| `export` | Write an aggregate JSON report, readable Markdown report and file-hash manifest |
| `evaluate` | Open an existing external outcome vault through the prospective manager and bind the resulting evaluation to the sealed predictions |

Run this offline rehearsal from an installed checkout/virtual environment:

```sh
python -m rival study example --output reports/study-input.json
python -m rival study prepare --input reports/study-input.json --workspace .rival-data/workflow-example
python -m rival study run --workspace .rival-data/workflow-example
python -m rival study status --workspace .rival-data/workflow-example
python -m rival study export --workspace .rival-data/workflow-example --output reports/workflow-example-export
```

The example has six generated seed records and 1,000 simulation draws. It calls
the existing heuristic provider and has zero model API cost. The report explicitly
labels the audience as generated and its confidence as unqualified. An existing
input file is never overwritten. Repeating `run` recovers the saved result;
repeating `export` to an identical destination is idempotent. Changed report
content, including a subsequent outcome comparison, requires a new destination.
See the [generated example report](examples/study-workflow/report.md) and its
[machine-readable export](examples/study-workflow/report.json).

## Inputs and boundaries

`rival.study-request.v1` contains the brief, audience records and controls, source
declarations, evidence role, managed-execution limits and optional preregistration.
The input requires stable study/person/source IDs, timezone-aware source dates and
an information cutoff. Every audience record must reference a declared source.
Source declarations must explicitly permit simulation. Unknown references,
duplicate people, outcome sources, post-cutoff sources, non-finite inputs,
unsupported population categories and nonconvergent controls stop preparation.
The existing outcome firewall filters history before the provider sees it.

These checks establish an input contract. They do not verify source permissions,
real-world representativeness, interview fidelity, complete semantic leakage
prevention or independent evidence groups. L08's v2 contract expands the evidence
and support checks as described above. Proprietary customer data is not a prerequisite.

## Managed execution

Change `execution.mode` to `managed` in a new input and supply all of:
`model`, `base_url`, `budget_usd`, `reservation_usd`, `max_attempts` and a
timezone-aware `not_after`. Model and endpoint are explicit, rather than inherited
from environment configuration. Request settings also include temperature,
retry count, timeout, history limit and output-token limit. API credentials are
read only when executing, from `RIVAL_API_KEY` or `OPENROUTER_API_KEY`.

Preparation still makes no requests and needs no API key. Each sampled seed
person's managed completion is reused across that person's draws. Budgets and
attempt limits apply over the saved journal's entire lifetime; a new `run`
invocation does not reset them. The final report reads the original journal's
total cost, including billed failed attempts. Reservation amounts are estimates;
the provider's spending controls remain the remote ceiling.

The model, question, audience, sources, evidence role and limits are fixed by
preparation. This first workflow does not implement budget/expiry renewal. A
stopped budget or expiry stays stopped; do not delete its journal to continue.
The [managed execution guide](MANAGED_EXECUTION.md) explains reconciliation using
actual provider billing evidence. Reconciliation does not authorize another call.

## Recovery and outcomes

A workspace contains an immutable signed preparation record, source/study ledger,
append-only execution events, local signing key and, for managed studies, a separate
attempt journal. It refuses a missing or replaced journal before any model request.
A partially completed study keeps its original planned denominator and exposes no
completed distribution. Timeouts and interrupted requests require reconciliation.
Already stored results and seals survive a crash between saving and finalizing;
recovery reuses them. The OS lock permits one workflow writer per workspace.

The local key custodian, host clock and database/filesystem remain trusted.
These are local integrity controls, not independently witnessed signatures.
Keep the workspace intact. Aggregate exports omit personas, raw responses and the
signing key; they include source declarations, execution identity and input hashes.

For outcome comparison, a custodian deposits the observed distribution into an
external `OutcomeVault`, bound to the already sealed manifest hash. Then set
`RIVAL_OUTCOME_KEY` in the environment and run:

```sh
python -m rival study evaluate --workspace .rival-data/workflow-example --vault PATH_TO_EXISTING_VAULT
```

The existing manager enforces the preregistered availability date, authenticated
reveal and metric recomputation. Repeated evaluation preserves the same evaluation
record. Training/calibration/evaluation roles are declared before prediction lock;
held-out roles are not fitted automatically. Group uniqueness is enforced within
the workspace's evidence registry, not across separately copied workspaces.
No research fit becomes qualified through this workflow.

## Verification

The regression suite includes the complete CLI rehearsal, real local HTTP retry
transport, billed failures, interrupted-request reconciliation, preserved failure
denominators, missing/replaced journals, altered inputs, saved-run/seal recovery,
protected evaluation, held-out roles and immutable exports. The installed-wheel
check exercises study preparation, execution, repeated execution and report export
outside the checkout. These tests use generated fixtures and no paid inference.

All 152 regression tests pass, including 16 new study-workflow tests. The final
`0.6.0.dev5` wheel passes 13 installation checks outside the checkout. The existing
integrity (10/10) and research-component (8/8) checks also pass; their research
results do not establish improvement in this workflow's predictive accuracy.
See [the verification checkpoint](verification/phase_two_workflow.json).

The L08 checkpoint adds 20 tests (172 total), verifies the installed v2 workflow,
and independently checks 2,058 demographic mappings from a pinned public snapshot.
The old L07 generated export remains byte-identical. See
[the L08 verification record](verification/phase_two_evidence.json).
