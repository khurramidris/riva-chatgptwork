# Phase two: a connected research pipeline

Phase one (L01–L06) completed at `c67da54`, with Linux and Windows CI passing.
Phase two develops L07–L10 on `codex/phase-two-study-workflow`, starting from that
verified commit. The development version is `0.6.0.dev5`.

| Task | Phase-two acceptance | Current position |
|---|---|---|
| L07 Integrated study workflow | Save and validate a versioned brief, audience, sources, evidence role, execution policy and preregistration; execute and recover through the request journal; seal predictions; export an aggregate report; optionally compare authenticated outcomes | Complete; 152 local tests and isolated installed-wheel verification pass |
| L08 Public evidence and population support | Import permitted public or licensed evidence with traceable versions and dates; enforce declared audience and condition support; expose missing support | Basic source references and cutoff checks are present; catalog/import and support coverage remain pending |
| L09 Model execution and elicitation | Run a pinned real model through the complete workflow; verify elicitation, reproducibility, failures, cost and latency | Managed HTTP transport works with test responses; real-model execution and elicitation work remain pending |
| L10 Runtime calibration | Bind the reference bank and calibration model to evidence and provider identities; keep fitting separate from held-out studies; integrate and compare calibration in the workflow | Pending |

L11–L12 establish qualified uncertainty and performance on untouched studies.
L13–L14 complete the operator interface and delivery rehearsal. Passing the new
workflow's engineering tests does not complete those later tasks.

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
prevention or independent evidence groups. L08 expands the evidence and support
checks. Proprietary customer data is not a prerequisite.

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
