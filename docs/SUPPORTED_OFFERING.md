# Rival's first supported offering

Rival is building a supervised research service for comparing concepts, messages
and proposed scenarios for a declared audience. Its initial quantitative output
is an aggregate choice distribution and differences between alternatives.
Individual synthetic responses are intermediate model outputs, not forecasts of
identified real people. A large simulated sample does not create more independent
human evidence.

## Current release contract

The development release runs in research mode. It provides a saved study workflow
from brief and audience through simulation, recovery and aggregate report export,
alongside an offline demonstration, protected prediction/outcome records, corrected
scoring and managed request accounting. It withholds decision-ready confidence and any
claim of customer-domain accuracy, individual fidelity or Aaru/Simile parity.
`rival.readiness.release_claims()` is the machine-readable contract used by the
health API and reports. Historical qualification files remain historical.

Inputs are a question, alternatives, audience records and optional population
controls, with source provenance and an information cutoff where applicable.
Component outputs include simulated choice shares, optional anchor corrections, diagnostics,
source and execution identities, limitations and comparison metrics when observed
outcomes are available. The built-in browser screen uses generated demo people,
anchors and outcomes; it is not yet the customer study interface.
The `study` operator commands implement the first integrated workflow. Their
reports contain raw simulated choice shares. The v4 contract adds calibration from
compatible protected training studies, preserving both outputs and scoring them
against the same outcomes. See [runtime calibration](RUNTIME_CALIBRATION.md). See [phase-two development](PHASE_TWO_DEVELOPMENT.md).

The v2 study contract binds public/licensed/human inputs to verified local evidence
imports and an explicit population support policy. Geography is applied to the
simulation records. Missing declared regions, conditions, attributes or subgroup
support block preparation, as do failed population controls and low effective
seed counts under the chosen policy. The report includes the import and support
audit. These checks establish which evidence was used and its declared scope;
they do not independently establish rights, representativeness, semantic relevance
or statistical confidence. See [evidence imports](EVIDENCE_IMPORTS.md).

The v3 contract adds explicit model/revision declarations, checked returned model
identities and selectable direct or text/SSR elicitation. Execution audits expose
failures, cost, tokens and latency; fresh-run comparisons measure repeatability
separately from cached recovery. This does not qualify model probabilities or
semantic scores against human behavior. See [model execution](MODEL_EXECUTION.md).

## Acceptance before offering decision support (L07–L14)

Before viewing qualification outcomes, record the intended audience, geography,
question family, decision, permitted data, independent evidence groups, strongest
relevant baseline, absolute error tolerance, required baseline improvement,
interval coverage, abstention policy, subgroup limits, failure denominator,
maximum cost and turnaround. Thresholds depend on the decision's consequences;
there is no honest universal accuracy threshold for all markets.

Use relevant public or licensed studies initially; proprietary customer data is
not a prerequisite. Keep development, calibration and final evaluation groups
separate. Confirm superiority or useful bounded performance on untouched studies,
including negative results and failures, before approving that narrow use.
Rehearse a complete study-to-report delivery and recovery process before launch.

Discovery conversations may explore customer needs now. Do not sell qualified
decision support until the empirical and operational gates pass. Multi-agent
social worlds, causal intervention claims, open-ended individual fidelity and
universal population representation need their own evidence and development.

## Phase-one completion criteria (L01–L06)

| Task | Required engineering acceptance |
|---|---|
| L01 | One explicit offering; research claims enforced in API/report/demo; no historical pilot badge presented as current qualification |
| L02 | Invalid probabilities and survey primitives rejected; tied ranks and fractional composites scored correctly; invalid/missing cases remain in denominators |
| L03 | Predictions sealed before authenticated reveal; exact outcomes and recomputed metrics bound to the stored phase chain; frozen studies preserved |
| L04 | No automatic confidence fitting; planned anchors cannot help; unqualified fits abstain; signed pre-lock evidence roles are immutable and deduplicated; held-out roles excluded from training |
| L05 | All current general network adapters and Mega v2 use durable per-attempt budgets, expiry, billing, caching and quarantine; no implicit unmanaged network call in the current interface |
| L06 | Wheel installed outside checkout can verify frozen resources and run offline; complete notices and current release hashes; failure exit codes are truthful |

These are engineering criteria. L11–L12 separately establish statistical coverage,
generalization, empirical independence and whether the system improves decisions.
