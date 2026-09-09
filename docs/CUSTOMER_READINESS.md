# Rival customer-readiness implementation

Started 2026-09-06 from main `902a3eb975b9e765f5a0c41e66afb6abe2ead5c6`.
Current version: `0.6.0.dev9`. Branch: `codex/phase-three-uncertainty`.

Rival's target remains population and behavior simulation for comparing proposed
concepts, messages and scenarios. The first delivery target is a supervised study
service. These changes improve engineering reliability; they do not establish
customer accuracy, Aaru/Simile parity, or readiness for paid decision support.

Phase one is L01–L06. Its engineering work completed at `c67da54`, with Linux and
Windows CI passing on [draft PR #5](https://github.com/khurramidris/riva-chatgptwork/pull/5).
Phase two is L07–L10. All four are implemented and locally verified:
**11 of 14 engineering tasks complete; 3 later tasks pending** as of 2026-09-09.
L11 adds protected uncertainty experiments and v5 reports; human-domain qualification remains L12. See the [phase-two plan](PHASE_TWO_DEVELOPMENT.md)
and [latest verification record](verification/phase_three_uncertainty.json). Statistical
qualification and customer launch remain later gates.

## Delivery board

| Work | Status | Completion evidence / next dependency |
|---|---|---|
| L01 Supported offering and claims | Complete, locally verified | Explicit offering and acceptance contract; current research claims in API/report/demo; historical readiness badges removed |
| L02 Response validation and scoring | Complete, locally verified | Primitive/PMF validation, tied ranks, numeric composite scoring, historical raw revalidation, JSON-safe undefined variance |
| L03 Outcome integrity | Complete, locally verified | Vault-mediated reveal, sealed date enforcement, signed phase evidence, exact outcome binding and metric recomputation; frozen files unchanged |
| L04 Confidence evidence | Complete, locally verified | All research fits abstain with uninformative bounds; signed pre-lock roles, unique study groups and runs; verified admission; held-out partitions excluded from training |
| L05 Request accounting and recovery | Complete, locally verified | General probability, behavioral and text/SSR providers plus Mega v2 use per-attempt journals; budget/expiry, physical attempt limits, cached recovery, unknown-billing quarantine and draw deduplication |
| L06 Installation and release bookkeeping | Complete, locally verified | Installed-wheel archive/resources, offline CLI/HTTP rehearsal, complete notices, truthful qualification exits and full wheel inventory/manifest |
| L07 One integrated study workflow | Complete, locally verified | Versioned input; signed preparation and evidence role; journaled execution and crash recovery; sealed predictions; aggregate report; protected outcome comparison; installed-package rehearsal |
| L08 Public evidence and population support | Complete, locally verified | Replayable pinned CSV/JSONL imports; v2 evidence binding; cutoff, geography, condition and subgroup checks; effective counts; 2,058-row public demographic import rehearsal |
| L09 Model execution and elicitation | Complete engineering milestone, locally verified | Pinned v3 execution, fair SSR ties, 196 tests and two real local SSR runs/repeat/recovery; small-model direct probabilities correctly rejected; no production accuracy claim |
| L10 Runtime calibration | Complete engineering milestone, locally verified | Signed training banks; model/evidence/panel binding; v4 prediction seals and protected comparisons; SYN-DIGITS objective parity; 215 tests and retrospective public-data comparison |
| L11 Qualified uncertainty and abstention | Complete engineering milestone; customer qualification remains L12 | Frozen study roster; separate ridge/conformal/evaluation stages; finite-sample coverage and selective-risk gates; v5 reports; generated rehearsal; customer confidence still abstains |
| L12 Frozen qualification | Pending | Relevant untouched public data, classical baselines, missingness and failure denominators, operational measurements |
| L13 Operator interface and reports | Pending | Complete study-to-report flow, uncertainty and unsupported-use presentation |
| L14 Delivery rehearsal and launch | Pending | Access controls, backup/recovery, cost/turnaround measurements and bounded pilot terms |

Statuses distinguish implementation from empirical qualification. No paid model
requests, customer messages, deployment, merge, or new performance claim were
part of this foundation batch. Discovery can proceed while engineering continues.

## What changed

The original Mega v1 implementation is itself hashed by the frozen manifest.
Its modules and scripts therefore remain unchanged. Corrected benchmark behavior
lives in `rival.mega_study_v2`; reports and new result rows carry
`rival.mega-runtime.v2`. Do not append v2 rows to existing Mega A–D results or
reinterpret a v2 report as the preregistered v1 result.

The v2 parser reads option positions and matrix row counts from the answer-free
instrument. It rejects invalid primitive positions before averaging, malformed
cardinality, missing/extra questions, duplicate JSON keys and nonstandard numeric
constants. It uses the preserved official outcome formulas. Composite metrics
remain numeric, even if a particular sample happens to contain only integers.

The core engine rejects invalid PMFs, rejects conflicting study IDs before model
requests, and uses tie-aware Spearman ranks. Undated history is excluded when a
cutoff applies, and paired S-RCT prediction goes through the same outcome firewall
in both arms. Single-pair variance is unevaluable; single human anchors receive
uninformative intervals with a null standard error. This does not qualify the
remaining small-sample, weighted, projected, or hybrid uncertainty formulas.

`engine.evaluate(..., learn_confidence=True)` now fails explicitly. Ordinary
evaluations can still be saved but cannot improve the operational confidence
model. A confidence evidence register now assigns signed, immutable training/calibration/
evaluation roles before prediction lock. Admission verifies protected outcomes and
recomputed evaluations; only distinct training-role studies enter the research fit.
Even a fitted model returns `unqualified`, abstains and exposes the full [0,1] TVD
range. Statistical independence and calibrated coverage still require L11–L12.

The prospective manager advances outcome reveal only by reading the authenticated
vault, with no caller clock override. Reveal evidence and evaluations are signed
and retained with their phase transitions. Evaluations must use the exact revealed
distribution, sealed preregistration and stored locked predictions. Core evaluation
and retrieval policies are versioned v2. Legacy phase chains remain historical
artifacts; missing evidence is not retrospectively manufactured.

The new request journal records an attempt before network access. Every response,
including a billed empty response, contributes its reported cost. Missing usage,
timeouts and interrupted requests retain a reservation and an unknown total;
they stop further spending. A reported zero cost is distinct from missing cost.
Budgets and expiry are checked before each retry, with bounded backoff. A restart
can export a completed response without making the request again. SQLite claims
and an OS file lock protect concurrent writers. A provider-side hard spending
limit remains necessary for a remote ceiling: local reservations use estimates.

## Verification

L08 passes **172 tests**, including 20 new import/support regressions. The v2
workflow also prepares a pinned public Twin-2K demographic source, with every one
of its 2,058 mappings independently checked against the official question catalog.
The previous L07 completed workspace still resumes and exports byte-identical
files. See the [L08 verification record](verification/phase_two_evidence.json),
[evidence guide](EVIDENCE_IMPORTS.md), and
[generated report with support checks](examples/evidence-support/generated-report/report.md).
No paid inference or new accuracy result is part of this increment.

Phase two's first increment passes **152 tests**, including a real local HTTP
retry/recovery test with generated responses, and the offline installed-wheel
check. The integrity and research reproduction checks also pass. The
[generated example report](examples/study-workflow/report.md) illustrates the
new input-to-report path; it is not a model accuracy result. See the
[phase-two verification record](verification/phase_two_workflow.json) for source,
wheel and evidence hashes. The results below remain the phase-one checkpoint.

Run from the repository checkout:

```sh
python -m unittest discover -s tests -v
python -m rival.mega_study_v2 --help
python -m rival qualify-integrity --output reports/integrity_dev3.json
```

Local result: **136 tests passed**, the ten-check integrity qualification and the
research reproduction checks passed, and 68 preserved files matched the starting
commit byte for byte. The wheel was installed offline outside the checkout and
exercised through the CLI, resource loader, freeze algorithm, demo and local HTTP
API. See [the phase-one verification record](verification/phase_one.json).
The earlier [dev3 verification record](verification/customer_readiness_foundation.json)
remains a historical checkpoint. GitHub CI runs the suite, integrity and research
checks, wheel build and isolated installation on Linux and Windows/Python 3.11;
the PR records the latest verified commit. The first publication run exposed a
Windows installation-check failure. The checker now resolves both installation
paths before comparison and retains child-process diagnostics on failure. A
local aliased-directory rehearsal reproduced the old path-check failure and
passed with the correction.
The [release receipt](verification/phase_one_release.json) records the verified
wheel hash, per-file inventory and evidence hashes. The concrete local release
bundle is `dist/phase-one`; its manifest verifies without changing the historical
root release manifest.

The regression suite covers primitive averaging attacks, fractional scoring,
historical raw-response revalidation, forged and premature reveals, substituted
outcomes/metrics, duplicate confidence training, planned anchors, one-observation
variance, invalid PMFs, history cutoffs, and retry/restart/concurrency accounting.
Network responses in these tests are mocked; no keys or paid inference are needed.

An additional local instrument check read only `survey_text` from one public
example per official Mega development study. It validated 63 junk-fee questions,
22 hiring questions and 5 privacy questions and constructed 3, 40 and 1 outcomes.
This is a format compatibility check, not a study performance result.

The frozen Mega manifest still verifies as
`5fa5cdf8ee9e1e802a18f7c03b0fb756b0359011add037df42728a425aff05c0`.
Both existing study directories, all original implementation witnesses and the
SYN-DIGITS E/F design remain byte-identical to the starting commit. Existing
Windows-local raw ledgers were not available and were not modified or evaluated.

## Explicit v2 commands

These commands work from the verified source checkout and an installed wheel.
The wheel carries exact archived v1 witnesses separately from the current runtime.
This is an installable research development release. On Windows, replace `python`
with `.venv\Scripts\python.exe`. See `python -m rival mega-v2 --help` and the
[managed execution guide](MANAGED_EXECUTION.md) for general model execution.

For already frozen results whose sealed outcomes have been materialized, create a
new corrected report without changing the original ledger or report:

```sh
python -m rival.mega_study_v2 reanalyze --stage-root PATH_TO_STAGE --results PATH_TO_FROZEN_RESULTS --freeze-marker PATH_TO_FREEZE --json-report reports/reanalysis_v2.json --markdown-report reports/reanalysis_v2.md
```

For a separately authorized **new v2 run**, use a new results path, explicit budget,
expiry and physical-attempt limit. API credentials come from `RIVAL_API_KEY` or
`OPENROUTER_API_KEY`; never put them in command arguments. The `run --help` command
lists the arguments. The new runner rejects existing v1 rows and stops if outcomes
were already materialized in that stage. It is not a command to resume or repair
the user's interrupted frozen v1 experiment.

The adjacent `RESULTS.jsonl.attempts.sqlite3` is authoritative accounting state.
Preserve it with the result file. A missing journal cannot be reconstructed from
result rows that only record final-attempt usage. `freeze` requires complete v2
results and resolved accounting and binds the journal hash in the new marker.

If a worker times out or is interrupted, first stop the original worker. Inspect
`AttemptJournal.attempts_for(work_id)` and the provider generation/billing record.
Use `reconcile` only with the actual cost and a durable evidence reference; supply
the recovered completion when available. Without a recovered completion the work
remains terminal. No automatic replacement is sent. A claim interrupted strictly
between attempts can be released with `release_interrupted_claim` only after the
worker has stopped and the journal confirms no unresolved remote attempt.

## Remaining limits relevant to this batch

- Frozen v1 runners remain historical witnesses and are excluded from the current
  managed CLI. General current probability, behavioral and text-generation adapters
  require an explicit execution session. L07 now connects the saved study workflow
  through operator commands; a complete browser interface remains L13 work.
- HMAC seals are deployment-local. Key custodians, host clock, filesystem and
  database access remain trusted; independent custody and external timestamping
  are separate release work.
- Automatic confidence fitting is disabled. The evidence registry supports a
  protected research fit, but cannot establish study independence or qualified
  coverage. L10 adds runtime calibration; shrinkage, coverage and selective-risk
  tests still need L11–L12 and untouched data.
- Frozen v1 benchmark reports, earlier qualification artifacts and their release
  labels remain historical. Reanalysis does not turn development data into
  confirmation evidence. No E/F reference calls or adapters were executed.

Source references for instrument/accounting contracts:
[official Mega dataset](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study),
[OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).


L09 connects direct/SSR elicitation and model identity checks to the saved workflow.
Its real CPU rehearsal includes 12 valid SSR completions and two rejected direct
completions; failures remain in the record. Six generated seed distributions
matched exactly in the fresh SSR repeat. API cost was zero, with no assertion of
zero computing cost or production provider qualification. All 68 protected files
and completed v1/v2 example exports are unchanged. See the
[model execution guide](MODEL_EXECUTION.md) and [verification record](verification/phase_two_model_execution.json).

## L11 uncertainty milestone

The [runtime uncertainty guide](RUNTIME_UNCERTAINTY.md) describes the new prospective
cohort workflow. It freezes settings and every study group before execution, keeps
training/calibration/evaluation separate, counts failures, and binds the resulting
research assessment to a v5 target before outcomes. Public research reports retain
missed bounds and harmful candidate acceptances. No generated test can unlock
customer confidence. A valid integer-valued outcome PMF now compares numerically
with the typed evaluation while its original vault receipt remains unchanged;
booleans, strings and altered probabilities still fail.

Verification passes 235 tests and 17 isolated installed-wheel checks. All 68
protected files and completed v1–v4 example exports are unchanged.

L12 is next: freeze a relevant untouched public-evidence qualification, compare
strong baselines and measure error, coverage, abstention, failures, cost and time.
L13–L14 complete the operator GUI and delivery rehearsal.
