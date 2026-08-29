# Rival Twin-2K-500 Mega-Study Development Protocol

Status: frozen before the first real API call  
Protocol: `rival-twin2k-500-mega-development-v1`  
Manifest identity: see `MEGA_STUDY_MANIFEST.json`

## Purpose

This is a new, parallel benchmark. It does not replace or reinterpret Rival's
existing 1,500-case Twin-2K Wave-4 experiment. The existing experiment asks
whether prior waves predict a later retest. This benchmark asks whether old,
person-specific information predicts the same real person's behavior in a
genuinely new later experiment.

For participant P:

1. use P's original Twin-2K history;
2. show an answer-free, later Mega-Study experiment;
3. predict P's structured survey response;
4. freeze every prediction;
5. only then open P's recorded Mega-Study response and score it.

The preserved Wave-4 protocol and cases are protected by the two hashes in the
manifest. Mega-Study code has no write path to them.

## Official benchmark snapshot

- Paper: [Twin-2K-500 Mega-Study](https://arxiv.org/abs/2509.19088)
- Official repository commit:
  `afe2bb933fce377ed196f441a4c12962cb55a53a`
- Mega-Study dataset revision:
  `0401b715a341ac4b5f98b4424b4aecf9d29570d0`
- Original Twin-2K persona revision:
  `f883165a3026fde855dfd448e0cd16443ab257b6`

Every downloaded Parquet or official reference file is size- and
SHA-256-verified. Exact URLs, sizes, licenses, and hashes are in the manifest.

## Development studies and cohort

The exact official split names are:

| Study | Eligible official rows | Frozen cases | Preregistered outcomes |
|---|---:|---:|---:|
| `junk_fees` | 400 | 100 | 3 |
| `hiring_algorithms` | 999 | 100 | 40 |
| `privacy` | 1,200 | 100 | 1 |

Selection is outcome-blind. A row is eligible when it has nonempty
`survey_text` and maps to an original Twin-2K full persona. Within each study,
eligible rows are ranked by `SHA-256(20260829|study_id|pid)` and the first 100
are retained. The frozen cohort contains 300 cases and 273 unique people.

The four variants are rotated within each study. Exactly 25 cases begin with
each variant. Calls are independent; no conversation state is shared.

## Controlled A/B/C/D conditions

All conditions use the same model, upstream provider, system instruction,
temperature, seed, maximum output, structured-output request, parser, and
retry policy. Only the person representation changes.

| Variant | Person-specific input |
|---|---|
| A — Generic | None |
| B — Demographics | The 14 pre-existing fields in the official Demographics block |
| C — Full Persona | The complete official `persona_text`; no truncation |
| D — Rival Retrieval | Demographics plus target-blind BM25/MMR retrieval from that person's earlier Twin-2K answers |

Rival retrieval uses only the answer-free new survey as its query. It selects
up to 24 historical items under a 16,000-character evidence limit. Selection,
scores, ranks, evidence IDs, and evidence hashes are recorded. The retrieval
API has no outcome argument.

No SSR, calibration, OSim, fine-tuning, multi-agent simulation, or architecture
tuning is part of this phase.

## Frozen model route

- API router: OpenRouter
- exact model: `deepseek/deepseek-v4-flash-0731`
- exact upstream: `DeepInfra`
- fallbacks: disabled
- temperature: `0.0`
- seed: `20260829`
- reasoning: disabled and excluded
- response format: JSON object
- maximum output: 16,384 tokens

The 1,200-prompt audit found a maximum prompt size of 174,453 characters. The
policy limit is 180,000 characters and no prompt is truncated. Full Persona
and Rival Retrieval remain distinct: their mean sizes are approximately
153,249 and 41,100 characters, respectively.

## Leakage firewall and phase order

Prediction preparation reads only `PID` and `survey_text` from each Mega-Study
split. It writes only answer-free cases and original Twin-2K personas. It does
not materialize a target response or target-derived hash.

The immutable source-file checksums, cohort IDs, and frozen extraction rules
commit the target set before calls without exposing target values.

The required order is:

1. `prepare`: build and audit prediction-only inputs;
2. `preflight`: call all four variants for one case;
3. `pilot`: complete all 1,200 frozen work items;
4. `freeze`: hash the immutable result ledger;
5. `evaluate`: only now read `survey_json_with_human_response` into a separate
   sealed directory and calculate metrics.

The prediction runner accepts no source-cache or outcome path. Evaluation
refuses to start without a valid freeze marker, and refuses a ledger modified
after freezing. Any leakage-audit failure is fatal.

## Output parsing and outcomes

One call returns the complete structured survey response, matching the
authors' full-survey convention. The parser requires a JSON object, contiguous
`Q1..Qn` survey labels, the expected answer structure, numeric positions, and
preregistered natural ranges. Unsupported or malformed output is recorded as
a failure; it is not removed from the denominator.

Outcome construction follows the official repository. The exact maps,
recodes, and ranges are in `OUTCOME_MAPPING.json` and implemented in
`rival/mega_study/outcomes.py`.

## Evaluation

The primary individual metric is the authors' normalized accuracy:

`1 - absolute error / natural outcome range`

It is computed by participant/outcome, then macro-averaged across outcomes and
studies. Pearson correlation is calculated across participants separately for
each outcome and aggregated with a Fisher-z mean. Spearman, exact accuracy,
balanced accuracy where applicable, MAE, RMSE, population-mean error, absolute
Glass delta, SD ratio, and discrete distribution TVD are also retained.

The six paired contrasts are:

1. Demographics − Generic
2. Full Persona − Generic
3. Rival Retrieval − Generic
4. Full Persona − Demographics
5. Rival Retrieval − Demographics
6. Rival Retrieval − Full Persona

Lift is the paired difference in normalized accuracy on the same
participant/outcome cells. Uncertainty uses 10,000 participant-cluster
bootstrap resamples and a 95% percentile interval. Results are reported both
by study and as a study-macro average.

All frozen cells remain in the denominator. API, parsing, missing-history,
missing-outcome, unsupported-question, and context failures are explicit.
Matched valid cells are used for paired lift, while coverage and failure counts
remain visible.

## Official-method compatibility check

Before any new model spending, Rival reproduces the authors' published
Digital Certification full-persona reference output. This study is not a
development or confirmation study. Rival exactly matches 600-person pairing,
DV1 and log-WTP construction, normalized accuracy where defined, both Pearson
correlations, human/twin means, and human/twin standard deviations (within
`1e-12`). The preparation audit also verifies all 300 persona mappings, all
survey renderings, all 1,200 prompts, provider identity, and the
prediction-only firewall.

## Calls, budget, and resumability

There are 300 cases × 4 variants = 1,200 calls. Calls are sequential. Results
append to a durable JSONL ledger and completed work IDs are skipped on resume.
The runner checks a user-supplied dollar ceiling and UTC expiry before each
call. It stops after the frozen retry policy and the configured error limit.

At the pinned public prices, a conservative estimate using mean prompt sizes
and maximum output on every call is `$5.494368`; the all-calls, maximum-context
and maximum-output bound is `$6.158352`. The recommended hard budget is `$7`.
Actual provider-reported cost is used whenever available.

## Commands

From the repository root with Python 3.11:

```console
.venv\Scripts\python.exe scripts\prepare_mega_study.py
```

After preparation reports `PASS`, run the four-call preflight:

```console
.venv\Scripts\python.exe scripts\run_mega_study_secure.py --phase preflight --budget-usd 0.10 --expiry-minutes 30
```

Then run in resumable 100-call checkpoints:

```console
.venv\Scripts\python.exe scripts\run_mega_study_secure.py --phase pilot --max-new-calls 100 --budget-usd 7.00 --expiry-minutes 120
```

Repeat the checkpoint command until `PILOT_PREDICTIONS_COMPLETE`, then:

```console
.venv\Scripts\python.exe scripts\freeze_mega_study.py
.venv\Scripts\python.exe scripts\evaluate_mega_study.py --no-download
```

The equivalent product entry point is `rival mega-study` with `prepare`,
`audit`, `run`, `freeze`, and `evaluate` subcommands. The secure script is
recommended because it prompts for the API key without echoing it.

Do not run the freeze or evaluation commands early. Do not inspect or tune on
partial results.
