# Runtime calibration (L10)

Rival can fit an adjustment from completed reference studies and apply it to a new
study. A v4 report shows raw simulated shares, calibrated shares and a historical
average baseline. Both predictions are fixed before target outcomes are opened;
the same authenticated outcomes score every method.

L10's engineering milestone is complete in `0.6.0.dev8`. Calibration remains
research output. Fitted weights and historical scores do not establish customer
accuracy, individual fidelity, qualified uncertainty or launch readiness.
L11–L14 remain: uncertainty, untouched qualification, operator GUI and delivery.

## Supported boundary

Calibration requires the same fixed seed panel, imported audience, controls,
model/endpoint/runtime settings and ordered choice schema. Question text may vary;
choice labels, features and SSR anchors cannot change. Different populations,
models, alternatives or response scales require a compatible new reference bank.
This does not establish semantic support for an arbitrary new question.

Direct probability and pinned semantic SSR responses are supported. Every eligible
seed must appear in every study; increase `sample_size` before preparation if a
random draw omits a seed. Repeated draws contribute one probability vector per
seed, not extra training observations. Calibration learns a nonnegative weight for
each seed plus shared base-choice weights. It operates on aggregate distributions.

## Operator workflow

Prepare and execute v3 reference studies using the evidence and model guides.
Assign `evidence.role: training` with a distinct `group_id` **before prediction
lock**. A custodian deposits reference outcomes in an external vault; the normal
`study evaluate` flow authenticates and records them. Development, calibration,
evaluation and unrevealed studies cannot enter a training bank. Generated training
fixtures are permitted, retain their generated labels and are not human evidence.

Use a new calibration catalog directory and at least two training groups:

```sh
python -m rival calibration bank --catalog .rival-data/calibration --workspace .rival-data/reference-a --workspace .rival-data/reference-b
python -m rival calibration fit --catalog .rival-data/calibration --bank BANK_SHA256
```

`bank` returns `bank_sha256`; `fit` returns `adapter_sha256` and diagnostics.
Fitting takes only the bank, makes no new model calls and has no target-outcome
argument. Repeating identical inputs/settings returns the same artifact hash.

Optional `fit --settings FILE.json` accepts `CalibrationSettings`. Defaults are
mean KL loss, 500 iterations, learning rate 1, persona/base L2 penalties `1e-6`,
gradient norm limit 10 and simplex-gap tolerance `1e-6`. Settings enter the artifact
identity. Choose them using training/development evidence before final evaluation.
An iteration limit is not convergence: actual gaps are retained and unconverged
fits produce a report warning.

In a new full study input, retain the v3 evidence and execution fields, set
`schema_version` to `rival.study-request.v4`, and add:

```json
{"calibration": {"adapter_sha256": "THE_64_CHARACTER_ADAPTER_HASH"}}
```

Use a new target study/group ID and question. Its information cutoff must be at or
after the authenticated reference reveals. Then:

```sh
python -m rival study check --input target.json --catalog .rival-data/evidence --calibration-catalog .rival-data/calibration
python -m rival study prepare --input target.json --workspace .rival-data/target --catalog .rival-data/evidence --calibration-catalog .rival-data/calibration
python -m rival study run --workspace .rival-data/target
python -m rival study export --workspace .rival-data/target --output reports/target-before-outcomes
```

Preparation makes no model calls. It snapshots the verified adapter into the
signed workspace. Its hash also enters the raw prediction's scenario metadata and
sealed manifest, without entering the model prompt. Execution retains managed
budgets, attempt journals and recovery. The calibrated result has its own signed
checkpoint bound to the raw simulation and completion record. Resume verifies
its exact derivation and makes no additional inference calls.

When independent target outcomes become available, use the existing vault flow:

```sh
python -m rival study evaluate --workspace .rival-data/target --vault PATH_TO_TARGET_VAULT
python -m rival study export --workspace .rival-data/target --output reports/target-after-outcomes
```

The custodian supplies `RIVAL_OUTCOME_KEY`. Evaluation recomputes raw, calibrated,
uniform-panel and training-mean TVD, Jensen–Shannon divergence and squared
distribution error against the same exact choice set. Negative TVD reduction
means calibration made that study worse. Evaluation never refits the adapter.
Confidence remains `unqualified` and abstains.

## Evidence and recovery

Banks are content addressed and locally signed. Harvest verifies protected
evaluations, pre-lock training assignments, complete denominators and identical
source/panel/model identities. Duplicate studies, groups, normalized question and
context fingerprints, runs, manifests or reused model response IDs are rejected.
Target groups/questions cannot overlap references, nor can target responses reuse
reference request IDs. Reference outcomes must precede the target cutoff.

Keys, host clocks, local files and custodians remain trusted. Local attestations
do not establish truthful source metadata, representative sampling, effective
human sample sizes or semantic independence. Renaming related questions and falsely
declaring independent groups cannot be reliably detected. There is no global
registry across independent catalogs or external timestamp witness.

Back up complete catalogs and workspaces. A prepared target can resume using its
saved adapter without the catalog; missing original keys cannot be replaced to
manufacture history. Aggregate exports exclude seed IDs, reference response
vectors, persona coefficients and keys. Existing v1/v2/v3 schemas and completed
exports are preserved.

## Relationship to SYN-DIGITS

The source is [SYN-DIGITS section 7](https://arxiv.org/html/2604.07513v1#S7)
and its [distribution-calibration code](https://github.com/yw3453/syn-digits/blob/db891b6f821c914455b11763a96679864bf4fc48/src/distribution_calibration.py)
at `db891b6f821c914455b11763a96679864bf4fc48`. The archived implementation,
licenses, research adapter, datasets and frozen studies remain unchanged.

The runtime uses `q[j,k] = base[k] + sum_i weight[i] * response[j,i,k]`, with the
combined persona/base coefficients on one simplex. One-hot responses recover the
published distributional ensemble. Soft direct/SSR responses are an explicit
Rival extension that preserves the full vectors instead of applying argmax.
This is not the paper's individual-level synthetic-control method or proof that
SSR scores are human probabilities.

The independent solver uses mean KL and separate persona/base L2 penalties. It
omits upstream's optional ordinal-MSE penalty, accepts explicit training groups
instead of randomly splitting internally, treats zero observed mass without
artificial counts, uses a `1e-12` prediction floor rather than `1e-6`, and adds
stable mirror steps, backtracking and a simplex-gap diagnostic. With MSE disabled,
the archived class's objective and gradient match five strictly interior one-hot
cases to numerical precision. Soft gradients pass finite differences; a small
fit agrees with independent SciPy SLSQP within `2e-6` objective value. These verify
the stated adaptation, not every upstream option or experiment.

## Development evidence

A retrospective five-fold comparison scores all 489 bundled public OpinionQA
questions, 464 lexical question families and 2,058 released personas. Each test
question uses weights fitted on other families. These data were already used in
development and are **not untouched qualification**.

| Method | Mean TVD; lower is better |
|---|---:|
| Uniform uncalibrated personas | 0.331125 |
| Training-family historical mean | 0.226775 |
| Prior research adapter, 150 iterations | 0.169495 |
| Runtime adapter, default 500 iterations | 0.166793 |

Runtime error is 49.63% lower than raw responses, 26.45% lower than the historical
mean and 1.59% lower than the previous adapter. It wins on 417/489, 358/489 and
269/489 questions respectively; losses remain in the report. Defaults differ,
so this comparison does not isolate a single cause. All five fits hit 500
iterations before the `1e-6` convergence tolerance. No settings were selected
using these results. See [all question results and diagnostics](verification/phase_two_calibration_artifacts/opinionqa-runtime.json).

These released responses lack the live provider attestations required by the
runtime bank. This measures the numerical component and does not promote old
responses into a current model's reference bank. Five-choice one-hot data cannot
qualify soft SSR outputs or customer domains.

The separate [workflow rehearsal](examples/calibration/rehearsal.json) uses generated
people, response fixtures and outcomes over real loopback HTTP. Two training and
two target studies complete 24 requests and 160 draws. One target improves and one
worsens; both include the historical baseline. There are zero real LLM calls in
this rehearsal. L09's real-model evidence remains a separate record.

```sh
python scripts/run_calibration_rehearsal.py --workspace .rival-data/calibration-example --output reports/calibration-example
python scripts/compare_runtime_calibration.py --output reports/calibration-development.json
```

Use new output paths. These scripts exercise the engineering workflow and reproduce
the already-used public-data comparison without inference. Neither opens frozen
Wave4/Mega studies or executes the E/F calibration design.
