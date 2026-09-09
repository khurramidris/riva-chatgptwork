# Study-level uncertainty and abstention (L11)

Version `0.6.0.dev9` connects a protected uncertainty experiment to saved studies.
It estimates the error of an aggregate raw or calibrated choice distribution,
sets a research upper bound using separate calibration studies, and tests one
predeclared acceptance policy on a third partition. A v5 study pins that evidence
before execution and exports the assessment alongside its prediction.

This is an engineering milestone. **Customer confidence still abstains and keeps
the full [0,1] error range.** Research bounds are a separate, clearly labelled
report field. Passing generated tests cannot enable decision support. L12 must
establish performance on relevant untouched human evidence; L13–L14 still cover
the complete operator interface and customer delivery.

## What the method measures

The target is total variation distance (TVD) between a saved aggregate prediction
and its protected observed study distribution. It is not individual accuracy, a
causal effect, a confidence interval for each choice share, or a guarantee about
the underlying population's true proportions. Sampling error in the human study
is not removed by this method.

One independent study group contributes one row. Repeated simulations, extra
draws or more personas do not increase the number of error-calibration examples.
The current cohort admits one question per group and requires the same provider,
revision, elicitation settings, visible panel, imports, audience/support policy,
choice schema, context, sample size and sampling seed. If distribution calibration
is used, its adapter hash must also match. The confidence cohort is disjoint from
that adapter's own training evidence.

The error model is a ridge regression trained on four outcome-free summaries:
distribution entropy, largest share, gap between the two largest shares and mean
disagreement between seed probability vectors. Scaling and observed feature
ranges are fitted only on training groups. Predicted errors are clipped to [0,1].
At least five training groups are required to fit; five is an engineering minimum,
not a statement that five studies suffice for useful confidence.

For each calibration group, the nonnegative score is `max(0, observed_error -
predicted_error)`. With `n` calibration groups and coverage `c`, use order statistic
`ceil((n+1)*c)`, counting from one. If the rank exceeds `n`, use the full range.
Otherwise the research upper bound is `min(1, predicted_error + score_quantile)`.
The nonnegative score deliberately avoids shrinking below the expected error.
Features outside the training range also receive the full range and abstention.

This is an independently implemented one-sided split-conformal construction,
following the score/quantile construction in Angelopoulos and Bates,
[A Gentle Introduction, arXiv:2107.07511v6](https://arxiv.org/pdf/2107.07511v6),
sections 1.1 and 3. No code was copied or newly vendored. Its coverage is marginal
over exchangeable calibration and new study groups. It is not conditional
coverage for a selected subgroup or for accepted predictions. Independence,
exchangeability, outcome truth and semantic relevance require evidence beyond
local hashes and declarations. A feature-range check is not a semantic novelty
detector; a changed market can produce an overconfident research estimate.

## Frozen policy and evaluation

Before **any** cohort execution, `uncertainty plan` freezes all study requests,
roles, independent group IDs, normalized question fingerprints and settings.
Every workspace receives an immutable cohort reservation. A second catalog
cannot reserve the same workspaces with different settings or a different roster.
Local key custody and the host clock remain trusted. Coordinated execution must
start after planning returns; an interrupted reservation is resumable with the
same unchanged roster and settings before execution.

Run training studies first, fit once, then run calibration studies. Their sealed
predictions must postdate the fitted artifact. After calibration freezes the bound,
run evaluation studies; their seals must postdate the bound. Every stage rechecks
prospective roles, completed predictions, protected outcome reveals and exact
error targets. Reused response IDs, runs, manifests, groups and questions fail.

Thresholds are decision-specific and required explicitly. Example settings below
are illustrative, not a recommended acceptance bar:

```json
{
  "max_tvd": 0.2,
  "required_coverage": 0.85,
  "max_bad_acceptance_rate": 0.1,
  "max_failure_rate": 0.1,
  "nominal_coverage": 0.9,
  "family_error_rate": 0.05,
  "ridge": 1.0
}
```

The candidate policy accepts only a supported feature vector whose research upper
bound is within `max_tvd`. No threshold is selected using evaluation outcomes.
Evaluation reports coverage, acceptance, average error, error among accepted
studies, and the fraction of accepted studies exceeding the declared tolerance.
Three one-sided Clopper–Pearson gates test minimum coverage, maximum bad-acceptance
rate and maximum execution/outcome failure rate. Each uses one third of the
declared family error budget (Bonferroni). The beta-quantile implementation is
checked against [SciPy's exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Every evaluation workspace in the frozen roster must be supplied. Incomplete
execution and missing protected outcomes are retained as failures, counted in
the total denominator and as noncoverage. Zero accepted studies cannot pass the
accepted-error gate. One immutable assessment is permitted per bound in the
catalog. Finalizing an incomplete cohort records that failure permanently; finish
the intended executions before invoking evaluation. Repeating the stage returns
the original artifact rather than replacing it with improved results.

The three-gate error budget applies to this one fixed policy and cohort, not to
searching across many separate experiments. Someone with custody of all keys and
files can copy or re-create studies elsewhere. There is no global trial registry,
external timestamp authority or automatic proof that two differently worded
questions or two declared study groups are statistically independent.

## Operator sequence

1. Prepare fresh pinned v3 or v4 workspaces with training, calibration and
   evaluation roles. For v4, fit the separate distribution adapter first.
2. Freeze the entire uncertainty roster, including the evaluation workspaces.
3. Run and evaluate all training studies through the existing protected vault
   workflow; fit the error model. Only then execute calibration studies.
4. Calibrate the bound, then execute the evaluation studies and assess the policy.
5. Pin the resulting assessment in a fresh v5 study; check, prepare, run and export.

```sh
python -m rival uncertainty plan --catalog UQ_ROOT --settings settings.json --workspace TRAIN_A --workspace TRAIN_B --workspace TRAIN_C --workspace TRAIN_D --workspace TRAIN_E --workspace CAL_A --workspace EVAL_A
python -m rival uncertainty fit --catalog UQ_ROOT --plan PLAN_HASH --workspace TRAIN_A --workspace TRAIN_B --workspace TRAIN_C --workspace TRAIN_D --workspace TRAIN_E
python -m rival uncertainty calibrate --catalog UQ_ROOT --fit FIT_HASH --workspace CAL_A
python -m rival uncertainty evaluate --catalog UQ_ROOT --bound BOUND_HASH --workspace EVAL_A
python -m rival uncertainty inspect --catalog UQ_ROOT --assessment ASSESSMENT_HASH
python -m rival study schema --version v5 --output schema-v5.json
```

The seven-workspace example illustrates arguments. With the settings above, one
calibration group yields an uninformative bound and one evaluation group cannot
establish the required gates. Declare enough independent studies for the intended
coverage and evidence strength. On Windows use `.venv\Scripts\python.exe`.

A v5 request uses the v3 fields plus
`"uncertainty": {"assessment_sha256": "ASSESSMENT_HASH"}`. It may also include
the v4 `calibration` pin. Its information cutoff must follow the completed
uncertainty assessment, and its group/question must be separate from every
training, calibration and evaluation member. Use `--uncertainty-catalog UQ_ROOT`
with `study check` and `study prepare`; include `--calibration-catalog` when using
a distribution adapter. Preparation makes no model calls.

Preparation embeds the verified artifact chain in the private signed workspace.
The uncertainty prediction is saved before outcome reveal and linked to the
completed study seal. Resume needs no original catalog and makes no additional
model calls for completed responses. Export recomputes the derivation before
accepting the saved assessment. Outcome comparison records missed bounds and bad
candidate acceptances without fitting again. Aggregate reports omit model
coefficients, per-person IDs, raw model responses and signing keys.

## Evidence in this milestone

Verification passes **235 tests**, including 20 new uncertainty and outcome-boundary
regressions, and **17 installed-wheel checks**. Integrity passes 10/10 and research
component reproduction 8/8. The 68 frozen/protected files remain byte-identical;
completed v1/v2/v3/v4 examples resume and export identical files.

The numerical tests cover exact finite-sample ranks, empty/small samples, ties,
independent binomial-interval parity, risk under selective acceptance, distribution
shift and all failure denominators. Workflow tests cover raw and calibrated
targets, split ordering, immutable settings, evidence reuse, drift, tampering and
crash recovery. The installed-wheel probe repeats the new workflow over real
loopback HTTP outside the checkout.

The [generated rehearsal](examples/uncertainty/rehearsal.json) freezes 5 training,
19 calibration and 60 evaluation groups, then executes two new targets. One
target follows the original process and one deliberately changes its outcomes.
The latter exposes a missed research bound even after the fixture evaluation
passes. Both retain customer abstention. These are generated responses and
outcomes, with zero real LLM calls, zero API fees and no claim of zero compute
cost or human accuracy. Run with new paths:

```sh
python scripts/run_uncertainty_rehearsal.py --workspace .rival-data/uncertainty-example --output reports/uncertainty-example
```

Frozen Wave-4/Mega experiments, the E/F design and archived upstream code are
preserved. The previous development-data calibration comparison is not reused as
untouched confidence evidence. See [verification](verification/phase_three_uncertainty.json).

The initial larger rehearsal exposed an integer/float representation mismatch
between an original vault outcome (`1`) and a typed evaluation (`1.0`). The new
comparison validates numeric probabilities, preserves the original receipt and
requires exact numeric equality; strings, booleans and altered probabilities
remain rejected. The initial failure log is retained in the verification artifacts.
The completed rehearsal records 516 HTTP fixture requests, 3,440 completed draws,
zero additional resume requests and approximately 49 seconds on the recorded host.
Its 60 generated evaluation groups all pass the fixed policy; the subsequent
changed-process target has TVD 0.925 against a research upper bound of 0.08.
That is an explicit failure of generalization under distribution shift, not an
accuracy claim or a reason to suppress the result.
