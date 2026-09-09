# Frozen qualification: L12 implementation increment

Rival now has a qualification workflow attached to the existing protected study
pipeline. It freezes the rules, human sample design and study roster; compares the
calibrated hybrid with four baselines; and publishes failed and missing studies.
**L12 remains in progress.** This increment supplies engineering, not fresh human
accuracy evidence, a completed prospective study, or permission to sell qualified
decision support. L01–L11 remain the 11 completed engineering milestones.

## What is compared

The candidate is the v5 study's pinned calibrated distribution. Its calibration
adapter, uncertainty assessment, model, visible seed panel, choices, evidence,
audience and execution settings retain the existing v3–v5 bindings.

| Comparator | Information permitted |
|---|---|
| Synthetic only | The same run's uncalibrated aggregate prediction |
| Weighted history | Human distributions from protected training-role v3 studies, averaged with frozen study weights |
| Classical multinomial regression | Those training distributions and explicitly declared outcome-free numerical study features |
| Human only | A separate, prospectively enumerated human sample under the same budget cap |

The classical baseline is Rival-written regularized multinomial regression,
fitted using weighted cross-entropy and L-BFGS-B. It is not a hierarchical model
and is not claimed to be the strongest baseline for every domain. Operators must
justify the selected features and comparator for their study family. Price and
other stimulus attributes available before asking a question can be legitimate
features; post-treatment trust, engagement, purchase intention or other answer
fields cannot be used as outcome-free predictors. The fitting code does not
authenticate that semantic judgment.

Features are standardized using training data only. The intercept is unpenalized;
slopes receive the locked ridge penalty. Constant features recover weighted
history. Weights are **study weights**, not claimed respondent counts or survey
degrees of freedom. Optimizer nonconvergence is retained and fails its gate.
See [SciPy's optimizer contract](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
No new third-party source extraction is part of this increment.

## Freeze and execute

1. Complete protected v3 training studies. Supply their numeric features, study
   weights, feature provenance, relevance justification and baseline fit settings.
   At least three distinct training groups are required for this software path;
   this minimum is not a scientific sample-size recommendation.
2. Prepare every final evaluation study as v5, with the same distribution adapter
   and uncertainty assessment. Build a protocol listing every study, source,
   instrument hash, rights reference, date, independence rationale, feature row,
   anchor unit, evaluation unit and unit weight. The protocol requires an audience,
   geography, intended decision, threshold rationale and subgroup scope.
   The audience description and ordered geography list must match the prepared
   study audience exactly; the qualification report cannot relabel its population.
3. Freeze the plan before **any** target model execution or outcome reveal.
   Source workspaces receive an immutable cohort reservation. Changing rules or
   using another catalog cannot overwrite that reservation. Finish planning before
   running workers. Interrupted reservations can resume with the identical plan.
4. Run each study through `rival study run`. Provide its separate human anchor
   sample and seal all four baseline forecasts before revealing its evaluation
   outcomes. An anchor response never adjusts the Rival candidate in this workflow.
5. A custodian deposits the evaluation distribution into the existing outcome
   vault, bound to the sealed study. Run `rival study evaluate` as usual. Retain the
   unit-level evaluation sample, including missing responses, for qualification.
6. Assess the entire frozen roster once and export aggregate JSON and Markdown.
   A study omitted from the assessment command remains as a failed study.

```sh
python -m rival qualification schema
python -m rival qualification fit --catalog CATALOG --design TRAINING_DESIGN.json --settings BASELINE_SETTINGS.json --workspace TRAIN_A --workspace TRAIN_B --workspace TRAIN_C
python -m rival qualification plan --catalog CATALOG --baselines BASELINES_SHA256 --protocol PROTOCOL.json --workspace TARGET_A --workspace TARGET_B
python -m rival study run --workspace TARGET_A
python -m rival qualification seal --catalog CATALOG --plan PLAN_SHA256 --workspace TARGET_A --anchors ANCHORS_A.json
python -m rival study evaluate --workspace TARGET_A --vault EXISTING_OUTCOME_VAULT
python -m rival qualification assess --catalog CATALOG --plan PLAN_SHA256 --workspace TARGET_A --workspace TARGET_B --outcomes OUTCOMES.json --operations OPERATIONS.json
python -m rival qualification export --catalog CATALOG --assessment ASSESSMENT_SHA256 --output NEW_REPORT_DIRECTORY
```

Repeat the run/seal/evaluate sequence for all targets before final assessment.
The example command's two targets illustrate syntax; two studies cannot satisfy
the human-evidence minimum or statistical gates. Never replace the final outcome
file with anchor results. Vault credentials stay in the existing environment-key
mechanism, not in command arguments.

`fit` returns `baselines_sha256`; `plan` returns `plan_sha256` and a feasibility
check. `seal` takes one workspace. `assess` returns `assessment_sha256` and the
report, exiting **1** on FAIL/UNEVALUABLE, **0** on PASS, and **2** on an invalid
operation. Inspect/export are read-only result operations and return 0 on success.
An assessment is immutable. Repeating it returns its original result, even if
previously unfinished studies have since completed. Finalize only when the planned
execution window has closed. Failed cases cannot disappear by omitting workspaces.

## Human samples and accounting

Each sample is `{"source_reference": "...", "rows": [...]}`. Each row contains
`unit_sha256`, `choice_id` (or `null` for missing), and `weight`. Unit hashes are
stable pseudonymous recruitment identifiers, not raw personal identifiers. The
protocol freezes their complete roster and weights. All units are unique across
anchor/evaluation samples and final independent study groups. Participant identity
and independence remain source-custodian responsibilities; hashing does not prove
that two records belong to different real people.

The outcome and operations files are objects keyed by study ID. Every evaluation
sample must reproduce the **exact** distribution authenticated by the protected
vault. Unplanned choices, duplicate/omitted units, altered weights or mismatched
outcome distributions are rejected. Missing values remain in count and weight
denominators. Reports include observed/effective counts and missing weight mass.

The human-only quota is `floor(budget_per_study_usd / human_unit_cost_usd)`. The
protocol requires that full quota, so operators cannot weaken this baseline by
providing an arbitrarily tiny sample. The unit cost must reflect the declared
recruitment design. Actual total human-only cost and turnaround are required at
assessment. Both methods must fit the same budget cap; equal actual spending is
not asserted. This is a historical sample-cost comparison unless actual fieldwork
and operating receipts establish otherwise.

Operations require `candidate_other_cost_usd`, `human_only_total_cost_usd`,
`human_only_turnaround_seconds`, and `receipt_reference`. Candidate API charges
come exclusively from the attempt journal, including failed attempts. The other
cost field must include compute, staff, data, amortized reference/model fitting
and any other charges outside the target API journal. Do not enter zero for an
unknown cost. Omitted operations make the cost/time gate UNEVALUABLE. Candidate
runtime is measured from qualification-plan freeze to model completion; complete
service turnaround, including prior preparation and delivery, still needs L14.

## Statistical and evidence gates

Each declared independent study contributes one observation. Synthetic draws,
seed personas and multiple questions from one human study cannot manufacture
independent study evidence. Semantic grouping still needs an actual source audit.

- The primary all-roster error is TVD, conservatively increased by missing human
  weight mass and capped at 1. This bounds missing-response sensitivity within the
  frozen eligible sample; it does not cure selection bias outside that sample.
- Paired baseline improvement is baseline TVD minus Rival TVD, reduced by twice
  missing weight mass for a conservative comparison on the eligible sample.
- Unscored groups receive error 1 and paired improvement −1 against every
  baseline. A saved acceptance with missing outcomes counts as a bad acceptance.
- One-sided Hoeffding bounds cover mean error and four paired improvements.
  Their ranges are [0,1] and [−1,1]. Four exact Clopper–Pearson gates cover observed
  distribution coverage, acceptance frequency, bad acceptance and failure rates.
  Bonferroni divides the locked family error across these nine statistical gates.
  These claims require independent studies; binomial interpretation additionally
  assumes exchangeable groups. They cover one fixed protocol, not repeated model
  search or choosing a favorable report after inspecting many tests.
- All operating, missingness, convergence and human-evidence gates must also pass.
  A subgroup request stays UNEVALUABLE: this increment has no protected subgroup
  forecast/outcome path. Empty priority groups mean an explicitly aggregate-only
  study, not implicit subgroup qualification.

The initial minimum remains three untouched historical human groups and one
genuinely prospective group. Historical means existing data whose outcomes the
model team had not inspected when freezing the test. Prospective means the sealed
model prediction predates declared fieldwork and authenticated reveal follows
declared outcome availability. Software verifies local timing, not the truth of
external fieldwork dates. Previously used OpinionQA/Twin-2K data and generated
fixtures cannot satisfy these gates.

Those counts are structural minima. **Four studies are generally too few to pass
the statistical gates.** Plan output reports best-case feasibility before spending
on execution. Hoeffding bounds can require many independent studies for modest
improvements; a merely feasible design is not necessarily well powered. Choose a
feasible, decision-relevant protocol before collecting outcomes. Do not weaken a
frozen threshold after a negative result.

Even PASS is evidence about one declared aggregate experiment. It does not flip
Rival's operational confidence badge or authorize launch. External custody,
source auditing, subgroup extensions where requested, and delivery gates remain.

## Public evidence and remaining work

[PUBLIC_QUALIFICATION_EVIDENCE.md](PUBLIC_QUALIFICATION_EVIDENCE.md) records the
metadata-only source screen. No new human outcome file was downloaded, examined,
or scored during this increment. No public source has been admitted as a locked
qualification set yet. This preserves the ability to review instruments and
freeze a defensible test before opening outcomes, without pretending an available
dataset is already a validated study family.

Next: pin an eligible source's answer-free instrument and coding; audit the exact
stimuli, sampling, response scale and semantic groups; prepare compatible training
and uncertainty evidence; freeze a sufficiently informative final roster; execute
the chosen model with operating receipts; and obtain future human fieldwork with
independent outcome custody. No paid model or fieldwork budget is assumed here.
L13 interface development can proceed while those empirical inputs are prepared.

## Generated verification example

Run `python scripts/run_qualification_rehearsal.py --workspace NEW_PRIVATE_ROOT
--output NEW_EXPORT_ROOT` from an installed checkout (or with `PYTHONPATH=.`).
The [example report](examples/qualification/report.md) retains a helpful
calibration, a harmful calibration, missing human responses and an unfinished
study. It correctly fails qualification. Generated monetary receipts are only
accounting fixtures. API calls go to a loopback fixture, not an actual LLM.

Catalogs and workspaces contain private sample hashes and local HMAC keys. Keep
them intact. Public exports contain aggregate statistics and evidence hashes;
they exclude unit rosters, raw model responses, coefficients and signing keys.
Custodians, clocks, source truth, feature semantics and non-journal costs remain
trusted. The original Wave-4, Mega v1 and E/F experiments remain unchanged.
