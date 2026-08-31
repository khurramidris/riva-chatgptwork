# SYN-DIGITS E/F Calibration Design

Status: **specified, not authorized for execution**

Upstream: [SYN-DIGITS](https://github.com/yw3453/syn-digits) at commit
db891b6f821c914455b11763a96679864bf4fc48. Paper:
[arXiv:2604.07513](https://arxiv.org/abs/2604.07513).

## Decision

Do not insert SYN-DIGITS into the raw A/B/C/D Mega-Study run. Establish the raw
baseline first and preserve it exactly. Add calibration only as a separately
hashed prediction supplement:

| Variant | Definition |
|---|---|
| E | Frozen C — Full Persona predictions transformed by SYN-DIGITS |
| F | Frozen D — Rival Retrieval predictions transformed by SYN-DIGITS |

E/F are not extra LLM answers to the Mega target surveys. They are calibrated
predictions derived from frozen C/D outputs. E is compared primarily with C and
F primarily with D. Raw A-D stays independently reportable if calibration
fails.

## Why the existing component is not enough

Rival already vendors the full MIT-licensed SyntheticControl component and
tests matrix completion and its ridge evaluation path. That proves software
availability, not the presence of a valid Mega-Study calibration matrix.

SYN-DIGITS needs matched human and digital-twin matrices with the same people
and reference outcomes. The current 300-result Wave-4 checkpoint cannot supply
this: its 50-person cohort overlaps the Mega cohort on only seven people, it is
incomplete, and its conditions are not Mega C/D. It must not be silently reused
as E/F training data.

## Historical reference bank

Create a new calibration-reference run without touching the paused Wave-4
ledger. Its rows are the 273 unique people in the frozen Mega cohort. Its 15
columns are the preregistered eligible Wave-4 targets from protocol
389be30eaa0ca7e2d342153b1641377ec6431f910fbc7b588e5539095f2ef06e:

- nonseparability ris _4
- 33_Q295
- Q159_3
- Q198_1
- False Cons. self _3
- Q163
- Q172
- Q194
- Q176
- Q183
- Q167
- Q192
- Q179
- Q161
- Q157

For every person/reference column, generate two answer-free predictions using
the exact Mega model route:

- C-reference: complete official Full Persona;
- D-reference: demographics plus target-blind Rival Retrieval.

This is 273 × 15 × 2 = 8,190 maximum reference calls. The reference prediction
runner must not import or accept human_outcomes.csv. After all reference
predictions are frozen, a separate calibration process may open only Wave-4
human reference values. Mega target and confirmation outcomes remain forbidden.

Wave-4 human cells are naturally incomplete, which SYN-DIGITS supports. Across
the 273 target people, observed human coverage per reference column ranges from
70 to 273. Synthetic reference predictions stay complete so matrix completion
has a consistent scaffold.

## Matrices

Build matrices separately for C→E and D→F. For each Mega target outcome:

1. retain only the 100 participants in that frozen target study;
2. use the 15 Wave-4 reference outcomes as donor columns;
3. append one raw Mega target-prediction column;
4. populate real donor columns from Wave-4 human responses;
5. populate synthetic donor and target columns from matching C or D output;
6. leave the entire real Mega target column NaN and unavailable.

Normalize each scalar outcome to 0–100 using its preregistered natural range.
This prevents arbitrary survey scales dominating the regression and fits the
SYN-DIGITS low-variance guard. Missing reference values are imputed only inside
the fixed procedure. Output is clipped to 0–100 and mapped back to the natural
scale.

The primary calibrated value stays continuous for normalized accuracy, MAE,
RMSE, correlation, and population means. A deterministic nearest-supported-code
projection is secondary and used only for exact or balanced accuracy.

## Frozen method

Use SYN-DIGITS target-column ridge impute-regress-transfer with regularization
multiplier 1e-6. Donors are restricted to the 15 reference columns. Imputation
rank is selected deterministically using synthetic-donor-only cross-validation;
no Mega human target is available to the selector.

Before Mega application, run leave-one-reference-column-out diagnostics. Each
reference column becomes a pseudo-target while the other 14 are donors. Report
calibrated-versus-raw error and coverage separately for C and D. This qualifies
the fixed method but does not permit tuning on Mega targets.

The vendored evaluator currently calculates the blind prediction internally
but exposes only evaluation summaries. A prediction-only adapter must be
implemented and parity-tested against it on observed reference columns. It must
accept an all-NaN real target column and return predictions without importing a
Mega outcome materializer.

## Required phase order

1. Freeze reference cohort, columns, prompts, provider, parser, budget, and
   source/code hashes.
2. Generate and freeze all C/D reference predictions.
3. Open Wave-4 reference outcomes only and build the paired matrices.
4. Run fixed leave-one-reference-column-out qualification.
5. Complete and freeze all 1,200 raw Mega A/B/C/D predictions.
6. Produce E/F while the Mega sealed directory does not exist.
7. Freeze E/F and bind it to raw-ledger and reference-matrix hashes.
8. Only then materialize development outcomes and evaluate A-F.

If Mega outcomes open before step 7, E/F loses prospective calibration status
and must be labeled exploratory.

## Evaluation

The raw preregistered A-D report remains unchanged. The supplement adds:

1. E − C: value of SYN-DIGITS after Full Persona;
2. F − D: value of SYN-DIGITS after Rival Retrieval;
3. F − E: calibrated Retrieval versus calibrated Full Persona;
4. E/F versus A and B as secondary product comparisons.

Use the raw benchmark's participant/outcome normalized accuracy, study
macro-averaging, failure accounting, and 10,000 participant-cluster bootstrap.
Reference failures and E/F abstentions remain visible. No confirmation claim is
allowed from these development studies.

## Execution blockers

This design does not pretend E/F is runnable. Four pieces still need separate
review and implementation:

- exact reference-case and cost manifests;
- the isolated same-route reference runner;
- the prediction-only SYN-DIGITS adapter and parity tests;
- E/F freezing and combined post-reveal evaluation.

These blockers do not prevent the raw A/B/C/D preflight. They mean only that
Mega human outcomes must stay closed after raw prediction freeze if a valid E/F
supplement will follow.
