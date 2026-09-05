# Confidence qualification v1 — frozen exploratory protocol

Date: 2026-09-05. Parent repository revision: `902a3eb975b9e765f5a0c41e66afb6abe2ead5c6`.

Question: can a cheap error predictor identify unreliable aggregate survey predictions on public human outcomes, and do independently calibrated upper error bounds transfer?

This is a retrospective qualification experiment using the existing Rival OpinionQA subset: 489 questions, 2,058 released synthetic persona answers, and released human choice counts. No new LLM predictions are generated. Human counts are normalized as released; these are not a reconstruction of survey-weighted Pew population estimates. Individual fidelity, customer performance, treatment effects, and proprietary Simile parity are outside its claim scope.

## Data roles and fixed splits

Four disjoint roles are mandatory: A fits the existing population calibrator; B fits error predictors; C calibrates upper error bounds; D is used only for final scoring. Semantic families use Rival's existing canonicalization and TF-IDF grouping at threshold 0.84. No family may cross active roles.

1. `family`: use the existing balanced family splitter with 20 folds and seed `rival-confidence-20260905-v1`; folds 0–9 are A, 10–13 B, 14–16 C, 17–19 D.
2. `wave`: A = waves 26,27,29,32,34,36,41; B = 42,43,45,49; C = 50,54; D = 82,92. For a family spanning roles, retain only its rows in the latest role and purge earlier rows. This is a later-wave stress test, not evidence of a prospective model training cutoff. The public base model may have seen these surveys.

The split manifest is written from question identifiers/texts before model fitting. Source file hashes, code hashes and the final result are recorded. There is no independent custodian or external timestamp; hashes provide reproducibility, not independent proof of blindness.

## Fixed models

Reuse `VectorizedPersonaCalibrator(max_iter=150, learning_rate=1.0)` without changing it. Compare its distribution against an unweighted persona aggregate, uniform probabilities, and mean historical choice-position proportions fitted on A. The last baseline is meaningful only within this five-option encoding and is not a semantic response model.

Target error is total variation distance (TVD) between a predicted distribution and the released normalized human counts. Error models are (i) the mean TVD on B, (ii) standardized numerical features with ridge alpha 10, (iii) word/unigram-bigram TF-IDF with ridge alpha 10, (iv) their concatenation with ridge alpha 10. The primary challenger is (iv); all other variants are prespecified comparisons, not candidates selected on D. Numeric features are entropy, maximum probability, top-two probability margin, probability concentration, log question length, maximum text similarity to A, and raw-to-calibrated TVD. Vectorizers and scalers are fitted only on their declared training inputs. No current human counts, sample sizes, or labels enter error features.

For each error model, use C only to compute one nonconformity score per family: the maximum positive residual `max(0, observed_TVD - predicted_TVD)` among its questions. Use the finite-sample order statistic `ceil((G+1)*0.9)` for a nominal 90% one-sided upper error bound. If the rank exceeds G, use the trivial upper bound 1. Bounds are clipped to [0,1].

Under exchangeability of independent families, this construction controls marginal coverage for all questions of a future family. Lexical families and survey waves need not satisfy that assumption. Empirical family coverage is reported; no conditional, subgroup, temporal, or post-selection coverage guarantee is claimed. These bounds describe discrepancy from empirical survey counts, not uncertainty about a population parameter.

## Scoring and decision rule

Report TVD, error-prediction RMSE, AUROC for TVD below 0.16, descriptive risk at 25/50/75/100% retained questions, upper-bound coverage per question and complete family, mean bound width, and number accepted at `upper_TVD <= 0.16`. The value 0.16 is an illustrative comparator from Simile's published error model, not a universal business tolerance. Rank-based retention is descriptive and provides no selective-risk guarantee. Constant-score ties use question IDs; they are not informative rankings.

Primary exploratory gate: the combined error model must beat the mean-error baseline in RMSE in BOTH splits; achieve at least 0.90 observed complete-family coverage in BOTH; and accept at least ten final questions under the illustrative 0.16 threshold in BOTH. Otherwise status is FAIL or UNEVALUABLE. Passing this small gate still would not qualify a commercial system. Do not tune or rerun changed hyperparameters against D. Any subsequent change requires a new development protocol and fresh external qualification data.

## Limitations to preserve in every result

Historical public benchmark; unknown foundation-model contamination; restricted five-option response format; repeated panel dependence cannot be recovered from aggregates; lexical grouping cannot certify conceptual independence; only two final waves; no survey design weights; no raw behavior events; no live provider or cost-quality evaluation; no externally held prospective outcomes. Report every variant and failure.

## Run

From the existing repository root, with its dependencies installed:

```bash
python -m unittest discover -s experiments/confidence_qualification_v1 -p 'test_*.py'
python experiments/confidence_qualification_v1/run.py --output-dir experiments/confidence_qualification_v1/results
```

The script refuses to overwrite an existing results directory. It does not modify Rival's frozen live Twin or Mega-Study protocols, models, or results.
