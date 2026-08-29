# Mega-Study Development Pre-registration

This document was frozen before the first real LLM call. The three studies are
development studies; their results may later inform Rival, but they are not
confirmation evidence.

## Scientific questions and directional hypotheses

1. Does knowing the person improve prediction?
   - B > A, C > A, and D > A in paired normalized accuracy.
2. Does rich history improve on demographics?
   - C > B.
3. Does target-specific retrieval use history better than dumping the full
   persona into the same model?
   - D > C. This is the central Rival architecture test.
4. Does Rival improve on demographics?
   - D > B.

These are directional scientific hypotheses, not pass/fail thresholds. A
positive point estimate without a compatible uncertainty interval is not
treated as decisive evidence.

## Frozen experimental unit

The unit is a participant-study pair. There are 100 pairs from each of three
official studies and four predictions per pair. Because 27 people appear in
more than one development study, inference clusters by participant rather than
pretending all 300 rows are independent.

The frozen cohort, dataset revisions, file checksums, selection seed, study
allocation, per-case call order, model route, prompts, retrieval policy,
parser, outcome maps, metrics, and failure policy are in the manifest.

## Primary estimand

For a participant/outcome cell, normalized accuracy is:

`s = 1 - |prediction - human| / (maximum - minimum)`

For a contrast X − Y, the primary paired estimand is the mean of `s_X - s_Y`
within outcome, macro-averaged over outcomes within study, then over the three
studies. Positive values favor X. A 95% participant-cluster bootstrap interval
uses 10,000 deterministic resamples.

The central preregistered contrast is Rival Retrieval − Full Persona. Rival −
Generic and Rival − Demographics are also primary product comparisons. The
remaining contrasts answer the representation-ablation questions.

## Secondary outcomes

- Pearson correlation across people for each outcome, Fisher-z macro average
- Spearman correlation
- MAE and RMSE
- exact accuracy and balanced accuracy when meaningful
- predicted versus human population means
- normalized population-mean error
- absolute Glass delta and SD ratio
- discrete total-variation distance
- valid coverage and failure counts

No probability score is claimed in phase 1 because the official full-survey
interface returns structured point responses rather than calibrated outcome
probabilities.

## Missingness and failures

Every frozen participant-study-variant-outcome cell remains eligible. A failed
API call, invalid JSON, missing expected answer, out-of-range value, missing
history, unsupported question, or context failure is reported and is not
silently removed. Absolute variant summaries use valid cells and show coverage.
Paired comparisons use the intersection of valid cells and report the matched
cell and participant counts.

The runtime stops after the configured error threshold to avoid turning a
provider incident into a large block of missing data. Its frozen retry policy
is applied consistently to every variant.

## No tuning during baseline establishment

No prompts, retrieval settings, parser rules, study allocation, model route,
or metrics may change after the first call. Partial results are not used to
change Rival. The complete A/B/C/D baseline report must be produced first.

If a defect is discovered after calls begin, the current run is retained and
marked invalid or unevaluable. A new versioned protocol and cohort must be
created; the old artifacts are not rewritten.

## Confirmation lock

Before development outcomes were inspected, these entire studies were reserved
as candidate confirmation studies:

- `accuracy_nudges`
- `context_effects`
- `preference_redistribution`
- `quantitative_intuition`

Their human outcomes may not be used for prompt, retrieval, feature, threshold,
model, or architecture selection. A later confirmation protocol must choose a
defensible subset, freeze it, and predict it without development-time access to
those outcomes. Row-level random splitting within the three development studies
does not qualify as confirmation.

## Interpretation rules

- Decent absolute accuracy is not sufficient for a Rival claim.
- D > A supports the value of Rival's complete system relative to model priors.
- D > B supports using rich person history beyond segmentation.
- D > C supports the specific claim that target-relevant retrieval exploits
  history better than a full-persona dump.
- D <= C means retrieval, as frozen here, has not earned that claim even if D's
  absolute performance is good.
- Strong population agreement with weak individual correlation supports only a
  population-simulation claim, not individual digital-twin accuracy.
- Development results are estimates for these studies, not general proof.

Weak or negative results are retained. They determine what must be diagnosed
before a separately preregistered follow-up; they are not repaired by excluding
studies or cases.
