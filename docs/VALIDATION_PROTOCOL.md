# Protected Validation Protocol

## Purpose

This protocol governs when Rival may make a predictive claim. A visually persuasive demo or retrospective reconstruction is not sufficient.

## Lock before outcome access

For every eligible study, record and hash:

- decision, estimand, population, geography and horizon;
- information cutoff and outcome availability timestamp;
- scenarios, stimuli and valid actions;
- model, prompt, data and population versions;
- metrics, thresholds, subgroup slices and abstention rule;
- all allowed tuning and exclusion criteria.

The model team must not possess outcome access until the run artifact is committed.

## Required baselines

1. weighted historical/base-rate forecast;
2. classical hierarchical, discrete-choice or Gaussian-copula model;
3. generic LLM without personas;
4. demographics/simple-persona LLM;
5. Rival evidence-grounded simulator;
6. a small probability sample of humans;
7. Rival hybrid simulator plus human anchor;
8. the conventional full study where available.

## Splits

Use simultaneous temporal, entity/brand, question-family, semantic-near-duplicate, geography and intervention splits. A question that is paraphrased from training is not held out.

The public qualification implements the question-family layer: canonical normalization, TF-IDF similarity, experimental-block collapse, explicit counterbalanced sibling sets, deterministic group folds, a manifest hash, and a programmatic no-crossing assertion. Rival v0.4 additionally enforces the declared information cutoff at the runtime retrieval boundary. It does not yet prove customer/entity or geographic isolation because the incorporated public tracks do not support a customer-prospective test.

## Metrics

- distributions: TVD, Jensen–Shannon divergence, percentage-point error and interval coverage;
- forecasts: log/Brier scores, calibration and sharpness;
- treatment effects: bias, RMSE, sign, coverage and decision regret;
- individual prediction: separate person-level calibration/correlation and variance ratio;
- free response: blinded human coding, theme prevalence, omissions and response-style bias;
- equity: worst-group error, calibration and within-group variance;
- operations: cost, latency, reproducibility and human fieldwork saved.

## Release gates

- Hybrid estimate beats synthetic-only and the equal-cost small-human baseline on protected tests.
- It beats or adds decision value beyond the relevant classical baseline.
- Nominal intervals achieve acceptable empirical coverage.
- Error falls monotonically as low-confidence cases are rejected.
- No priority subgroup exceeds its preregistered error threshold.
- Results remain within a sensitivity band across seeds, prompt versions and two eligible model backends.
- At least three prospective design-partner studies are complete, with misses retained.

## v0.4 evidence decision

| Claim | Evidence | Decision |
|---|---|---|
| Five-choice population distribution calibration | OpinionQA: mean TVD 0.331 unweighted / 0.227 global-history → 0.169 calibrated over 489 family-held-out questions; 85.1%/71.6% win rates | Qualified for bounded pilots |
| Released-model bias correction with 80 human anchors | Twin-2K: mean categorical TVD 0.279 → 0.074 on non-anchor respondents | Promising, but not equal-cost-baseline qualified |
| Novel individual question prediction | Twin-2K transfer accuracy 44.0% vs 52.8% population-mode baseline | Failed; research only |
| Universal human simulation | No qualifying benchmark | Not claimed |
| Prospective integrity controls | 10 deterministic checks covering context, firewall, sealing, vault, reveal and phase chain | Engineering-qualified; not predictive evidence |

The OpinionQA result is retrospective public-data evidence, not a prospective customer result. “Bounded pilots” means a controlled evaluation with human review, not autonomous production decisions.

## Publication rule

Report all registered eligible studies, denominators, exclusions, aggregate and worst-case errors, confidence coverage and failures. Customer confidentiality may hide names/stimuli; it must not hide whether a registered test missed.
