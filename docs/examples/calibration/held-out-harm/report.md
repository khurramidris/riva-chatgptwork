# Generated calibration rehearsal: held-out-harm

**Research simulation — confidence unqualified.**

Decision being explored: Explore how two delivery plans compare in a generated audience.

Question: Which delivery plan suits held-out-harm?

## Simulated choices

| Choice | Share | Difference from reference |
|---|---:|---:|
| Standard delivery | 52.5% | +0.0 percentage points |
| Flexible delivery | 40.0% | -12.5 percentage points |
| Neither plan | 7.5% | -45.0 percentage points |

Reference: standard. These differences are not causal effects.

## Audience and execution

- Audience: Six generated example records; no real consumers represented.
- Declared geography: example-only
- Eligible seed records: 6
- Effective seed records under the weights: 6.0
- Unique seed records sampled: 6
- Simulation draws: 40
- Execution mode: managed
- Accounted request attempts: 6
- Total recorded request cost: $0.000000
- Information cutoff: 2026-09-08T10:08:24.938605+00:00

## Evidence

- Rival generated workflow example (synthetic); rights reference: Generated in Rival for testing.

## Declared population support

- Support checks passed: True
- Seed records excluded by geography and audience filters: 0
- Source files and conversions were verified at preparation.
- Condition IDs: grocery-delivery-plans
- Value segment: 3 seeds; effective count 3.0.
- Convenience segment: 3 seeds; effective count 3.0.
- Checks describe supplied evidence; they do not establish representative sampling or predictive accuracy.
- Protected outcome comparison TVD: 0.0000; verified against the local sealed ledger.

## Model execution

- Requested model: generated-calibration-fixture; elicitation: direct.
- Accepted seed requests: 6/6.
- Measured HTTP time: 0.01 seconds across 6 attempts.
- Saved model outputs are reused on resume. A fresh repeat is a separate study and may differ.
- This measures model execution; accuracy against real people remains unqualified.

## Calibration

Fitted to 2 training groups using a fixed panel of 6 seeds. Both predictions were saved before outcome reveal.

| Choice | Raw simulation | Calibrated | Training mean baseline |
|---|---:|---:|---:|
| Standard delivery | 52.5% | 25.0% | 25.0% |
| Flexible delivery | 40.0% | 50.0% | 50.0% |
| Neither plan | 7.5% | 25.0% | 25.0% |

| Compared with protected outcomes | TVD (lower is better) |
|---|---:|
| raw | 0.0000 |
| calibrated | 0.2750 |
| uniform\_panel | 0.0000 |
| historical\_mean | 0.2750 |

TVD reduction versus raw: -0.2750. A negative value means calibration made this study worse.

## Limitations

- This study includes generated audience records; it is an engineering demonstration.
- Choice shares describe model outputs for the supplied audience; customer-domain accuracy is not established.
- Resampled draws do not add independent people or human evidence.
- Differences between choice shares are not causal effects of an intervention.
- Source rights, dates, geography and independence are operator declarations, not independently verified facts.
- Confidence remains unqualified; no decision-ready interval or recommendation is issued.
- Declared support checks do not establish population representativeness or validate the scenario&\#x27;s meaning.
- declared revision with checked response metadata; remote weights are not independently attested
- journal replay reuses outputs; fresh repeatability requires a separately executed study
- Calibration was fitted only to its declared reference training groups. New-question support and improvement are unqualified.

Study: l10-held-out-harm. Input fingerprint: `0c05d4de7cf7b9f927257273fd7fef179e314d183ee79d49fb091e3e745c1640`.
