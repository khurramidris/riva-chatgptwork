# Local ssr model engineering rehearsal

**Research simulation — confidence unqualified.**

Decision being explored: Explore how two delivery plans compare in a generated audience.

Question: Which grocery delivery plan would you choose?

## Simulated choices

| Choice | Share | Difference from reference |
|---|---:|---:|
| Standard delivery | 47.8% | +0.0 percentage points |
| Flexible delivery | 52.2% | +4.4 percentage points |
| Neither plan | 0.0% | -47.8 percentage points |

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
- Information cutoff: 2026-09-01T00:00:00+00:00

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
- No observed-outcome comparison is available for this study.

## Model execution

- Requested model: Qwen/Qwen2.5-0.5B-Instruct; elicitation: ssr.
- Accepted seed requests: 6/6.
- Measured HTTP time: 46.48 seconds across 6 attempts.
- Saved model outputs are reused on resume. A fresh repeat is a separate study and may differ.
- This measures model execution; accuracy against real people remains unqualified.

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

Study: l09-local-ssr-2. Input fingerprint: `fb7a82d116523f1a603a55f7bdc2327485f5083416b893ffb42a0db1566cde58`.
