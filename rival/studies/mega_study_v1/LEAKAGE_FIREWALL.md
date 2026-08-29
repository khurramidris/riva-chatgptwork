# Mega-Study Leakage Firewall

The protected field is `survey_json_with_human_response`. The permitted
prediction field is `survey_text`, which contains the survey and a masked answer
template but no participant answer.

## Capability boundary

| Component | Permitted inputs | Protected response accepted? |
|---|---|---:|
| cohort selection | study ID, PID, safe-survey presence, persona presence | No |
| prediction staging | `PID`, `survey_text`, original Twin-2K persona | No |
| demographics | original pre-Mega persona JSON | No |
| retrieval | answer-free new survey + old persona JSON | No |
| prompt construction | safe case + permitted persona representation | No |
| provider | system prompt + rendered user prompt | No |
| result ledger | prediction and transport metadata | No |
| model/feature selection | frozen manifest only | No |
| post-freeze outcome materializer | verified immutable sources + frozen cohort | Yes |
| evaluator | frozen predictions + sealed outcomes | Yes |

The prediction runner has no source-cache argument and does not import the
outcome materializer. Prediction-stage JSONL files have explicit allowed-key
sets. Protected field-name collisions are fatal. The cohort contains no target
value and no per-target response hash.

The full official Parquet files necessarily contain both safe and protected
columns. They live in a checksum-verified source cache outside the prediction
package. Prediction staging requests only `PID` and `survey_text` through
Parquet column projection. The protected column is not read into a Rival object
until the post-freeze materialization command.

## Pre-call automated proof

`prepare_mega_study.py` verifies:

1. all official file sizes and SHA-256 hashes;
2. the manifest and cohort self-consistency;
3. the unchanged hashes of both legacy Wave-4 artifacts;
4. exact allowed keys in 300 safe cases and 273 persona rows;
5. absence of protected keys in prediction inputs;
6. no sealed outcome directory is produced;
7. all 1,200 prompts render below the frozen context policy;
8. all prompt templates and provider identity match the manifest;
9. target-blind retrieval is deterministic and auditable;
10. the official Digital Certification reference metrics reproduce exactly;
    this fixture has no development- or confirmation-study overlap.

## Post-prediction gate

`freeze_mega_study.py` requires a terminal ledger entry for all 1,200 frozen
work IDs and records the exact result-file hash with `outcomes_opened: false`.

`evaluate_mega_study.py` refuses to materialize answers unless:

- the protocol and prediction-stage hashes match;
- the freeze marker's self-hash verifies;
- the result ledger still has the frozen hash;
- the marker certifies that outcomes were unopened;
- every official source file still matches its preregistered hash.

The resulting outcome file and its manifest are placed under `sealed/` with
restricted permissions where supported. Evaluation binds its report to the
protocol, stage, freeze, results, and outcome hashes.

Any violation is a fatal scientific error. It is not converted into a warning
or silently repaired.
