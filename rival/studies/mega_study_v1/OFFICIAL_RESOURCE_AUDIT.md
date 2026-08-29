# Official Mega-Study Resource Audit

Audit date: 2026-08-29. No real Rival API call had been made.

## Resources inspected

The paper, official Apache-2.0 repository at commit
`afe2bb933fce377ed196f441a4c12962cb55a53a`, released Mega-Study dataset,
original Twin-2K full-persona dataset, study configurations, simulation code,
evaluation code, post-metric code, and published reference outputs were
inspected before implementation.

The Mega-Study splits expose exactly three columns: `PID`, answer-bearing
`survey_json_with_human_response`, and answer-free `survey_text`. The official
simulation convention sends a complete persona and complete survey in one
model call, asks for a structured JSON survey response, consolidates human and
twin answers, constructs study-specific dependent variables, and evaluates
individual and population agreement.

## Verified development splits

| Exact split | Rows | Answer-free survey size | Official dependent variables |
|---|---:|---|---|
| `junk_fees` | 400 | median ~43,219 characters | percent correct, fairness average, regulation support |
| `hiring_algorithms` | 999 | ~23,170 characters | eight direct ratings and 32 job-profile ratings |
| `privacy` | 1,200 | ~5,459 characters | privacy-protection value (PPV) |

All selected PIDs map to the original Twin-2K personas. The seven full-persona
Parquet shards contain 2,058 people with `pid`, `persona_text`,
`persona_summary`, and `persona_json`. For the selected people, `persona_text`
averages approximately 128,000 characters and `persona_json` exposes roughly
236 usable earlier questions. The official Demographics block contains 14
answered fields.

## Authors' methods adopted

- full-survey, one-call response generation
- authors' system instruction and structured survey convention
- official dependent-variable definitions and recodes
- normalized accuracy `1 - MAD / natural range`
- per-outcome human/twin Pearson correlation
- Fisher-z correlation aggregation
- population means, absolute Glass delta, and SD ratio
- official published Digital Certification output as a non-target compatibility reference

Rival's evaluation also retains transparent coverage, MAE/RMSE, Spearman,
exact/balanced accuracy where useful, and distribution TVD. These additions do
not replace the authors' primary metrics.

## Intentional differences

The official paper's original GPT-4.1 configuration used temperature 0.7, and
the repository also contains temperature-zero configurations. This protocol
uses temperature zero because all four representation conditions must be
compared with minimal decoding noise. It freezes the same model and upstream
route for every condition.

The official full-persona baseline is retained unchanged in spirit: all
available prior persona text is supplied with no truncation. Rival adds three
controlled ablations—Generic, Demographics, and target-blind Retrieval—to
measure the incremental value of person representation. It does not add other
Rival research components in this phase.

## Reference replication

The official
`digital_certification/full_persona_without_reasoning_2025-06-23` files
reproduce the following without touching a development- or confirmation-study
outcome:

| Quantity | DV1 | log WTP |
|---|---:|---:|
| people | 600 | 600 |
| Pearson correlation | 0.14395874126249344 | 0.11925467362764719 |
| normalized accuracy | 0.929074074074074 | not defined by authors |
| human mean | 4.735 | 5.130115806998002 |
| twin mean | 4.647222222222222 | 5.85481852949553 |
| human standard deviation | 1.5257503452791543 | 0.8414248869068346 |
| twin standard deviation | 0.8134190743339643 | 0.2847101152271372 |

These agree with the published reference within `1e-12`. This establishes
compatibility of reference-file parsing, participant pairing, matched invalid
WTP exclusion, DV1/log-WTP construction, primary accuracy, correlation, and
population summaries before new spending.

## Provenance and reuse

No large upstream runtime was vendored for this module. Rival uses the authors'
data conventions, exact scientific formulas, and small independently organized
implementations of the required parsers/evaluators. Source attribution,
licenses, revisions, URLs, and file hashes are retained in the manifest and
repository notices.
