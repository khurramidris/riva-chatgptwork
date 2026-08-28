# Authorized Upstream Integration

## Incorporation decision

Rival now incorporates research components where the data contract and benchmark are clear, while keeping product runtime interfaces Rival-owned.

| Upstream | Incorporated component | Product boundary | Evidence produced |
|---|---|---|---|
| SYN-DIGITS | Full calibration source snapshot; OpinionQA human counts and released persona answers | Memory-safe vectorized calibrator in `rival/research/calibration.py` | Five family-held-out folds, per-question TVD/JS, source and split hashes |
| Twin-2K-500 | Question catalog, mapping, three aligned wave/model response tables | Typed loader, family firewall, baselines, ridge-transfer and anchor evaluator | Individual accuracy/NMAE/correlation plus aggregate TVD and explicit failure gate |
| PopulationSim | Method/QA reference | Existing support-checked raking compiler | Margin error and effective sample size |
| AI-Augmented Estimation | Estimator reference | Existing categorical human-residual correction | Held-out anchor evaluation on Twin-2K |
| Adaptive Query | Selection reference | Existing uncertainty/coverage selector | Awaiting prospective propensity-aware validation |

## Source isolation

The upstream SYN-DIGITS module is retained unmodified under `vendor/syn_digits/` with its license and README. Rival does not import it at runtime because it constructs a dense mapping matrix per question and brings notebook/plotting dependencies into the server. `VectorizedPersonaCalibrator` implements the same documented KL objective using indexed accumulation and is covered by a golden synthetic-mixture test.

Research datasets live under `rival/datasets/` so editable installs and built wheels reproduce the qualification command. Loaders validate identifiers, code ranges, duplicate participants, common-person intersections, mapped columns, and exact source hashes.

## Leakage firewall

Question splitting is a first-class component, not a notebook convention:

1. normalize text, numbers, bracketed options, and experimental condition labels;
2. union exact canonical duplicates;
3. union high-similarity TF-IDF documents;
4. collapse repeated QuestionIDs and normalized experimental blocks;
5. explicitly bind known counterbalanced Twin-2K sibling sets;
6. assign whole families to deterministic, balanced folds;
7. hash the item/family/fold manifest and assert that no family crosses folds.

For Twin-2K novel transfer, the same family map removes target siblings from the donor matrix. Wave-4 outcomes are evaluation-only.

## Import procedure for the next component

1. Pin the commit and signed permission/license reference.
2. Inventory source, data, weights, and hosted-service terms separately.
3. Copy the smallest coherent component under `vendor/` with original notices.
4. Record every incorporated file and SHA-256 in `upstreams.lock.json`.
5. Write a narrow Rival adapter and schema validator.
6. Add parity, leakage, accuracy, latency, cost, and failure tests.
7. Produce a protected baseline comparison before enabling the component in customer studies.
8. Deploy behind a feature flag and retain the previous qualified version.

Copying code accelerates implementation; it does not transfer the upstream paper's empirical claim to Rival. The release gate remains attached to Rival's own data, split, model, and outcome hashes.
