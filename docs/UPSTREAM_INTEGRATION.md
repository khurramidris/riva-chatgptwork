# Authorized Upstream Integration

## v0.5 integration map

Rival uses a narrow product boundary around each research component. Vendor snapshots remain auditable; Rival-owned adapters validate inputs, record identities, and expose stable APIs.

| Research line | Integration class | Product component | Qualification evidence |
|---|---|---|---|
| PyMC Labs Semantic Similarity Rating | Verbatim source + adapter | `elicitation.py`: free text → embeddings → SSR PMF → temperature scaling | Valid-PMF and denominator-zero tests; pinned implementation identity |
| SYN-DIGITS distribution calibration | Verbatim audit snapshot + independently vectorized runtime | `research/calibration.py` | Five family-held-out OpinionQA folds and per-question metrics |
| SYN-DIGITS synthetic control | Full vendored class with portability patches | `research/synthetic_control.py` | Hard/soft/ALS completion invariants plus full ridge evaluation path |
| UQ Survey Simulation | Patched vendor source + formula-level wrapper | `uncertainty.py` | Numerical parity with upstream CLT, Hoeffding, and Bernstein functions |
| Twin-2K-500 | Licensed data/protocol | Typed loader, leakage firewall, baselines, anchor correction | Individual and aggregate retrospective qualification; negative transfer result retained |
| Twin-2K Mega Study | Verbatim MAD evaluator + lazy wrapper | `research/mad.py` | Official MAD summary smoke test |
| H&M demand/pricing | Verbatim source snapshot + adapted runtime | `pricing.py` | Simplex persona/no-buy fit, calibrated probabilities, revenue/CVaR price test |
| S-RCT | Paper-derived Rival implementation | `experiments/srct.py` | Weighted paired effect and pre-period residual invariants |
| 1,000-person interview study | Conceptual design, Rival implementation | `personas.py` | Typed JSONL/CSV ingestion, deterministic provenance, protected-outcome rejection |
| Centauri | Adapter only | `CentauriProvider` | Revision/endpoint/corpus/license-bound identity; no credential serialization |
| Socrates / SocSci210 | Adapter only | `SocratesProvider` | Same identity controls; no code or weights claimed |

The machine-readable source of truth is `upstreams.lock.json`. `THIRD_PARTY_NOTICES.md` records the public terms and the one repository where use relies on separate permission because no public LICENSE file was present.

## Portability modifications

The full SYN-DIGITS `SyntheticControl` file is kept intact except for documented product-runtime patches:

1. replace `causaltensor.matlib.SVD` with a local NumPy rank-k reconstruction;
2. replace interactive `plt.show()` calls with headless figure closing;
3. redirect diagnostic output to a temporary or `RIVAL_SYN_DIGITS_OUTPUT_DIR` directory;
4. bound two expensive evaluation imputation call sites at 100 iterations.

The UQ source has one import-only patch: CI helpers remain usable when `tqdm` is absent. The scientific formulas are unchanged and parity-tested.

The H&M demand source snapshot is retained for audit. Its product runtime is adapted in Rival-owned code because the upstream execution path depends on CVXPY/solver selection and notebook-oriented progress tooling. Rival preserves the substantive structure—monotone logit calibration, a simplex over personas plus a dummy no-buy mass, binomial or zero-truncated likelihood—and uses SciPy for deployment.

## Runtime sequence

1. Strict product schemas validate scenarios, populations, transcripts, matrices, and numeric bounds.
2. The outcome firewall removes post-cutoff history and rejects protected outcomes before model access.
3. A provider or research adapter executes against a pinned component/model identity.
4. Results cross a finite/probability/shape invariant before entering the engine or API.
5. Prediction contexts, request identities, manifests, and qualification reports bind the implementation version.

## Leakage and evidence rules

Question splitting remains a first-class component:

1. normalize text, numbers, bracketed options, and condition labels;
2. union exact canonical duplicates and high-similarity documents;
3. collapse repeated QuestionIDs, experiment blocks, and known counterbalanced siblings;
4. assign complete families to deterministic folds and hash the manifest;
5. remove target families from Twin-2K donor matrices;
6. keep wave-4 outcomes evaluation-only.

Component parity is not predictive validity. The v0.5 integration qualification proves that code is wired, licensed snapshots are hash-stable, and bounded numerical invariants hold. Only protected customer studies can establish accuracy, causal usefulness, cost advantage, subgroup behavior, and transfer.

## Procedure for future upstream changes

1. Pin the exact commit and permission/license reference.
2. Inventory code, data, weights, hosted-service terms, and transitive assets separately.
3. Copy the smallest coherent component and preserve its notices.
4. Record every incorporated file and SHA-256 in `upstreams.lock.json`.
5. Document every patch and keep research logic distinct from transport/UI code.
6. Add parity, leakage, accuracy, latency, cost, and failure tests.
7. Run a protected baseline comparison before enabling a new claim.
8. Ship behind a revision-bound interface and retain the previous qualified release.
