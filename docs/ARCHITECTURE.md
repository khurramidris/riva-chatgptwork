# Rival v0.4 Architecture

## Design rule

Rival estimates a bounded conditional distribution:

\[
P(\text{choice}\mid\text{target population, scenario, time, information})
\]

It does not treat a plausible agent transcript as validation. Every quantitative result must be traceable to a population build, scenario, model artifact, random seed and—when available—a protected outcome.

## Runtime sequence

1. Parse a typed `ScenarioSpec`; reject duplicate actions and unsupported interaction modes.
2. Filter the seed population to the declared segment.
3. Rake seed weights to declared target controls; fail if target support is absent.
4. Reject protected outcome keys, exclude history after the information cutoff, and emit a per-person retrieval audit.
5. Bind scenario, population, targets, audit, provider/model configuration and code version into a deterministic `PredictionContext`.
6. Draw a reproducible Monte Carlo population and record a request/cache identity for every provider call.
7. Aggregate probabilities and preserve sampled choices only for qualitative/runtime use.
8. Assess novelty, entropy, provider disagreement, population fit and anchor coverage.
9. Seal the unchanged prediction plus preregistered evaluation plan and append the `prediction_locked` phase.
10. Keep outcomes encrypted in a separate vault until the declared reveal time, then append reveal and evaluation events.

## Qualification sequence

1. Load a pinned dataset through an alignment- and range-checking adapter.
2. Canonicalize question text and experiment metadata into semantic families.
3. Assign complete families to deterministic folds and hash the manifest.
4. Fit calibration or transfer components without access to held-out families/outcomes.
5. Score every eligible question and retain per-question results, not only averages.
6. Compare against historical, population, human-retest, and human-anchor baselines.
7. Publish a machine-readable release decision, including failed gates.

## Modules

| Module | Responsibility | Extension boundary |
|---|---|---|
| `schemas.py` | Strict product/data contracts | Version schemas; never hide unknown fields |
| `population.py` | Filtering, raking, support checks, sampling, ESS | Add PopulationSim adapter or learned joint generator |
| `providers.py` | Offline and OpenAI-compatible behavior probability providers | Add Socrates/Centaur/Centauri/domain models |
| `behavior.py` | Model registry/router | Route by domain, policy, evidence and cost |
| `adaptive.py` | Anchor selection | Replace with authorized CAT/MIRT implementation while logging propensities |
| `hybrid.py` | Prediction-powered residual correction and intervals | Add cross-fitting, complex survey weights and multi-arm treatment effects |
| `evaluation.py` | Distribution metrics and interval coverage | Add proper forecast scores, subgroup and trajectory suites |
| `confidence.py` | Cold-start policy and ridge error prediction | Train on protected workload outcomes; calibrate risk/coverage |
| `store.py` | Append-only provenance/evidence ledger | Replace with tenant-isolated Postgres/object store |
| `integrity.py` | Outcome firewall, contexts, manifest sealing and phase orchestration | Add external timestamp/notary and asymmetric signatures |
| `outcome_vault.py` | Separate AES-GCM outcome custody and time gate | Move to custodian-owned KMS/HSM and isolated service account |
| `engine.py` | Orchestration and lineage | Queue workers, cost controls and replay |
| `server.py` | Local API/static UI | Replace standard-library transport without changing engine interfaces |
| `research/datasets.py` | OpinionQA/Twin-2K validation and alignment | Add customer survey connectors with the same typed checks |
| `research/firewall.py` | Semantic families, deterministic folds and split hashes | Add embedding review and human adjudication |
| `research/calibration.py` | Vectorized SYN-DIGITS KL persona calibration | Add constrained/subgroup and incremental variants |
| `research/opinionqa.py` | Aggregate distribution qualification | Extend to other choice formats and temporal holdouts |
| `research/twin2k.py` | Longitudinal individual/anchor baselines | Replace failed research transfer only after protected evidence |

## Immediate production hardening

- Move SQLite to tenant-isolated Postgres and immutable object storage.
- Add authentication, RBAC, audit log, rate limiting and regional encryption controls.
- Add schema migrations, task queue, provider budgets, idempotency keys and cancellation.
- Store model/prompt/data hashes separately from large prediction payloads.
- Separate the outcome vault from model-team credentials.
- Add panel-provider consent/deletion workflow and cross-customer training policy.
- Add classical discrete-choice and Gaussian-copula baselines.
- Add cross-fitting and survey-design-aware standard errors.
- Add protected temporal/entity/geography splits alongside the implemented question-family scan.

## Multi-agent extension

Interaction remains deliberately absent from the qualified v0.4 scope. A future world runtime must declare state, clock, information visibility, network edges, action effects and termination. It should ship only when an interaction-dependent benchmark beats an independent-agent and classical dynamic baseline.
