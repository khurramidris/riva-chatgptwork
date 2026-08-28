# Rival v0.5 Build Report

**Build date:** 28 August 2026

**Status:** planned research-component development complete; external customer qualification pending

**Qualified scope:** bounded five-choice population-distribution pilots and deterministic study controls

**Not qualified:** universal simulation, reliable novel individual prediction, causal validity, or Simile/Aaru empirical parity

## Material delivered

### Research runtimes

- Apache-2.0 PyMC Semantic Similarity Rating source and a Rival provider that maps free-text intent to choice probabilities;
- MIT UQ Survey source plus formula-parity CLT, Hoeffding, Bernstein, sample-width, coverage, and residual-corrected intervals;
- the full MIT SYN-DIGITS `SyntheticControl` class with documented portability patches, matrix-completion API, and row/column evaluation wrapper;
- paper-derived paired S-RCT estimation with weighted uncertainty and pre-period residual calibration;
- licensed/permissioned H&M persona-mixture demand source, a SciPy-only calibrated no-buy mixture, revenue CVaR, and price selection;
- interview-grounded persona ingestion from JSONL/CSV with deterministic transcript hashes and protected-outcome rejection;
- Apache-2.0 Twin-2K Mega Study MAD evaluation wrapper;
- Centauri and Socrates/SocSci210 inference adapters that bind model revision, endpoint, training-corpus declaration, and deployment license without storing credentials or bundling weights.

### Product surfaces

- `qualify-research-components` CLI gate with eight independent checks;
- REST endpoints for SSR, uncertainty, S-RCT, pricing, personas, and SYN-DIGITS matrix completion;
- health inventory for nine research capabilities;
- v0.5 qualification summary, component register, upstream integration guide, notices, and SHA-256 source lock;
- existing v0.4 outcome firewall, deterministic prediction context, provider-call identity, HMAC manifest seal, append-only phase chain, and separate AES-GCM outcome vault.

## Research-component gate

| Check | Result | Meaning |
|---|---|---|
| Semantic Similarity Rating | PASS | Valid PMF plus degenerate-denominator guard |
| UQ Survey numerical parity | PASS | CLT, Hoeffding, and Bernstein match licensed functions |
| Full SYN-DIGITS component | PASS | Finite completion with observed-value invariants |
| Paired S-RCT/pre-period | PASS | Weighted paired effect and residual adjustment invariants |
| Persona demand/pricing | PASS | Simplex/no-buy fit, bounded predictions, deterministic price choice |
| Interview-grounded personas | PASS | Typed ingestion and protected-outcome rejection |
| Behavioral-model adapters | PASS | Revision-bound identities with no secret material |
| Twin-2K official MAD | PASS | Official summary path executes and returns expected statistics |

Decision: **8/8 PASS** for wiring and numerical parity. This gate explicitly makes no external predictive or causal claim.

## Retained real-data evidence

### OpinionQA aggregate distributions

The benchmark contains 489 five-choice items and 2,058 released personas. Related variants remain together across five deterministic families/folds.

| Metric | Unweighted personas | Global-history baseline | Calibrated |
|---|---:|---:|---:|
| Mean TVD | 0.3311 | 0.2268 | **0.1695** |
| Relative reduction | — | — | **48.8% vs raw; 25.3% vs history** |
| Question win rate | — | — | **85.1% vs raw; 71.6% vs history** |

Decision: bounded aggregate pilots in a similar setting. This remains retrospective, not customer-transfer evidence.

### Twin-2K individual/anchor evidence

| Categorical individual metric | Accuracy |
|---|---:|
| Released LLM output | 54.1% |
| Human test–retest | **68.6%** |
| Population mode | 52.8% |
| Leakage-safe novel transfer | **44.0%** |

The target-family-excluded transfer fails the population baseline and stays research-only. With an 80-person held-out anchor, aggregate TVD improves from 0.2792 to 0.0735, but the human-only and repeated-history baselines remain stronger on this dataset.

## Verification completed

| Check | Result |
|---|---|
| Source inventory | PASS: every incorporated file matches `upstreams.lock.json` |
| Prospective integrity controls | **10/10 PASS** |
| Research-component integration | **8/8 PASS** |
| Full unit/integration/research suite | **54/54 PASS** |
| Python compilation | PASS |
| Full qualification regeneration | PASS |
| Wheel build | PASS: `rival_sim-0.5.0-py3-none-any.whl` |
| Wheel SHA-256 | `426e22297dcc79125ef60b16472837a0a2ff369a7933581c2497887d5130cbaf` |
| Wheel inventory | PASS: runtimes, datasets, summary, and four public license files present |
| Isolated wheel smoke | PASS: v0.5 import, health, and component qualification |
| HTTP research smoke | PASS: health plus six new endpoints |
| Release-manifest verification | PASS |

## Reproduce

```bash
python -m unittest discover -s tests -v
python -m rival qualify-research-components
python -m rival qualify-integrity
python -m rival qualify-all --output-dir reports
python -m rival verify-release
python -m rival serve --port 8080
```

Canonical evidence is in `reports/opinionqa_qualification.json`, `reports/twin2k_qualification.json`, `reports/integrity_qualification.json`, `reports/research_components_qualification.json`, and `reports/qualification_summary.json`. The wheel and reports are hash-bound by `RELEASE_MANIFEST.json`.

## Known limitations

- The new component gate tests software and numerical parity, not customer prediction or causal validity.
- Both public-data qualification tracks are retrospective; the model team has seen their outcomes.
- The default hashing embedder is an offline development fallback. Production SSR requires a pinned, domain/language-qualified embedding model.
- Centauri and Socrates adapters are present, but model weights and live inference are not bundled or benchmarked.
- S-RCT has not been tested on a newly randomized customer intervention.
- Pricing has no protected customer demand, inventory, competitive, or temporal benchmark yet.
- The H&M repository had no public LICENSE at the pinned revision; use relies on separate permission and must be verified before redistribution.
- Independent outcome custody still requires separate people, credentials, infrastructure, and key management in a live study.
- Authentication, tenant isolation, managed storage, queues, billing, and enterprise deployment remain outside this local build.
- Simile and Aaru have proprietary data, training, and customer results unavailable for direct parity testing.

## Production decision

Freeze v0.5 as the component-complete development baseline. Start a narrow, human-supervised, independently custodied customer pilot. Require Rival to beat weighted history, a relevant classical model, synthetic-only, and an equal-cost human-only estimate under preregistered thresholds. Do not sell individual digital twins, universal behavior prediction, or causal lift until those locked results exist.
