# Rival v0.4 Build Report

**Build date:** 28 August 2026
**Status:** prospective-integrity kernel on the real-data qualification foundation
**Qualified scope:** five-choice population response distributions  
**Not qualified:** universal simulation or reliable novel individual prediction

## Material delivered

- deterministic, self-hashing `PredictionContext` over scenario, calibrated population, targets, retrieval audit, provider configuration and code version;
- fail-closed protected-outcome field detection and information-cutoff filtering with per-person audit hashes;
- provider/model/endpoint/request/cache identity on every retained agent prediction, without secret serialization;
- HMAC-SHA256 symmetric manifest sealing over predictions and preregistered metrics/thresholds;
- append-only phase events enforcing `draft → prediction_locked → outcomes_revealed → evaluated`;
- separate AES-GCM outcome vault with HKDF key derivation, manifest-bound authenticated data, time-gated reveal and access events;
- `/api/prediction-context` and `/api/studies/lock`, with no outcome reveal surface in the simulation server;
- deterministic 10-control integrity qualification, release-hash verifier, and 16 new integrity/release tests;
- incorporated, hash-pinned OpinionQA and Twin-2K-500 research subsets;
- full authorized SYN-DIGITS distribution-calibration source snapshot with MIT notice;
- wheel-safe dataset adapters with identifier, support, range, duplicate, and alignment checks;
- deterministic semantic question-family firewall and no-cross-fold assertion;
- vectorized KL mirror-descent persona/base calibration;
- five-fold OpinionQA qualification with per-question outputs;
- Twin-2K direct-model, human-retest, population, novel-transfer, and human-anchor tracks;
- strict machine-readable reports and bundled release summary;
- `qualify-opinionqa`, `qualify-twin2k`, `qualify-integrity`, and `qualify-all` commands;
- `/api/qualification` and a product Validation workspace;
- expanded provenance lock, attribution, validation protocol, and v0.5 pilot plan;
- original population compiler, provider adapter, hybrid estimator, confidence policy, evidence ledger, API, and study interface.

## Real-data result 1: OpinionQA

The benchmark contains 489 five-choice Pew items, 2,058 released persona responses per item, and 1,853,229 human responses in aggregate. Related lexical/semantic variants are assigned together, producing 464 families over five balanced folds.

| Metric | Unweighted personas | Global-history baseline | Calibrated |
|---|---:|---:|---:|
| Mean TVD | 0.3311 | 0.2268 | **0.1695** |
| Median TVD | 0.3193 | 0.2052 | **0.1582** |
| 90th-percentile TVD | 0.5419 | 0.3925 | **0.2816** |

- 48.8% relative mean error reduction versus unweighted personas;
- 25.3% relative mean error reduction versus global history;
- calibrated wins on 85.1% of questions versus unweighted personas;
- calibrated wins on 71.6% of questions versus global history;
- split manifest: `2fae83e0f1e1e1d703ee18e670bc0c6da7fe7a909bad134ad5416d24c19521d6`.

Decision: this clears the v0.2 gate for controlled population-distribution pilots in a similar five-choice setting. It is retrospective domain evidence, not proof of customer or future-event transfer.

## Real-data result 2: Twin-2K

The benchmark aligns 2,058 people across wave-1/3 history, wave-4 outcomes, and released GPT-4.1-mini outputs for 126 mapped items (108 categorical, 18 continuous). Target items, same-QuestionID rows, normalized experiment blocks, and known counterbalanced siblings are excluded from each novel-transfer donor matrix.

| Categorical individual metric | Accuracy | Mean distribution TVD |
|---|---:|---:|
| Released LLM output | 54.1% | 0.2797 |
| Human test–retest | **68.6%** | **0.0248** |
| Population mode | 52.8% | 0.4717 |
| Leakage-safe novel transfer | 44.0% | 0.2442 |

The novel-transfer method fails the classical individual-accuracy baseline and remains research-only. This negative result is retained in the dashboard and release summary.

With a deterministic 80-person wave-4 anchor and evaluation on all other valid respondents:

- raw released-model mean TVD: 0.2792;
- prediction-powered hybrid mean TVD: 0.0735 (73.7% lower; better on 98.1% of items);
- human-anchor-only mean TVD: 0.0623;
- prior-wave historical mean TVD: 0.0254.

The anchor correction removes most model bias, but it does not beat human-only or repeated-history baselines on this dataset. It therefore demonstrates correction mechanics, not fieldwork savings.

## Verification completed

| Check | Result |
|---|---|
| Dataset alignment | Pass: 489×2,058 OpinionQA; 2,058×126 Twin-2K |
| Source hashes | Pass against `upstreams.lock.json` |
| Family leakage assertion | Pass; known counterbalanced siblings joined |
| Full qualification rerun | Pass; strict JSON reports regenerated |
| Prospective integrity controls | **10/10 pass** |
| Outcome firewall | Pass; protected fields fail before provider calls |
| Locked-context mismatch | Pass; changed inputs rejected before provider calls |
| Manifest seal/tamper test | Pass |
| Outcome vault | Pass; ciphertext-only at rest, early/wrong-key reveal rejected |
| Phase chain | Pass through evaluated state |
| Python compilation | Pass |
| JavaScript syntax | Pass |
| Unit/integration/research tests | **36/36 pass** |
| Wheel build | Pass: `rival_sim-0.4.0-py3-none-any.whl` |
| Wheel dataset/report inventory | Pass |
| HTTP health endpoint | Pass: v0.4.0; locking configuration disclosed without secrets |
| HTTP qualification endpoint | Pass: 489-question summary |
| Static Validation workspace | Served successfully |
| HTTP end-to-end demo | Pass |

Wheel SHA-256 at verification: `3c7c70270e029e4fa72c2d52d840134d598663d54420ad58769090cdf51b9590`.

## Reproduce

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
python3 -m rival qualify-integrity
python3 -m rival qualify-all --output-dir reports
python3 -m rival verify-release
python3 -m rival serve --port 8080
```

The canonical artifacts are `reports/opinionqa_qualification.json`, `reports/twin2k_qualification.json`, `reports/integrity_qualification.json`, `reports/qualification_summary.json`, and `rival/qualification/summary.json`.

## Known limitations

- Both qualifications are retrospective public-data analyses; the current model team has seen their outcomes.
- The integrity qualification tests engineering controls only; it is not a customer-prospective accuracy result.
- Manifest sealing is symmetric and deployment-local, not an independent public-key signature or trusted external timestamp.
- The encrypted outcome vault is separate in code and storage, but independent custody still requires separate people, credentials, infrastructure and key management in a live study.
- The OpinionQA result covers five-choice aggregate distributions only.
- The released model outputs are upstream artifacts, not live calls through Rival's provider adapter.
- Twin-2K repeats prior questions, is not a customer market panel, and does not establish novel scenario fidelity.
- The semantic firewall is conservative and auditable but cannot prove causal independence.
- Confidence intervals, subgroup fairness, survey-design variance, temporal drift, and decision regret need customer-study validation.
- Authentication, tenant isolation, managed storage, queues, billing, and enterprise deployment remain outside this local build.
- No external provider credential was available, so cost/latency and cross-provider stability were not measured.

## Production decision

Start a narrow, human-supervised concept/message survey pilot using the aggregate calibration and anchor workflow. Use the v0.4 context, manifest, phase chain and independently custodied vault before collecting the outcome. Do not sell individual digital twins, autonomous forecasting, or universal behavior simulation. Require Rival to beat weighted history, a relevant classical model, and an equal-cost human-only estimate under preregistered thresholds.
