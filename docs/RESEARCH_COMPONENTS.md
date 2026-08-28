# Rival v0.5 Research Component Register

This register answers three separate questions for each research line: what code is physically present, what Rival can execute, and what has actually been validated.

| Component | Code incorporation | Executable Rival surface | v0.5 evidence | Remaining qualification |
|---|---|---|---|---|
| Semantic Similarity Rating | Two Apache-2.0 source modules at commit `86dcd259…` | `SemanticSimilarityRater`, `SSRElicitationProvider`, `/api/research/ssr` | PMF and numerical edge-case checks | Pin/qualify a production sentence encoder by language/domain |
| SYN-DIGITS distribution calibration | Original source snapshot plus Rival vectorized implementation | OpinionQA calibration pipeline | 489 questions, five family-held-out folds, mean TVD 0.169 | Customer-domain temporal/entity holdout |
| SYN-DIGITS synthetic control | Full MIT `SyntheticControl` at commit `db891b6f…`, with four portability patches | Matrix completion and row/column evaluation wrapper/API | Hard/soft/ALS invariants and full ridge path | Workload-specific method/rank selection without outcome leakage |
| UQ Survey Simulation | MIT `evaluations.py` at commit `fe3eb191…` | Survey and residual confidence intervals/API | Formula parity for CLT, Hoeffding, Bernstein | Empirical coverage on locked customer outcomes |
| Twin-2K-500 | CC BY 4.0 data subset and benchmark protocol | Longitudinal baselines, transfer, anchor correction | Direct 54.1%; leakage-safe transfer 44.0%; transfer gate fails | Do not market novel individual twins; test new data prospectively |
| Twin-2K Mega MAD | Apache-2.0 evaluator at commit `afe2bb93…` | Official MAD wrapper | Summary invariant passes | Freeze task mapping for each customer instrument |
| H&M demand/pricing | Three source modules at commit `b56a7c0…`; separate user permission, no public LICENSE found | Calibrated persona/no-buy model, CVaR price optimizer/API | Simplex, bounded probability, and deterministic optimizer checks | Real products, prices, inventory, competitors, and future demand |
| S-RCT | Rival-written from paper | Weighted paired estimator and pre-period adjustment/API | Effect/adjustment invariants | Randomized or quasi-randomized prospective outcome study |
| Interview grounding | Rival-written from 1,000-person study concept | Typed JSONL/CSV ingestion and persona builder/API | Outcome-key rejection and provenance tests | Consent, deletion, PII minimization, retrieval quality |
| Centauri | Adapter only; no code/weights | `CentauriProvider` | Identity binds revision, endpoint, corpus, and declared license | Operator must pin authorized weights and benchmark them |
| Socrates / SocSci210 | Adapter only; no code/weights | `SocratesProvider` | Same identity/credential controls | Obtain authorized checkpoint/data terms and benchmark it |

## What “development complete” means here

The planned scientific components have product-facing boundaries, provenance records, tests, CLI/API surfaces, and failure guards. Rival no longer has a known repository component waiting to be wired from this plan.

It does **not** mean Simile/Aaru parity is empirically demonstrated. Their proprietary data, model training, customer workflows, and prospective results are unavailable. Rival can now run the corresponding categories of experiment; it still needs locked customer outcomes to demonstrate accuracy and commercial advantage.

## Qualification commands

```bash
python -m rival qualify-research-components
python -m rival qualify-integrity
python -m rival qualify-all --output-dir reports
python -m unittest discover -s tests -v
```

The component command must remain separate from predictive qualification so a passing software/parity check cannot be mistaken for real-world validity.
