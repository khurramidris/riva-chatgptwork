# Rival frozen qualification

Result: **FAIL**

one frozen aggregate study-family evaluation under declared custody and independence assumptions

Customer qualification: **not granted**.

Study family: Generated delivery preference cases

Audience: Six generated example records; no real consumers represented.

Evidence origins: generated.

Planned groups: 4; scored: 3; failed: 1.

| Gate | Result |
|---|---|
| absolute\_error | FAIL |
| acceptance\_rate | FAIL |
| accepted\_error\_risk | FAIL |
| classical\_fit\_converged | PASS |
| coverage | FAIL |
| failure\_rate | FAIL |
| improvement\_over\_classical\_multinomial | FAIL |
| improvement\_over\_human\_only | FAIL |
| improvement\_over\_synthetic\_only | FAIL |
| improvement\_over\_weighted\_history | FAIL |
| missingness | FAIL |
| operating\_cost\_and\_time | FAIL |
| priority\_subgroups | PASS |
| prospective\_human\_evidence | FAIL |
| untouched\_historical\_evidence | FAIL |

| Baseline | Conservative mean improvement | Simultaneous lower bound |
|---|---:|---:|
| classical\_multinomial | -0.3750 | -1.0000 |
| human\_only | -0.3750 | -1.0000 |
| synthetic\_only | -0.3208 | -1.0000 |
| weighted\_history | -0.3750 | -1.0000 |

Positive improvement favors Rival. Failures receive maximum error and minimum paired improvement.

| Study | Result | Rival TVD |
|---|---|---:|
| l12-benefit | SCORED | 0.0000 |
| l12-harm | SCORED | 0.2500 |
| l12-missing-response | SCORED | 0.1667 |
| l12-unfinished | incomplete\_execution | — |

- Public historical evidence can still be in model pretraining; local freezing does not prove contamination freedom.
- Human roster, source truth, weights, semantic independence and non-journal costs are custodian declarations.
- Each declared study group counts once. More synthetic draws do not increase independent evidence.
- Coverage refers to the observed human distribution; it is not a latent population or subgroup interval.
- Missingness bounds concern the frozen eligible sample, not selection bias outside that sample.
- Human-only uses a separate full-quota sample under the same budget cap; realized spending need not be equal.
- Mean-error and improvement bounds use Hoeffding; binomial gates assume independent exchangeable study groups.
- Priority subgroup claims require separate protected evidence; this increment cannot qualify them.
- No model, threshold, or feature search may reuse this final evaluation set.
