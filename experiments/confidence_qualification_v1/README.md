# Confidence qualification v1

**Result: FAIL.** This isolated retrospective experiment tests an error predictor and separately calibrated upper error bounds against released human survey counts. It does not change Rival's production confidence code or any frozen Twin/Mega experiment.

The protocol was written before this run. All four variants and both fixed splits are retained in `results/report.json`. Five protocol tests passed. The run made zero provider calls and incurred zero provider charges.

| Final-test measure | Question-family split | Later-wave split |
|---|---:|---:|
| Final questions / families | 73 / 68 | 75 / 71 |
| Raw persona TVD | 0.3416 | 0.3209 |
| Historical choice-position baseline TVD | 0.2333 | 0.2195 |
| Existing calibrated persona TVD | 0.1798 | 0.1758 |
| Mean-error baseline RMSE | 0.0877 | 0.0816 |
| Primary combined-ridge error RMSE | 0.0838 | 0.0784 |
| Primary error-ranking AUROC | 0.7129 | 0.6743 |
| Observed complete-family upper-bound coverage | 85.3% | 95.8% |
| Mean upper TVD | 0.2727 | 0.3130 |
| Accepted questions at upper TVD <= 0.16 | 0 | 0 |
| Prespecified gate | FAIL | FAIL |

These results show a modest descriptive improvement in error prediction on these splits. They do not establish statistical superiority, dependable selective risk, or individual behavioral fidelity. Complete-family coverage missed the illustrative 90% gate in one split. Wide bounds admitted no questions in either split. We therefore do not promote this estimator to production or describe it as a qualified confidence system.

The population calibration result is useful but narrower: it reuses already released synthetic responses and predicts historical aggregate survey proportions. It does not train or evaluate a new live human-behavior model. The later-wave test cannot rule out contamination of the original foundation model. Human proportions are normalized released counts, not reconstructed survey-weighted national estimates. Lexical grouping does not establish independent experiments; only two final waves are available.

The four roles are population fitting, error-model fitting, bound calibration, and final scoring. The primary model and hyperparameters were fixed in `PROTOCOL.md`; no variant was selected after seeing the final results. The final sets are now spent qualification sets and must not be reused to qualify a revised method. The upper bounds require exchangeability of independent families; they do not establish subgroup, drift, or post-selection coverage.

## Reproduction

Run from the repository root with the dependencies listed in `results/environment.json`. Use a fresh directory because the program refuses to overwrite results:

```bash
python -m unittest discover -s experiments/confidence_qualification_v1 -p 'test_*.py' -v
python experiments/confidence_qualification_v1/run.py --output-dir /path/to/new/results
```

Source hashes and split assignments are in `results/manifest.json`. Predictions were written before final scoring, and their hashes are recorded in `results/report.json`. There was no independent custodian or external preregistration timestamp: these records establish reproducibility, not externally verified blindness.

Parent repository revision: `902a3eb975b9e765f5a0c41e66afb6abe2ead5c6`. Tested on Linux/Python 3.12; Windows execution was not tested in this session. This directory contains a research experiment, not a replacement application.
