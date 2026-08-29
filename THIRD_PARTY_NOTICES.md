# Third-Party Research, Code, and Data Notices

The user confirmed commercial permission for the research papers, repositories, and data used in this build. Rival preserves public attribution and license information below. Public license texts and any separate signed permission remain authoritative. Exact incorporated paths, commits, modifications, and SHA-256 hashes are recorded in `upstreams.lock.json`.

## Code incorporated in v0.5

| Source | Pinned revision | Public terms at revision | What is bundled | Rival integration |
|---|---|---|---|---|
| [PyMC Labs Semantic Similarity Rating](https://github.com/pymc-labs/semantic-similarity-rating) | `86dcd2597c7824e4fd6546b884c5500c43a4b022` | Apache-2.0 | `compute.py`, `response_rater.py`, license | The PMF computation is called by `rival/elicitation.py`; Rival adds pluggable embeddings/text generation and numerical guards. |
| [SYN-DIGITS](https://github.com/yw3453/syn-digits) | `db891b6f821c914455b11763a96679864bf4fc48` | MIT | Distribution calibration snapshot, full `SyntheticControl`, license, OpinionQA research artifacts | Full component retained with four portability changes: local NumPy truncated SVD, headless plot closing, temporary/configurable diagnostics directory, and bounded evaluation imputation iterations. |
| [UQ LLM Survey Simulation](https://github.com/yw3453/uq-llm-survey-simulation) | `fe3eb19111a2d9327e9ec051bd96f68750a7895d` | MIT | `evaluations.py`, license | Optional-`tqdm` import patch; Rival exposes CLT, Hoeffding, and Bernstein intervals and tests them against upstream `CI()`/`synthetic_CI()`. |
| [LLM Demand Simulator](https://github.com/khurramidris/LLM-demand-simulator) | `b56a7c0acad7406bff81b7cdf179314894b2fa97` | No LICENSE file found | `llm_mix.py`, `llm_mix_cal.py`, `metrics.py` | Incorporation relies on the user's separate permission. `rival/pricing.py` adapts the calibrated persona/no-buy mixture to a SciPy-only runtime and adds revenue/CVaR price selection. Do not describe this repository as MIT or another public license without an authoritative license grant. |
| [Twin-2K-500 Mega Study](https://github.com/TianyiPeng/Twin-2K-500-Mega-Study) | `afe2bb933fce377ed196f441a4c12962cb55a53a` | Apache-2.0 | Official MAD evaluation module, license | `rival/research/mad.py` provides lazy column/task/summary wrappers. The parallel Mega-Study benchmark also follows the authors' survey and outcome conventions in Rival-written code. |

## Data incorporated

### OpinionQA through SYN-DIGITS

Rival bundles the 489-question five-choice artifact and 2,058-persona response matrix from the pinned SYN-DIGITS snapshot. The upstream project documents that the question text and human counts originate from Pew Research survey material. Redistribution and commercial use in this package rely on the user's confirmed permission in addition to repository terms.

### Twin-2K-500

- Dataset: [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
- Snapshot: `f883165a3026fde855dfd448e0cd16443ab257b6`
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Authors: Olivier Toubia, George Z. Gui, Tianyi Peng, Daniel J. Merlau, Ang Li, and Haozhe Chen
- Paper: [Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People](https://arxiv.org/abs/2505.17479)

Rival bundles a question catalog, wave-4 mapping, aligned wave-1/3 human history, wave-4 outcomes, and released GPT-4.1-mini predictions. Files were renamed and aligned by `TWIN_ID`; numeric response codes were not semantically altered. No endorsement by the authors is implied.

### Twin-2K-500 Mega-Study

- Dataset: [Twin-2K-500 Mega-Study](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study)
- Snapshot: `0401b715a341ac4b5f98b4424b4aecf9d29570d0`
- License: Apache-2.0
- Paper: [Twin-2K-500 Mega-Study](https://arxiv.org/abs/2509.19088)

The development benchmark downloads, but does not redistribute, the three
official study Parquet files and seven original full-persona shards. Exact
source URLs, sizes, SHA-256 hashes, prompt identities, and reference-output
files are recorded in `rival/studies/mega_study_v1/MEGA_STUDY_MANIFEST.json`.
Selected human target answers are not committed to the repository.

## Paper-derived or adapter-only integrations

These components contain Rival-written code, not copied repository code:

| Source | Rival use | What is not bundled |
|---|---|---|
| [S-RCT](https://arxiv.org/abs/2608.02345) | Paired surrogate-experiment estimator and pre-period residual adjustment | No public repository code was copied. |
| [1,000-person interview-grounded agents study](https://arxiv.org/abs/2411.10109) | Typed interview-to-person-state ingestion and provenance | No study code was copied. |
| [Centauri](https://github.com/socius-org/Centauri), revision `6ef71bc4ce5661df106d31f0727497af15268914` | Revision-, endpoint-, corpus-, and license-bound inference adapter | No Centauri code or weights are bundled. Repository code is MIT; model weights may have separate terms. |
| [Socrates / SocSci210](https://arxiv.org/abs/2509.05830) | Equivalent inference adapter for an operator-hosted checkpoint | No code, dataset, or model weights are bundled. |

## Reviewed but not bundled

| Upstream | Snapshot | Public license | Rival use |
|---|---|---|---|
| ActivitySim PopulationSim | `cc22d25499e7c54ee5ea184a7ecd0f9ee7f20231` | BSD-3-Clause | Calibration/QA reference; Rival-owned compact raking |
| AI-Augmented Estimation | `b7e3f9a89690b981076498a0b4a272bbe47de2d2` | MIT code; data separate | Prediction-powered residual-correction design |
| Adaptive Querying with AI Persona Priors | `fbd8e19eed6af960b64e3afae13e3bcf4020b73f` | MIT | Human-anchor selection design |
| LLM Economist | `8d1c9295ab69ed4b819fc58ef0684c70a8c73a53` | MIT | Structured-output/provider reference |

Model weights, hosted APIs, training datasets, source survey instruments, and pretrained backbones may carry terms separate from repository code. Before redistribution or deployment, verify the user's permissions against the exact inventory and deployment-specific model/data terms.
