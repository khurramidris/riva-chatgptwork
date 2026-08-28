# Third-Party Research, Code, and Data Notices

The user confirmed commercial permission for the research papers, repositories, and data used in this build. Rival also preserves the public attribution and license information below. Public license texts and any separate signed permission are authoritative.

## Incorporated in v0.2

### SYN-DIGITS

- Source: <https://github.com/yw3453/syn-digits>
- Snapshot: `db891b6f821c914455b11763a96679864bf4fc48`
- Public repository license: MIT, copyright (c) 2026 yw3453
- Bundled source: `vendor/syn_digits/distribution_calibration.py`, its upstream README, and MIT license
- Bundled research artifacts: the 489-question OpinionQA five-choice file and 2,058-persona response matrix
- Rival change: the production runtime independently vectorizes the released KL/simplex objective to avoid constructing hundreds of dense `A_j` matrices. The unmodified upstream module remains beside it for audit and parity work.

The OpinionQA questions and human answer counts originate from Pew Research survey material, as documented by SYN-DIGITS. Redistribution and commercial use in this package rely on the user's confirmed permission in addition to the repository terms.

### Twin-2K-500

- Dataset: <https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500>
- Snapshot: `f883165a3026fde855dfd448e0cd16443ab257b6`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>
- Authors: Olivier Toubia, George Z. Gui, Tianyi Peng, Daniel J. Merlau, Ang Li, and Haozhe Chen
- Paper/data citation: *Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People* (2025), <https://arxiv.org/abs/2505.17479>
- Bundled artifacts: question catalog, wave-4 mapping, wave-1/3 human responses, wave-4 human outcomes, and released GPT-4.1-mini simulation results
- Rival changes: selected files were renamed to stable product-facing names; response rows are aligned by `TWIN_ID`; question columns are mapped to catalog metadata; numeric codes are not semantically altered.

CC BY 4.0 permits sharing and adaptation with attribution, a license link, and an indication of changes. No endorsement by the dataset authors is implied.

## Reviewed but not directly bundled

| Upstream | Snapshot | Public license | Rival use |
|---|---|---|---|
| ActivitySim PopulationSim | `cc22d25499e7c54ee5ea184a7ecd0f9ee7f20231` | BSD-3-Clause | Calibration/QA reference; Rival-owned compact raking |
| AI-Augmented Estimation | `b7e3f9a89690b981076498a0b4a272bbe47de2d2` | MIT code; data separate | Prediction-powered residual correction design |
| Adaptive Querying with AI Persona Priors | `fbd8e19eed6af960b64e3afae13e3bcf4020b73f` | MIT | Human-anchor selection design |
| Centauri | `6ef71bc4ce5661df106d31f0727497af15268914` | MIT | Reserved specialized-model adapter; no weights bundled |
| LLM Economist | `8d1c9295ab69ed4b819fc58ef0684c70a8c73a53` | MIT | Provider and structured-output patterns |
| Twin-2K-500 Mega Study code | `afe2bb933fce377ed196f441a4c12962cb55a53a` | Apache-2.0 | Evaluation schema reference |

Model weights, hosted APIs, source datasets, and pretrained backbones may carry terms separate from repository code. Before external redistribution, verify the signed permissions against the exact SHA-256 inventory in `upstreams.lock.json`.
