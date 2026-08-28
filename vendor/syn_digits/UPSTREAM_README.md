# SYN-DIGITS

This repository contains code and data for the paper **"SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation"** ([arXiv](https://arxiv.org/abs/2604.07513)).

## Repository Structure

```
syn-digits/
├── data/                                      # Datasets (see data/data.md for details)
│   ├── 2K500/                                 # Twin-2K-500 dataset
│   │   └── synthetic_control/
│   │       └── <persona_construction>/        # One folder per persona construction
│   │           ├── real.csv                   # Real survey responses
│   │           └── LLM.csv                    # Digital twins' responses
│   ├── MovieLens-20M/
│   │   ├── distribution_calibration/          # Distribution calibration data
│   │   │   ├── top_movies_df.csv
│   │   │   └── persona_ratings.csv
│   │   └── synthetic_control/                 # Synthetic control data
│   │       ├── real.csv
│   │       ├── LLM.csv
│   │       └── LLM_in_context_other_ratings_full.csv
│   ├── OpinionQA/
│   │   └── distribution_calibration/
│   │       ├── Qs_likert_scale_5_choices.json
│   │       └── persona_answers.csv
│   ├── personas.json                          # Persona descriptions (2,058 personas)
│   └── data.md                                # Detailed data documentation
│
├── notebooks/
│   ├── distribution_calibration/              # Distribution calibration experiments
│   │   ├── MovieLens_experiments.ipynb
│   │   └── OpinionQA_experiments.ipynb
│   └── synthetic_control/                     # Synthetic control experiments
│       ├── 2K500_experiments.ipynb            # Single-run exploration
│       ├── 2K500_experiments_batch.ipynb      # Batch benchmark across persona constructions
│       ├── MovieLens_experiments.ipynb        # Single-run exploration
│       └── MovieLens_experiments_batch.ipynb  # Batch benchmark
│
└── src/                                       # Core Python modules
    ├── distribution_calibration.py            # Distribution calibration optimization
    ├── synthetic_control.py                   # Synthetic control implementation
    ├── synthetic_control_batch.py             # Batch benchmarking across methods
    └── process_and_diagnostics.py             # SVD diagnostics and preprocessing
```

## Setup

**Requirements:** Python >= 3.10

**Installation using [uv](https://docs.astral.sh/uv/) (recommended):**
```bash
uv sync
```

**Installation using pip:**
```bash
pip install -e .
```

## Usage

### Experiments I: Synthetic Control

**Single-run notebooks** explore one persona construction at a time with diagnostics and plots:
```bash
jupyter notebook notebooks/synthetic_control/2K500_experiments.ipynb
jupyter notebook notebooks/synthetic_control/MovieLens_experiments.ipynb
```

**Batch notebooks** benchmark all methods across multiple persona constructions:
```bash
jupyter notebook notebooks/synthetic_control/2K500_experiments_batch.ipynb
jupyter notebook notebooks/synthetic_control/MovieLens_experiments_batch.ipynb
```

**What it does:**
1. Performs SVD diagnostics on real vs. LLM-generated data
2. Implements multiple methods (ridge, lasso, elastic net, synthetic control, neural net, matrix completion variants, synthetic intervention) and reports Pearson correlation

**Core modules:** `src/synthetic_control.py`, `src/synthetic_control_batch.py`

### Experiments II: Distribution Calibration

**Run experiments:**
```bash
jupyter notebook notebooks/distribution_calibration/OpinionQA_experiments.ipynb
jupyter notebook notebooks/distribution_calibration/MovieLens_experiments.ipynb
```

**What it does:**
1. Learns optimal weights for personas to match target distributions
2. Compares multiple divergence metrics (TV, KL, Chi-squared, L1, L2, etc.)
3. Evaluates fitting modes and generates convergence plots, variance ratio plots, and distribution comparisons

**Core module:** `src/distribution_calibration.py`

## Data

See [`data/data.md`](data/data.md) for detailed documentation of all datasets, including file schemas, column descriptions, and examples.

## Outputs

All results are saved to the `outputs/` directory (auto-created when running notebooks):

- **Figures:** Convergence plots, correlation heatmaps, distribution comparisons (PDF)
- **Tables:** CSV files with evaluation metrics across different methods
- **Naming convention:** Results are organized by dataset, method, and direction (column/row)

## Citation
```bibtex
@misc{fan2026syndigits,
      title={SYN-DIGITS: A Synthetic Control Framework for Calibrated Digital Twin Simulation}, 
      author={Grace Jiarui Fan and Chengpiao Huang and Tianyi Peng and Kaizheng Wang and Yuhang Wu},
      year={2026},
      eprint={2604.07513},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.07513}, 
}
```
