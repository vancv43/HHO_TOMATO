# HHO_TOMATO
# HHO-XQI: Tomato Grading

Code accompanying the paper *"HHO-XQI: Explainable Quality Index for Tomato Grading"* (submitted to the *Journal of Data Science and Intelligent Systems*).

This repository contains the feature-extraction, training, optimization, ablation, statistical-testing, and figure-generation scripts used to produce every result reported in the paper. **No paper text, no journal template files, no trained model weights, and no precomputed feature files are included** — these are excluded intentionally (see [What's excluded](#whats-excluded)) so that results can be independently reproduced from raw images rather than from shipped artifacts.

## What this study does

- Extracts a 125-dimensional interpretable color-statistical descriptor (RGB/HSV/Lab channel statistics + histograms + 17 ripeness/damage indices) from each tomato image.
- Trains four heterogeneous classical learners (logistic regression, random forest, extra trees, k-nearest neighbors) on these features.
- Uses **Harris Hawks Optimization (HHO)**, a swarm-intelligence metaheuristic, to search the soft-voting fusion weights across the four learners, producing a continuous **Tomato Quality Index (TQI)**.
- Benchmarks the fused classifier against a fine-tuned ResNet18 CNN on accuracy, latency, and training-time trade-offs.
- Applies **SHAP** to explain the fused classifier and **Grad-CAM** to explain the CNN.
- Reports repeated cross-validation with Holm-corrected Wilcoxon significance tests, a two-axis ablation study, an HHO hyperparameter sensitivity analysis, and a convergence study.

## Dataset

This project uses the **Tomatoes Dataset** (Damaged / Old / Ripe / Unripe classes, 7,226 images), publicly available on Kaggle:

- https://www.kaggle.com/datasets/enalis/tomatoes-dataset

This repo does not redistribute the dataset. Download it and arrange it as:

```
data/Tomatoes Dataset/
├── train/
│   ├── Damaged/*.jpg
│   ├── Old/*.jpg
│   ├── Ripe/*.jpg
│   └── Unripe/*.jpg
└── val/
    ├── Damaged/*.jpg
    ├── Old/*.jpg
    ├── Ripe/*.jpg
    └── Unripe/*.jpg
```

(matching the dataset's original train/validation partition), placed as a sibling of `code/` — or point `TOMATO_DATA_ROOT` (see [Configuration](#configuration)) at wherever you extracted it.

## What's excluded

Per the paper's data/code release policy, this repository intentionally omits:

- **Trained model weights** (`weights/cnn_best.pt`, ~43 MB) — retrain with `train_cnn.py`.
- **Precomputed feature cache** (`features.npz`, ~3.5 MB) — regenerate with `extract_all.py`.

Everything else needed to reproduce every number, table, and figure in the paper from the raw images is included.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10 on Apple Silicon (PyTorch MPS backend for the CNN branch); no CUDA GPU is required, though one will be faster for `train_cnn.py`.

## Configuration

Two environment variables control where data is read from and results are written to (both default to sensible relative paths, so most users won't need to set them):

| Variable | Default | Purpose |
|---|---|---|
| `TOMATO_DATA_ROOT` | `../data/Tomatoes Dataset` (relative to `code/`) | Root folder containing `train/` and `val/` class subfolders. |
| `TOMATO_OUT_DIR` | `../outputs` (relative to `code/`) | Where every script reads/writes intermediate artifacts (features, trained weights, JSON results, figures). |

## Pipeline

Run scripts from `code/` in this order:

| Script | Purpose |
|---|---|
| `features.py` | The 125-D color-statistical feature extractor (imported by other scripts, not run directly). |
| `extract_all.py` | Extracts features for the full train/val split into `features.npz`. |
| `train_baseline.py` | Trains the four base learners, runs HHO to find fusion weights, evaluates the HHO-XQI ensemble. |
| `ablation.py` | Feature-space and ensemble-composition ablation study. |
| `convergence.py` | Multi-seed HHO convergence study. |
| `sensitivity.py` | HHO population-size / iteration-budget sensitivity sweep. |
| `cv_stats.py` | Repeated stratified cross-validation with Holm-corrected Wilcoxon significance tests against each baseline. |
| `complexity.py` | Per-model training time, inference latency, and asymptotic complexity. |
| `train_cnn.py` | Fine-tunes a ResNet18 CNN benchmark on the same split. |
| `cnn_latency.py` | CPU vs. MPS/GPU inference latency for the CNN. |
| `gradcam.py` | Grad-CAM visualizations for the CNN branch. |
| `shap_analysis.py` | SHAP feature-attribution analysis on the fused classifier. |
| `fig_architecture.py`, `fig_cnn_results.py`, `fig_dataset_and_baseline.py` | Regenerate the paper's figures from the JSON results above. |

`hho.py` is a self-contained NumPy implementation of Harris Hawks Optimization (Heidari et al., 2019 — exploration, soft/hard besiege, Lévy-flight dive), with no dependency on the rest of the pipeline; it can be reused standalone for any continuous weight-vector optimization problem.

The `*_results.json` files in this repo are the actual saved outputs from the runs reported in the paper, included so that `fig_*.py` and the analysis scripts can be re-run without recomputing everything from scratch.

## Key results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic regression | 0.8952 | 0.8717 |
| Random forest | 0.9338 | 0.9119 |
| Extra trees | 0.9434 | 0.9266 |
| K-nearest neighbors | 0.8966 | 0.8727 |
| **HHO-XQI (ours)** | **0.9462** | **0.9290** |
| ResNet18 CNN | 0.9931 | 0.9918 |

HHO-XQI's improvement over the strongest single learner (extra trees) was **not** statistically significant under repeated cross-validation (Wilcoxon $p = 0.071$ after Holm correction), while it was significant against the other three baselines ($p < 10^{-4}$). The CNN reaches higher accuracy but requires ~147× longer training time (575.7 s vs. 3.92 s) for comparable CPU inference latency (11.31 ms vs. 12.05 ms). See the paper for the full discussion, including two honest null/negative findings reported rather than omitted: the hand-designed ripeness indices slightly *reduce* ensemble accuracy in the feature-space ablation, and HHO-XQI is not statistically distinguishable from extra trees alone.

## Citation

If you use this code, please cite the paper (details to be updated upon publication):

```bibtex
@article{chau2026hhoxqi,
  title   = {HHO-XQI: Explainable Quality Index for Tomato Grading},
  author  = {Chau, Van Van and Meesad, Phayung and Nguyen, Minh Tuan},
  journal = {Journal of Data Science and Intelligent Systems},
  year    = {2026}
}
```

## License

Code released under the MIT License. The Tomatoes Dataset is separately licensed by its Kaggle publisher and must be obtained from the official source linked above.
