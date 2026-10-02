# Neonatal encephalopathy prediction from cardiotocography

Deep learning models that predict neonatal encephalopathy (NE) in term infants from intrapartum
cardiotocography (CTG): the fetal heart rate (FHR) and uterine contraction (TOCO) signals recorded
during labor.

The task is binary classification of 30-minute CTG windows sampled at 4 Hz, on a small and
imbalanced clinical cohort (roughly 290 subjects, 13% positive). The code is built around that
constraint: every split is made at the subject level, and results are reported over
cross-validation folds.

Preliminary results were presented at the Pediatric Academic Societies (PAS) Meeting, 2025.

## Data

**No data or trained weights are included in this repository.** The clinical dataset is private and
cannot be shared. `.gitignore` excludes data files and checkpoints so they are not committed by
accident.

To run the code, place a NumPy archive in `dataset/` with these arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `X` | `(N, 2, 7200)` | CTG windows, channels `[FHR, TOCO]`, 30 minutes at 4 Hz |
| `y` | `(N,)` | Label per window, inherited from the subject (0 = normal, 1 = NE) |
| `subject_ids` | `(N,)` | Subject identifier per window, used to group windows when splitting |

The public [CTU-UHB Intrapartum Cardiotocography Database](https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
(552 recordings) can be converted to the same format for pretraining or for testing the pipeline.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train and evaluate one model with 5-fold cross-validation:

```bash
python run.py --config configs/cnn_lstm.yaml
```

Command-line flags override values in the YAML file:

```bash
python run.py --config configs/inception_time.yaml --seed 1 --learning_rate 5e-5
```

Train several models on identical folds and evaluate their averaged predictions:

```bash
python run_ensemble.py --configs configs/cnn_lstm.yaml configs/rocket.yaml --weights 0.6 0.4
```

Per-fold metrics and their mean and standard deviation are written to
`checkpoints/<experiment>/cv_results.json`.

## Models

| Model | Config | Description |
| --- | --- | --- |
| CNN-LSTM | `configs/cnn_lstm.yaml` | Convolutional feature extractor followed by an LSTM |
| InceptionTime | `configs/inception_time.yaml` | Residual inception blocks with multi-scale kernels |
| PatchTransformer | `configs/patch_transformer.yaml` | Transformer encoder over fixed-length patches |
| PatchMLP | `configs/ctg_patchmlp.yaml` | MLP over patch embeddings |
| ROCKET | `configs/rocket.yaml` | Random convolutional kernels with a linear classifier |

## Evaluation protocol

- **Subject-level splits.** Each subject contributes several windows, so all windows from one
  subject stay in the same split. Outer folds use `StratifiedGroupKFold`; within each fold, 15% of
  the training subjects are held out for validation.
- **Model selection on validation only.** Early stopping and the decision threshold are both chosen
  on the validation subjects, then applied unchanged to the test fold.
- **Class imbalance.** Focal loss is available through `use_focal`, `focal_alpha`, and `focal_gamma`.
- **Reproducibility.** Seeds are fixed for Python, NumPy, and PyTorch, and the fold assignment is
  deterministic.

## Results

<!-- Fill in from checkpoints/<experiment>/cv_results.json. Report mean ± std over the 5 folds. -->

| Model | AUROC | Accuracy |
| --- | --- | --- |
| CNN-LSTM | | |
| InceptionTime | | |
| PatchTransformer | | |
| PatchMLP | | |
| ROCKET | | |

## Repository structure

```
run.py                   Entry point: cross-validated training and evaluation
run_ensemble.py          Ensemble of several models over identical folds
extract_windows.py       Cuts full recordings into non-overlapping 30-minute windows
dataPreprocessing.ipynb  Cleaning and segmentation of the raw recordings (outputs stripped)
configs/                 One YAML file per model, plus shared defaults
data_provider/           Dataset classes and data loaders
exp/                     Training, validation, and test loops
model/                   Model definitions
layers/                  Shared layers (patch embeddings)
utils/                   Metrics, split helpers, early stopping
```
