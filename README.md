# Predicting intrapartum fetal compromise from cardiotocography

Deep learning models that predict adverse neonatal outcomes from intrapartum cardiotocography (CTG):
the fetal heart rate (FHR) and uterine contraction (UC) signals recorded during labor.

The public benchmark is the [CTU-UHB Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)
(552 recordings, 4 Hz, ending near delivery), with fetal acidemia (umbilical artery pH below a
threshold, 7.05 by default) as the target. Everything in this repository runs end to end on that
public data. The code also supports a private clinical cohort for neonatal encephalopathy (NE);
those data are not included and cannot be shared.

Preliminary results on the clinical cohort were presented at the Pediatric Academic Societies (PAS)
Meeting, 2025.

## Design principles

- **Know where each window sits in labor.** Windows are cut backwards from a fixed anchor (end of
  stage I, or end of recording), and every window records its distance to birth.
- **Missing signal is missing, not zero.** Signal loss is kept as a mask; scaling uses fixed
  physiological constants, so nothing is fit on the evaluation data.
- **Check for shortcuts before training.** The prepare script compares cases and controls on
  properties that are not physiology (recording length, signal loss, stage II coverage, delivery
  type) and flags large differences.
- **Split by subject.** All windows from one recording stay in the same fold.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (public data)

Download CTU-UHB from PhysioNet (about 40 MB) and build window files for both labels:

```bash
python scripts/prepare_ctu_uhb.py --wfdb_dir data/ctu-uhb --download --ph_threshold 7.05
python scripts/prepare_ctu_uhb.py --wfdb_dir data/ctu-uhb --ph_threshold 7.15
```

pH < 7.05 is the primary label (40 positives of 552); pH < 7.15 (105 positives) is a secondary
analysis with more stable estimates.

Train with repeated cross-validation (5 folds x 5 repeats), run the shortcut baselines on the
same folds, and evaluate:

```bash
D=ctu_uhb_stage1_end_w30m_n1_v0.2_ph7.05
python run.py --config configs/ctu_uhb_cnn_lstm.yaml --data_path $D.npz --results_dir results/$D/cnn_lstm_wbce
python scripts/baselines.py --data dataset/$D.npz --n_repeats 5 --out_dir results/$D/baselines
python scripts/evaluate.py --data dataset/$D.npz \
    --runs cnn_lstm_wbce=results/$D/cnn_lstm_wbce \
           missingness=results/$D/baselines/missingness \
           missingness_length=results/$D/baselines/missingness_length \
    --reference missingness_length --subset all vaginal --out results/$D/table.csv
```

On a cluster, run each repeat as its own job with `--only_repeat r --results_dir results/$D/cnn_lstm/r<r>`;
`evaluate.py` merges every `oof_predictions.csv` under the run directory.

`prepare_ctu_uhb.py` writes into `dataset/` (git-ignored):

- `ctu_uhb_store.npz` and `ctu_uhb_meta.csv`: every recording at full length, unscaled, with
  missing samples as NaN, plus the clinical header fields. Built once and reused.
- `ctu_uhb_<anchor>_w<minutes>m_n<windows>_v<min valid>_ph<threshold>.npz`: the windows a model
  trains on.

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--anchor` | `stage1_end` | Count windows back from the start of stage II (`stage1_end`) or the end of the recording (`end`) |
| `--window_min` | `30` | Window length in minutes |
| `--n_windows` | `1` | Windows per recording, moving back from the anchor |
| `--min_valid_frac` | `0.2` | Drop windows with less observed FHR than this (see below) |
| `--ph_threshold` | `7.05` | Label is 1 when pH is below this |
| `--no_mask_channels` | off | Leave out the observed-mask channels (input becomes 2 channels) |

The script prints the exact `run.py` command for the file it wrote, including `--seq_len` and
`--in_channels`.

Window file contents:

| Key | Shape | Description |
| --- | --- | --- |
| `X` | `(N, C, T)` | Scaled `[FHR, UC, FHR observed, UC observed]`, missing samples set to 0 |
| `mask` | `(N, 2, T)` | True where FHR / UC were observed |
| `y` | `(N,)` | 1 if the recording's pH is below the threshold |
| `subject_ids` | `(N,)` | Recording id, used to keep windows of one recording in one fold |
| `min_before_birth` | `(N,)` | Minutes from the end of the window to birth |
| `ph`, `delivery_type`, `recording_min`, `rec_fhr_missing_frac` | `(N,)` | Recording-level values used by the baselines and subset analyses |

## Shortcuts in CTU-UHB, and how the evaluation handles them

The prepare step compares positives and negatives on properties that are not fetal physiology.
On CTU-UHB at pH < 7.05:

| Signal that is not physiology | AUROC on its own |
| --- | --- |
| Fraction of FHR missing over the recording | 0.71 |
| Fraction of FHR missing in the 30-minute window | 0.67 |
| Recording length | 0.66 |

Cesarean deliveries are also over-represented among positives (15 of 46 cesareans, against 25 of
506 vaginal deliveries). Because of this the evaluation:

- **keeps poor-quality windows.** A 50% observed-FHR filter would remove 10 of the 40 positives
  but only 24 of 512 negatives; the default of 20% keeps 38 of 40. The model sees the
  observed-masks as input channels instead.
- **includes shortcut baselines.** `baselines.py` fits logistic regressions on missingness and
  recording length using the same folds. `evaluate.py --reference` reports each model's AUROC
  difference from a baseline with a paired bootstrap interval.
- **reports a vaginal-delivery subset** (`--subset vaginal`).
- **reports subject-level metrics** pooled across folds and averaged over repeats, with 95%
  stratified bootstrap intervals. With 40 positives, expect intervals of roughly ±0.08 AUROC.

## Pretrained time-series encoders (frozen embeddings + linear probe)

Two general-purpose pretrained encoders are tested without fine-tuning:
[MOMENT](https://github.com/moment-timeseries-foundation-model/moment) (base, large) and
[Mantis / MantisV2](https://github.com/vfeofanov/mantis) (final layer, and layer 2 as the
Mantis authors recommend for classification).

- **Full resolution.** Both encoders take 512 samples per channel. Each 30-minute window is
  cut into 14 consecutive 512-sample chunks (about 2 minutes each) instead of being
  downsampled, which would erase short-term FHR variability. Chunk embeddings are pooled
  (mean, or mean and max) per channel.
- **Missing signal.** MOMENT receives the observed-mask; Mantis gets gaps linearly
  interpolated. Chunks with under half the signal observed are left out of the pooling.
- **Probe.** `scripts/probe.py` fits a standardized L2 logistic regression on the same
  repeated folds as every other model. Its regularization is chosen by an inner
  cross-validation on the training recordings only.
- **Beyond the shortcuts?** `--add_shortcuts` adds the signal-loss and recording-length
  features, with the embedding reduced to 32 PCA components inside each training fold so the
  two groups compete on equal terms. If this run beats the shortcut baseline, the embedding
  carries information the shortcuts do not.

```bash
python scripts/extract_embeddings.py --data dataset/$D.npz --model moment-base \
    --out embeddings/${D%_ph*}__moment-base.npz           # GPU recommended
python scripts/probe.py --data dataset/$D.npz --embeddings embeddings/${D%_ph*}__moment-base.npz \
    --pool meanmax --n_repeats 5 --out_dir results/$D/probe_moment-base_meanmax
```

Embeddings do not depend on the label, so one extraction serves both pH thresholds.
Extraction needs `momentfm` (install with `--no-deps`; its pinned versions are outdated),
`mantis-tsfm`, `transformers`, `einops` and `datasets`.

## Running on CHTC

`chtc/` runs the repeated cross-validation as one GPU job per window file and repeat
(2 label thresholds x 5 repeats = 10 jobs), then evaluates everything together.

```bash
# on the submit node, from the repository root, with both window files in dataset/
mkdir -p chtc/logs
condor_submit chtc/cnn_lstm.sub

# when the jobs finish: unpack outputs, run the baselines on the same folds, evaluate
bash chtc/collect_and_evaluate.sh
```

| File | Role |
| --- | --- |
| `chtc/run_cv.sh` | Job wrapper: installs missing Python packages into the job directory, runs one repeat of `run.py`, packs the results |
| `chtc/cnn_lstm.sub` | Submit file; set `container_image` to your own image if you have one |
| `chtc/jobs.txt` | One line per job: window file stem, repeat |
| `chtc/collect_and_evaluate.sh` | Unpacks job outputs (`results_*.tar.gz`, `emb_*.tar.gz`), runs `baselines.py` if needed, then `evaluate.py` per window file |
| `chtc/extract.sub`, `chtc/run_extract.sh` | One GPU job per encoder in `extract_models.txt`; writes `embeddings/` |
| `chtc/probe.sub`, `chtc/run_probe.sh` | One CPU job per line of `probe_jobs.txt` (threshold x encoder x with/without shortcuts) |

Pretrained-encoder workflow:

```bash
condor_submit chtc/extract.sub           # 6 GPU jobs, one per encoder
bash chtc/collect_and_evaluate.sh        # unpacks embeddings/
condor_submit chtc/probe.sub             # 24 CPU jobs
bash chtc/collect_and_evaluate.sh        # final tables, both thresholds
```

To add a model, copy `cnn_lstm.sub`, change `run_name` and `config`, and submit; the
collection script picks up every run directory under `results/<window file>/`.

## Tests

```bash
python -m pytest tests -q
```

The tests build synthetic recordings in the CTU-UHB file format and check header parsing, missing
value handling, window placement relative to stage II and birth, resumable downloads, subject-level
splits without leakage, the baselines and evaluation, and one short CPU training run (skipped if
PyTorch is not installed).

## Training options

Command-line flags override values in the YAML file:

```bash
python run.py --config configs/ctu_uhb_cnn_lstm.yaml --learning_rate 5e-5 --n_repeats 3
```

Train several models on identical folds and evaluate their averaged predictions:

```bash
python run_ensemble.py --configs configs/cnn_lstm.yaml configs/rocket.yaml --weights 0.6 0.4
```

Each run writes `cv_results.json` (per-fold metrics) and `oof_predictions.csv` (one row per
test window per repeat) to `--results_dir`. Model weights are deleted after each fold unless
`--keep_checkpoints` is set.

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
- **Repeated cross-validation.** `--n_repeats` reruns the 5 folds with different subject
  assignments (seed `fold_seed + r`); every script that splits the data uses the same function,
  so models and baselines see identical folds.
- **Reproducibility.** Seeds are fixed for Python, NumPy, and PyTorch, and the fold assignment is
  deterministic.

## Results

<!-- Fill in from checkpoints/<experiment>/cv_results.json. Report mean ± std over the 5 folds. -->

Results on CTU-UHB (541 recordings with a usable window): subject-level AUROC, 5-fold
cross-validation x 5 repeats, 95% bootstrap intervals, from `scripts/evaluate.py`. The shortcut
baselines never look at the heart rate pattern; a model has to beat them to show it reads the trace.

| Model | AUROC, pH < 7.05 | AUROC, pH < 7.15 | vs. best shortcut baseline |
| --- | --- | --- | --- |
| Shortcut: missingness | 0.683 [0.588, 0.771] | 0.574 [0.510, 0.634] | |
| Shortcut: missingness + length | 0.716 [0.624, 0.800] | 0.638 [0.580, 0.693] | reference |
| CNN-LSTM, focal loss, early stop on val loss (`cnn_lstm_focal`) | 0.601 [0.513, 0.696] | 0.458 [0.429, 0.486] | worse |
| CNN-LSTM, weighted BCE, early stop on val AUROC (`cnn_lstm_wbce`) | | | |
| MOMENT-base, frozen + linear probe | | | |
| Mantis, frozen + linear probe | | | |

What the first CNN-LSTM runs showed:

- **At pH < 7.05 the network learned delivery mode.** Its scores separate cesarean from
  vaginal deliveries with AUROC 0.92 and correlate with signal loss; on vaginal deliveries
  only it is at chance (0.47). Its overall 0.60 is the delivery-mode shortcut, and it is
  still below the shortcut baseline.
- **At pH < 7.15 it did not train.** Within every fold its predictions varied by about
  0.002, so it had collapsed to a constant; the below-0.5 AUROC is noise in that constant.
  The cause was a focal-loss weight fixed for a different prevalence plus early stopping on
  validation loss. The corrected configuration weights the positive class from each training
  fold and keeps the checkpoint with the best validation AUROC; training logs now flag
  near-constant predictions. A regression test checks that this setup learns a planted signal.

Report the vaginal-delivery subset alongside the full cohort: it removes the delivery-mode
shortcut by design.

## Repository structure

```
run.py                   Entry point: cross-validated training and evaluation
run_ensemble.py          Ensemble of several models over identical folds
scripts/prepare_ctu_uhb.py   Downloads CTU-UHB, builds the store and window files, checks for confounds
scripts/baselines.py         Shortcut baselines (missingness, recording length) on the same folds
scripts/extract_embeddings.py  Frozen MOMENT / Mantis embeddings at full resolution (14 chunks per window)
scripts/probe.py             Linear probe on embeddings, same folds, regularization tuned inside each fold
scripts/evaluate.py          Subject-level AUROC/AUPRC with bootstrap intervals and baseline comparison
scripts/audit_raw_data.py    Per-file audit of raw clinical exports (format, truncation, timing)
extract_windows.py       Cuts full clinical recordings into non-overlapping 30-minute windows
dataPreprocessing.ipynb  Cleaning and segmentation of the clinical recordings (outputs stripped)
chtc/                    HTCondor submit file, job wrapper and result collection for CHTC
tests/                   Tests for the CTU-UHB pipeline and evaluation
configs/                 One YAML file per model, plus shared defaults
data_provider/           CTU-UHB loader (ctu_uhb.py), dataset classes and data loaders
exp/                     Training, validation, and test loops
model/                   Model definitions
layers/                  Shared layers (patch embeddings)
utils/                   Metrics, repeated subject-level CV splits, early stopping
```

## Data citation

CTU-UHB is released under the Open Data Commons Attribution License v1.0. If you use it, cite
Chudáček V, et al. Open access intrapartum CTG database. *BMC Pregnancy and Childbirth* 14:16 (2014),
and PhysioNet.
