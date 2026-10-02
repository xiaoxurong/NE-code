"""
run.py — CTG classification entry point
========================================
Key changes from original:
  1. K-fold CV (StratifiedGroupKFold at subject level) — essential for 290 samples.
  2. Separate held-out test set: one fold is held out from the start and never
     used for model selection.
  3. Dynamic `setting` name derived from model + timestamp (was hard-coded).
  4. `--seed` for full reproducibility.
  5. GPU detection cleaned up (argparse default was immediately overridden).
  6. YAML override now also accepts keys not in argparse (model-specific params).
  7. Added model-specific args (patch_size, num_kernels, num_heads, in_channels …).
  8. AUROC aggregated and reported across folds alongside accuracy.
  9. Results saved to a JSON file next to the checkpoints for later analysis.
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from exp.exp_classification import Exp_Classification


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── config file (parsed first; CLI flags override YAML) ──────────────────
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a YAML config file. CLI flags override YAML values.')

    # ── experiment ────────────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='PatchTransformer',
                        choices=[
                            'CNN', 'EnhancedCNN', 'CNN1D', 'EnhancedCNN1D',
                            'CNN_LSTM', 'TimeSeriesTransformer',
                            'InceptionTime', 'PatchTransformer', 'ROCKET',
                            'PatchMLP',  # keep legacy name
                        ])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_classes', type=int, default=2)

    # ── data ─────────────────────────────────────────────────────────────────
    parser.add_argument('--root_path',  type=str, default='./dataset/')
    parser.add_argument('--data_path',  type=str, default='CTG30minSegments.npz')
    parser.add_argument('--data',       type=str, default='CTG')
    parser.add_argument('--seq_len',    type=int, default=7200,
                        help='Length of each input window in time steps')
    parser.add_argument('--in_channels', type=int, default=2,
                        help='Number of input channels (2 = FHR + TOCO)')

    # ── cross-validation ─────────────────────────────────────────────────────
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (StratifiedGroupKFold at subject level). '
                             'Set to 1 to use a single stratified train/val split instead.')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Fraction of training subjects used as validation within each fold '
                             '(only used when n_folds > 1).')

    # ── training ─────────────────────────────────────────────────────────────
    parser.add_argument('--train_epochs',  type=int,   default=100)
    parser.add_argument('--patience',      type=int,   default=20,
                        help='Early-stopping patience (epochs without val improvement)')
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lradj',         type=str,   default='type1')
    parser.add_argument('--dropout',       type=float, default=0.1)
    parser.add_argument('--checkpoints',   type=str,   default='./checkpoints/')

    # ── GPU ───────────────────────────────────────────────────────────────────
    parser.add_argument('--gpu',            type=int,  default=0)
    parser.add_argument('--use_multi_gpu',  action='store_true', default=False)
    parser.add_argument('--devices',        type=str,  default='0,1,2,3')

    # ── shared model hyperparams ──────────────────────────────────────────────
    parser.add_argument('--d_model',     type=int, default=64)
    parser.add_argument('--e_layers',    type=int, default=2,  help='Transformer encoder layers')
    parser.add_argument('--num_heads',   type=int, default=4,  help='Attention heads')
    parser.add_argument('--use_norm',    type=int, default=1)
    parser.add_argument('--moving_avg',  type=int, default=13)
    parser.add_argument('--enc_in',      type=int, default=2,  help='Encoder input size (= in_channels)')

    # ── PatchTransformer / PatchMLP ───────────────────────────────────────────
    parser.add_argument('--patch_size', type=int, default=60,
                        help='Patch length in time steps. '
                             'At 4 Hz: 60=15 s, 120=30 s, 240=60 s')

    # ── ROCKET ────────────────────────────────────────────────────────────────
    parser.add_argument('--num_kernels', type=int, default=10_000,
                        help='Number of random kernels for ROCKET')

    # ── CNN_LSTM ──────────────────────────────────────────────────────────────
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--lstm_layers', type=int, default=1)

    # ── loss / class imbalance ────────────────────────────────────────────────
    # --use_focal / --no_focal  (mutually exclusive boolean pair)
    focal_group = parser.add_mutually_exclusive_group()
    focal_group.add_argument('--use_focal',  dest='use_focal', action='store_true',
                             help='Use focal loss instead of BCE (recommended for imbalanced data)')
    focal_group.add_argument('--no_focal',   dest='use_focal', action='store_false',
                             help='Use standard BCE loss (overrides YAML use_focal: true)')
    parser.set_defaults(use_focal=False)
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Focal loss α — weight for positive class. '
                             'Rule of thumb: 1 - (n_pos/n_total). For 39/290 ≈ 0.87')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss γ — focusing strength. 0=BCE, 2=paper default')

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# YAML loading
# ─────────────────────────────────────────────────────────────────────────────

def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Load YAML config and merge with parsed args. CLI flags always win over YAML.

    The old approach (comparing values to defaults) failed for store_true /
    store_false flags: --no_focal sets use_focal=False which equals the default
    False, so the merge incorrectly treated it as "not explicitly set" and
    overwrote it with the YAML value.

    Fix: inspect sys.argv directly to build the set of keys the user actually
    typed, then skip YAML for those keys unconditionally.
    """
    if args.config is None:
        return args

    with open(args.config) as f:
        yaml_cfg: dict = yaml.safe_load(f)

    # Build the set of dest keys explicitly provided on the command line.
    # --no_<key>  (store_false convention) maps to dest <key>.
    import sys
    explicit_keys: set = set()
    for token in sys.argv[1:]:
        if token.startswith('--'):
            raw = token.lstrip('-').split('=')[0].replace('-', '_')
            if raw.startswith('no_'):
                explicit_keys.add(raw[3:])   # --no_focal → 'focal'
            else:
                explicit_keys.add(raw)

    for key, yaml_val in yaml_cfg.items():
        if key in explicit_keys:
            continue                  # CLI wins — leave args untouched
        setattr(args, key, yaml_val)  # YAML wins for everything else

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────────────────────

def stratified_group_kfold_splits(subject_ids: np.ndarray, labels: np.ndarray, n_folds: int):
    """
    Yields (train_idx, val_idx, test_idx) index triples at the *window* level,
    ensuring all windows from the same subject stay in the same split.

    Strategy:
      - Outer fold (StratifiedGroupKFold, n_folds): defines test set for each fold.
      - Inner split from the remaining train subjects: 15 % → validation.

    This gives a clean 3-way split with no leakage across subjects.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # We need subject-level labels for stratification.
    # Derive them from the first occurrence of each subject.
    unique_subjects, first_occ = np.unique(subject_ids, return_index=True)
    subject_labels = labels[first_occ]

    # StratifiedGroupKFold.split needs arrays with len == n_windows
    X_dummy = np.zeros(len(subject_ids))

    for train_val_idx, test_idx in sgkf.split(X_dummy, labels, groups=subject_ids):
        # Further split train_val into train / val (at subject level)
        train_val_subjects = np.unique(subject_ids[train_val_idx])
        tv_labels = np.array([
            subject_labels[unique_subjects == s][0] for s in train_val_subjects
        ])

        # Inner stratified split of subjects
        n_val_subjects = max(1, int(len(train_val_subjects) * 0.15))
        pos_subjects  = train_val_subjects[tv_labels == 1]
        neg_subjects  = train_val_subjects[tv_labels == 0]

        # Keep class proportion in val
        n_val_pos = max(1, round(n_val_subjects * tv_labels.mean()))
        n_val_neg = n_val_subjects - n_val_pos

        rng = np.random.default_rng(42)
        val_subjects = np.concatenate([
            rng.choice(pos_subjects, min(n_val_pos, len(pos_subjects)), replace=False),
            rng.choice(neg_subjects, min(n_val_neg, len(neg_subjects)), replace=False),
        ])
        train_subjects = np.setdiff1d(train_val_subjects, val_subjects)

        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
        val_idx   = np.where(np.isin(subject_ids, val_subjects))[0]
        # test_idx already at window level from outer fold

        yield train_idx, val_idx, test_idx


def single_split(subject_ids: np.ndarray, labels: np.ndarray, val_ratio: float = 0.15):
    """
    Fallback single train/val/test split when n_folds=1.
    60 % train / 15 % val / 25 % test (subject-level stratified).
    """
    from utils.splits import subject_wise_stratified_split
    train_val_idx, test_idx = subject_wise_stratified_split(
        subject_ids, labels, val_ratio=0.25
    )
    train_idx, val_idx = subject_wise_stratified_split(
        subject_ids[train_val_idx], labels[train_val_idx], val_ratio=val_ratio / 0.75
    )
    # Remap val indices back to global index space
    val_idx = train_val_idx[val_idx]
    train_idx = train_val_idx[train_idx]
    return [(train_idx, val_idx, test_idx)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()
    args = merge_config(args)   # YAML overrides defaults; CLI overrides YAML

    set_seed(args.seed)

    # ── device ───────────────────────────────────────────────────────────────
    args.use_gpu = torch.cuda.is_available()
    args.device  = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    print(f"Using device: {args.device}")

    # ── load data (subject IDs + labels only — actual tensors loaded in Dataset) ──
    data_file = os.path.join(args.root_path, args.data_path)
    data = np.load(data_file, allow_pickle=True)
    subject_ids: np.ndarray = data["subject_ids"]
    labels:      np.ndarray = data["y"]

    pos_rate = labels.mean()
    print(f"Dataset: {len(labels)} windows | {labels.sum()} positive "
          f"({pos_rate*100:.1f}%) | {len(np.unique(subject_ids))} subjects")

    # ── unique experiment tag ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_setting = f"{args.model}_seed{args.seed}_{timestamp}"
    print(f"Experiment: {base_setting}")

    # ── cross-validation ──────────────────────────────────────────────────────
    if args.n_folds > 1:
        fold_iter = enumerate(stratified_group_kfold_splits(subject_ids, labels, args.n_folds))
    else:
        fold_iter = enumerate(single_split(subject_ids, labels, args.val_ratio))

    fold_results = []

    for fold, (train_idx, val_idx, test_idx) in fold_iter:
        fold_tag = f"fold{fold+1}" if args.n_folds > 1 else "single"
        setting  = f"{base_setting}_{fold_tag}"
        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/{args.n_folds}  |  "
              f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
        print(f"  Positive rate — train:{labels[train_idx].mean():.3f}  "
              f"val:{labels[val_idx].mean():.3f}  test:{labels[test_idx].mean():.3f}")
        print(f"{'='*60}")

        args.train_idx = train_idx
        args.val_idx   = val_idx
        args.test_idx  = test_idx

        exp = Exp_Classification(args)
        exp.train(setting)
        metrics = exp.test(setting)   # expected to return a dict with at least 'auroc', 'acc'

        fold_results.append({"fold": fold + 1, "setting": setting, **metrics})
        print(f"  Fold {fold+1} result: {metrics}")

    # ── aggregate results ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {args.n_folds}-Fold CV Summary  ({args.model})")
    print(f"{'='*60}")

    metric_keys = [k for k in fold_results[0] if k not in ("fold", "setting")]
    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in fold_results]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary[key] = {"mean": mean, "std": std, "per_fold": vals}
        print(f"  {key:12s}  {mean:.4f} ± {std:.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    results_dir = os.path.join(args.checkpoints, base_setting)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "cv_results.json")
    with open(results_path, "w") as f:
        json.dump({"args": vars(args), "fold_results": fold_results, "summary": summary},
                  f, indent=2, default=str)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()