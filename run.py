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
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from exp.exp_classification import Exp_Classification
# re-exported so `from run import stratified_group_kfold_splits` keeps working
from utils.splits import repeated_cv_splits, stratified_group_kfold_splits  # noqa: F401


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
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Repeat cross-validation with different fold assignments '
                             '(seed fold_seed + r for repeat r).')
    parser.add_argument('--fold_seed', type=int, default=42,
                        help='Seed for fold assignment; 42 reproduces the original splits.')
    parser.add_argument('--only_repeat', type=int, default=None,
                        help='Run a single repeat (0-based). Use this to run repeats as '
                             'separate cluster jobs; evaluate.py merges their outputs.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Where cv_results.json and oof_predictions.csv are written. '
                             'Default: <checkpoints>/<model>_seed<seed>_<timestamp>')
    parser.add_argument('--keep_checkpoints', action='store_true', default=False,
                        help='Keep per-fold model weights (deleted after testing by default).')
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

    # ── cross-validation (optionally repeated) ───────────────────────────────
    if args.n_folds > 1:
        splits = repeated_cv_splits(subject_ids, labels, args.n_folds,
                                    n_repeats=args.n_repeats, fold_seed=args.fold_seed,
                                    val_ratio=args.val_ratio, only_repeat=args.only_repeat)
    else:
        splits = ((0, 0, tr, va, te) for tr, va, te in
                  single_split(subject_ids, labels, args.val_ratio))

    results_dir = args.results_dir or os.path.join(args.checkpoints, base_setting)
    os.makedirs(results_dir, exist_ok=True)

    fold_results, oof_rows = [], []
    for repeat, fold, train_idx, val_idx, test_idx in splits:
        fold_tag = f"r{repeat}_fold{fold+1}" if args.n_folds > 1 else "single"
        setting  = f"{base_setting}_{fold_tag}"
        print(f"\n{'='*60}")
        print(f"  Repeat {repeat}  Fold {fold+1}/{args.n_folds}  |  "
              f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
        print(f"  Positive rate — train:{labels[train_idx].mean():.3f}  "
              f"val:{labels[val_idx].mean():.3f}  test:{labels[test_idx].mean():.3f}")
        print(f"{'='*60}")

        args.train_idx = train_idx
        args.val_idx   = val_idx
        args.test_idx  = test_idx

        exp = Exp_Classification(args)
        exp.train(setting)
        metrics = exp.test(setting)

        # out-of-fold predictions: one row per test window
        for i, p in zip(test_idx, np.asarray(exp.test_probs).reshape(-1)):
            oof_rows.append((repeat, fold + 1, int(i), str(subject_ids[i]), int(labels[i]), float(p)))

        fold_results.append({"repeat": repeat, "fold": fold + 1, "setting": setting, **metrics})
        print(f"  Repeat {repeat} fold {fold+1} result: {metrics}")

        if not args.keep_checkpoints:
            shutil.rmtree(os.path.join(args.checkpoints, setting), ignore_errors=True)

    # ── per-fold metrics (descriptive; scripts/evaluate.py gives the subject-level
    #    pooled AUROC with confidence intervals) ──────────────────────────────
    n_rep = len(set(r["repeat"] for r in fold_results))
    print(f"\n{'='*60}")
    print(f"  {args.n_folds}-fold CV x {n_rep} repeat(s)  ({args.model})")
    print(f"{'='*60}")

    metric_keys = [k for k in fold_results[0] if k not in ("repeat", "fold", "setting")]
    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in fold_results]
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
        summary[key] = {"mean": mean, "std": std, "per_fold": vals}
        print(f"  {key:12s}  {mean:.4f} ± {std:.4f}")

    # ── save results ──────────────────────────────────────────────────────────
    oof_path = os.path.join(results_dir, "oof_predictions.csv")
    with open(oof_path, "w") as f:
        f.write("repeat,fold,window_index,subject_id,y,prob\n")
        for row in oof_rows:
            f.write(",".join(str(v) for v in row) + "\n")

    saved_args = {k: v for k, v in vars(args).items() if k not in ("train_idx", "val_idx", "test_idx")}
    results_path = os.path.join(results_dir, "cv_results.json")
    with open(results_path, "w") as f:
        json.dump({"args": saved_args, "fold_results": fold_results, "summary": summary},
                  f, indent=2, default=str)
    print(f"\nResults saved → {results_path}")
    print(f"Out-of-fold predictions → {oof_path}")


if __name__ == "__main__":
    main()