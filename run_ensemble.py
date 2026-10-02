"""
run_ensemble.py — Ensemble of multiple models over the same CV folds.
=====================================================================
Trains each member model on identical fold splits, averages their predicted
probabilities, then evaluates the ensemble. Threshold is tuned on the
averaged val-set probabilities (not individual models).

Usage:
    python run_ensemble.py \
        --configs configs/cnn_lstm.yaml configs/rocket.yaml \
        --weights 0.6 0.4

    # Equal weights (default):
    python run_ensemble.py \
        --configs configs/cnn_lstm.yaml configs/rocket.yaml
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from exp.exp_classification import Exp_Classification
from utils.metrics import binary_classification_metrics, find_optimal_threshold
from run import stratified_group_kfold_splits, single_split, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_args_from_yaml(config_path: str) -> argparse.Namespace:
    """Load a YAML file into an argparse.Namespace with sensible defaults."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)

    # GPU
    args.use_gpu       = torch.cuda.is_available()
    args.gpu           = getattr(args, 'gpu', 0)
    args.device        = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    args.use_multi_gpu = getattr(args, 'use_multi_gpu', False)
    args.device_ids    = [0]

    # Defaults that might be missing from YAML
    args.data          = getattr(args, 'data',        'CTG')
    args.num_workers   = getattr(args, 'num_workers', 4)
    args.checkpoints   = getattr(args, 'checkpoints', './checkpoints/')
    args.lradj         = getattr(args, 'lradj',       'type1')
    args.use_norm      = getattr(args, 'use_norm',    1)
    args.moving_avg    = getattr(args, 'moving_avg',  13)
    args.enc_in        = getattr(args, 'enc_in', getattr(args, 'in_channels', 2))
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--configs', nargs='+', required=True,
                        help='YAML config files for each ensemble member, in order.')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Probability averaging weights (one per config). '
                             'Automatically normalised. Defaults to equal weights.')
    cli = parser.parse_args()

    # ── Weights ──────────────────────────────────────────────────────────────
    n_members = len(cli.configs)
    weights   = cli.weights or [1.0] * n_members
    assert len(weights) == n_members, "--weights must have one value per --configs entry"
    total   = sum(weights)
    weights = [w / total for w in weights]

    print(f"Ensemble members:")
    for path, w in zip(cli.configs, weights):
        print(f"  {os.path.basename(path):40s}  weight={w:.3f}")

    # ── Base settings (data / CV) taken from first config ────────────────────
    base_args = load_args_from_yaml(cli.configs[0])
    set_seed(getattr(base_args, 'seed', 42))
    print(f"\nUsing device: {base_args.device}")

    # ── Load subject IDs and labels for fold splitting ────────────────────────
    data_file   = os.path.join(base_args.root_path, base_args.data_path)
    data        = np.load(data_file, allow_pickle=True)
    subject_ids = data['subject_ids']
    labels      = data['y']

    pos_rate = labels.mean()
    print(f"Dataset: {len(labels)} windows | {labels.sum()} positive "
          f"({pos_rate*100:.1f}%) | {len(np.unique(subject_ids))} subjects")

    # ── Fold iterator ─────────────────────────────────────────────────────────
    n_folds  = getattr(base_args, 'n_folds', 5)
    val_ratio = getattr(base_args, 'val_ratio', 0.15)
    if n_folds > 1:
        fold_iter = list(enumerate(
            stratified_group_kfold_splits(subject_ids, labels, n_folds)
        ))
    else:
        fold_iter = list(enumerate(
            single_split(subject_ids, labels, val_ratio)
        ))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_tag  = f"Ensemble_seed{getattr(base_args, 'seed', 42)}_{timestamp}"

    fold_results = []

    for fold, (train_idx, val_idx, test_idx) in fold_iter:
        fold_tag = f"fold{fold+1}" if n_folds > 1 else "single"
        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/{n_folds}  |  "
              f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
        print(f"{'='*60}")

        val_probs_per_model  = []
        test_probs_per_model = []
        val_trues  = None
        test_trues = None

        # ── Train each member on this fold ────────────────────────────────────
        for config_path, weight in zip(cli.configs, weights):
            model_args = load_args_from_yaml(config_path)

            # Inject shared fold indices and data settings
            model_args.train_idx   = train_idx
            model_args.val_idx     = val_idx
            model_args.test_idx    = test_idx
            model_args.root_path   = base_args.root_path
            model_args.data_path   = base_args.data_path
            model_args.data        = base_args.data
            model_args.seq_len     = base_args.seq_len
            model_args.num_classes = base_args.num_classes
            model_args.n_folds     = n_folds

            setting = f"{base_tag}_{fold_tag}_{model_args.model}"
            print(f"\n  [{model_args.model}] training ...")

            exp       = Exp_Classification(model_args)
            exp.train(setting)
            criterion = exp._select_criterion()

            # Val probabilities (for threshold tuning on averaged output)
            _, _, vp, vt = exp.validate(exp._get_data('val')[1],  criterion)
            # Test probabilities
            _, _, tp, tt = exp.validate(exp._get_data('test')[1], criterion)

            val_probs_per_model.append(vp)
            test_probs_per_model.append(tp)
            val_trues  = vt
            test_trues = tt

            indiv = binary_classification_metrics(tt, tp, threshold=0.5)
            print(f"  [{model_args.model}] AUROC={indiv['auroc']:.4f}  "
                  f"Sens={indiv['sensitivity']:.4f}  Spec={indiv['specificity']:.4f}")

        # ── Weighted average ──────────────────────────────────────────────────
        avg_val_probs  = sum(w * p for w, p in zip(weights, val_probs_per_model))
        avg_test_probs = sum(w * p for w, p in zip(weights, test_probs_per_model))

        # ── Threshold tuning on averaged val predictions ──────────────────────
        threshold = find_optimal_threshold(val_trues, avg_val_probs)
        print(f"\n  Ensemble optimal threshold (val): {threshold:.3f}")

        # ── Evaluate ensemble on test set ─────────────────────────────────────
        metrics = binary_classification_metrics(test_trues, avg_test_probs,
                                                threshold=threshold)
        fold_results.append({'fold': fold + 1, **metrics})
        print(f"  Ensemble AUROC={metrics['auroc']:.4f}  "
              f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
              f"F1={metrics['f1']:.4f}")

    # ── Aggregate across folds ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Ensemble {n_folds}-Fold CV Summary")
    print(f"{'='*60}")

    metric_keys = [k for k in fold_results[0] if k not in ('fold',)]
    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in fold_results]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        summary[key] = {'mean': mean, 'std': std, 'per_fold': vals}
        print(f"  {key:15s}  {mean:.4f} ± {std:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir  = os.path.join(base_args.checkpoints, base_tag)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'ensemble_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'configs':      cli.configs,
            'weights':      weights,
            'fold_results': fold_results,
            'summary':      summary,
        }, f, indent=2, default=str)
    print(f"\nResults saved → {results_path}")


if __name__ == '__main__':
    main()