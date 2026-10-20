"""
Combine the out-of-fold predictions of two or more runs by averaging their ranks.

Within each repeat, every run's predictions are converted to percentile ranks (ties get
the mid-rank) and the percentiles are averaged with fixed weights (equal by default). The
result is written as a new run that scripts/evaluate.py reads like any other.

Why ranks: AUROC depends only on the ordering of scores, and two models' probabilities are
usually on different scales (a run whose probabilities spread over 0.2-0.8 would dominate
a plain average with one that stays within 0.45-0.55). Ranks give each run equal say.

What it does not do: it never reads the labels and fits nothing, so it cannot overfit the
test folds. Ranks are computed over the pooled out-of-fold predictions of a repeat, which
uses the distribution of test-set scores (but not their labels). To deploy, a new score
would be converted to a percentile against a reference set, e.g. training predictions.

Every run must come from the same window file and the same folds (same --fold_seed and
--n_repeats); this is checked.

Usage
  python scripts/combine.py \
      --runs results/<stem>/probe_mantis_meanmax results/<stem>/baselines/missingness_length \
      --out_dir results/<stem>/combo_mantis+shortcut
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_run  # noqa: E402

KEY = ["repeat", "window_index"]


def rank_average(runs: list, weights=None) -> pd.DataFrame:
    """runs: list of out-of-fold DataFrames (repeat, fold, window_index, subject_id, y, prob)."""
    weights = np.ones(len(runs)) if weights is None else np.asarray(weights, dtype=float)
    if len(weights) != len(runs) or (weights < 0).any() or weights.sum() == 0:
        raise ValueError("need one non-negative weight per run")
    weights = weights / weights.sum()

    base = runs[0].sort_values(KEY).reset_index(drop=True)
    score = np.zeros(len(base))
    for i, df in enumerate(runs):
        df = df.sort_values(KEY).reset_index(drop=True)
        if len(df) != len(base) or not (df[KEY].to_numpy() == base[KEY].to_numpy()).all():
            raise ValueError(f"run {i} does not cover the same windows and repeats as run 0")
        if not (df["y"].to_numpy() == base["y"].to_numpy()).all():
            raise ValueError(f"run {i} has different labels; was it built from another window file?")
        if "fold" in df and not (df["fold"].to_numpy() == base["fold"].to_numpy()).all():
            raise ValueError(f"run {i} used different folds; use the same --fold_seed and --n_repeats")
        pct = df.groupby("repeat")["prob"].rank(method="average", pct=True).to_numpy()
        score += weights[i] * pct
    out = base[["repeat", "fold", "window_index", "subject_id", "y"]].copy()
    out["prob"] = score
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories (or csv files)")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Fixed weights, one per run (default: equal). Never tune these on the "
                         "same predictions you report.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if len(args.runs) < 2:
        sys.exit("give at least two runs")

    try:
        out = rank_average([load_run(p) for p in args.runs], args.weights)
    except ValueError as e:
        sys.exit(str(e))
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(os.path.join(args.out_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "combine.json"), "w") as fh:
        json.dump({"method": "rank_average", "runs": args.runs,
                   "weights": args.weights or [1.0] * len(args.runs)}, fh, indent=2)
    print(f"Wrote {args.out_dir}/oof_predictions.csv ({out['repeat'].nunique()} repeats, "
          f"{len(out)} rows)")


if __name__ == "__main__":
    main()
