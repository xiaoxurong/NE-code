"""
Shortcut baselines: how well can the label be predicted WITHOUT reading the heart rate
pattern?

Each baseline is a logistic regression on properties of the recording that are not
fetal physiology (how much signal is missing, how long the recording is). It is trained
on exactly the same repeated cross-validation folds as run.py and writes out-of-fold
predictions in the same format, so scripts/evaluate.py can put it in the same table.

A deep model only shows that it reads the trace if it beats these baselines.

Usage
  python scripts/baselines.py --data dataset/<window file>.npz --n_repeats 5 \
      --out_dir results/<window file stem>/baselines
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.splits import repeated_cv_splits  # noqa: E402

FEATURE_SETS = {
    # signal loss only
    "missingness": ["window_fhr_missing", "window_uc_missing", "rec_fhr_missing_frac"],
    # signal loss and recording length
    "missingness_length": ["window_fhr_missing", "window_uc_missing", "rec_fhr_missing_frac",
                           "recording_min"],
}


def window_features(d) -> dict:
    mask = d["mask"].astype(bool)
    return {
        "window_fhr_missing": 1.0 - mask[:, 0].mean(axis=1),
        "window_uc_missing": 1.0 - mask[:, 1].mean(axis=1),
        "rec_fhr_missing_frac": d["rec_fhr_missing_frac"],
        "recording_min": d["recording_min"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Window file written by prepare_ctu_uhb.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=1)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    y, sid = d["y"], d["subject_ids"]
    feats = window_features(d)

    for name, cols in FEATURE_SETS.items():
        X = np.column_stack([feats[c] for c in cols]).astype(float)
        rows = []
        for r, f, tr, va, te in repeated_cv_splits(sid, y, args.n_folds, args.n_repeats,
                                                   args.fold_seed, args.val_ratio):
            fit_idx = np.concatenate([tr, va])  # no early stopping, so validation is training data
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=1000))
            model.fit(X[fit_idx], y[fit_idx])
            prob = model.predict_proba(X[te])[:, 1]
            rows += [(r, f + 1, int(i), str(sid[i]), int(y[i]), float(p)) for i, p in zip(te, prob)]

        out = os.path.join(args.out_dir, name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "oof_predictions.csv"), "w") as fh:
            fh.write("repeat,fold,window_index,subject_id,y,prob\n")
            for row in rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        with open(os.path.join(out, "baseline.json"), "w") as fh:
            json.dump({"features": cols, **vars(args)}, fh, indent=2)
        print(f"{name:20s} features={cols} -> {out}/oof_predictions.csv")


if __name__ == "__main__":
    main()
