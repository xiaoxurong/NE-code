"""
Subject-level evaluation of out-of-fold predictions, with confidence intervals.

For each run (a model or a baseline) this:
  1. loads every oof_predictions.csv under the given directory (several repeats may come
     from separate cluster jobs),
  2. averages window probabilities to one probability per subject within each repeat,
  3. computes AUROC and AUPRC over all subjects pooled across folds, per repeat,
  4. reports the mean over repeats with a 95% stratified bootstrap interval over subjects,
  5. optionally reports the paired difference in AUROC against a reference run
     (e.g. a shortcut baseline), on the same bootstrap samples.

Usage
  python scripts/evaluate.py --data dataset/<window file>.npz \
      --runs cnn_lstm=results/<stem>/cnn_lstm \
             missingness=results/<stem>/baselines/missingness \
      --reference missingness --subset all vaginal
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties handled by mid-ranks."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_run(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "**", "oof_predictions.csv"), recursive=True))
    if os.path.isfile(path):
        files = [path]
    if not files:
        sys.exit(f"No oof_predictions.csv found under {path}")
    df = pd.concat([pd.read_csv(f, dtype={"subject_id": str}) for f in files], ignore_index=True)
    dup = df.duplicated(["repeat", "window_index"])
    if dup.any():
        sys.exit(f"{path}: {int(dup.sum())} windows predicted twice within a repeat; "
                 f"are two jobs writing the same repeat?")
    return df


def subject_matrix(df: pd.DataFrame, subjects: pd.Index) -> np.ndarray:
    """(n_repeats, n_subjects) subject-level probabilities, NaN where a subject is missing."""
    per = df.groupby(["repeat", "subject_id"])["prob"].mean().unstack("subject_id")
    return per.reindex(columns=subjects).to_numpy(dtype=float)


def per_repeat(metric, y, P):
    vals = []
    for row in P:
        ok = np.isfinite(row)
        vals.append(metric(y[ok], row[ok]))
    return np.array(vals)


def bootstrap_indices(y: np.ndarray, n_boot: int, seed: int):
    """Stratified bootstrap: resample positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def summarize(y, P, n_boot, seed, P_ref=None):
    out = {}
    au = per_repeat(auroc, y, P)
    ap = per_repeat(average_precision_score, y, P)
    out["auroc"], out["auroc_sd_repeats"] = np.nanmean(au), np.nanstd(au)
    out["auprc"] = np.nanmean(ap)

    boots_au, boots_ap, boots_d = [], [], []
    for idx in bootstrap_indices(y, n_boot, seed):
        yb, Pb = y[idx], P[:, idx]
        a = np.nanmean(per_repeat(auroc, yb, Pb))
        boots_au.append(a)
        boots_ap.append(np.nanmean(per_repeat(average_precision_score, yb, Pb)))
        if P_ref is not None:
            boots_d.append(a - np.nanmean(per_repeat(auroc, yb, P_ref[:, idx])))
    out["auroc_ci"] = np.nanpercentile(boots_au, [2.5, 97.5])
    out["auprc_ci"] = np.nanpercentile(boots_ap, [2.5, 97.5])
    if P_ref is not None:
        ref = np.nanmean(per_repeat(auroc, y, P_ref))
        out["delta"] = out["auroc"] - ref
        out["delta_ci"] = np.nanpercentile(boots_d, [2.5, 97.5])
        out["p_not_better"] = float(np.mean(np.array(boots_d) <= 0))
    return out


def fold_assignment(df: pd.DataFrame) -> pd.Series:
    return df.set_index(["repeat", "window_index"])["fold"].sort_index()


def warn_on_different_folds(runs: dict):
    """Paired comparisons are tightest when every run used the same folds; say so if not."""
    names = [n for n, df in runs.items() if "fold" in df]
    if len(names) < 2:
        return
    ref = fold_assignment(runs[names[0]])
    for n in names[1:]:
        other = fold_assignment(runs[n])
        common = ref.index.intersection(other.index)
        if len(common) and not (ref.loc[common] == other.loc[common]).all():
            print(f"WARNING: {n} and {names[0]} were run on different cross-validation folds "
                  f"(different --fold_seed, or scikit-learn < 1.8 vs >= 1.8). Each is still a "
                  f"valid out-of-fold estimate, but their paired comparison is noisier.",
                  file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Window file the runs were trained on")
    ap.add_argument("--runs", nargs="+", required=True, metavar="NAME=PATH",
                    help="Run name and a directory searched for oof_predictions.csv")
    ap.add_argument("--reference", default=None, help="Run name to compare every other run against")
    ap.add_argument("--subset", nargs="+", default=["all"], choices=["all", "vaginal"],
                    help="'vaginal' restricts to vaginal deliveries (delivery_type == 1)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional CSV for the results table")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    win = pd.DataFrame({"subject_id": d["subject_ids"].astype(str), "y": d["y"]})
    if "delivery_type" in d.files:
        win["delivery_type"] = d["delivery_type"]
    subj = win.groupby("subject_id").first()

    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        df = load_run(path)
        mismatch = (df["y"].to_numpy() != d["y"][df["window_index"].to_numpy()]).sum()
        if mismatch:
            sys.exit(f"{name}: labels in the predictions do not match {args.data}; "
                     f"was it trained on a different window file?")
        runs[name] = df
    if args.reference and args.reference not in runs:
        sys.exit(f"--reference {args.reference} is not one of the runs")
    warn_on_different_folds(runs)

    rows = []
    for subset in args.subset:
        if subset == "vaginal":
            if "delivery_type" not in subj:
                sys.exit("This window file has no delivery_type; rebuild it with prepare_ctu_uhb.py")
            keep = subj.index[subj["delivery_type"] == 1]
        else:
            keep = subj.index
        y = subj.loc[keep, "y"].to_numpy().astype(int)
        mats = {n: subject_matrix(df, keep) for n, df in runs.items()}
        P_ref = mats.get(args.reference)

        for name, P in mats.items():
            covered = np.isfinite(P).all(axis=1).mean()
            res = summarize(y, P, args.n_boot, args.seed,
                            None if name == args.reference or P_ref is None else P_ref)
            row = {
                "subset": subset, "run": name, "subjects": len(y), "positives": int(y.sum()),
                "repeats": P.shape[0], "full_coverage": f"{covered:.0%}",
                "AUROC": f"{res['auroc']:.3f} [{res['auroc_ci'][0]:.3f}, {res['auroc_ci'][1]:.3f}]",
                "AUROC sd (repeats)": f"{res['auroc_sd_repeats']:.3f}",
                "AUPRC": f"{res['auprc']:.3f} [{res['auprc_ci'][0]:.3f}, {res['auprc_ci'][1]:.3f}]",
                "prevalence": f"{y.mean():.3f}",
            }
            if "delta" in res:
                row[f"dAUROC vs {args.reference}"] = (
                    f"{res['delta']:+.3f} [{res['delta_ci'][0]:+.3f}, {res['delta_ci'][1]:+.3f}]")
                row["P(not better)"] = f"{res['p_not_better']:.3f}"
            rows.append(row)

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.to_string(index=False))
    print("\nAUROC/AUPRC: mean over repeats of the subject-level metric pooled across folds; "
          "[95% stratified bootstrap over subjects]. AUPRC baseline = prevalence.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
