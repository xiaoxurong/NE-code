"""
Linear probe on frozen embeddings, on exactly the same repeated cross-validation folds as
run.py and baselines.py.

For every outer fold, a standardized L2 logistic regression is fit on the training
recordings. Its regularization strength C is chosen by an inner cross-validation (AUROC)
that only ever sees those training recordings, so nothing from the test fold influences
the model. Out-of-fold predictions are written in the format scripts/evaluate.py reads.

Feature sets
  --pool mean        mean of the chunk embeddings
  --pool meanmax     mean and max of the chunk embeddings, concatenated
  --channels fhr     FHR channel only
  --channels fhr_uc  FHR and UC
  --add_shortcuts    also include the shortcut-baseline features (signal loss, recording
                     length), to test whether the embedding adds information beyond them.
                     The embedding is then reduced to --pca components (fit on the training
                     fold only), so a handful of shortcut features is not drowned out by
                     ~1,000 embedding dimensions under one penalty.

Usage
  python scripts/probe.py --data dataset/<window file>.npz \
      --embeddings embeddings/<window stem>__moment-base.npz \
      --pool meanmax --channels fhr_uc --n_repeats 5 \
      --out_dir results/<window file stem>/probe_moment-base_meanmax
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from utils.splits import repeated_cv_splits  # noqa: E402
from baselines import FEATURE_SETS, window_features  # noqa: E402

DEFAULT_CS = np.logspace(-4, 1, 11)


def load_features(data, emb, pool: str, channels: str, add_shortcuts: bool):
    """Embedding features aligned to the windows of the data file, and the number of
    shortcut columns appended at the end (0 if none)."""
    key_d = [f"{s}|{w}" for s, w in zip(data["subject_ids"], data["window_index"])]
    key_e = {f"{s}|{w}": i for i, (s, w) in enumerate(zip(emb["subject_ids"], emb["window_index"]))}
    missing = [k for k in key_d if k not in key_e]
    if missing:
        sys.exit(f"{len(missing)} windows of the data file have no embedding "
                 f"(e.g. {missing[:3]}); extract embeddings from a file with the same windows")
    order = np.array([key_e[k] for k in key_d])

    ch = {"fhr": [0], "fhr_uc": [0, 1]}[channels]
    parts = [emb["emb_mean"][order][:, ch].reshape(len(order), -1)]
    if pool == "meanmax":
        parts.append(emb["emb_max"][order][:, ch].reshape(len(order), -1))
    n_short = 0
    if add_shortcuts:
        f = window_features(data)
        cols = FEATURE_SETS["missingness_length"]
        parts.append(np.column_stack([f[c] for c in cols]))
        n_short = len(cols)
    return np.concatenate(parts, axis=1).astype(np.float64), n_short


def make_model(C: float, n_features: int, n_short: int, pca: int):
    """Standardize; with shortcut columns, PCA-reduce only the embedding block."""
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    if n_short == 0 or pca <= 0:
        return make_pipeline(StandardScaler(), lr)
    n_emb = n_features - n_short
    pre = ColumnTransformer([
        ("emb", make_pipeline(StandardScaler(), PCA(n_components=min(pca, n_emb))), list(range(n_emb))),
        ("short", StandardScaler(), list(range(n_emb, n_features))),
    ])
    return make_pipeline(pre, lr)


def fit_with_inner_cv(X, y, groups, Cs, inner_folds, seed, n_short=0, pca=0):
    """Choose C by inner group-aware CV on the training recordings only, then refit."""
    inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = list(inner.split(X, y, groups))
    scores = []
    for C in Cs:
        aucs = []
        for tr, va in splits:
            if len(np.unique(y[va])) < 2:
                continue
            m = make_model(C, X.shape[1], n_short, pca)
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        scores.append(np.mean(aucs) if aucs else -np.inf)
    best = float(Cs[int(np.argmax(scores))])
    model = make_model(best, X.shape[1], n_short, pca)
    model.fit(X, y)
    return model, best, float(np.max(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Window file (provides labels and folds)")
    ap.add_argument("--embeddings", required=True, help="Output of extract_embeddings.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pool", choices=["mean", "meanmax"], default="meanmax")
    ap.add_argument("--channels", choices=["fhr", "fhr_uc"], default="fhr_uc")
    ap.add_argument("--add_shortcuts", action="store_true")
    ap.add_argument("--pca", type=int, default=32,
                    help="Embedding PCA components when --add_shortcuts is set (0 = no PCA)")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=1)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    e = np.load(args.embeddings, allow_pickle=False)
    X, n_short = load_features(d, e, args.pool, args.channels, args.add_shortcuts)
    y, sid = d["y"], d["subject_ids"]
    print(f"features {X.shape} | positives {int(y.sum())} | model {str(e['model'])}")

    rows, chosen = [], []
    for r, f, tr, va, te in repeated_cv_splits(sid, y, args.n_folds, args.n_repeats,
                                               args.fold_seed, args.val_ratio):
        fit_idx = np.concatenate([tr, va])  # no early stopping, so validation is training data
        model, C, inner_auc = fit_with_inner_cv(X[fit_idx], y[fit_idx], sid[fit_idx], DEFAULT_CS,
                                                args.inner_folds, seed=args.fold_seed + r,
                                                n_short=n_short, pca=args.pca)
        prob = model.predict_proba(X[te])[:, 1]
        rows += [(r, f + 1, int(i), str(sid[i]), int(y[i]), float(p)) for i, p in zip(te, prob)]
        chosen.append({"repeat": r, "fold": f + 1, "C": C, "inner_auroc": inner_auc})
        print(f"  repeat {r} fold {f + 1}: C={C:.2g} inner AUROC={inner_auc:.3f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "oof_predictions.csv"), "w") as fh:
        fh.write("repeat,fold,window_index,subject_id,y,prob\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    with open(os.path.join(args.out_dir, "probe.json"), "w") as fh:
        json.dump({**vars(args), "model": str(e["model"]), "n_features": int(X.shape[1]),
                   "folds": chosen}, fh, indent=2)
    print(f"Wrote {args.out_dir}/oof_predictions.csv")


if __name__ == "__main__":
    main()
