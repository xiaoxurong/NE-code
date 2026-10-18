"""
Tests for the repeated cross-validation splits, the shortcut baselines and the
subject-level evaluation. These do not need PyTorch.

Run: python -m pytest tests -q
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from utils.splits import repeated_cv_splits, stratified_group_kfold_splits  # noqa: E402
from evaluate import auroc  # noqa: E402


def toy_cohort(n_subjects=120, n_pos=15, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, n_subjects)
    sid = np.repeat([f"S{i:03d}" for i in range(n_subjects)], counts)
    y = np.array([int(int(s[1:]) < n_pos) for s in sid])
    return sid, y


def test_splits_keep_subjects_apart_and_cover_everyone():
    sid, y = toy_cohort()
    seen_test = []
    for tr, va, te in stratified_group_kfold_splits(sid, y, 5, seed=7):
        a, b, c = set(sid[tr]), set(sid[va]), set(sid[te])
        assert not (a & b or a & c or b & c)
        assert len(tr) + len(va) + len(te) == len(y)
        assert y[te].sum() > 0 and y[va].sum() > 0
        seen_test += list(te)
    assert sorted(seen_test) == list(range(len(y)))


def test_repeats_are_reproducible_and_differ():
    sid, y = toy_cohort()
    a = [te for *_, te in repeated_cv_splits(sid, y, 5, n_repeats=2, fold_seed=42)]
    b = [te for *_, te in repeated_cv_splits(sid, y, 5, n_repeats=2, fold_seed=42)]
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert not all(np.array_equal(p, q) for p, q in zip(a[:5], a[5:]))
    only = [te for *_, te in repeated_cv_splits(sid, y, 5, n_repeats=2, fold_seed=42, only_repeat=1)]
    assert all(np.array_equal(p, q) for p, q in zip(only, a[5:]))


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 1)  # many ties
    assert auroc(y, p) == pytest.approx(roc_auc_score(y, p))


@pytest.fixture
def window_file(tmp_path):
    """A window file where missingness predicts the label and the 'model' is perfect."""
    rng = np.random.default_rng(0)
    sid, y = toy_cohort(n_subjects=80, n_pos=20, seed=1)
    n, T = len(y), 40
    mask = np.ones((n, 2, T), dtype=bool)
    for i in range(n):  # positives lose more signal
        k = rng.integers(15, 30) if y[i] else rng.integers(0, 15)
        mask[i, :, :k] = False
    rec_missing = 1 - mask[:, 0].mean(1)
    delivery = np.where(rng.random(n) < 0.2, 2.0, 1.0)
    path = tmp_path / "w.npz"
    np.savez_compressed(path, X=np.zeros((n, 4, T), np.float32), y=y, subject_ids=sid, mask=mask,
                        rec_fhr_missing_frac=rec_missing.astype(np.float32),
                        recording_min=np.full(n, 70, np.float32), delivery_type=delivery)
    return path, sid, y


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, check=True)


def test_baselines_and_evaluate_end_to_end(window_file, tmp_path):
    path, sid, y = window_file
    out = tmp_path / "baselines"
    run("scripts/baselines.py", "--data", str(path), "--n_repeats", "2", "--out_dir", str(out))
    oof = pd.read_csv(out / "missingness" / "oof_predictions.csv", dtype={"subject_id": str})
    assert len(oof) == 2 * len(y)                       # every window once per repeat
    assert not oof.duplicated(["repeat", "window_index"]).any()

    # a "model" that knows the answer, written as two separate repeat jobs
    for r in (0, 1):
        d = tmp_path / "perfect" / f"r{r}"
        d.mkdir(parents=True)
        pd.DataFrame({"repeat": r, "fold": 1, "window_index": np.arange(len(y)),
                      "subject_id": sid, "y": y, "prob": y * 0.9 + 0.05}).to_csv(
            d / "oof_predictions.csv", index=False)

    table = tmp_path / "table.csv"
    res = run("scripts/evaluate.py", "--data", str(path),
              "--runs", f"perfect={tmp_path / 'perfect'}", f"missingness={out / 'missingness'}",
              "--reference", "missingness", "--subset", "all", "vaginal",
              "--n_boot", "200", "--out", str(table))
    t = pd.read_csv(table).set_index(["subset", "run"])
    assert t.loc[("all", "perfect"), "AUROC"].startswith("1.000")
    assert t.loc[("all", "perfect"), "repeats"] == 2
    missing_auc = float(t.loc[("all", "missingness"), "AUROC"].split()[0])
    assert missing_auc > 0.8                             # the planted shortcut is found
    assert t.loc[("vaginal", "perfect"), "subjects"] < t.loc[("all", "perfect"), "subjects"]
    assert "dAUROC vs missingness" in t.columns
    assert "AUPRC baseline = prevalence" in res.stdout


def test_evaluate_rejects_predictions_from_a_different_file(window_file, tmp_path):
    path, sid, y = window_file
    d = tmp_path / "wrong"
    d.mkdir()
    pd.DataFrame({"repeat": 0, "fold": 1, "window_index": np.arange(len(y)),
                  "subject_id": sid, "y": 1 - y, "prob": 0.5}).to_csv(d / "oof_predictions.csv", index=False)
    with pytest.raises(subprocess.CalledProcessError) as e:
        run("scripts/evaluate.py", "--data", str(path), "--runs", f"wrong={d}")
    assert "do not match" in e.value.stderr


def test_run_py_writes_out_of_fold_predictions(window_file, tmp_path):
    """One quick CPU training pass through run.py (skipped if PyTorch is not installed)."""
    pytest.importorskip("torch")
    path, sid, y = window_file
    res_dir = tmp_path / "res"
    run("run.py", "--config", "configs/ctu_uhb_cnn_lstm.yaml",
        "--root_path", str(path.parent) + "/", "--data_path", path.name,
        "--seq_len", "40", "--in_channels", "4", "--enc_in", "4",
        "--train_epochs", "1", "--patience", "1", "--n_folds", "3", "--n_repeats", "1",
        "--num_workers", "0", "--batch_size", "16",
        "--checkpoints", str(tmp_path / "ckpt") + "/", "--results_dir", str(res_dir))
    oof = pd.read_csv(res_dir / "oof_predictions.csv", dtype={"subject_id": str})
    assert sorted(oof["window_index"]) == list(range(len(y)))
    assert oof["prob"].between(0, 1).all()
    assert not list((tmp_path / "ckpt").rglob("*.pth"))   # weights cleaned up by default
