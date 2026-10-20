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


def test_cnn_lstm_learns_a_planted_signal_with_weighted_bce(tmp_path):
    """
    Regression test for the collapse seen with a fixed focal alpha: with class-weighted BCE
    and AUROC-based checkpoint selection, the CNN-LSTM must learn an obvious planted signal
    (positives have a raised, oscillating FHR) at a minority prevalence.
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n, T = 160, 240
    sid = np.array([f"S{i:03d}" for i in range(n)])
    y = (np.arange(n) < 32).astype(int)                      # 20% positive
    X = rng.standard_normal((n, 4, T)).astype(np.float32) * 0.5
    t = np.arange(T)
    X[y == 1, 0] += 1.0 + np.sin(t / 6.0)                    # the planted FHR pattern
    X[:, 2:] = 1.0                                           # mask channels: all observed
    path = tmp_path / "planted.npz"
    np.savez_compressed(path, X=X, y=y, subject_ids=sid, mask=np.ones((n, 2, T), bool),
                        rec_fhr_missing_frac=np.zeros(n, np.float32),
                        recording_min=np.full(n, 70, np.float32), delivery_type=np.ones(n))
    res_dir = tmp_path / "res"
    out = run("run.py", "--config", "configs/ctu_uhb_cnn_lstm.yaml",
              "--root_path", str(tmp_path) + "/", "--data_path", path.name,
              "--seq_len", str(T), "--in_channels", "4", "--enc_in", "4",
              "--train_epochs", "25", "--patience", "8", "--learning_rate", "1e-3",
              "--n_folds", "3", "--n_repeats", "1", "--num_workers", "0", "--batch_size", "16",
              "--checkpoints", str(tmp_path / "ckpt") + "/", "--results_dir", str(res_dir))
    assert "pos_weight=" in out.stdout and "pred sd" in out.stdout
    weights = [float(line.split("pos_weight=")[1].split()[0])
               for line in out.stdout.splitlines() if "pos_weight=" in line]
    assert all(3.0 < w < 5.0 for w in weights)              # about 128/32 per training fold
    oof = pd.read_csv(res_dir / "oof_predictions.csv")
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(oof["y"], oof["prob"]) > 0.9
    assert oof["prob"].std() > 0.01                          # not collapsed


def test_rank_average_is_label_free_and_checks_folds():
    from combine import rank_average
    rng = np.random.default_rng(0)
    n = 60
    base = pd.DataFrame({"repeat": np.repeat([0, 1], n), "fold": np.tile(np.arange(n) % 5 + 1, 2),
                         "window_index": np.tile(np.arange(n), 2),
                         "subject_id": np.tile([f"S{i}" for i in range(n)], 2),
                         "y": np.tile((np.arange(n) < 15).astype(int), 2)})
    a = base.assign(prob=rng.random(2 * n))
    b = base.assign(prob=rng.random(2 * n) * 0.01 + 0.5)          # different scale
    out = rank_average([a, b])
    # equal say: each run's percentile is in (0, 1], so the average is too
    assert out["prob"].between(0, 1).all()
    pa = a.groupby("repeat")["prob"].rank(pct=True)
    pb = b.groupby("repeat")["prob"].rank(pct=True)
    np.testing.assert_allclose(out.sort_values(["repeat", "window_index"])["prob"].to_numpy(),
                               ((pa + pb) / 2).to_numpy())
    # never reads labels: flipping them in every input leaves the scores unchanged
    flipped = rank_average([a.assign(y=1 - a["y"]), b.assign(y=1 - b["y"])])
    np.testing.assert_allclose(flipped["prob"], out["prob"])
    # refuses runs on different folds
    with pytest.raises(ValueError, match="different folds"):
        rank_average([a, b.assign(fold=b["fold"] % 5 + 1)])


def test_sklearn_version_guard(monkeypatch):
    import sklearn
    from utils import splits
    monkeypatch.setattr(sklearn, "__version__", "1.7.2")
    with pytest.raises(RuntimeError, match="scikit-learn>=1.8"):
        splits.check_sklearn_folds()
    monkeypatch.setattr(sklearn, "__version__", "1.9.1")
    splits.check_sklearn_folds()


def test_folds_are_pinned():
    """The fold assignment every result in this repo uses (scikit-learn >= 1.8, seed 42)."""
    sid, y = toy_cohort(n_subjects=50, n_pos=10, seed=3)
    te = [sorted(str(s) for s in set(sid[t]))
          for *_, t in stratified_group_kfold_splits(sid, y, 5, seed=42)]
    assert sorted(s for fold in te for s in fold) == sorted(set(sid))
    # if this changes, a library update has changed the folds: rerun everything or pin it
    assert te[0] == ["S002", "S005", "S010", "S016", "S023", "S024", "S028", "S030", "S041", "S049"]
