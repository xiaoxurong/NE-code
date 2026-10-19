"""
Tests for scripts/extract_embeddings.py and scripts/probe.py.

The pretrained weights are not downloaded here: the pipeline is tested with stub encoders,
and the real MOMENT and Mantis code paths with small randomly initialized networks
(skipped if momentfm / mantis-tsfm are not installed).
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import extract_embeddings as ex  # noqa: E402


class StubEncoder:
    """Embeds a chunk as [mean of observed values, observed fraction, 1]."""
    def __init__(self):
        self.calls = 0

    def __call__(self, x, m):
        self.calls += 1
        obs = m.sum(1)
        mean = np.where(obs > 0, (x * m).sum(1) / np.maximum(obs, 1), 0)
        return np.column_stack([mean, m.mean(1), np.ones(len(x))]).astype(np.float32)


def test_chunks_align_to_the_end_of_the_window():
    T = 7200
    X = np.arange(T, dtype=np.float32)[None, None, :].repeat(2, 1)
    Xc, Mc = ex.chunk_windows(X, np.ones_like(X, bool), 512)
    assert Xc.shape == (1, 2, 14, 512)
    assert Xc[0, 0, -1, -1] == T - 1                 # last sample kept
    assert Xc[0, 0, 0, 0] == T - 14 * 512            # oldest 32 samples dropped


def test_embed_windows_skips_chunks_with_too_little_signal():
    N, T, L = 2, 2048, 512
    X = np.ones((N, 2, T), np.float32)
    M = np.ones((N, 2, T), bool)
    M[0, 0, :L] = False                               # window 0: first FHR chunk empty
    M[1, 0, :] = False                                # window 1: no FHR at all
    X[0, 0, L:] = 5.0
    mean, mx, n_valid = ex.embed_windows(StubEncoder(), X, M, chunk_len=L, min_observed=0.5,
                                         log_every=0)
    assert n_valid[0].tolist() == [3, 4]
    assert n_valid[1].tolist() == [0, 4]
    assert mean[0, 0, 0] == pytest.approx(5.0)        # the empty chunk did not dilute the mean
    assert (mean[1, 0] == 0).all()                    # no usable chunk -> zeros, flagged by n_valid
    assert np.isfinite(mean).all() and np.isfinite(mx).all()


def test_interpolate_gaps():
    x = np.array([[0., 99., 99., 3.], [7., 7., 7., 7.]])
    m = np.array([[True, False, False, True], [False] * 4])
    out = ex.interpolate_gaps(x, m)
    np.testing.assert_allclose(out[0], [0, 1, 2, 3])
    assert (out[1] == 0).all()


def tiny_moment():
    pytest.importorskip("momentfm")
    from momentfm import MOMENTPipeline
    cfg = dict(task_name="embedding", seq_len=512, patch_len=8, patch_stride_len=8, d_model=32,
               transformer_type="encoder_only", transformer_backbone="google/flan-t5-small",
               randomly_initialize_backbone=True, enable_gradient_checkpointing=False,
               t5_config=dict(d_model=32, d_ff=64, num_layers=1, num_heads=2, d_kv=16,
                              vocab_size=32128, feed_forward_proj="gated-gelu",
                              is_encoder_decoder=True, model_type="t5"))
    m = MOMENTPipeline(cfg, model_kwargs={"task_name": "embedding"})
    m.init()
    return m, cfg


def test_moment_encoder_ignores_values_under_the_mask():
    model, _ = tiny_moment()
    enc = ex.MomentEncoder("unused", "cpu", model=model)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 512)).astype(np.float32)
    m = np.ones((3, 512), bool)
    m[1, :100] = False
    z1 = enc(x, m)
    x2 = x.copy()
    x2[1, :100] = 1e6
    z2 = enc(x2, m)
    assert z1.shape == (3, 32)
    np.testing.assert_allclose(z1, z2, atol=1e-5)


def test_moment_manual_loading(tmp_path, monkeypatch):
    """The fallback loader builds the model from config.json + weights."""
    model, cfg = tiny_moment()
    import json

    import torch
    import huggingface_hub
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")

    def fake_download(repo, filename, **kw):
        p = tmp_path / filename
        if not p.exists():
            raise FileNotFoundError(filename)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    loaded = ex.load_moment_manually("any/repo", {"task_name": "embedding"})
    for (k, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
        assert torch.equal(a, b), k


def test_mantis_encoder_shapes():
    pytest.importorskip("mantis")
    from mantis.architecture import MantisV1
    enc = ex.MantisEncoder("mantis_v1", "unused", "cpu", network=MantisV1(device="cpu"))
    x = np.random.default_rng(0).standard_normal((4, 512)).astype(np.float32)
    m = np.ones((4, 512), bool)
    m[0, 10:50] = False
    z = enc(x, m)
    assert z.shape == (4, 256) and np.isfinite(z).all()


# ---------------------------------------------------------------- probe

def make_files(tmp_path, signal=True):
    rng = np.random.default_rng(0)
    n, T = 150, 64
    sid = np.array([f"R{i:03d}" for i in range(n)])
    y = (np.arange(n) < 25).astype(int)
    mask = np.ones((n, 2, T), bool)
    data = tmp_path / "w.npz"
    np.savez_compressed(data, X=np.zeros((n, 4, T), np.float32), y=y, subject_ids=sid,
                        window_index=np.zeros(n, int), mask=mask,
                        rec_fhr_missing_frac=np.zeros(n, np.float32),
                        recording_min=np.full(n, 70, np.float32), delivery_type=np.ones(n))
    D = 8
    e_mean = rng.standard_normal((n, 2, D)).astype(np.float32)
    if signal:
        e_mean[:, 0, 0] += 2.0 * y                    # one embedding dimension carries the label
    perm = rng.permutation(n)                         # stored in a different order on purpose
    emb = tmp_path / "e.npz"
    np.savez_compressed(emb, subject_ids=sid[perm], window_index=np.zeros(n, int)[perm],
                        emb_mean=e_mean[perm], emb_max=e_mean[perm],
                        n_valid_chunks=np.full((n, 2), 14)[perm], model="stub")
    return data, emb, y


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, check=True)


def test_probe_finds_planted_signal_and_aligns_by_window(tmp_path):
    data, emb, y = make_files(tmp_path)
    out = tmp_path / "probe"
    run("scripts/probe.py", "--data", str(data), "--embeddings", str(emb), "--out_dir", str(out),
        "--pool", "meanmax", "--channels", "fhr_uc", "--n_repeats", "2")
    oof = pd.read_csv(out / "oof_predictions.csv")
    assert len(oof) == 2 * len(y)
    from sklearn.metrics import roc_auc_score
    for r, g in oof.groupby("repeat"):
        assert roc_auc_score(g["y"], g["prob"]) > 0.75   # about 0.5 if windows were misaligned


def test_probe_without_signal_is_near_chance(tmp_path):
    data, emb, y = make_files(tmp_path, signal=False)
    out = tmp_path / "probe"
    run("scripts/probe.py", "--data", str(data), "--embeddings", str(emb), "--out_dir", str(out),
        "--channels", "fhr")
    oof = pd.read_csv(out / "oof_predictions.csv")
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(oof["y"], oof["prob"]) - 0.5) < 0.2


def test_probe_refuses_embeddings_from_other_windows(tmp_path):
    data, emb, y = make_files(tmp_path)
    e = dict(np.load(emb))
    e["subject_ids"] = np.array([s + "x" for s in e["subject_ids"]])
    np.savez_compressed(emb, **e)
    with pytest.raises(subprocess.CalledProcessError) as err:
        run("scripts/probe.py", "--data", str(data), "--embeddings", str(emb),
            "--out_dir", str(tmp_path / "p"))
    assert "have no embedding" in err.value.stderr


def test_extract_cli_end_to_end_then_probe(tmp_path):
    """extract_embeddings.py -> probe.py -> evaluate.py on a small window file (random Mantis)."""
    pytest.importorskip("mantis")
    rng = np.random.default_rng(1)
    n, T = 40, 1024
    sid = np.array([f"R{i:03d}" for i in range(n)])
    y = (np.arange(n) < 10).astype(int)
    mask = rng.random((n, 2, T)) > 0.1
    data = tmp_path / "w.npz"
    np.savez_compressed(data, X=rng.standard_normal((n, 4, T)).astype(np.float32), y=y,
                        subject_ids=sid, window_index=np.zeros(n, int), mask=mask,
                        rec_fhr_missing_frac=np.full(n, 0.1, np.float32),
                        recording_min=np.full(n, 70, np.float32), delivery_type=np.ones(n))
    emb = tmp_path / "emb.npz"
    run("scripts/extract_embeddings.py", "--data", str(data), "--model", "mantis",
        "--out", str(emb), "--device", "cpu", "--random_init")
    e = np.load(emb)
    assert e["emb_mean"].shape == (n, 2, 256) and e["n_valid_chunks"].max() == 2
    out = tmp_path / "res" / "probe_mantis"
    run("scripts/probe.py", "--data", str(data), "--embeddings", str(emb), "--out_dir", str(out),
        "--n_folds", "3", "--inner_folds", "3")
    run("scripts/evaluate.py", "--data", str(data), "--runs", f"probe={out}", "--n_boot", "50")
