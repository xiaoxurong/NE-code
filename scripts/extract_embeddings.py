"""
Frozen embeddings from pretrained time-series encoders (MOMENT, Mantis) for every window
in a window file written by prepare_ctu_uhb.py.

Why chunks: both encoders take 512 samples per channel. A 30-minute window at 4 Hz has
7,200 samples, and squeezing it into 512 would sample about every 3.5 s, which removes the
short-term FHR variability that matters most in CTG. Instead each window is cut into
consecutive 512-sample chunks (about 2.1 min each, aligned to the end of the window, so the
oldest few samples are dropped), each chunk is embedded at full resolution, and the chunk
embeddings are pooled.

Missing signal:
  MOMENT  takes an observed-mask; missing samples are masked, not filled.
  Mantis  has no mask; gaps are linearly interpolated within the chunk.
  A chunk with less than --min_observed of a channel observed is left out of that
  channel's pooling. The count of usable chunks is saved but is NOT part of the embedding,
  so missingness only enters a probe when it is asked for explicitly.

The encoder never sees labels, so the output can be reused for any label threshold.

Output (.npz): subject_ids, window_index, emb_mean (N, 2, D), emb_max (N, 2, D),
n_valid_chunks (N, 2), channels, model, chunk_len, min_observed.

Usage
  python scripts/extract_embeddings.py --data dataset/<window file>.npz \
      --model moment-base --out embeddings/<window stem>__moment-base.npz
"""

import argparse
import os
import sys
import time

import numpy as np

MODELS = {
    "moment-small": ("moment", "AutonLab/MOMENT-1-small"),
    "moment-base": ("moment", "AutonLab/MOMENT-1-base"),
    "moment-large": ("moment", "AutonLab/MOMENT-1-large"),
    "mantis": ("mantis_v1", "paris-noah/Mantis-8M"),
    "mantis-l2": ("mantis_v1_l2", "paris-noah/Mantis-8M"),
    "mantisv2": ("mantis_v2", "paris-noah/MantisV2"),
    "mantisv2-l2": ("mantis_v2_l2", "paris-noah/MantisV2"),
}
CHANNELS = ["FHR", "UC"]


# ----------------------------------------------------------------------------- encoders

class MomentEncoder:
    """MOMENT in embedding mode; each channel is encoded on its own with its own mask."""

    def __init__(self, repo: str, device: str, model=None):
        import torch
        self.torch = torch
        self.device = device
        self.model = model if model is not None else self._load(repo)
        self.model.to(device).eval()

    @staticmethod
    def _load(repo):
        from momentfm import MOMENTPipeline
        kwargs = {"task_name": "embedding"}
        try:
            model = MOMENTPipeline.from_pretrained(repo, model_kwargs=dict(kwargs))
        except Exception as e:  # hub-mixin API differences across huggingface_hub versions
            print(f"MOMENTPipeline.from_pretrained failed ({type(e).__name__}: {e}); "
                  f"loading the weights manually", flush=True)
            model = load_moment_manually(repo, kwargs)
        model.init()
        return model

    def __call__(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        """x, m: (B, L). Returns (B, D); rows with no observed patch come back as NaN."""
        t = self.torch
        with t.no_grad():
            xe = t.from_numpy(np.where(m, x, 0.0).astype(np.float32))[:, None, :].to(self.device)
            mk = t.from_numpy(m.astype(np.float32)).to(self.device)
            out = self.model.embed(x_enc=xe, input_mask=mk, reduction="mean").embeddings
        return out.float().cpu().numpy()


def load_moment_manually(repo: str, model_kwargs: dict):
    """Fallback: download config + weights and build MOMENTPipeline directly."""
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from momentfm import MOMENTPipeline

    with open(hf_hub_download(repo, "config.json")) as f:
        config = json.load(f)
    model = MOMENTPipeline(config, model_kwargs=dict(model_kwargs))
    state = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(repo, name)
        except Exception:
            continue
        if name.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu")
        break
    if state is None:
        raise RuntimeError(f"no weights file found in {repo}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    encoder_missing = [k for k in missing if k.startswith(("encoder.", "patch_embedding."))]
    if encoder_missing:
        raise RuntimeError(f"encoder weights missing after load: {encoder_missing[:5]}")
    print(f"loaded {repo}: {len(missing)} missing (head only), {len(unexpected)} unexpected keys")
    return model


class MantisEncoder:
    """Mantis / MantisV2. No mask: gaps are linearly interpolated within each chunk."""

    def __init__(self, kind: str, repo: str, device: str, network=None):
        from mantis.trainer import MantisTrainer
        if network is None:
            from mantis.architecture import MantisV1, MantisV2
            cls = MantisV2 if "v2" in kind else MantisV1
            layer_kwargs = {"return_transf_layer": 2, "output_token": "combined"} if kind.endswith("_l2") else {}
            network = cls(device=device, **layer_kwargs).from_pretrained(repo)
        self.trainer = MantisTrainer(device=device, network=network)

    def __call__(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        x = interpolate_gaps(x, m)
        z = self.trainer.transform(x[:, None, :].astype(np.float32))
        return np.asarray(z, dtype=np.float32)


def interpolate_gaps(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Linear interpolation over unobserved samples, row by row (edges held constant)."""
    out = np.zeros_like(x, dtype=np.float32)
    idx = np.arange(x.shape[1])
    for i in range(x.shape[0]):
        obs = m[i]
        if obs.any():
            out[i] = np.interp(idx, idx[obs], x[i, obs])
    return out


def build_encoder(name: str, device: str, random_init: bool = False):
    kind, repo = MODELS[name]
    if kind == "moment":
        if random_init:
            raise ValueError("--random_init is only supported for the Mantis models")
        return MomentEncoder(repo, device)
    if random_init:  # for smoke tests without downloading weights
        from mantis.architecture import MantisV1, MantisV2
        return MantisEncoder(kind, repo, device,
                             network=(MantisV2 if "v2" in kind else MantisV1)(device=device))
    return MantisEncoder(kind, repo, device)


# ----------------------------------------------------------------------------- pipeline

def chunk_windows(X: np.ndarray, M: np.ndarray, chunk_len: int):
    """(N, C, T) -> (N, C, K, chunk_len), aligned to the end of each window."""
    N, C, T = X.shape
    K = T // chunk_len
    if K == 0:
        raise ValueError(f"window of {T} samples is shorter than one chunk ({chunk_len})")
    start = T - K * chunk_len
    return (X[:, :, start:].reshape(N, C, K, chunk_len),
            M[:, :, start:].reshape(N, C, K, chunk_len))


def embed_windows(encoder, X, M, chunk_len=512, min_observed=0.5, batch_size=256, log_every=20):
    """
    X: (N, 2, T) channel values (missing samples arbitrary), M: (N, 2, T) bool observed.
    Returns emb_mean (N, 2, D), emb_max (N, 2, D), n_valid (N, 2).
    """
    Xc, Mc = chunk_windows(X, M, chunk_len)
    N, C, K, L = Xc.shape
    flat_x, flat_m = Xc.reshape(-1, L), Mc.reshape(-1, L)
    valid = flat_m.mean(axis=1) >= min_observed

    emb = None
    rows = np.flatnonzero(valid)
    t0 = time.time()
    for b, s in enumerate(range(0, len(rows), batch_size)):
        r = rows[s:s + batch_size]
        z = encoder(flat_x[r], flat_m[r])
        if emb is None:
            emb = np.full((len(flat_x), z.shape[1]), np.nan, dtype=np.float32)
        emb[r] = z
        if log_every and b % log_every == 0:
            print(f"  chunks {s + len(r)}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    if emb is None:
        raise ValueError("no chunk had enough observed signal to embed")

    emb = emb.reshape(N, C, K, -1)
    ok = np.isfinite(emb).all(axis=-1)                      # (N, C, K)
    n_valid = ok.sum(axis=2)
    D = emb.shape[-1]
    emb_mean = np.zeros((N, C, D), np.float32)
    emb_max = np.zeros((N, C, D), np.float32)
    for i in range(N):
        for c in range(C):
            e = emb[i, c][ok[i, c]]
            if len(e):
                emb_mean[i, c], emb_max[i, c] = e.mean(axis=0), e.max(axis=0)
    return emb_mean, emb_max, n_valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_len", type=int, default=512)
    ap.add_argument("--min_observed", type=float, default=0.5,
                    help="Minimum observed fraction of a channel in a chunk for it to be pooled")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    ap.add_argument("--random_init", action="store_true",
                    help="Randomly initialized Mantis instead of pretrained weights (smoke tests only)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data, allow_pickle=False)
    if "mask" not in d.files:
        sys.exit(f"{args.data} has no 'mask' array; rebuild it with prepare_ctu_uhb.py")
    # channels 0, 1 of X are scaled FHR and UC; any extra channels are the masks themselves
    X, M = d["X"][:, :2].astype(np.float32), d["mask"].astype(bool)
    print(f"{args.data}: {X.shape[0]} windows of {X.shape[-1]} samples | model {args.model} | device {device}")

    encoder = build_encoder(args.model, device, args.random_init)
    emb_mean, emb_max, n_valid = embed_windows(encoder, X, M, args.chunk_len, args.min_observed,
                                               args.batch_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, subject_ids=d["subject_ids"], window_index=d["window_index"],
                        emb_mean=emb_mean, emb_max=emb_max, n_valid_chunks=n_valid,
                        channels=np.array(CHANNELS), model=args.model, chunk_len=args.chunk_len,
                        min_observed=args.min_observed)
    K = X.shape[-1] // args.chunk_len
    print(f"Wrote {args.out}: embedding dim {emb_mean.shape[-1]} per channel | "
          f"usable chunks per window (FHR) median {np.median(n_valid[:, 0]):.0f} of {K} | "
          f"windows with no usable FHR chunk: {int((n_valid[:, 0] == 0).sum())}")


if __name__ == "__main__":
    main()
