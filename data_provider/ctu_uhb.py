"""
CTU-UHB Intrapartum Cardiotocography Database (PhysioNet, ctu-uhb-ctgdb 1.0.0).

552 recordings of fetal heart rate (FHR, bpm) and uterine contractions (UC) at 4 Hz,
each lasting at most 90 minutes and starting no more than 90 minutes before delivery.
Umbilical artery pH is available for every record.

Pipeline
--------
1. `build_store(wfdb_dir)` reads every record once into a canonical store:
   full-length signals (unnormalised, missing values kept as NaN) plus a metadata
   table parsed from the header comments. Nothing is windowed or scaled here.
2. `make_windows(store, ...)` cuts fixed-length windows counted backwards from an
   anchor (end of recording, or end of stage I) and returns raw values, masks, labels,
   and each window's distance from the anchor and from birth.
3. `scale_fixed(...)` applies a fixed physiological scaling. It uses constants, not
   statistics computed from the data, so it cannot leak information across folds.

Header fields used (as written in the .hea comments; record 1001 for reference):
    pH            umbilical artery pH
    Pos. II.st.   sample index where stage II starts (14400 in record 1001, i.e. 60 min
                  into an 80-min record whose II.stage field is 20 min)
    Sig2Birth     minutes from the end of the signal to birth
    Rec. type     FHR acquisition (as documented on PhysioNet, ultrasound or scalp ECG)
Fields in the '!NotReadyYet!' section of the header (NICU days, Seizures, HIE, ...) are
skipped: PhysioNet marks them as not ready.

Missing samples: zeros (and NaN) in FHR or UC are treated as missing.

Cite when using the data: Chudacek et al., BMC Pregnancy and Childbirth 2014, 14:16,
and PhysioNet. License: Open Data Commons Attribution v1.0.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

FS_HZ = 4
SAMPLES_PER_MIN = FS_HZ * 60
PHYSIONET_DB = "ctu-uhb-ctgdb"

# Clearer names for the header keys the pipeline relies on. Everything else is slugified.
_RENAME = {
    "ph": "ph",
    "bdecf": "bdecf",
    "pco2": "pco2",
    "be": "be",
    "apgar1": "apgar1",
    "apgar5": "apgar5",
    "gest_weeks": "gest_weeks",
    "weight_g": "weight_g",
    "i_stage": "stage1_min",
    "ii_stage": "stage2_min",
    "deliv_type": "delivery_type",
    "rec_type": "rec_type",
    "pos_ii_st": "stage2_start_sample",
    "sig2birth": "sig_to_birth_min",
    "dbid": "db_id",
}


def _slug(key: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_")


def _to_number(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def parse_header_comments(comments: Sequence[str]) -> Dict[str, object]:
    """
    Parse the '#key   value' lines that follow the signal spec in a CTU-UHB .hea file.
    Section lines start with '-' after the '#'. Keys in a section whose title contains
    'NotReadyYet' are skipped.
    """
    meta: Dict[str, object] = {}
    skipping = False
    for raw in comments:
        line = raw.strip().lstrip("#").strip()
        if not line:
            continue
        if line.startswith("-"):
            skipping = "notreadyyet" in line.lower()
            continue
        if skipping:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        slug = _slug(key)
        meta[_RENAME.get(slug, slug)] = _to_number(value)
    return meta


@dataclass
class CTGStore:
    """Full-length recordings, concatenated without pickling."""
    record_ids: np.ndarray      # (n,) str
    offsets: np.ndarray         # (n + 1,) int64; record i is [offsets[i], offsets[i+1])
    fhr: np.ndarray             # (total,) float32, NaN = missing
    uc: np.ndarray              # (total,) float32, NaN = missing
    meta: pd.DataFrame          # one row per record, index = record_id
    fs: int = FS_HZ

    def __len__(self) -> int:
        return len(self.record_ids)

    def signals(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.fhr[a:b], self.uc[a:b]

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def save(self, npz_path: str, meta_csv_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        np.savez_compressed(npz_path, record_ids=self.record_ids, offsets=self.offsets,
                            fhr=self.fhr, uc=self.uc, fs=self.fs)
        self.meta.to_csv(meta_csv_path)

    @classmethod
    def load(cls, npz_path: str, meta_csv_path: str) -> "CTGStore":
        d = np.load(npz_path, allow_pickle=False)
        meta = pd.read_csv(meta_csv_path, index_col=0, dtype={0: str})
        meta.index = meta.index.astype(str)
        rec = d["record_ids"].astype(str)
        return cls(rec, d["offsets"], d["fhr"], d["uc"], meta.loc[rec], int(d["fs"]))


def _record_ok(wfdb_dir: str, name: str) -> bool:
    """True if both files exist and the record reads back with its full length."""
    import wfdb

    base = os.path.join(wfdb_dir, name)
    if not all(os.path.isfile(f"{base}.{ext}") and os.path.getsize(f"{base}.{ext}") > 0
               for ext in ("hea", "dat")):
        return False
    try:
        rec = wfdb.rdrecord(base)
        return rec.p_signal is not None and rec.p_signal.shape[0] == rec.sig_len
    except Exception:
        return False


def download(dl_dir: str, attempts: int = 5, pause_s: float = 5.0) -> List[str]:
    """
    Download CTU-UHB from PhysioNet one record at a time, with retries.

    PhysioNet occasionally answers with transient errors (e.g. 502 Bad Gateway), and
    wfdb.dl_database stops at the first one. Here each record is retried with a growing
    pause, records that already read back correctly are skipped, and an interrupted run
    resumes where it stopped when the same command is run again.
    """
    import time

    from wfdb.io import download as wfdb_download

    def retry(fn, what):
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:  # network errors come in several exception types
                if attempt == attempts:
                    raise RuntimeError(
                        f"{what} failed after {attempts} attempts ({e}). PhysioNet may be "
                        f"temporarily unavailable; rerun the same command to resume.") from e
                wait = pause_s * attempt
                print(f"  {what}: {type(e).__name__}; retrying in {wait:.0f}s "
                      f"({attempt}/{attempts})", flush=True)
                time.sleep(wait)

    import posixpath

    os.makedirs(dl_dir, exist_ok=True)
    version = retry(lambda: wfdb_download.get_version(PHYSIONET_DB), "looking up the version")
    base_url = posixpath.join(wfdb_download.config.db_index_url, PHYSIONET_DB, version)
    records = retry(lambda: wfdb_download.get_record_list(f"{PHYSIONET_DB}/{version}"),
                    "listing records")
    with open(os.path.join(dl_dir, "RECORDS"), "w") as f:
        f.write("\n".join(records) + "\n")

    todo = [r for r in records if not _record_ok(dl_dir, r)]
    print(f"{len(records)} records listed, {len(records) - len(todo)} already present, "
          f"{len(todo)} to download", flush=True)
    for i, name in enumerate(todo, 1):
        def fetch():
            for ext in ("hea", "dat"):
                target = os.path.join(dl_dir, f"{name}.{ext}")
                # write to a temporary name first so a failed transfer never leaves a
                # partial file that looks complete
                wfdb_download.dl_full_file(posixpath.join(base_url, f"{name}.{ext}"),
                                           target + ".part")
                os.replace(target + ".part", target)
            if not _record_ok(dl_dir, name):
                raise IOError(f"record {name} incomplete after download")
        retry(fetch, f"record {name}")
        if i % 50 == 0 or i == len(todo):
            print(f"  downloaded {i}/{len(todo)}", flush=True)

    missing = [r for r in records if not _record_ok(dl_dir, r)]
    if missing:
        raise RuntimeError(f"{len(missing)} records still incomplete: {missing[:10]}")
    return records


def list_records(wfdb_dir: str) -> List[str]:
    """Record names from the RECORDS file, or every .hea file if there is none."""
    rec_file = os.path.join(wfdb_dir, "RECORDS")
    if os.path.exists(rec_file):
        with open(rec_file) as f:
            return [r.strip() for r in f if r.strip()]
    return sorted(os.path.splitext(f)[0] for f in os.listdir(wfdb_dir) if f.endswith(".hea"))


def _missing_to_nan(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x[(x == 0) | ~np.isfinite(x)] = np.nan
    return x


def read_record(wfdb_dir: str, name: str):
    """Return (fhr, uc, meta) for one record. Missing samples become NaN."""
    import wfdb  # imported here so the rest of the module works without it

    rec = wfdb.rdrecord(os.path.join(wfdb_dir, name))
    if rec.fs != FS_HZ:
        raise ValueError(f"record {name}: expected {FS_HZ} Hz, got {rec.fs}")
    names = [s.upper() for s in rec.sig_name]
    if "FHR" not in names or "UC" not in names:
        raise ValueError(f"record {name}: expected FHR and UC, got {rec.sig_name}")
    sig = rec.p_signal
    fhr = _missing_to_nan(sig[:, names.index("FHR")])
    uc = _missing_to_nan(sig[:, names.index("UC")])
    meta = parse_header_comments(rec.comments or [])
    meta["n_samples"] = len(fhr)
    return fhr, uc, meta


def build_store(wfdb_dir: str, records: Optional[Sequence[str]] = None) -> CTGStore:
    """Read every record into a CTGStore. Fails loudly if a record has no pH."""
    records = list(records) if records is not None else list_records(wfdb_dir)
    if not records:
        raise FileNotFoundError(f"no WFDB records found in {wfdb_dir}")
    fhrs, ucs, metas = [], [], []
    for name in records:
        fhr, uc, meta = read_record(wfdb_dir, name)
        if "ph" not in meta:
            raise KeyError(f"record {name}: no pH in header; keys found: {sorted(meta)}")
        fhrs.append(fhr)
        ucs.append(uc)
        metas.append(meta)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in fhrs])
    meta = pd.DataFrame(metas, index=pd.Index([str(r) for r in records], name="record_id"))
    return CTGStore(np.array(records, dtype=str), offsets,
                    np.concatenate(fhrs), np.concatenate(ucs), meta)


def anchor_index(meta_row: pd.Series, n_samples: int, anchor: str) -> int:
    """
    Sample index windows are counted back from.
    'end'        : end of the recording
    'stage1_end' : start of stage II if it falls inside the recording, else the end
    """
    if anchor == "end":
        return n_samples
    if anchor == "stage1_end":
        pos = meta_row.get("stage2_start_sample", np.nan)
        if pd.notna(pos) and 0 < int(pos) < n_samples:
            return int(pos)
        return n_samples
    raise ValueError(f"unknown anchor {anchor!r}; use 'end' or 'stage1_end'")


def make_windows(
    store: CTGStore,
    window_min: float = 30,
    n_windows: int = 1,
    step_min: Optional[float] = None,
    anchor: str = "end",
    min_valid_frac: float = 0.2,
    ph_threshold: float = 7.05,
):
    """
    Cut up to `n_windows` windows per record, counted backwards from the anchor.
    Window k (k = 0 nearest the anchor) covers [anchor - k*step - T, anchor - k*step).
    Windows with less than `min_valid_frac` observed FHR are dropped.

    Returns a dict:
        X                 (N, 2, T) float32, raw FHR (bpm) and UC, NaN = missing
        mask              (N, 2, T) bool, True = observed
        y                 (N,) int64, 1 if pH < ph_threshold
        subject_ids       (N,) str, record id (one record per subject in CTU-UHB)
        window_index      (N,) int, k
        min_before_anchor (N,) float, minutes from the window's end to the anchor
        min_before_birth  (N,) float, minutes from the window's end to birth
        ph, delivery_type, recording_min, rec_fhr_missing_frac
                          (N,) per-recording values repeated for each of its windows,
                          used by the shortcut baselines and subset analyses
    """
    T = int(round(window_min * SAMPLES_PER_MIN))
    step = T if step_min is None else int(round(step_min * SAMPLES_PER_MIN))
    if T <= 0 or step <= 0:
        raise ValueError("window_min and step_min must be positive")

    out = {k: [] for k in ("X", "mask", "y", "subject_ids", "window_index",
                           "min_before_anchor", "min_before_birth", "ph",
                           "delivery_type", "recording_min", "rec_fhr_missing_frac")}
    for i, rid in enumerate(store.record_ids):
        fhr, uc = store.signals(i)
        row = store.meta.loc[rid]
        n = len(fhr)
        a = anchor_index(row, n, anchor)
        sig_to_birth = float(row.get("sig_to_birth_min", 0) or 0)
        label = int(float(row["ph"]) < ph_threshold)
        rec_fhr_missing = float(np.isnan(fhr).mean())
        for k in range(n_windows):
            end = a - k * step
            start = end - T
            if start < 0:
                break
            seg = np.stack([fhr[start:end], uc[start:end]])
            m = np.isfinite(seg)
            if m[0].mean() < min_valid_frac:
                continue
            out["X"].append(seg)
            out["mask"].append(m)
            out["y"].append(label)
            out["subject_ids"].append(rid)
            out["window_index"].append(k)
            out["min_before_anchor"].append((a - end) / SAMPLES_PER_MIN)
            out["min_before_birth"].append((n - end) / SAMPLES_PER_MIN + sig_to_birth)
            out["ph"].append(float(row["ph"]))
            out["delivery_type"].append(float(row.get("delivery_type", np.nan)))
            out["recording_min"].append(n / SAMPLES_PER_MIN)
            out["rec_fhr_missing_frac"].append(rec_fhr_missing)

    if not out["X"]:
        raise ValueError("no windows survived; check window_min, anchor and min_valid_frac")
    return {
        "X": np.stack(out["X"]).astype(np.float32),
        "mask": np.stack(out["mask"]),
        "y": np.array(out["y"], dtype=np.int64),
        "subject_ids": np.array(out["subject_ids"], dtype=str),
        "window_index": np.array(out["window_index"], dtype=np.int64),
        "min_before_anchor": np.array(out["min_before_anchor"], dtype=np.float32),
        "min_before_birth": np.array(out["min_before_birth"], dtype=np.float32),
        "ph": np.array(out["ph"], dtype=np.float32),
        "delivery_type": np.array(out["delivery_type"], dtype=np.float32),
        "recording_min": np.array(out["recording_min"], dtype=np.float32),
        "rec_fhr_missing_frac": np.array(out["rec_fhr_missing_frac"], dtype=np.float32),
    }


# Fixed physiological scaling: constants only, so nothing is fit on the data.
FHR_CENTER, FHR_SCALE = 140.0, 25.0   # bpm; normal baseline is roughly 110-160
UC_CENTER, UC_SCALE = 0.0, 50.0


def scale_fixed(X: np.ndarray, mask: np.ndarray, add_mask_channels: bool = False) -> np.ndarray:
    """Scale FHR and UC with fixed constants and set missing samples to 0 (the centre)."""
    Z = np.empty_like(X, dtype=np.float32)
    Z[:, 0] = (X[:, 0] - FHR_CENTER) / FHR_SCALE
    Z[:, 1] = (X[:, 1] - UC_CENTER) / UC_SCALE
    Z[~mask] = 0.0
    if add_mask_channels:
        Z = np.concatenate([Z, mask.astype(np.float32)], axis=1)
    return Z


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def confound_report(store: CTGStore, ph_threshold: float = 7.05) -> pd.DataFrame:
    """
    Compare positives and negatives on properties that are not physiology: recording
    length, signal loss, stage II coverage, acquisition type, time to birth. A large
    standardized mean difference (|SMD| > 0.2) means a model could separate the classes
    partly through that property instead of physiology. Delivery type matters in
    particular: cesarean deliveries have no stage II and may be decided because of a
    worrying trace.
    """
    rows = []
    for i, rid in enumerate(store.record_ids):
        fhr, uc = store.signals(i)
        r = store.meta.loc[rid]
        n = len(fhr)
        pos = r.get("stage2_start_sample", np.nan)
        stage2_min = (n - pos) / SAMPLES_PER_MIN if pd.notna(pos) and 0 < pos < n else 0.0
        rows.append({
            "label": int(float(r["ph"]) < ph_threshold),
            "recording_min": n / SAMPLES_PER_MIN,
            "fhr_missing_frac": float(np.isnan(fhr).mean()),
            "uc_missing_frac": float(np.isnan(uc).mean()),
            "stage2_min_in_recording": stage2_min,
            "sig_to_birth_min": float(r.get("sig_to_birth_min", np.nan)),
            "rec_type": float(r.get("rec_type", np.nan)),
            "delivery_type": float(r.get("delivery_type", np.nan)),
        })
    df = pd.DataFrame(rows)
    pos, neg = df[df.label == 1], df[df.label == 0]
    report = []
    for col in df.columns.drop("label"):
        smd = standardized_mean_difference(pos[col], neg[col])
        mean_pos, mean_neg = float(pos[col].mean()), float(neg[col].mean())
        # ignore differences too small to matter (e.g. 0.01% vs 0.02% missing)
        material = abs(mean_pos - mean_neg) > 1e-3 * max(1.0, abs(float(df[col].mean())))
        report.append({
            "property": col,
            "mean_pos": round(mean_pos, 3),
            "mean_neg": round(mean_neg, 3),
            "smd": round(smd, 3),
            "flag": "CHECK" if abs(smd) > 0.2 and material else "",
        })
    return pd.DataFrame(report)
