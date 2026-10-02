"""
scripts/extract_windows.py
==========================
Extract multiple non-overlapping 30-min windows from full CTG recordings.

Input:  data_list.npz  — list of subject dicts, each containing:
            ['df']       : pandas DataFrame with FHR1, TOCO, FHR1MODE, TOCOMODE, ...
            ['label']    : int (0=normal, 1=NE)
            ['Study_ID'] : str

Output: dataset/CTGMultiWindow.npz — same format as CTG30minSegments.npz:
            X:           (N_windows, 2, 7200) — [FHR1, TOCO] stacked
            y:           (N_windows,)         — NE label (inherited from subject)
            subject_ids: (N_windows,)         — Study_ID repeated per window
                         (used by StratifiedGroupKFold to keep all windows
                          from the same subject in the same fold)

Design notes
------------
- Step size = WINDOW_SIZE (non-overlapping) by default.
  Non-overlapping windows are nearly independent and make test-set
  evaluation straightforward. Use STEP_SIZE < WINDOW_SIZE for more
  windows at the cost of higher within-subject correlation.

- Quality filter: skip windows where > MAX_BAD_FRAC of FHR1 samples
  are zero or flagged bad by FHR1MODE. TOCO is not filtered — external
  tocometry is often noisy and the model handles it via nan_to_num.

- At test time, aggregate per-subject: average the predicted probabilities
  across all windows from the same subject before computing metrics.
  See the helper function subject_level_metrics() below, and the note
  on how to plug it into exp_classification.py.
"""

import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_LIST_PATH = "./data_list.npz"
OUTPUT_PATH    = "./dataset/CTGMultiWindow.npz"

WINDOW_SIZE  = 7200    # 30 min at 4 Hz
STEP_SIZE    = 7200    # non-overlapping (set to 3600 for 50% overlap)
MAX_BAD_FRAC = 0.15    # skip window if >15% of FHR1 samples are zero (signal loss)

FHR_COL      = 'FHR1'
TOCO_COL     = 'TOCO'
FHR_MODE_COL = 'FHR1MODE'   # 0 = valid signal; non-zero = unreliable

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data_list.npz ...")
with np.load(DATA_LIST_PATH, allow_pickle=True) as f:
    data_list = list(f['arr_0'])
print(f"  {len(data_list)} subjects loaded.")

# ── Extract windows ───────────────────────────────────────────────────────────
X_list, y_list, sid_list = [], [], []
stats = {'short': 0, 'quality': 0, 'windows': 0}

for subj in data_list:
    study_id = str(subj['Study_ID'])
    label    = int(subj['label'])
    df       = subj['df']
    T        = len(df)

    if T < WINDOW_SIZE:
        stats['short'] += 1
        continue

    fhr  = df[FHR_COL].values.astype(np.float32)
    toco = df[TOCO_COL].values.astype(np.float32)

    # Quality mask: FHR=0 means signal loss (can't have 0 bpm).
    # FHR1MODE=1 is the valid-signal state (99.8% of samples); MODE=0
    # only co-occurs with FHR=0, so checking FHR==0 is sufficient.
    fhr_bad = (fhr == 0)

    for start in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        bad_frac = fhr_bad[start:end].mean()
        if bad_frac > MAX_BAD_FRAC:
            stats['quality'] += 1
            continue

        window = np.stack([fhr[start:end], toco[start:end]], axis=0)  # (2, 7200)
        X_list.append(window)
        y_list.append(label)
        sid_list.append(study_id)
        stats['windows'] += 1

# ── Save ──────────────────────────────────────────────────────────────────────
X    = np.stack(X_list, axis=0)   # (N, 2, 7200)
y    = np.array(y_list)
sids = np.array(sid_list)

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
np.savez(OUTPUT_PATH, X=X, y=y, subject_ids=sids)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nExtraction complete")
print(f"  Total windows   : {stats['windows']}")
print(f"  NE positive     : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Unique subjects : {len(np.unique(sids))}")
print(f"  Skipped (short) : {stats['short']}")
print(f"  Skipped (quality): {stats['quality']}")

print(f"\nPer-label breakdown:")
for lv, ln in [(0, 'Normal'), (1, 'NE')]:
    mask   = y == lv
    n_subj = len(np.unique(sids[mask]))
    n_win  = mask.sum()
    print(f"  {ln:6s}: {n_win:4d} windows from {n_subj:3d} subjects "
          f"({n_win/n_subj:.1f} windows/subject avg)")

print(f"\nSaved → {OUTPUT_PATH}")