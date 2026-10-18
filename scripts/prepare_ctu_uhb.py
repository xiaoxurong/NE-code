"""
Prepare the public CTU-UHB dataset for training.

Steps
  1. (optional) download the database from PhysioNet
  2. read every record into a canonical store (full-length, unscaled, missing = NaN)
  3. cut windows anchored at the end of the recording or the end of stage I
  4. write a window file that run.py can train on directly
  5. print a dataset summary and a case/control comparison of non-physiological
     properties (recording length, signal loss, stage II coverage, time to birth)

Usage
  python scripts/prepare_ctu_uhb.py --wfdb_dir data/ctu-uhb --download
  python scripts/prepare_ctu_uhb.py --wfdb_dir data/ctu-uhb \
      --window_min 30 --n_windows 1 --anchor stage1_end --ph_threshold 7.05

Then train with
  python run.py --config configs/ctu_uhb_cnn_lstm.yaml
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider.ctu_uhb import (  # noqa: E402
    PHYSIONET_DB, CTGStore, build_store, confound_report, download, make_windows, scale_fixed,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wfdb_dir", default="data/ctu-uhb", help="Folder with the .dat/.hea files")
    ap.add_argument("--download", action="store_true", help="Download the database from PhysioNet first")
    ap.add_argument("--out_dir", default="dataset", help="Where processed files are written (git-ignored)")
    ap.add_argument("--rebuild_store", action="store_true", help="Re-read the WFDB files even if a store exists")
    ap.add_argument("--window_min", type=float, default=30)
    ap.add_argument("--n_windows", type=int, default=1, help="Windows per record, counted back from the anchor")
    ap.add_argument("--step_min", type=float, default=None, help="Default: window_min (no overlap)")
    ap.add_argument("--anchor", choices=["end", "stage1_end"], default="stage1_end")
    ap.add_argument("--min_valid_frac", type=float, default=0.2,
                    help="Drop windows with less observed FHR than this. Kept low on purpose: "
                         "on CTU-UHB a 0.5 threshold removes a quarter of the positives")
    ap.add_argument("--ph_threshold", type=float, default=7.05)
    ap.add_argument("--no_mask_channels", dest="add_mask_channels", action="store_false",
                    help="Do not append the FHR/UC observed-masks as channels 3 and 4")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.download:
        print(f"Downloading {PHYSIONET_DB} to {args.wfdb_dir} ...")
        download(args.wfdb_dir)

    store_npz = os.path.join(args.out_dir, "ctu_uhb_store.npz")
    meta_csv = os.path.join(args.out_dir, "ctu_uhb_meta.csv")
    if os.path.exists(store_npz) and not args.rebuild_store:
        store = CTGStore.load(store_npz, meta_csv)
        print(f"Loaded store: {store_npz}")
    else:
        store = build_store(args.wfdb_dir)
        store.save(store_npz, meta_csv)
        print(f"Built store from {args.wfdb_dir} -> {store_npz}")

    # ---- dataset summary ---------------------------------------------------------
    lengths_min = store.lengths() / (store.fs * 60)
    ph = store.meta["ph"].astype(float)
    n_pos = int((ph < args.ph_threshold).sum())
    print(f"\nRecords: {len(store)} | positives (pH < {args.ph_threshold}): {n_pos} "
          f"({100 * n_pos / len(store):.1f}%)")
    print(f"Recording length (min): median {np.median(lengths_min):.1f}, "
          f"range {lengths_min.min():.1f}-{lengths_min.max():.1f}")
    print(f"FHR missing overall: {100 * np.isnan(store.fhr).mean():.1f}% | "
          f"UC missing overall: {100 * np.isnan(store.uc).mean():.1f}%")

    # ---- confound check ----------------------------------------------------------
    print("\nCase/control comparison of non-physiological properties "
          "(|SMD| > 0.2 is flagged):")
    print(confound_report(store, args.ph_threshold).to_string(index=False))

    # ---- windows -----------------------------------------------------------------
    w = make_windows(store, window_min=args.window_min, n_windows=args.n_windows,
                     step_min=args.step_min, anchor=args.anchor,
                     min_valid_frac=args.min_valid_frac, ph_threshold=args.ph_threshold)
    X = scale_fixed(w["X"], w["mask"], add_mask_channels=args.add_mask_channels)
    channels = ["FHR", "UC"] + (["FHR_observed", "UC_observed"] if args.add_mask_channels else [])

    n_rec = len(np.unique(w["subject_ids"]))
    dropped = sorted(set(store.record_ids) - set(w["subject_ids"]))
    tag = (f"ctu_uhb_{args.anchor}_w{args.window_min:g}m_n{args.n_windows}"
           f"_v{args.min_valid_frac:g}_ph{args.ph_threshold:g}"
           f"{'' if args.add_mask_channels else '_nomask'}")
    out_npz = os.path.join(args.out_dir, f"{tag}.npz")
    np.savez_compressed(
        out_npz,
        X=X, y=w["y"], subject_ids=w["subject_ids"], mask=w["mask"],
        window_index=w["window_index"], min_before_anchor=w["min_before_anchor"],
        min_before_birth=w["min_before_birth"], ph=w["ph"], delivery_type=w["delivery_type"],
        recording_min=w["recording_min"], rec_fhr_missing_frac=w["rec_fhr_missing_frac"],
        channels=np.array(channels),
        fs=store.fs, window_minutes=args.window_min,
    )
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
        json.dump({**vars(args), "n_windows_total": int(len(w["y"])), "n_records": n_rec,
                   "n_records_dropped": len(dropped), "channels": channels,
                   "X_shape": list(X.shape)}, f, indent=2)

    rec_y = (store.meta["ph"].astype(float) < args.ph_threshold)
    dropped_pos = int(rec_y.loc[dropped].sum()) if dropped else 0
    print(f"\nWindows: {len(w['y'])} from {n_rec} records "
          f"({len(dropped)} records had no window with >= {args.min_valid_frac:.0%} FHR observed: "
          f"{dropped_pos} of {int(rec_y.sum())} positives, "
          f"{len(dropped) - dropped_pos} of {int((~rec_y).sum())} negatives)")
    print(f"Positive windows: {int(w['y'].sum())} | X shape {X.shape} | channels {channels}")
    mbb = w["min_before_birth"]
    print(f"Minutes before birth at window end: median {np.median(mbb):.1f} "
          f"(positives {np.median(mbb[w['y'] == 1]):.1f}, negatives {np.median(mbb[w['y'] == 0]):.1f})")
    print(f"\nWrote {out_npz}")
    print(f"Train with: python run.py --config configs/ctu_uhb_cnn_lstm.yaml "
          f"--root_path {args.out_dir}/ --data_path {os.path.basename(out_npz)} "
          f"--seq_len {X.shape[-1]} --in_channels {X.shape[1]} --enc_in {X.shape[1]}")


if __name__ == "__main__":
    main()
