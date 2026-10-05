"""
Audit the raw CTG export files before any preprocessing.

For every .xls / .xlsx recording in a folder this writes one row to a CSV with:
  - file format, sheet count, row count, and whether the sheet stops exactly at
    the legacy .xls row limit (65,536 rows), which means the export was truncated
  - the signal columns present and where the header row sits
  - recording duration from the TM column (ms since previous sample), including gaps
  - hours from recording start to birth, and from the END of the recording to birth
  - FHR signal loss, signal-quality flags, and transducer modes
  - the fill colour of the Study_ID cell (currently the source of the label)

No signal values are written, only per-file summaries.

Usage:
    python scripts/audit_raw_data.py --raw_dir /path/to/raw_files --out raw_audit.csv
"""

import argparse
import csv
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

XLS_ROW_LIMIT = 65536
FS_HZ = 4
SAMPLE_MS = 1000 // FS_HZ


def find_header_row(raw: pd.DataFrame, max_scan: int = 60) -> int:
    """Row whose first cell is 'TM' and which names FHR1 and TOCO."""
    for i in range(min(max_scan, len(raw))):
        vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if vals and vals[0] == "TM" and "FHR1" in vals and "TOCO" in vals:
            return i
    return -1


def find_time(raw: pd.DataFrame, key: str, max_scan: int = 20):
    """Parse '# CTG start time : ...' / '# birth time : ...' from the header block."""
    pat = re.compile(key + r"\s*:\s*(.+)", re.IGNORECASE)
    for i in range(min(max_scan, len(raw))):
        for v in raw.iloc[i].tolist():
            m = pat.search(str(v))
            if m:
                return pd.to_datetime(m.group(1).strip(), errors="coerce")
    return pd.NaT


def find_study_id(raw: pd.DataFrame, max_scan: int = 10) -> str:
    for i in range(min(max_scan, len(raw) - 1)):
        if str(raw.iat[i, 0]).strip() == "Study_ID":
            return str(raw.iat[i + 1, 0]).strip()
    return ""


def study_id_cell_colour(path: str) -> str:
    """Fill colour of the Study_ID value cell (Excel row 4, column A)."""
    try:
        if path.lower().endswith(".xlsx"):
            import openpyxl
            ws = openpyxl.load_workbook(path).worksheets[0]
            return str(ws.cell(row=4, column=1).fill.fgColor.rgb)
        import xlrd
        book = xlrd.open_workbook(path, formatting_info=True)
        sh = book.sheet_by_index(0)
        xf = book.xf_list[sh.cell_xf_index(3, 0)]
        rgb = book.colour_map.get(xf.background.pattern_colour_index)
        return "none" if rgb is None else "%02X%02X%02X" % rgb
    except Exception as e:  # colour is best-effort; never fail the audit on it
        return f"error:{type(e).__name__}"


def frac(mask) -> float:
    return float(np.mean(mask)) if len(mask) else float("nan")


def audit_file(path: str, read_colour: bool) -> dict:
    row = {"file": os.path.basename(path), "ext": os.path.splitext(path)[1].lower()}
    xl = pd.ExcelFile(path)
    row["n_sheets"] = len(xl.sheet_names)
    raw = xl.parse(xl.sheet_names[0], header=None)
    row["sheet_rows"] = len(raw)
    row["at_xls_row_limit"] = len(raw) == XLS_ROW_LIMIT
    row["study_id"] = find_study_id(raw)

    h = find_header_row(raw)
    row["header_row"] = h
    if h < 0:
        row["error"] = "no TM/FHR1/TOCO header row found"
        return row

    cols = [str(c).strip() for c in raw.iloc[h].tolist()]
    d = raw.iloc[h + 1:].copy()
    d.columns = cols
    d = d.loc[:, [c for c in cols if c and c != "nan"]].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["TM", "FHR1"]).reset_index(drop=True)
    row["columns"] = " ".join(d.columns)
    row["n_samples"] = len(d)

    # --- timing ---
    tm = d["TM"].to_numpy()
    gaps = tm[1:][tm[1:] > SAMPLE_MS]
    row["first_tm_s"] = round(float(tm[0]) / 1000, 1) if len(tm) else np.nan
    row["n_gaps"] = int(len(gaps))
    row["gap_total_min"] = round(float((gaps - SAMPLE_MS).sum()) / 60000, 2)
    row["samples_h"] = round(len(d) / FS_HZ / 3600, 3)
    elapsed_h = float(tm.sum()) / 3.6e6
    row["elapsed_h"] = round(elapsed_h, 3)

    start, birth = find_time(raw, "CTG start time"), find_time(raw, "birth time")
    row["has_start_time"], row["has_birth_time"] = pd.notna(start), pd.notna(birth)
    if pd.notna(start) and pd.notna(birth):
        start_to_birth_h = (birth - start).total_seconds() / 3600
        row["start_to_birth_h"] = round(start_to_birth_h, 2)
        # > 0: the recording stops this many hours BEFORE birth. < 0: it runs past birth.
        row["end_to_birth_h"] = round(start_to_birth_h - elapsed_h, 2)

    # --- signal quality ---
    fhr = d["FHR1"].to_numpy()
    valid = fhr > 0
    row["fhr_zero_frac"] = round(frac(~valid), 3)
    if valid.any():
        last_valid = int(np.flatnonzero(valid)[-1])
        row["trailing_zero_min"] = round((len(fhr) - 1 - last_valid) / FS_HZ / 60, 1)
        row["fhr_median"] = float(np.median(fhr[valid]))
    for minutes in (30, 60):
        n = minutes * 60 * FS_HZ
        row[f"valid_frac_last_{minutes}min"] = round(frac(valid[-n:]), 3)
    if "SIG1" in d:
        for name, code in (("red", 0), ("yellow", 1), ("green", 2)):
            row[f"sig1_{name}_frac"] = round(frac(d["SIG1"] == code), 3)
    if "FHR1MODE" in d:
        row["fhr_us_frac"] = round(frac(d["FHR1MODE"] == 1), 3)
        row["fhr_scalp_ecg_frac"] = round(frac(d["FHR1MODE"] == 2), 3)
    if "TOCOMODE" in d:
        row["toco_external_frac"] = round(frac(d["TOCOMODE"] == 9), 3)
        row["toco_iup_frac"] = round(frac(d["TOCOMODE"] == 8), 3)
    row["toco_zero_frac"] = round(frac(d["TOCO"] == 0), 3) if "TOCO" in d else np.nan
    row["has_tocosig"] = "TOCOSIG" in d
    row["has_mhr"] = bool("MHR" in d and (d["MHR"] > 0).any())

    if read_colour:
        row["study_id_cell_colour"] = study_id_cell_colour(path)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", required=True, help="Folder with the raw .xls / .xlsx files")
    ap.add_argument("--out", default="raw_audit.csv")
    ap.add_argument("--no_colour", action="store_true", help="Skip reading the cell colour (faster)")
    args = ap.parse_args()

    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(args.raw_dir)
        for f in names
        if f.lower().endswith((".xls", ".xlsx")) and not f.startswith("~$")
    )
    if not files:
        sys.exit(f"No .xls/.xlsx files under {args.raw_dir}")

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(path)}", flush=True)
        try:
            rows.append(audit_file(path, read_colour=not args.no_colour))
        except Exception as e:
            rows.append({"file": os.path.basename(path), "error": f"{type(e).__name__}: {e}"})

    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    print(f"\nWrote {args.out}  ({len(df)} files)")
    if "error" in df:
        print(f"  files with errors:                {int(df['error'].notna().sum())}")
    if "at_xls_row_limit" in df:
        print(f"  stopped at the .xls row limit:    {int(df['at_xls_row_limit'].fillna(False).sum())}")
    if "end_to_birth_h" in df:
        e = df["end_to_birth_h"].dropna()
        print(f"  recording ends >1 h before birth: {int((e > 1).sum())} of {len(e)}")
        print(f"  recording runs past birth:        {int((e < -0.25).sum())} of {len(e)}")
        print(f"  end-to-birth hours (median):      {e.median():.2f}")
    if "study_id_cell_colour" in df:
        print("  Study_ID cell colours:", df["study_id_cell_colour"].value_counts(dropna=False).to_dict())
    dup = df["study_id"].dropna()
    dup = dup[dup != ""]
    print(f"  duplicated study IDs:             {int(dup.duplicated().sum())}")


if __name__ == "__main__":
    main()
