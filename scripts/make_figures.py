"""
Figures for the README, drawn from the evaluation tables and out-of-fold predictions.

  docs/figures/auroc_overview.png    AUROC [95% CI] of every model family, both pH thresholds
  docs/figures/missingness_corr.png  how strongly each model's scores follow signal loss

Also copies the evaluation tables to docs/tables/ so the numbers in the README are
versioned with the code.

Usage (after chtc/collect_and_evaluate.sh or scripts/evaluate.py has written table.csv):
  python scripts/make_figures.py
"""

import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_run  # noqa: E402

STEM = "ctu_uhb_stage1_end_w30m_n1_v0.2_ph{}"
THRESHOLDS = ["7.05", "7.15"]

# family -> color (validated categorical slots 1-3 plus a neutral for the shortcut baselines)
FAMILY = {
    "shortcut": ("Shortcut baseline (no heart-rate pattern)", "#8a8984"),
    "scratch": ("Trained from scratch", "#eb6834"),
    "probe": ("Pretrained encoder, frozen + linear probe", "#2a78d6"),
    "combo": ("Combination (exploratory)", "#1baf7a"),
}
ROWS = [  # (run name in table.csv, label, family) top to bottom
    ("missingness", "Signal loss", "shortcut"),
    ("missingness_length", "Signal loss + recording length", "shortcut"),
    ("cnn_lstm_wbce", "CNN-LSTM", "scratch"),
    ("probe_moment-large_meanmax", "MOMENT-large", "probe"),
    ("probe_moment-base_meanmax", "MOMENT-base", "probe"),
    ("probe_mantisv2_meanmax", "MantisV2", "probe"),
    ("probe_mantis_meanmax", "Mantis", "probe"),
    ("probe_mantis_meanmax_sc", "Mantis + shortcut features (one probe)", "combo"),
    ("combo_mantis+shortcut", "Mantis and shortcut baseline, rank-averaged", "combo"),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def parse(cell: str):
    """'0.705 [0.633, 0.768]' -> (0.705, 0.633, 0.768)"""
    head, ci = cell.split(" [")
    lo, hi = ci.rstrip("]").split(", ")
    return float(head), float(lo), float(hi)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def auroc_overview(tables: dict, out: str, subset: str = "all"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    ypos = np.arange(len(ROWS))[::-1]
    for ax, t in zip(axes, THRESHOLDS):
        tab = tables[t].query("subset == @subset").set_index("run")
        style(ax)
        ref = parse(tab.loc["missingness_length", "AUROC"])[0]
        ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (2, 3)))
        ax.axvline(ref, color=FAMILY["shortcut"][1], linewidth=1, linestyle=(0, (5, 3)))
        for y, (run, label, fam) in zip(ypos, ROWS):
            m, lo, hi = parse(tab.loc[run, "AUROC"])
            color = FAMILY[fam][1]
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(m, y, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(hi + 0.008, y, f"{m:.2f}", va="center", ha="left", fontsize=9, color=INK)
        n_pos = int(tab["positives"].iloc[0])
        n = int(tab["subjects"].iloc[0])
        ax.set_title(f"pH < {t}  ({n_pos} of {n} recordings)", loc="left", fontsize=11, color=INK)
        ax.set_xlim(0.40, 0.88)
        ax.set_xlabel("AUROC (mean of 5 x 5-fold CV, 95% bootstrap CI)", fontsize=9, color=INK2)
    axes[0].set_yticks(ypos, [r[1] for r in ROWS], fontsize=9, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=8, label=l)
               for l, c in FAMILY.values()]
    handles.append(plt.Line2D([], [], color=FAMILY["shortcut"][1], linestyle=(0, (5, 3)),
                              label="Best shortcut baseline"))
    handles.append(plt.Line2D([], [], color=INK2, linestyle=(0, (2, 3)), label="Chance (0.5)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def missingness_corr(out: str, t: str = "7.05"):
    stem = STEM.format(t)
    d = np.load(f"dataset/{stem}.npz", allow_pickle=False)
    miss = d["rec_fhr_missing_frac"]
    rows = [r for r in ROWS if r[0] in ("missingness_length", "cnn_lstm_wbce",
                                        "probe_moment-base_meanmax", "probe_mantis_meanmax")]
    vals = []
    for run, label, fam in rows:
        path = f"results/{stem}/baselines/{run}" if fam == "shortcut" else f"results/{stem}/{run}"
        df = load_run(path)
        P = df.pivot_table(index="window_index", columns="repeat", values="prob")
        rho = np.mean([spearmanr(P[r].to_numpy(), miss[P.index.to_numpy()]).statistic for r in P])
        vals.append((label, fam, rho))
    fig, ax = plt.subplots(figsize=(7, 2.6), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(vals))[::-1]
    for yi, (label, fam, rho) in zip(y, vals):
        ax.barh(yi, rho, height=0.55, color=FAMILY[fam][1])
        ax.text(rho + 0.015, yi, f"{rho:+.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [v[0] for v in vals], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Spearman correlation of predicted risk with FHR signal loss (pH < {t})",
                  fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="docs")
    args = ap.parse_args()
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)
    os.makedirs(f"{args.out_dir}/tables", exist_ok=True)
    tables = {}
    for t in THRESHOLDS:
        src = f"results/{STEM.format(t)}/table.csv"
        tables[t] = pd.read_csv(src)
        shutil.copy(src, f"{args.out_dir}/tables/table_ph{t}.csv")
    auroc_overview(tables, f"{args.out_dir}/figures/auroc_overview.png")
    missingness_corr(f"{args.out_dir}/figures/missingness_corr.png")
    print(f"Wrote {args.out_dir}/figures/ and {args.out_dir}/tables/")


if __name__ == "__main__":
    main()
