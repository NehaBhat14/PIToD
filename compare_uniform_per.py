"""
compare_uniform_per.py
======================
Load progress.txt logs from Uniform and PER runs (H1 experiment on Hopper-v2)
and produce three figures:

  figure/H1_learning_curves.pdf   -- TestEpRet vs env steps, mean ± 1 std
  figure/H1_auc_final.pdf         -- AUC and final-performance bar charts
  figure/H1_sps.pdf               -- Steps-per-second (H3 context)

Column names produced by EpochLogger in dynamic-main-TH.py:
  AverageTestEpRet, StdTestEpRet   (from log_tabular('TestEpRet', with_min_and_max=True))
  TotalEnvInteracts                (direct scalar)
  SPS                              (direct scalar)
  ReplayMode                       (direct scalar, string)

Usage:
  python compare_uniform_per.py                    # looks in runs/H1/
  python compare_uniform_per.py --data_dir runs/H1 --out_dir figure
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving PDFs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Styling (mirrors plot_main_results_pitod.py) ──────────────────────────────
sns.set(style="white")
sns.set_context("paper", 2.0, {"lines.linewidth": 2.5})
PALETTE = {"uniform": "#4878CF", "per": "#E84646"}   # blue, red
LABELS  = {"uniform": "Uniform", "per": "PER"}
ALPHA_BAND = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_mode_seed(folder_name: str, env: str) -> tuple:
    """
    Extract (mode, seed) from a subfolder name produced by setup_logger_kwargs.

    Example:  'redq_sac_Hopper-v2_uniform_s0'  →  ('uniform', 0)
              'redq_sac_Hopper-v2_per_s1'       →  ('per', 1)

    The structure is: redq_sac_{env}_{mode}_s{seed}
    """
    prefix = f"redq_sac_{env}_"
    if not folder_name.startswith(prefix):
        return None, None
    remainder = folder_name[len(prefix):]   # e.g. "uniform_s0"
    # split from the right on '_s' to get (mode, seed_str)
    parts = remainder.rsplit("_s", maxsplit=1)
    if len(parts) != 2:
        return None, None
    mode, seed_str = parts
    try:
        seed = int(seed_str)
    except ValueError:
        return None, None
    return mode, seed


def load_runs(data_dir: str, env: str, modes: list) -> pd.DataFrame:
    """
    Scan data_dir for progress.txt files and return a single DataFrame
    with extra columns: mode, seed.
    """
    pattern = os.path.join(data_dir, "**", "progress.txt")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"[ERROR] No progress.txt files found under '{data_dir}'.")
        print("        Run ./run_uniform_per.sh first.")
        sys.exit(1)

    dfs = []
    for path in sorted(files):
        folder = os.path.basename(os.path.dirname(path))
        mode, seed = parse_mode_seed(folder, env)
        if mode is None or mode not in modes:
            continue
        df = pd.read_csv(path, sep="\t")
        df["mode"] = mode
        df["seed"] = seed
        dfs.append(df)

    if not dfs:
        print(f"[ERROR] Found progress.txt files but none matched modes={modes} for env={env}.")
        print("        Check that -info H1 was used when running dynamic-main-TH.py.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def smoothed_mean_std(df: pd.DataFrame, x_col: str, y_col: str,
                      mode: str, n_interp: int = 200):
    """
    For a given mode, pivot seeds into columns, interpolate onto a common
    x-grid, and return (x, mean, std).
    """
    grp = df[df["mode"] == mode]
    seeds = sorted(grp["seed"].unique())

    # build a common x-grid from the union of all x values
    x_min = grp[x_col].min()
    x_max = grp[x_col].max()
    x_grid = np.linspace(x_min, x_max, n_interp)

    interped = []
    for seed in seeds:
        s = grp[grp["seed"] == seed].sort_values(x_col)
        y_interp = np.interp(x_grid, s[x_col].values, s[y_col].values)
        interped.append(y_interp)

    interped = np.array(interped)   # shape: (n_seeds, n_interp)
    mean = interped.mean(axis=0)
    std  = interped.std(axis=0)
    return x_grid, mean, std


# ── Figure 1: Learning curves ─────────────────────────────────────────────────

def plot_learning_curves(df: pd.DataFrame, modes: list, out_path: str,
                         x_col="TotalEnvInteracts",
                         y_col="AverageTestEpRet") -> None:
    """
    Plot mean test return ± 1 std for each mode.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for mode in modes:
        if mode not in df["mode"].values:
            print(f"  [WARN] no data for mode='{mode}', skipping.")
            continue
        x, mean, std = smoothed_mean_std(df, x_col, y_col, mode)
        color = PALETTE[mode]
        ax.plot(x, mean, color=color, label=LABELS[mode])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=ALPHA_BAND)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Test Return")
    ax.set_title("Hopper-v2 — Uniform vs PER (5 seeds)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}K"))
    ax.legend(frameon=True)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: AUC and final performance bar charts ───────────────────────────

def compute_summary(df: pd.DataFrame, modes: list,
                    x_col="TotalEnvInteracts",
                    y_col="AverageTestEpRet",
                    final_frac: float = 0.1) -> pd.DataFrame:
    """
    For each mode and seed compute:
      - AUC  (trapezoid under the return curve, normalised by x range)
      - final_perf  (mean return over the last final_frac of training)
    Returns a DataFrame with columns: mode, seed, auc, final_perf
    """
    rows = []
    for mode in modes:
        grp = df[df["mode"] == mode]
        for seed in sorted(grp["seed"].unique()):
            s = grp[grp["seed"] == seed].sort_values(x_col)
            x = s[x_col].values.astype(float)
            y = s[y_col].values.astype(float)
            # AUC (normalised so units are "return")
            auc = np.trapz(y, x) / (x[-1] - x[0]) if x[-1] > x[0] else np.nan
            # Final performance: mean over the last final_frac of steps
            cutoff = x[0] + (x[-1] - x[0]) * (1.0 - final_frac)
            final_perf = y[x >= cutoff].mean() if (x >= cutoff).any() else y[-1]
            rows.append({"mode": mode, "seed": seed,
                         "auc": auc, "final_perf": final_perf})
    return pd.DataFrame(rows)


def plot_auc_final(summary: pd.DataFrame, modes: list, out_path: str) -> None:
    """
    Two-panel bar chart: left = AUC, right = final performance.
    Error bars = 1 std across seeds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, metric, title in zip(
        axes,
        ["auc", "final_perf"],
        ["AUC (normalised)", "Final Performance\n(last 10% of training)"]
    ):
        means = [summary[summary["mode"] == m][metric].mean() for m in modes]
        stds  = [summary[summary["mode"] == m][metric].std()  for m in modes]
        colors = [PALETTE[m] for m in modes]
        labels = [LABELS[m]  for m in modes]

        bars = ax.bar(labels, means, yerr=stds, color=colors,
                      capsize=6, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Return")
        sns.despine(ax=ax)

        # annotate bar tops
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + max(means) * 0.01,
                    f"{mean:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Hopper-v2 — Uniform vs PER (5 seeds)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: SPS comparison ──────────────────────────────────────────────────

def plot_sps(df: pd.DataFrame, modes: list, out_path: str) -> None:
    """
    Bar chart of median SPS per mode (proxy for wall-clock cost, H3 context).
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    medians = []
    stds    = []
    for mode in modes:
        grp = df[df["mode"] == mode]["SPS"].dropna()
        medians.append(grp.median())
        stds.append(grp.std())

    bars = ax.bar([LABELS[m] for m in modes], medians, yerr=stds,
                  color=[PALETTE[m] for m in modes],
                  capsize=6, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Steps per Second (SPS)")
    ax.set_title("Throughput — Uniform vs PER\n(lower SPS = higher overhead)")
    sns.despine(ax=ax)

    for bar, med in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2,
                med + max(medians) * 0.01,
                f"{med:.0f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Console summary table ─────────────────────────────────────────────────────

def print_summary_table(summary: pd.DataFrame, modes: list) -> None:
    print("\n" + "=" * 52)
    print(f"  {'Mode':<12} {'AUC mean':>12} {'AUC std':>10} {'Final mean':>12} {'Final std':>10}")
    print("-" * 52)
    for mode in modes:
        grp = summary[summary["mode"] == mode]
        print(f"  {LABELS[mode]:<12}"
              f"  {grp['auc'].mean():>10.1f}"
              f"  {grp['auc'].std():>10.1f}"
              f"  {grp['final_perf'].mean():>10.1f}"
              f"  {grp['final_perf'].std():>10.1f}")
    print("=" * 52 + "\n")


def print_sps_table(df: pd.DataFrame, modes: list) -> None:
    print("  SPS (Steps per Second):")
    print(f"  {'Mode':<12} {'Median':>10} {'Std':>10}")
    print("  " + "-" * 34)
    for mode in modes:
        grp = df[df["mode"] == mode]["SPS"].dropna()
        print(f"  {LABELS[mode]:<12}  {grp.median():>10.1f}  {grp.std():>10.1f}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Uniform vs PER results")
    parser.add_argument("--data_dir", default="runs/H1",
                        help="Directory containing run subdirectories (default: runs/H1)")
    parser.add_argument("--env", default="Hopper-v2",
                        help="Environment name used in folder names (default: Hopper-v2)")
    parser.add_argument("--out_dir", default="figure",
                        help="Directory to write PDF figures into (default: figure)")
    parser.add_argument("--modes", nargs="+", default=["uniform", "per"],
                        help="Modes to compare (default: uniform per)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading runs from: {args.data_dir}")
    df = load_runs(args.data_dir, args.env, args.modes)

    # Sanity: list what we found
    found = df.groupby(["mode", "seed"]).size().reset_index(name="epochs")
    print("\nRuns found:")
    for _, row in found.iterrows():
        print(f"  mode={row['mode']}  seed={row['seed']}  epochs={row['epochs']}")

    # Check the return column exists (logger uses 'Average' prefix)
    ret_col = "AverageTestEpRet"
    if ret_col not in df.columns:
        # Older runs may use 'TestEpRet' directly — fall back gracefully
        if "TestEpRet" in df.columns:
            ret_col = "TestEpRet"
            print(f"\n  [INFO] Using column 'TestEpRet' (no 'Average' prefix found).")
        else:
            available = [c for c in df.columns if "Test" in c or "Ret" in c]
            print(f"\n  [ERROR] Neither 'AverageTestEpRet' nor 'TestEpRet' found.")
            print(f"          Available return-related columns: {available}")
            sys.exit(1)

    print(f"\nGenerating figures → {args.out_dir}/")

    # Figure 1: learning curves
    plot_learning_curves(
        df, args.modes,
        out_path=os.path.join(args.out_dir, "H1_learning_curves.pdf"),
        y_col=ret_col,
    )

    # Figure 2: AUC + final performance
    summary = compute_summary(df, args.modes, y_col=ret_col)
    plot_auc_final(
        summary, args.modes,
        out_path=os.path.join(args.out_dir, "H1_auc_final.pdf"),
    )

    # Figure 3: SPS
    if "SPS" in df.columns:
        plot_sps(df, args.modes,
                 out_path=os.path.join(args.out_dir, "H1_sps.pdf"))
    else:
        print("  [WARN] SPS column not found, skipping throughput plot.")

    # Console tables
    print_summary_table(summary, args.modes)
    if "SPS" in df.columns:
        print_sps_table(df, args.modes)

    print("Done.")


if __name__ == "__main__":
    main()
