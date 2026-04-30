#!/usr/bin/env python3
"""
Compare Dynamic PIToD screening runs and emit learning / diagnostic plots.

Example:
    python analyze_dynamic_pitod_study.py \
        --env Hopper-v2 \
        --spec uniform=screen_uniform:uniform \
        --spec static=screen_static:static_pitod \
        --spec dyn_base=screen_dyn_base:dynamic_pitod \
        --spec dyn_early=screen_dyn_early:dynamic_pitod \
        --analysis-seed 0 \
        --return-threshold 1000 \
        --output-dir figure/dynamic_screen
"""

from __future__ import annotations

import argparse
import bz2
import glob
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DYNAMIC_COLUMNS = {
    "score_mean": "AverageDynPIToD/ScoreMean",
    "score_std": "AverageDynPIToD/ScoreStd",
    "epsilon": "AverageDynPIToD/Epsilon",
    "num_active": "AverageDynPIToD/NumActive",
    "num_evicted": "AverageDynPIToD/NumEvicted",
    "newly_evicted": "AverageDynPIToD/NewlyEvicted",
    "buffer_active_frac": "AverageDynPIToD/BufferActiveFrac",
    "refresh_wallclock": "AverageDynPIToD/RefreshWallclock",
    "num_refreshed": "AverageDynPIToD/NumRefreshed",
    "schedule_k": "AverageDynPIToD/ScheduleK",
    "schedule_b": "AverageDynPIToD/ScheduleB",
    "pruning_enabled": "AverageDynPIToD/PruningEnabled",
}


@dataclass(frozen=True)
class Spec:
    label: str
    top_level_dir: str
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Dynamic PIToD comparison studies under runs/.")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing experiment top-level dirs.")
    parser.add_argument("--env", required=True, help="Environment name, e.g. Hopper-v2.")
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="Comparison spec in the form label=top_level_dir:replay_mode. Repeat per curve.",
    )
    parser.add_argument("--analysis-seed", type=int, default=0, help="Seed to use for H2 plots, if present.")
    parser.add_argument("--return-threshold", type=float, default=1000.0, help="Return threshold for crossing stats.")
    parser.add_argument("--output-dir", default="figure/dynamic_study", help="Where to save figures/tables.")
    return parser.parse_args()


def parse_spec(text: str) -> Spec:
    if "=" not in text or ":" not in text:
        raise ValueError(f"Invalid --spec {text!r}; expected label=top_level_dir:replay_mode")
    label, rest = text.split("=", 1)
    top_level_dir, mode = rest.rsplit(":", 1)
    label = label.strip()
    top_level_dir = top_level_dir.strip()
    mode = mode.strip()
    if not label or not top_level_dir or not mode:
        raise ValueError(f"Invalid --spec {text!r}; empty label/top-level/mode")
    return Spec(label=label, top_level_dir=top_level_dir, mode=mode)


def trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def find_run_dirs(runs_root: Path, env_name: str, spec: Spec) -> List[Path]:
    pattern = (
        runs_root
        / spec.top_level_dir
        / f"redq_sac_{env_name}_{spec.mode}"
        / f"redq_sac_{env_name}_{spec.mode}_s*"
    )
    return sorted(Path(p) for p in glob.glob(str(pattern)))


def extract_seed(path: Path) -> int:
    match = re.search(r"_s(\d+)$", path.name)
    if not match:
        raise ValueError(f"Could not parse seed from run dir {path}")
    return int(match.group(1))


def load_progress(path: Path) -> pd.DataFrame:
    df = pd.read_table(path)
    required = {"Epoch", "Time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    if "AverageTestEpRet" not in df.columns:
        if "TestEpRet" in df.columns:
            df["AverageTestEpRet"] = df["TestEpRet"]
        else:
            raise ValueError(f"Missing AverageTestEpRet in {path}")
    return df


def first_crossing(series: pd.Series, x: pd.Series, threshold: float) -> float:
    crossed = series >= threshold
    if not bool(crossed.any()):
        return float("nan")
    idx = int(np.argmax(crossed.to_numpy()))
    return float(x.iloc[idx])


def summarize_runs(
    runs_root: Path,
    env_name: str,
    specs: Iterable[Spec],
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, int], Path]]:
    per_seed_rows: List[Dict[str, float]] = []
    curve_rows: List[pd.DataFrame] = []
    run_lookup: Dict[Tuple[str, int], Path] = {}

    for spec in specs:
        run_dirs = find_run_dirs(runs_root, env_name, spec)
        if not run_dirs:
            raise FileNotFoundError(
                f"No runs found for spec {spec.label!r} under {runs_root/spec.top_level_dir}"
            )
        for run_dir in run_dirs:
            seed = extract_seed(run_dir)
            progress = load_progress(run_dir / "progress.txt").copy()
            progress["label"] = spec.label
            progress["mode"] = spec.mode
            progress["seed"] = seed
            curve_rows.append(progress)
            run_lookup[(spec.label, seed)] = run_dir
            per_seed_rows.append(
                {
                    "label": spec.label,
                    "mode": spec.mode,
                    "seed": seed,
                    "final_return": float(progress["AverageTestEpRet"].iloc[-1]),
                    "best_return": float(progress["AverageTestEpRet"].max()),
                    "auc_return": trapz_auc(
                        progress["Epoch"].to_numpy(),
                        progress["AverageTestEpRet"].to_numpy(),
                    ),
                    "final_time_s": float(progress["Time"].iloc[-1]),
                    "threshold_epoch": first_crossing(
                        progress["AverageTestEpRet"], progress["Epoch"], threshold
                    ),
                    "threshold_time_s": first_crossing(
                        progress["AverageTestEpRet"], progress["Time"], threshold
                    ),
                }
            )

    curves = pd.concat(curve_rows, ignore_index=True)
    per_seed = pd.DataFrame(per_seed_rows).sort_values(["label", "seed"]).reset_index(drop=True)
    aggregate = (
        per_seed.groupby(["label", "mode"], as_index=False)
        .agg(
            final_mean=("final_return", "mean"),
            final_std=("final_return", "std"),
            best_mean=("best_return", "mean"),
            best_std=("best_return", "std"),
            auc_mean=("auc_return", "mean"),
            auc_std=("auc_return", "std"),
            time_mean_s=("final_time_s", "mean"),
            time_std_s=("final_time_s", "std"),
            threshold_epoch_mean=("threshold_epoch", "mean"),
            threshold_epoch_std=("threshold_epoch", "std"),
            threshold_time_mean_s=("threshold_time_s", "mean"),
            threshold_time_std_s=("threshold_time_s", "std"),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )
    return curves, per_seed, aggregate, run_lookup


def print_summary(per_seed: pd.DataFrame, aggregate: pd.DataFrame, threshold: float) -> None:
    print("\n== Per-seed summary ==")
    for row in per_seed.itertuples(index=False):
        print(
            f"{row.label:16s} seed={row.seed} final={row.final_return:8.1f} "
            f"best={row.best_return:8.1f} auc={row.auc_return:10.1f} "
            f"time={row.final_time_s:8.0f}s cross@{threshold:.0f}="
            f"{'never' if np.isnan(row.threshold_epoch) else f'epoch {row.threshold_epoch:.0f} / {row.threshold_time_s:.0f}s'}"
        )

    print("\n== Aggregate summary ==")
    for row in aggregate.itertuples(index=False):
        crossing = (
            "never"
            if np.isnan(row.threshold_epoch_mean)
            else f"epoch {row.threshold_epoch_mean:.1f} +/- {row.threshold_epoch_std:.1f}, "
            f"{row.threshold_time_mean_s:.0f}s +/- {row.threshold_time_std_s:.0f}s"
        )
        print(
            f"{row.label:16s} final={row.final_mean:8.1f} +/- {row.final_std:6.1f} "
            f"best={row.best_mean:8.1f} +/- {row.best_std:6.1f} "
            f"auc={row.auc_mean:10.1f} +/- {row.auc_std:7.1f} "
            f"time={row.time_mean_s:8.0f}s +/- {row.time_std_s:5.0f}s "
            f"threshold={crossing}"
        )


def _plot_mean_std(ax, df: pd.DataFrame, x_col: str, y_col: str, label: str) -> None:
    grouped = (
        df.groupby(x_col, as_index=False)
        .agg(y_mean=(y_col, "mean"), y_std=(y_col, "std"))
        .sort_values(x_col)
    )
    x = grouped[x_col].to_numpy()
    y_mean = grouped["y_mean"].to_numpy()
    y_std = np.nan_to_num(grouped["y_std"].to_numpy(), nan=0.0)
    ax.plot(x, y_mean, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.18)


def save_learning_plots(curves: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label in sorted(curves["label"].unique()):
        subset = curves[curves["label"] == label]
        _plot_mean_std(axes[0], subset, "Epoch", "AverageTestEpRet", label)
        _plot_mean_std(axes[1], subset, "Epoch", "Time", label)

    axes[0].set_title("AverageTestEpRet vs epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("return")
    axes[0].legend()
    axes[1].set_title("Wall-clock vs epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_and_time.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for label in sorted(curves["label"].unique()):
        subset = curves[curves["label"] == label]
        grouped = (
            subset.groupby("Epoch", as_index=False)
            .agg(return_mean=("AverageTestEpRet", "mean"), time_mean=("Time", "mean"))
            .sort_values("Epoch")
        )
        ax.plot(grouped["time_mean"], grouped["return_mean"], label=label)
    ax.set_title("Return vs wall-clock")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("AverageTestEpRet")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "return_vs_wallclock.png", dpi=140)
    plt.close(fig)


def save_dynamic_diagnostics(curves: pd.DataFrame, output_dir: Path) -> None:
    labels_with_dynamic = [
        label
        for label in sorted(curves["label"].unique())
        if DYNAMIC_COLUMNS["score_mean"] in curves.columns
        and curves[curves["label"] == label][DYNAMIC_COLUMNS["score_mean"]].notna().any()
    ]
    if not labels_with_dynamic:
        return

    panels = [
        ("Score mean", DYNAMIC_COLUMNS["score_mean"]),
        ("Score std", DYNAMIC_COLUMNS["score_std"]),
        ("Active fraction", DYNAMIC_COLUMNS["buffer_active_frac"]),
        ("Active groups", DYNAMIC_COLUMNS["num_active"]),
        ("Evicted groups", DYNAMIC_COLUMNS["num_evicted"]),
        ("Refresh wall-clock", DYNAMIC_COLUMNS["refresh_wallclock"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (title, col) in zip(axes.flatten(), panels):
        plotted = False
        for label in labels_with_dynamic:
            subset = curves[curves["label"] == label]
            if col not in subset.columns or not subset[col].notna().any():
                continue
            _plot_mean_std(ax, subset.dropna(subset=[col]), "Epoch", col, label)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_diagnostics.png", dpi=140)
    plt.close(fig)


def save_h2_plots(run_lookup: Dict[Tuple[str, int], Path], analysis_seed: int, output_dir: Path) -> None:
    for (label, seed), run_dir in sorted(run_lookup.items()):
        if seed != analysis_seed:
            continue
        path = run_dir / "h2_dynamic_scores.bz2"
        if not path.is_file():
            continue
        with bz2.BZ2File(path, "rb") as fh:
            payload = pickle.load(fh)
        records = payload.get("records", [])
        if not records:
            continue
        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for gid in sorted(df["group_id"].unique()):
            subset = df[df["group_id"] == gid].sort_values("env_step")
            ax.plot(subset["env_step"], subset["score"], label=f"group {gid}")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_title(f"H2 score traces: {label} seed {seed}")
        ax.set_xlabel("env_step")
        ax.set_ylabel("group score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"h2_{label}_seed{seed}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    specs = [parse_spec(text) for text in args.spec]
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves, per_seed, aggregate, run_lookup = summarize_runs(
        runs_root=runs_root,
        env_name=args.env,
        specs=specs,
        threshold=args.return_threshold,
    )
    per_seed.to_csv(output_dir / "per_seed_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)
    print_summary(per_seed, aggregate, args.return_threshold)

    save_learning_plots(curves, output_dir)
    save_dynamic_diagnostics(curves, output_dir)
    save_h2_plots(run_lookup, args.analysis_seed, output_dir)

    print(f"\nSaved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
