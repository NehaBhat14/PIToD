"""
Compare static_pitod vs dynamic_pitod for screen_dyn_early_vs_static.

Dynamic data comes from the console log (progress.txt was empty locally).
Static data comes from the local progress.txt files.

Usage:
  python plot_screen_comparison.py \
      --log newPitod_runLogs.md \
      --run_dir runs/screen_dyn_early_vs_static \
      --out_dir figure/screen_comparison
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


# ── 1. Parse dynamic metrics from console log ──────────────────────────────

def parse_log(log_path: str) -> pd.DataFrame:
    row_re = re.compile(r'^\|\s+(.+?)\s+\|\s+(.+?)\s+\|$')
    sep_re = re.compile(r'^-{10,}')
    records, current = [], {}
    with open(log_path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip()
            if sep_re.match(line):
                if 'Epoch' in current:
                    records.append(dict(current))
                current = {}
                continue
            m = row_re.match(line)
            if m:
                key, val = m.group(1).strip(), m.group(2).strip()
                try:
                    current[key] = float(val)
                except ValueError:
                    current[key] = val
    if current and 'Epoch' in current:
        records.append(current)
    df = pd.DataFrame(records)
    df['Epoch'] = df['Epoch'].astype(int)
    df = df.sort_values('Epoch').reset_index(drop=True)
    print(f"[log] dynamic_pitod: {len(df)} epochs "
          f"(epoch {df['Epoch'].min()}–{df['Epoch'].max()}), seed 0 only")
    return df


# ── 2. Load static progress.txt files ─────────────────────────────────────

def load_progress_files(paths: list) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_table(p)
        df['_seed'] = p
        dfs.append(df)
        print(f"[static] {p}: {len(df)} epochs")
    combined = pd.concat(dfs, ignore_index=True)
    mean = combined.groupby('Epoch', as_index=False).mean(numeric_only=True)
    std  = combined.groupby('Epoch', as_index=False).std(numeric_only=True)
    return mean, std, len(dfs)


def find_static_progress(run_dir: str) -> list:
    static_root = os.path.join(run_dir, 'redq_sac_Hopper-v2_static_pitod')
    paths = []
    if os.path.isdir(static_root):
        for seed_dir in sorted(os.listdir(static_root)):
            p = os.path.join(static_root, seed_dir, 'progress.txt')
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                paths.append(p)
    return paths


# ── 3. Summary stats ───────────────────────────────────────────────────────

def trapz(x, y):
    fn = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(fn(np.nan_to_num(y), x))

def first_cross(xs, ys, thresh):
    idx = np.flatnonzero(np.asarray(ys) >= thresh)
    return float(xs[idx[0]]) if idx.size else float('nan')

def print_summary(dyn: pd.DataFrame, st_mean: pd.DataFrame, st_n: int,
                  col: str = 'AverageTestEpRet', thresh: float = 1000.0):
    shared = sorted(set(dyn['Epoch']) & set(st_mean['Epoch']))
    max_ep = shared[-1] if shared else None

    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'dynamic (s0)':>15} {'static (n={})'.format(st_n):>15}")
    print(f"{'='*60}")

    dyn_ret  = dyn[col].to_numpy()
    st_ret   = st_mean[col].to_numpy()
    dyn_ep   = dyn['Epoch'].to_numpy()
    st_ep    = st_mean['Epoch'].to_numpy()

    dyn_auc  = trapz(dyn_ep, dyn_ret)
    st_auc   = trapz(st_ep,  st_ret)
    dyn_best = float(np.nanmax(dyn_ret))
    st_best  = float(np.nanmax(st_ret))
    dyn_fin  = float(dyn_ret[-1])
    st_fin   = float(st_ret[-1])

    dyn_cross = first_cross(dyn_ep, dyn_ret, thresh)
    st_cross  = first_cross(st_ep,  st_ret,  thresh)

    def fmt(v): return f'{v:>15.1f}' if np.isfinite(v) else f"{'never':>15}"
    print(f"{'Final return':<30}{fmt(dyn_fin)}{fmt(st_fin)}")
    print(f"{'Best return':<30}{fmt(dyn_best)}{fmt(st_best)}")
    print(f"{'AUC (return)':<30}{fmt(dyn_auc)}{fmt(st_auc)}")
    print(f"{'First cross {:.0f}'.format(thresh):<30}{fmt(dyn_cross)}{fmt(st_cross)}")

    if max_ep is not None:
        dv = float(dyn.loc[dyn['Epoch']==max_ep, col].iloc[0])
        sv = float(st_mean.loc[st_mean['Epoch']==max_ep, col].iloc[0])
        print(f"{'At epoch ' + str(max_ep):<30}{fmt(dv)}{fmt(sv)}")
    print('='*60)

    # Q-bias
    if 'AverageQBias' in dyn.columns and 'AverageQBias' in st_mean.columns:
        print(f"\n{'='*60}")
        print(f"{'Q-Bias metric':<30} {'dynamic (s0)':>15} {'static (n={})'.format(st_n):>15}")
        print(f"{'='*60}")
        dyn_qb = dyn['AverageQBias'].to_numpy()
        st_qb  = st_mean['AverageQBias'].to_numpy()
        print(f"{'Mean |QBias| (all epochs)':<30}{float(np.nanmean(np.abs(dyn_qb))):>15.3f}{float(np.nanmean(np.abs(st_qb))):>15.3f}")
        print(f"{'Final QBias':<30}{float(dyn_qb[-1]):>15.3f}{float(st_qb[-1]):>15.3f}")
        if 'AverageNormQBias' in dyn.columns and 'AverageNormQBias' in st_mean.columns:
            print(f"{'Final NormQBias':<30}{float(dyn['AverageNormQBias'].iloc[-1]):>15.4f}{float(st_mean['AverageNormQBias'].iloc[-1]):>15.4f}")
        print('='*60)


# ── 4. Plots ───────────────────────────────────────────────────────────────

COLORS = {'dynamic_pitod': '#1f77b4', 'static_pitod': '#ff7f0e'}

def plot_learning_and_bias(dyn: pd.DataFrame,
                           st_mean: pd.DataFrame,
                           st_std: pd.DataFrame,
                           st_n: int,
                           out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── return ──
    ax = axes[0]
    x_d = dyn['Epoch'].to_numpy()
    ax.plot(x_d, dyn['AverageTestEpRet'], color=COLORS['dynamic_pitod'],
            label='dynamic_pitod (partial, 1 seed)', linewidth=2)

    x_s = st_mean['Epoch'].to_numpy()
    ax.plot(x_s, st_mean['AverageTestEpRet'], color=COLORS['static_pitod'],
            label=f'static_pitod (n={st_n} seed{"s" if st_n>1 else ""})', linewidth=2)
    if st_n > 1 and 'AverageTestEpRet' in st_std.columns:
        m = st_mean['AverageTestEpRet'].to_numpy()
        s = np.nan_to_num(st_std['AverageTestEpRet'].to_numpy())
        ax.fill_between(x_s, m - s, m + s, alpha=0.2, color=COLORS['static_pitod'])

    ax.axvline(x_d[-1], color='grey', linestyle='--', linewidth=0.9,
               label=f'dynamic cutoff (epoch {x_d[-1]})')
    ax.set_title('Average Test Return')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Return')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── q-bias ──
    ax2 = axes[1]
    if 'AverageQBias' in dyn.columns:
        ax2.plot(x_d, dyn['AverageQBias'], color=COLORS['dynamic_pitod'],
                 label='dynamic_pitod', linewidth=2)
    if 'AverageQBias' in st_mean.columns:
        ax2.plot(x_s, st_mean['AverageQBias'], color=COLORS['static_pitod'],
                 label=f'static_pitod (n={st_n})', linewidth=2)
        if st_n > 1 and 'AverageQBias' in st_std.columns:
            m2 = st_mean['AverageQBias'].to_numpy()
            s2 = np.nan_to_num(st_std['AverageQBias'].to_numpy())
            ax2.fill_between(x_s, m2 - s2, m2 + s2, alpha=0.2, color=COLORS['static_pitod'])
    ax2.axhline(0, color='black', linestyle=':', linewidth=0.8)
    ax2.axvline(x_d[-1], color='grey', linestyle='--', linewidth=0.9)
    ax2.set_title('Average Q-Bias  (0 = unbiased)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Q-Bias')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, 'return_and_qbias.png')
    fig.savefig(p, dpi=130)
    print(f'\n[saved] {p}')
    plt.close(fig)


def plot_dynamic_diagnostics(dyn: pd.DataFrame, out_dir: str):
    cols = [
        ('DynPIToD/Epsilon',         'Epsilon (score threshold)'),
        ('DynPIToD/NumActive',        'Num Active Groups'),
        ('DynPIToD/BufferActiveFrac', 'Buffer Active Fraction'),
        ('DynPIToD/ScoreMean',        'Score Mean'),
        ('DynPIToD/RefreshWallclock', 'Refresh Wall-clock (s)'),
        ('DynPIToD/NumRefreshed',     'Num Groups Refreshed'),
    ]
    avail = [(c, l) for c, l in cols if c in dyn.columns]
    if not avail:
        print('[warn] No DynPIToD columns found in log.')
        return

    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4))
    if n == 1:
        axes = [axes]
    x = dyn['Epoch'].to_numpy()
    for ax, (col, label) in zip(axes, avail):
        ax.plot(x, dyn[col].to_numpy(), color=COLORS['dynamic_pitod'],
                linewidth=2, marker='.')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
    plt.suptitle('Dynamic PIToD Internals (seed 0)', y=1.02)
    plt.tight_layout()
    p = os.path.join(out_dir, 'dynamic_diagnostics.png')
    fig.savefig(p, dpi=130, bbox_inches='tight')
    print(f'[saved] {p}')
    plt.close(fig)


def plot_normbias(dyn: pd.DataFrame, st_mean: pd.DataFrame,
                  st_std: pd.DataFrame, st_n: int, out_dir: str):
    if 'AverageNormQBias' not in dyn.columns or 'AverageNormQBias' not in st_mean.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x_d = dyn['Epoch'].to_numpy()
    ax.plot(x_d, dyn['AverageNormQBias'], color=COLORS['dynamic_pitod'],
            label='dynamic_pitod (partial, 1 seed)', linewidth=2)
    x_s = st_mean['Epoch'].to_numpy()
    ax.plot(x_s, st_mean['AverageNormQBias'], color=COLORS['static_pitod'],
            label=f'static_pitod (n={st_n})', linewidth=2)
    if st_n > 1 and 'AverageNormQBias' in st_std.columns:
        m = st_mean['AverageNormQBias'].to_numpy()
        s = np.nan_to_num(st_std['AverageNormQBias'].to_numpy())
        ax.fill_between(x_s, m - s, m + s, alpha=0.2, color=COLORS['static_pitod'])
    ax.axvline(x_d[-1], color='grey', linestyle='--', linewidth=0.9)
    ax.set_title('Normalised Q-Bias (lower = better correction)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NormQBias')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'norm_qbias.png')
    fig.savefig(p, dpi=130)
    print(f'[saved] {p}')
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log',     required=True,
                        help='Console log file (newPitod_runLogs.md)')
    parser.add_argument('--run_dir', required=True,
                        help='Run directory (runs/screen_dyn_early_vs_static)')
    parser.add_argument('--out_dir', default='figure/screen_comparison',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dyn = parse_log(args.log)

    static_paths = find_static_progress(args.run_dir)
    if not static_paths:
        print('[warn] No static progress.txt files found — skipping static comparison.')
        st_mean = st_std = None
        st_n = 0
    else:
        st_mean, st_std, st_n = load_progress_files(static_paths)

    if st_mean is not None:
        print_summary(dyn, st_mean, st_n)
        plot_learning_and_bias(dyn, st_mean, st_std, st_n, args.out_dir)
        plot_normbias(dyn, st_mean, st_std, st_n, args.out_dir)

    plot_dynamic_diagnostics(dyn, args.out_dir)
    print(f'\nDone. Figures saved to: {args.out_dir}/')


if __name__ == '__main__':
    main()
