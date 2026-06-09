"""Phase 3c: HMM State Fractions by Genotype.

Compares HMM state usage (fraction of trials in each state) across D1 vs D2
mice. Also examines per-state behavioral metrics.

Requires: HMM fitting results from fit_hmm.py (results/hmm/<subject>/)

Figures:
  1. State fractions: D1 vs D2 (grouped bar + individual mice)
  2. Per-state d': D1 vs D2
  3. Learning trajectories per genotype (state fractions over sessions)
  4. Per-state psychometric curves (from model weights)

Usage:
    py scripts/analysis/photometry/07_hmm_state_fractions.py
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from visdetect_photom.core.constants import (
    SUBJECT_GENOTYPE, GENOTYPE_COLORS,
)
from visdetect_photom.analysis.hmm import (
    GLMHMM, auto_label_states, rename_labels,
    HMM_STATE_COLORS, HMM_STATE_ORDER,
)
from visdetect_photom.analysis.hmm_downstream import (
    compute_state_behavioral_metrics, compute_learning_trajectory,
)
from visdetect_photom.analysis.group_statistics import permutation_test

# ── Configuration ─────────────────────────────────────────────
HMM_RESULTS = "results/hmm"
FIGURES_ROOT = "FIGURES/phase3_state_fractions"


# ── Load all subjects' results ────────────────────────────────

def load_all_results():
    """Load HMM results for all subjects.

    Returns dict: subject_id → {model, assignments, labels, metrics, genotype}
    """
    results = {}
    for sid in sorted(os.listdir(HMM_RESULTS)):
        sid_dir = os.path.join(HMM_RESULTS, sid)
        if not os.path.isdir(sid_dir):
            continue
        model_path = os.path.join(sid_dir, "model.pkl")
        assign_path = os.path.join(sid_dir, "state_assignments.csv")
        labels_path = os.path.join(sid_dir, "state_labels.json")
        metrics_path = os.path.join(sid_dir, "state_metrics.csv")

        if not all(os.path.exists(p) for p in [model_path, assign_path, labels_path]):
            continue

        model = GLMHMM.load(model_path)
        assignments = pd.read_csv(assign_path)
        with open(labels_path) as f:
            state_labels = json.load(f)
        genotype = SUBJECT_GENOTYPE.get(sid, "?")

        metrics = None
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)

        results[sid] = {
            'model': model,
            'assignments': assignments,
            'labels': state_labels,
            'metrics': metrics,
            'genotype': genotype,
            'n_states': model.n_states,
        }

    return results


# ── Figure 1: State Fractions D1 vs D2 ──────────────────────

def plot_state_fractions_by_genotype(results, out_dir):
    """Grouped bar: fraction of trials per state, D1 vs D2."""
    # Collect per-subject state fractions
    # Use the canonical 3-state labels; if a subject has different K,
    # map as best we can
    rows = []
    for sid, res in results.items():
        labels = res['labels']
        assignments = res['assignments']
        n_total = len(assignments)
        for k, lbl in enumerate(labels):
            n_state = (assignments['hmm_state'] == k).sum()
            rows.append({
                'subject_id': sid,
                'genotype': res['genotype'],
                'state': lbl,
                'fraction': n_state / n_total if n_total > 0 else 0,
            })

    df = pd.DataFrame(rows)

    # Canonical state order
    state_order = [s for s in HMM_STATE_ORDER if s in df['state'].values]
    if not state_order:
        state_order = sorted(df['state'].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    genotypes = sorted(df['genotype'].unique())
    x = np.arange(len(state_order))
    width = 0.35

    for i, geno in enumerate(genotypes):
        gdf = df[df['genotype'] == geno]
        means = []
        sems = []
        for state in state_order:
            vals = gdf[gdf['state'] == state].groupby('subject_id')['fraction'].mean().values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        color = GENOTYPE_COLORS.get(geno, f'C{i}')
        bars = ax.bar(x + i * width, means, width, yerr=sems,
                       label=geno, color=color, edgecolor='k', linewidth=0.5,
                       alpha=0.8, capsize=3)

        # Individual mice as dots
        for j, state in enumerate(state_order):
            vals = gdf[gdf['state'] == state].groupby('subject_id')['fraction'].mean().values
            jitter = np.random.RandomState(42).uniform(-0.05, 0.05, len(vals))
            ax.scatter([x[j] + i * width] * len(vals) + jitter, vals,
                       color=color, edgecolor='k', s=30, zorder=5, alpha=0.7)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(state_order)
    ax.set_ylabel("Fraction of trials")
    ax.set_title("HMM State Fractions: D1 vs D2")
    ax.legend()
    ax.set_ylim(0, None)

    # Stats: permutation test per state
    stats_rows = []
    for state in state_order:
        d1_vals = df[(df['genotype'] == 'D1') & (df['state'] == state)].groupby('subject_id')['fraction'].mean().values
        d2_vals = df[(df['genotype'] == 'D2') & (df['state'] == state)].groupby('subject_id')['fraction'].mean().values
        if len(d1_vals) >= 2 and len(d2_vals) >= 2:
            res = permutation_test(d1_vals, d2_vals)
            stats_rows.append({'state': state, **res, 'n_D1': len(d1_vals), 'n_D2': len(d2_vals)})
            # Annotate
            if res['p'] < 0.1:
                ymax = max(np.mean(d1_vals), np.mean(d2_vals)) + max(
                    np.std(d1_vals) / np.sqrt(len(d1_vals)),
                    np.std(d2_vals) / np.sqrt(len(d2_vals))
                ) + 0.02
                ax.text(state_order.index(state) + width / 2, ymax,
                        f"p={res['p']:.3f}", ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "state_fractions_d1_vs_d2.png"), dpi=150)
    plt.close(fig)

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(os.path.join(out_dir, "state_fraction_stats.csv"), index=False)
        print("[INFO] State fraction stats:")
        print(stats_df.to_string(index=False))

    return df


# ── Figure 2: Per-State d' by Genotype ───────────────────────

def plot_state_dprime_by_genotype(results, out_dir):
    """Per-state d' comparison: D1 vs D2."""
    rows = []
    for sid, res in results.items():
        if res['metrics'] is None:
            continue
        for _, mrow in res['metrics'].iterrows():
            if pd.isna(mrow.get('dprime')):
                continue
            rows.append({
                'subject_id': sid,
                'genotype': res['genotype'],
                'state': mrow['label'],
                'dprime': mrow['dprime'],
                'hit_rate': mrow.get('hit_rate_go', np.nan),
                'fa_rate': mrow.get('catch_lick_rate', np.nan),
                'early_lick_rate': mrow.get('early_lick_rate', np.nan),
            })

    if not rows:
        return
    df = pd.DataFrame(rows)

    state_order = [s for s in HMM_STATE_ORDER if s in df['state'].values]
    metrics_to_plot = [
        ('dprime', "d'"),
        ('hit_rate', 'Hit Rate (go)'),
        ('early_lick_rate', 'Early Lick Rate'),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    genotypes = sorted(df['genotype'].unique())
    x = np.arange(len(state_order))
    width = 0.35

    for ax, (col, ylabel) in zip(axes, metrics_to_plot):
        for i, geno in enumerate(genotypes):
            gdf = df[df['genotype'] == geno]
            means = []
            sems = []
            for state in state_order:
                vals = gdf[gdf['state'] == state][col].dropna().values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            color = GENOTYPE_COLORS.get(geno, f'C{i}')
            ax.bar(x + i * width, means, width, yerr=sems,
                   label=geno, color=color, edgecolor='k', linewidth=0.5,
                   alpha=0.8, capsize=3)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(state_order, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.legend()
        if col == 'dprime':
            ax.axhline(0, color='k', lw=0.5)

    fig.suptitle("Per-State Behavioral Metrics: D1 vs D2", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "state_dprime_by_genotype.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Figure 3: Learning Trajectories per Genotype ─────────────

def plot_learning_by_genotype(results, out_dir):
    """State fractions over sessions, averaged per genotype."""
    # Collect learning trajectories per subject
    all_trajs = []
    for sid, res in results.items():
        traj_path = os.path.join(HMM_RESULTS, sid, "learning_trajectory.csv")
        if not os.path.exists(traj_path):
            continue
        traj = pd.read_csv(traj_path)
        traj['subject_id'] = sid
        traj['genotype'] = res['genotype']
        # Normalize session index per mouse (fraction of total sessions)
        n_sess = len(traj)
        traj['session_frac'] = np.linspace(0, 1, n_sess)
        all_trajs.append(traj)

    if not all_trajs:
        return
    df = pd.concat(all_trajs, ignore_index=True)

    genotypes = sorted(df['genotype'].unique())
    state_labels = results[list(results.keys())[0]]['labels']
    state_order = [s for s in HMM_STATE_ORDER if s in state_labels]
    if not state_order:
        state_order = state_labels

    fig, axes = plt.subplots(1, len(genotypes), figsize=(6 * len(genotypes), 4), sharey=True)
    if len(genotypes) == 1:
        axes = [axes]

    # Bin sessions into early/mid/late thirds
    for ax, geno in zip(axes, genotypes):
        gdf = df[df['genotype'] == geno]

        # Per-mouse learning curves binned into 5 bins
        n_bins = 5
        gdf['bin'] = pd.cut(gdf['session_frac'], bins=n_bins, labels=range(n_bins))

        for state in state_order:
            frac_col = f"frac_{state}"
            if frac_col not in gdf.columns:
                continue
            color = HMM_STATE_COLORS.get(state, 'gray')

            # Mean per bin across mice
            binned = gdf.groupby('bin')[frac_col].agg(['mean', 'sem']).reset_index()
            bin_x = np.arange(n_bins)
            ax.plot(bin_x, binned['mean'], 'o-', color=color, label=state)
            ax.fill_between(bin_x, binned['mean'] - binned['sem'],
                           binned['mean'] + binned['sem'], color=color, alpha=0.2)

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f"{int(i/n_bins*100)}-{int((i+1)/n_bins*100)}%" for i in range(n_bins)],
                          fontsize=8)
        ax.set_xlabel("Session progression (%)")
        ax.set_ylabel("State fraction")
        ax.set_title(f"{geno} (n={len(gdf['subject_id'].unique())} mice)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    fig.suptitle("Learning Trajectory: State Fractions Over Sessions", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_trajectory_by_genotype.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Figure 4: State Psychometrics per Subject ────────────────

def plot_psychometrics_by_genotype(results, out_dir):
    """State psychometric curves: D1 vs D2."""
    genotypes = sorted(set(r['genotype'] for r in results.values()))

    fig, axes = plt.subplots(1, len(genotypes), figsize=(5 * len(genotypes), 4))
    if len(genotypes) == 1:
        axes = [axes]

    for ax, geno in zip(axes, genotypes):
        subjects = [sid for sid, r in results.items() if r['genotype'] == geno]
        state_labels = None

        for sid in subjects:
            model = results[sid]['model']
            labels = results[sid]['labels']
            if state_labels is None:
                state_labels = labels
            psych = model.state_psychometrics()

            for k, lbl in enumerate(labels):
                sub = psych[psych['state'] == k]
                color = HMM_STATE_COLORS.get(lbl, f'C{k}')
                ax.plot(sub['stimulus'], sub['p_lick'], '-', color=color,
                        alpha=0.3, linewidth=1)

        # Average per state
        if state_labels:
            stim_vals = np.array([0, 0.32, 0.43, 0.58, 1.0, 2.0])
            for k, lbl in enumerate(state_labels):
                all_curves = []
                for sid in subjects:
                    model = results[sid]['model']
                    if k >= model.n_states:
                        continue
                    curve = []
                    for sv in stim_vals:
                        x = np.zeros(model.n_features)
                        x[0] = 1.0; x[1] = sv
                        from scipy.special import expit
                        curve.append(expit(model.weights[k] @ x))
                    all_curves.append(curve)
                if all_curves:
                    mean_curve = np.mean(all_curves, axis=0)
                    color = HMM_STATE_COLORS.get(lbl, f'C{k}')
                    ax.plot(stim_vals, mean_curve, 'o-', color=color,
                            linewidth=2.5, label=lbl, markersize=6)

        ax.set_xlabel("Stimulus (log2 change_size)")
        ax.set_ylabel("P(lick)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{geno} (n={len(subjects)})")
        ax.legend(fontsize=8)

    fig.suptitle("State Psychometric Curves", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "psychometrics_by_genotype.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_ROOT, exist_ok=True)

    if not os.path.exists(HMM_RESULTS):
        print("[ERROR] HMM results not found. Run fit_hmm.py first.")
        return

    results = load_all_results()
    print(f"[INFO] Loaded HMM results for {len(results)} subjects")
    for sid, res in sorted(results.items()):
        print(f"  {sid} ({res['genotype']}): K={res['n_states']}, labels={res['labels']}")

    if not results:
        return

    plot_state_fractions_by_genotype(results, FIGURES_ROOT)
    plot_state_dprime_by_genotype(results, FIGURES_ROOT)
    plot_learning_by_genotype(results, FIGURES_ROOT)
    plot_psychometrics_by_genotype(results, FIGURES_ROOT)

    print(f"\n[INFO] All figures saved to {FIGURES_ROOT}/")


if __name__ == "__main__":
    main()
