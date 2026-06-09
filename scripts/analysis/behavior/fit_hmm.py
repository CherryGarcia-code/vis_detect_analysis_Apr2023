"""Phase 3a: Fit Bernoulli GLM-HMM to behavioral data per subject.

For each of the 12 subjects, fits GLM-HMMs with K=2,3,4,5 states using
20 random restarts per K. Selects best K by BIC. Saves:
  - Fitted model (pickle)
  - State assignments CSV (trial-level)
  - State labels JSON
  - Model selection summary
  - Diagnostic plots (psychometrics, transition matrix, GLM weights,
    session state timelines, learning trajectory)

Usage:
    py scripts/analysis/behavior/fit_hmm.py [--subject BG_013] [--n-restarts 20]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from visdetect_photom.core import io as io_mod
from visdetect_photom.core.io import find_all_sessions
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import (
    SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES,
)
from visdetect_photom.analysis.hmm import (
    GLMHMM, GLMHMMConfig, prepare_session_data, fit_best_model,
    auto_label_states, rename_labels, decode_session,
    HMM_STATE_COLORS, HMM_STATE_ORDER, FEATURE_NAMES,
)
from visdetect_photom.analysis.hmm_downstream import (
    compute_state_behavioral_metrics,
    compute_learning_trajectory,
)

# ── Configuration ─────────────────────────────────────────────
DATA_ROOT = "photom_data"
FIGURES_ROOT = "FIGURES/phase3_hmm"
RESULTS_ROOT = "results/hmm"
STAGING_MANIFEST = "results/staging_manifest.csv"


# ── Plotting helpers ──────────────────────────────────────────

def plot_model_selection(selection_df, out_dir, subject_id):
    """BIC and AIC vs K."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(selection_df["K"], selection_df["bic"], 'o-', label="BIC", color='#1f77b4')
    ax.plot(selection_df["K"], selection_df["aic"], 's--', label="AIC", color='#ff7f0e')
    best_k = int(selection_df.loc[selection_df["bic"].idxmin(), "K"])
    ax.axvline(best_k, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("Number of states (K)")
    ax.set_ylabel("Information criterion")
    ax.set_title(f"{subject_id}: Model Selection")
    ax.legend()
    ax.set_xticks(selection_df["K"].values)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subject_id}_model_selection.png"), dpi=150)
    plt.close(fig)


def plot_state_psychometrics(model, state_labels, out_dir, subject_id):
    """P(lick) vs stimulus per state."""
    psych_df = model.state_psychometrics()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for k in range(model.n_states):
        sub = psych_df[psych_df["state"] == k]
        lbl = state_labels[k]
        color = HMM_STATE_COLORS.get(lbl, f"C{k}")
        ax.plot(sub["stimulus"], sub["p_lick"], 'o-', label=lbl, color=color)
    ax.set_xlabel("Stimulus (log2 change_size)")
    ax.set_ylabel("P(lick)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{subject_id}: State Psychometrics (K={model.n_states})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subject_id}_psychometrics.png"), dpi=150)
    plt.close(fig)


def plot_transition_matrix(model, state_labels, out_dir, subject_id):
    """Heatmap of transition matrix."""
    A = model.transition_matrix
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    sns.heatmap(A, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=state_labels, yticklabels=state_labels,
                ax=ax, vmin=0, vmax=1, square=True)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(f"{subject_id}: Transition Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subject_id}_transitions.png"), dpi=150)
    plt.close(fig)


def plot_glm_weights(model, state_labels, out_dir, subject_id):
    """Bar chart of GLM weights per state."""
    K, D = model.n_states, model.n_features
    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 3.5), sharey=True)
    if K == 1:
        axes = [axes]
    features = model.feature_names
    for k in range(K):
        ax = axes[k]
        lbl = state_labels[k]
        color = HMM_STATE_COLORS.get(lbl, f"C{k}")
        ax.barh(range(D), model.weights[k], color=color, edgecolor='k', linewidth=0.5)
        ax.set_yticks(range(D))
        if k == 0:
            ax.set_yticklabels(features)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(lbl)
        ax.set_xlabel("Weight")
    fig.suptitle(f"{subject_id}: GLM Weights", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subject_id}_weights.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_behavioral_metrics(metrics_df, out_dir, subject_id):
    """Bar charts of per-state behavioral metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    metrics_to_plot = [
        ("dprime", "d'"),
        ("hit_rate_go", "Hit Rate (go)"),
        ("catch_lick_rate", "Catch Lick Rate"),
        ("early_lick_rate", "Early Lick Rate"),
    ]
    for ax, (col, title) in zip(axes, metrics_to_plot):
        if col not in metrics_df.columns:
            continue
        labels = metrics_df["label"].values
        vals = metrics_df[col].values
        colors = [HMM_STATE_COLORS.get(l, 'gray') for l in labels]
        ax.bar(range(len(labels)), vals, color=colors, edgecolor='k', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(title)
    fig.suptitle(f"{subject_id}: Per-State Behavioral Metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subject_id}_behavioral_metrics.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_learning_trajectory(trajectory_df, state_labels, n_states, out_dir, subject_id):
    """Stacked area: state fractions + d' over sessions."""
    if trajectory_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Panel A: State fractions
    ax = axes[0]
    x = trajectory_df["session_idx"].values
    frac_cols = [f"frac_{lbl}" for lbl in state_labels]
    fracs = trajectory_df[frac_cols].values.T  # (K, n_sessions)
    colors = [HMM_STATE_COLORS.get(lbl, f"C{k}") for k, lbl in enumerate(state_labels)]
    ax.stackplot(x, fracs, labels=state_labels, colors=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("State fraction")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{subject_id}: Learning Trajectory")

    # Panel B: d' per state
    ax = axes[1]
    for k, lbl in enumerate(state_labels):
        col = f"dprime_{lbl}"
        if col in trajectory_df.columns:
            color = HMM_STATE_COLORS.get(lbl, f"C{k}")
            ax.plot(x, trajectory_df[col].values, 'o-', label=lbl, color=color, markersize=3)
    # Overall d'
    if "overall_dprime" in trajectory_df.columns:
        ax.plot(x, trajectory_df["overall_dprime"].values, 'k--', label="Overall", linewidth=2)
    ax.set_ylabel("d'")
    ax.set_xlabel("Session index")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subject_id}_learning.png"), dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────

def fit_subject(subject_id, sessions, args):
    """Fit GLM-HMM for one subject."""
    out_dir = os.path.join(FIGURES_ROOT, subject_id)
    results_dir = os.path.join(RESULTS_ROOT, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Prepare behavioral data (no photometry needed for HMM fitting)
    sessions_data = []
    for file_paths in sessions:
        try:
            session = load_session_from_files(file_paths)
            sd = prepare_session_data(session)
            if len(sd["y"]) >= 10:  # minimum trial count
                sessions_data.append(sd)
        except Exception as e:
            print(f"  [WARN] Failed to load: {e}")
            continue

    print(f"  {subject_id}: {len(sessions_data)} sessions, "
          f"{sum(len(s['y']) for s in sessions_data)} trials")

    if len(sessions_data) < 3:
        print(f"  [SKIP] Too few sessions for {subject_id}")
        return None

    # Fit
    config = GLMHMMConfig(
        max_iter=200,
        tol=1e-4,
        n_restarts=args.n_restarts,
        verbose=False,
    )

    best_model, selection_df, all_models = fit_best_model(
        sessions_data,
        K_range=[2, 3, 4, 5],
        config=config,
        verbose=True,
        n_workers=args.n_workers,
        seed=42,
    )

    # Label states
    raw_labels = auto_label_states(best_model)
    state_labels = rename_labels(raw_labels)
    n_states = best_model.n_states

    # Save model + selection
    best_model.save(os.path.join(results_dir, "model.pkl"))
    selection_df.to_csv(os.path.join(results_dir, "model_selection.csv"), index=False)
    with open(os.path.join(results_dir, "state_labels.json"), "w") as f:
        json.dump(state_labels, f)

    # Save all K models with full assignments/labels/metrics
    for K, model_k in all_models.items():
        model_k.save(os.path.join(results_dir, f"model_K{K}.pkl"))
        raw_k = auto_label_states(model_k)
        labels_k = rename_labels(raw_k)
        with open(os.path.join(results_dir, f"state_labels_K{K}.json"), "w") as f:
            json.dump(labels_k, f)

        # Decode all sessions for this K
        assigns_k = []
        for sd in sessions_data:
            df = sd["df"].copy()
            states_k = model_k.most_likely_states(sd)
            post_k = model_k.state_posteriors(sd)
            df["hmm_state"] = states_k
            df["hmm_state_label"] = [labels_k[s] for s in states_k]
            for kk in range(model_k.n_states):
                df[f"p_state_{kk}"] = post_k[:, kk]
            df["session_name"] = sd["session_name"]
            df["subject_id"] = sd.get("subject_id", subject_id)
            assigns_k.append(df)
        assigns_k_df = pd.concat(assigns_k, ignore_index=True)
        assigns_k_df.to_csv(os.path.join(results_dir, f"state_assignments_K{K}.csv"), index=False)

        metrics_k = compute_state_behavioral_metrics(assigns_k_df, labels_k, K)
        metrics_k.to_csv(os.path.join(results_dir, f"state_metrics_K{K}.csv"), index=False)

        traj_k = compute_learning_trajectory(assigns_k_df, labels_k, K)
        traj_k.to_csv(os.path.join(results_dir, f"learning_trajectory_K{K}.csv"), index=False)

    # Determine which K to use as the "default" (no suffix) result
    if args.force_k is not None and args.force_k in all_models:
        use_K = args.force_k
        print(f"  Using forced K={use_K} (BIC-best was K={best_model.n_states})")
    else:
        use_K = best_model.n_states

    use_model = all_models[use_K]
    raw_labels = auto_label_states(use_model)
    state_labels = rename_labels(raw_labels)
    n_states = use_model.n_states

    # Save default (no-suffix) results
    use_model.save(os.path.join(results_dir, "model.pkl"))
    selection_df.to_csv(os.path.join(results_dir, "model_selection.csv"), index=False)
    with open(os.path.join(results_dir, "state_labels.json"), "w") as f:
        json.dump(state_labels, f)

    # Decode all sessions → default assignments
    all_assignments = []
    for sd in sessions_data:
        df = sd["df"].copy()
        states = use_model.most_likely_states(sd)
        posteriors = use_model.state_posteriors(sd)
        df["hmm_state"] = states
        df["hmm_state_label"] = [state_labels[s] for s in states]
        for k in range(n_states):
            df[f"p_state_{k}"] = posteriors[:, k]
        df["session_name"] = sd["session_name"]
        df["subject_id"] = sd.get("subject_id", subject_id)
        all_assignments.append(df)

    assignments_df = pd.concat(all_assignments, ignore_index=True)
    assignments_df.to_csv(os.path.join(results_dir, "state_assignments.csv"), index=False)

    # Default metrics + trajectory (recompute for the selected K)
    metrics_df = compute_state_behavioral_metrics(assignments_df, state_labels, n_states)
    metrics_df.to_csv(os.path.join(results_dir, "state_metrics.csv"), index=False)

    trajectory_df = compute_learning_trajectory(assignments_df, state_labels, n_states)
    trajectory_df.to_csv(os.path.join(results_dir, "learning_trajectory.csv"), index=False)

    # Plots (using the selected K model)
    plot_model_selection(selection_df, out_dir, subject_id)
    plot_state_psychometrics(use_model, state_labels, out_dir, subject_id)
    plot_transition_matrix(use_model, state_labels, out_dir, subject_id)
    plot_glm_weights(use_model, state_labels, out_dir, subject_id)
    plot_behavioral_metrics(metrics_df, out_dir, subject_id)
    plot_learning_trajectory(trajectory_df, state_labels, n_states, out_dir, subject_id)

    print(f"  {subject_id}: K={n_states}, labels={state_labels}")
    print(f"  Fractions: {dict(zip(state_labels, [f'{v:.1%}' for v in metrics_df['fraction'].values]))}")
    if 'dprime' in metrics_df.columns:
        print(f"  d': {dict(zip(state_labels, [f'{v:.2f}' for v in metrics_df['dprime'].values]))}")

    return {
        "subject_id": subject_id,
        "genotype": SUBJECT_GENOTYPE.get(subject_id, "?"),
        "n_sessions": len(sessions_data),
        "n_trials": sum(len(s["y"]) for s in sessions_data),
        "bic_best_K": int(selection_df.loc[selection_df["bic"].idxmin(), "K"]),
        "used_K": n_states,
        "state_labels": state_labels,
        "bic": float(selection_df.loc[selection_df["bic"].idxmin(), "bic"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit GLM-HMM per subject")
    parser.add_argument("--subject", type=str, default=None,
                        help="Fit a single subject (e.g. BG_013)")
    parser.add_argument("--n-restarts", type=int, default=20,
                        help="Random restarts per K (default: 20)")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Parallel workers for fitting K values (default: 4)")
    parser.add_argument("--force-k", type=int, default=None,
                        help="Force a specific K for default outputs (e.g. --force-k 3)")
    parser.add_argument("--stages", type=str, nargs='+',
                        default=['Learning', 'Expert'],
                        help="Include only these stages (default: Learning Expert)")
    parser.add_argument("--no-stage-filter", action="store_true",
                        help="Skip stage filtering (use all sessions)")
    parser.add_argument("--manifest", type=str, default=STAGING_MANIFEST,
                        help="Path to staging manifest CSV")
    args = parser.parse_args()

    os.makedirs(FIGURES_ROOT, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # Load staging manifest for session filtering
    allowed_sessions = None
    if not args.no_stage_filter:
        if not os.path.exists(args.manifest):
            print(f"[ERROR] Staging manifest not found: {args.manifest}")
            print("  Run: py scripts/data_management/stage_sessions.py")
            print("  Or use --no-stage-filter to skip filtering.")
            return
        manifest_df = pd.read_csv(args.manifest)
        stage_mask = manifest_df['stage'].isin(args.stages)
        allowed_sessions = set(manifest_df.loc[stage_mask, 'session_name'].values)
        print(f"[INFO] Staging filter: {args.stages} -> "
              f"{len(allowed_sessions)} sessions allowed")

    # Discover sessions
    print("[INFO] Discovering sessions...")
    all_sessions = find_all_sessions(
        DATA_ROOT, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    print(f"[INFO] Discovered {len(all_sessions)} sessions")

    # Group by subject (extract subject_id from trials path)
    subject_sessions = defaultdict(list)
    for fp in all_sessions:
        trials_path = fp.get('trials', '')
        sid_num, date_str = io_mod.infer_session_keys_from_paths(trials_path)
        if sid_num is None:
            continue
        sid = f"BG_{sid_num.zfill(3)}"
        if sid not in SUBJECT_GENOTYPE:
            continue

        # Filter by staging manifest
        if allowed_sessions is not None:
            session_name = f"{sid_num}_{date_str}" if date_str else None
            # Also check with BG_ prefix format
            session_name_full = f"{sid}_{date_str}" if date_str else None
            if session_name not in allowed_sessions and session_name_full not in allowed_sessions:
                continue

        subject_sessions[sid].append(fp)

    if args.subject:
        if args.subject not in subject_sessions:
            print(f"[ERROR] Subject {args.subject} not found. Available: {sorted(subject_sessions.keys())}")
            return
        subject_sessions = {args.subject: subject_sessions[args.subject]}

    print(f"[INFO] Fitting {len(subject_sessions)} subjects...")
    for sid in sorted(subject_sessions.keys()):
        print(f"[INFO] Sessions for {sid}: {len(subject_sessions[sid])}")

    # Fit each subject
    summary_rows = []
    for sid in sorted(subject_sessions.keys()):
        print(f"\n{'='*60}")
        print(f"  Fitting {sid} ({SUBJECT_GENOTYPE.get(sid, '?')}) — "
              f"{len(subject_sessions[sid])} sessions")
        print(f"{'='*60}")
        result = fit_subject(sid, subject_sessions[sid], args)
        if result:
            summary_rows.append(result)

    # Summary table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(RESULTS_ROOT, "hmm_summary.csv"), index=False)
        print(f"\n{'='*60}")
        print("HMM Fitting Summary")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\nResults saved to {RESULTS_ROOT}/")
        print(f"Figures saved to {FIGURES_ROOT}/")


if __name__ == "__main__":
    main()
