"""Phase 3b: State-Conditioned Photometry PETHs.

For each HMM state (Disengaged, Engaged, Impulsive), extract PETHs
aligned to change onset (Hit, Miss) and lick (Hit-lick, FA-lick).
Compare how neural signals differ across behavioral states within
each region × genotype combination.

Requires: HMM fitting results from fit_hmm.py (results/hmm/<subject>/)

Usage:
    py scripts/analysis/photometry/06_state_conditioned_peths.py [--no-qc]
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
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from visdetect_photom.core import io as io_mod
from visdetect_photom.core.io import find_all_sessions
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import (
    SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES, PETH_WINDOW, SAMPLING_FREQ,
    CATCH_THRESHOLD, FA_RT_SPLIT, get_roi_region,
    GENOTYPE_COLORS, REGION_COLORS,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement,
    merge_hemispheres,
)
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.hmm import (
    GLMHMM, prepare_session_data, HMM_STATE_COLORS,
)
from visdetect_photom.analysis.group_statistics import permutation_test

# ── Configuration ─────────────────────────────────────────────
DATA_ROOT = "photom_data"
FIGURES_ROOT = "FIGURES/phase3_state_peths"
HMM_RESULTS = "results/hmm"

# Event types and their alignment rules
EVENT_CONFIGS = {
    'change_hit': {'outcome': 'Hit', 'align': 'change', 'label': 'Change (Hit)'},
    'change_miss': {'outcome': 'Miss', 'align': 'change', 'label': 'Change (Miss)'},
    'hit_lick': {'outcome': 'Hit', 'align': 'lick', 'label': 'Hit Lick'},
    'fa_lick': {'outcome': 'FA', 'align': 'lick', 'label': 'FA Lick'},
}


# ── Helpers ───────────────────────────────────────────────────

def get_event_times(session, event_type):
    """Get event times for a given event type."""
    times = []
    for t in session.trials:
        if event_type == 'change_hit' and t.outcome == 'Hit':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)
        elif event_type == 'change_miss' and t.outcome == 'Miss':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)
        elif event_type == 'hit_lick' and t.outcome == 'Hit':
            if t.absolute_reaction_time is not None:
                times.append(t.absolute_reaction_time)
        elif event_type == 'fa_lick' and t.outcome == 'FA':
            if t.absolute_reaction_time is not None:
                times.append(t.absolute_reaction_time)
    return np.array(times) if times else np.array([])


def get_state_trial_mask(session, state_assignments_df, state_int, event_type):
    """Get event times for trials in a specific HMM state.

    Returns event times only for trials matching both the outcome criterion
    AND the HMM state.
    """
    # Build mapping from trial_index → HMM state
    state_map = {}
    for _, row in state_assignments_df.iterrows():
        state_map[int(row['trial_index'])] = int(row['hmm_state'])

    times = []
    for t in session.trials:
        if t.trial_index not in state_map:
            continue
        if state_map[t.trial_index] != state_int:
            continue

        if event_type == 'change_hit' and t.outcome == 'Hit':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)
        elif event_type == 'change_miss' and t.outcome == 'Miss':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)
        elif event_type == 'hit_lick' and t.outcome == 'Hit':
            if t.absolute_reaction_time is not None:
                times.append(t.absolute_reaction_time)
        elif event_type == 'fa_lick' and t.outcome == 'FA':
            if t.absolute_reaction_time is not None:
                times.append(t.absolute_reaction_time)
    return np.array(times) if times else np.array([])


def aggregate_traces_permouse(trace_list):
    """Aggregate (subject_id, trace) into mean ± SEM over per-mouse averages."""
    from collections import defaultdict
    mouse_traces = defaultdict(list)
    for sid, trace in trace_list:
        mouse_traces[sid].append(trace)

    mouse_means = []
    for sid in sorted(mouse_traces.keys()):
        traces = np.array(mouse_traces[sid])
        mouse_means.append(np.nanmean(traces, axis=0))

    if not mouse_means:
        return None, None, 0

    arr = np.array(mouse_means)
    n_mice = arr.shape[0]
    grand_mean = np.nanmean(arr, axis=0)
    if n_mice > 1:
        grand_sem = np.nanstd(arr, axis=0) / np.sqrt(n_mice)
    else:
        grand_sem = np.zeros_like(grand_mean)
    return grand_mean, grand_sem, n_mice


# ── Main collection ──────────────────────────────────────────

def collect_state_peths(args):
    """Collect PETHs split by HMM state, region, genotype, event."""
    all_sessions = find_all_sessions(
        DATA_ROOT, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    print(f"[INFO] Discovered {len(all_sessions)} sessions")

    # data[genotype][region][event_type][state_label] = [(subject_id, trace), ...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    time_axis = None

    for file_paths in tqdm(all_sessions, desc="Loading sessions"):
        trials_path = file_paths.get('trials', '')
        sid_num, _ = io_mod.infer_session_keys_from_paths(trials_path)
        if sid_num is None:
            continue
        subject_id = f"BG_{sid_num.zfill(3)}"
        genotype = SUBJECT_GENOTYPE.get(subject_id)
        if genotype is None:
            continue

        # Load HMM results for this subject
        hmm_dir = os.path.join(HMM_RESULTS, subject_id)
        assignments_path = os.path.join(hmm_dir, "state_assignments.csv")
        labels_path = os.path.join(hmm_dir, "state_labels.json")
        if not os.path.exists(assignments_path) or not os.path.exists(labels_path):
            continue  # HMM not fitted for this subject

        with open(labels_path) as f:
            state_labels = json.load(f)
        full_assignments = pd.read_csv(assignments_path)

        # Load session
        try:
            session = load_session_from_files(file_paths)
        except Exception:
            continue

        if len(session.trials) < 10:
            continue

        # Behavioral engagement QC
        behav_qc = check_behavioral_engagement(session)
        if not behav_qc['pass']:
            continue

        # Get this session's assignments
        session_name = session.session_id
        sess_assign = full_assignments[full_assignments['session_name'] == session_name]
        if sess_assign.empty:
            continue

        # Get merged photometry trace per region (with QC)
        qc_results = compute_session_roi_qc(session) if not args.no_qc else None
        merged_regions = merge_hemispheres(session, qc_results)

        for region_base, region_data in merged_regions.items():
            signal = region_data['signal']
            timestamps = region_data['timestamps']

            # Extract PETHs per event × state
            for event_type, ecfg in EVENT_CONFIGS.items():
                for state_idx, state_label in enumerate(state_labels):
                    event_times = get_state_trial_mask(
                        session, sess_assign, state_idx, event_type
                    )
                    if len(event_times) < 3:
                        continue

                    t_ax, peth = extract_peth(
                        signal, timestamps, event_times,
                        window=PETH_WINDOW, fs=SAMPLING_FREQ,
                        normalize='subtract',
                    )
                    if peth is None or len(peth) == 0:
                        continue
                    if time_axis is None:
                        time_axis = t_ax

                    mean_trace = np.nanmean(peth, axis=0)
                    data[genotype][region_base][event_type][state_label].append(
                        (subject_id, mean_trace)
                    )

    return data, time_axis, state_labels


# ── Plotting ─────────────────────────────────────────────────

def plot_state_peths(data, time_axis, state_labels, out_dir):
    """Create figures: one per region, with genotypes as columns and events as rows.

    Within each panel, traces are split by HMM state (color-coded).
    """
    regions = sorted(set(
        r for g in data.values() for r in g.keys()
    ))
    genotypes = sorted(data.keys())
    event_types = list(EVENT_CONFIGS.keys())

    for region in regions:
        n_rows = len(event_types)
        n_cols = len(genotypes)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                                  sharex=True, sharey='row')
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for col, genotype in enumerate(genotypes):
            for row, event_type in enumerate(event_types):
                ax = axes[row, col]
                ecfg = EVENT_CONFIGS[event_type]

                any_plotted = False
                for state_label in state_labels:
                    trace_list = data.get(genotype, {}).get(region, {}).get(event_type, {}).get(state_label, [])
                    if not trace_list:
                        continue

                    mean, sem, n_mice = aggregate_traces_permouse(trace_list)
                    if mean is None:
                        continue

                    color = HMM_STATE_COLORS.get(state_label, 'gray')
                    ax.plot(time_axis, mean, color=color, linewidth=1.5,
                            label=f"{state_label} (n={n_mice})")
                    ax.fill_between(time_axis, mean - sem, mean + sem,
                                    color=color, alpha=0.2)
                    any_plotted = True

                ax.axvline(0, color='k', ls='--', lw=0.5, alpha=0.5)
                ax.axhline(0, color='k', ls='-', lw=0.3, alpha=0.3)

                if row == 0:
                    ax.set_title(f"{genotype}", fontsize=12, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f"{ecfg['label']}\n" + r"$\Delta$ z-dF/F")
                if row == n_rows - 1:
                    ax.set_xlabel("Time from event (s)")
                if any_plotted:
                    ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"{region}: State-Conditioned PETHs", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"state_peths_{region}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved: {region}")


def plot_state_peak_comparison(data, time_axis, state_labels, out_dir):
    """Bar plot: peak response per state × genotype × region for Hit trials."""
    from visdetect_photom.analysis.group_statistics import extract_peak_latency

    rows = []
    event = 'change_hit'
    for genotype in sorted(data.keys()):
        for region in sorted(data[genotype].keys()):
            for state_label in state_labels:
                trace_list = data[genotype][region].get(event, {}).get(state_label, [])
                if not trace_list:
                    continue
                for sid, trace in trace_list:
                    # Peak in post-stimulus window [0, 2]s
                    t_mask = (time_axis >= 0) & (time_axis <= 2.0)
                    peak = np.max(np.abs(trace[t_mask])) * np.sign(trace[t_mask][np.argmax(np.abs(trace[t_mask]))])
                    rows.append({
                        'genotype': genotype, 'region': region,
                        'state': state_label, 'subject_id': sid,
                        'peak': peak,
                    })

    if not rows:
        return
    df = pd.DataFrame(rows)

    # Per-mouse averages
    mouse_avg = df.groupby(['genotype', 'region', 'state', 'subject_id'])['peak'].mean().reset_index()

    regions = sorted(df['region'].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4), sharey=True)
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        sub = mouse_avg[mouse_avg['region'] == region]
        states = [s for s in state_labels if s in sub['state'].values]
        x_pos = np.arange(len(states))
        width = 0.35
        for i, geno in enumerate(sorted(sub['genotype'].unique())):
            gsub = sub[sub['genotype'] == geno]
            means = []
            sems = []
            for state in states:
                vals = gsub[gsub['state'] == state]['peak'].values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            color = GENOTYPE_COLORS.get(geno, f'C{i}')
            ax.bar(x_pos + i * width, means, width, yerr=sems, label=geno,
                   color=color, edgecolor='k', linewidth=0.5, alpha=0.8, capsize=3)

        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels(states, rotation=30, ha='right')
        ax.set_title(f"{region}: Change-Hit Peak")
        ax.set_ylabel("Peak " + r"$\Delta$ z-dF/F")
        ax.legend()
        ax.axhline(0, color='k', lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "state_peak_comparison.png"), dpi=150)
    plt.close(fig)
    print("[INFO] Saved: state_peak_comparison")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qc", action="store_true")
    args = parser.parse_args()

    os.makedirs(FIGURES_ROOT, exist_ok=True)

    # Check HMM results exist
    if not os.path.exists(HMM_RESULTS):
        print("[ERROR] HMM results not found. Run fit_hmm.py first.")
        return

    # Get ALL unique state labels across subjects
    all_state_labels = set()
    for sid in sorted(os.listdir(HMM_RESULTS)):
        lpath = os.path.join(HMM_RESULTS, sid, "state_labels.json")
        if os.path.exists(lpath):
            with open(lpath) as f:
                labels = json.load(f)
                all_state_labels.update(labels)
    if not all_state_labels:
        print("[ERROR] No state labels found.")
        return
    # Order: Disengaged variants, Engaged, Impulsive variants, then others
    priority = {"Disengaged": 0, "Disengaged_2": 1, "Engaged": 2, "Impulsive": 3, "Impulsive_2": 4}
    state_labels = sorted(all_state_labels, key=lambda s: priority.get(s, 10))

    print(f"[INFO] State labels: {state_labels}")

    data, time_axis, _ = collect_state_peths(args)

    if time_axis is None:
        print("[ERROR] No PETHs extracted.")
        return

    # Summary
    for geno in sorted(data.keys()):
        for region in sorted(data[geno].keys()):
            for event in sorted(data[geno][region].keys()):
                for state in data[geno][region][event]:
                    n = len(data[geno][region][event][state])
                    mice = len(set(s for s, _ in data[geno][region][event][state]))
                    print(f"  {geno}/{region}/{event}/{state}: {n} sessions, {mice} mice")

    plot_state_peths(data, time_axis, state_labels, FIGURES_ROOT)
    plot_state_peak_comparison(data, time_axis, state_labels, FIGURES_ROOT)

    print(f"\n[INFO] All figures saved to {FIGURES_ROOT}/")


if __name__ == "__main__":
    main()
