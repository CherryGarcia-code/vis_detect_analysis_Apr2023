"""
C1: D1 vs D2 Population Response Profiles — Region-Based

Grand-mean photometry traces by genotype (D1 vs D2) for key task events,
organized by brain region (DMS, VMS, VLS) with QC filtering and hemisphere
merging.

Panels per region:
  A: Change-aligned PETH (Hit trials) — D1 vs D2
  B: Change-aligned PETH (Miss trials) — D1 vs D2
  C: FA-lick-aligned PETH (All) — D1 vs D2
  D: Peak z-dF/F bar plots — D1 vs D2, 3 events, Mann-Whitney U
  E: Hit − Miss difference traces — D1 vs D2
  F: FA Lick (Early ≤ 3s) — D1 vs D2

Usage:
    py scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py
    py scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py --no-qc
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import (
    GENOTYPE_COLORS, SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES,
    PETH_WINDOW, PETH_BASELINE, FA_RT_SPLIT, CHANGE_SIZES,
    get_roi_region,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres,
)
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.group_statistics import (
    mannwhitney_with_effect_size, permutation_test, bootstrap_ci, format_stats_table,
    extract_signed_peak,
)
from visdetect_photom.analysis.group_utils import (
    get_genotype, get_region, _get_event_times, compute_session_summary,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────
WINDOW = (-2.0, 4.0)
BASELINE = (-2.0, 0.0)
PEAK_WINDOW = (0.0, 1.5)  # Window for peak z-dF/F extraction (post-event)
EVENT_TYPES = ['change_hit', 'change_miss', 'fa_lick', 'fa_early', 'fa_late', 'hit_lick']
EVENT_LABELS = {
    'change_hit': 'Change (Hit)',
    'change_miss': 'Change (Miss)',
    'fa_lick': 'FA Lick (All)',
    'fa_early': 'FA Lick (Early)',
    'fa_late': 'FA Lick (Late)',
    'hit_lick': 'Hit Lick',
}


# ── Data collection ───────────────────────────────────────────

def collect_peths_by_region(
    session_files_list: list,
    use_qc: bool = True,
    max_sessions: int = None,
) -> dict:
    """
    Load sessions, apply QC + hemisphere merging, extract PETHs grouped
    by genotype, region, and event type.

    Returns:
        data[genotype][region][event_type] = list of (subject_id, 1D_trace)
        time_axis: 1D array of relative time points
        summaries: list of per-session summary dicts
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    summaries = []
    time_axis = None
    n_loaded = 0
    n_skipped_geno = 0
    n_skipped_behav = 0

    for i, sf in enumerate(session_files_list):
        if max_sessions and n_loaded >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.warning(f"Skipped session {sf.get('trials', '?')}: {e}")
            continue

        genotype = get_genotype(sess.subject_id)
        if genotype == 'Unknown':
            n_skipped_geno += 1
            continue

        # Resolve full subject ID for region mapping
        subject_id = sess.subject_id
        if not subject_id.startswith('BG_') and subject_id.isdigit():
            subject_id_full = f'BG_{subject_id.zfill(3)}'
        else:
            subject_id_full = subject_id

        # Behavioral QC
        if use_qc:
            behav_qc = check_behavioral_engagement(sess)
            if not behav_qc['pass']:
                n_skipped_behav += 1
                continue

        # Compute summary
        try:
            summaries.append(compute_session_summary(sess))
        except Exception:
            pass

        # Signal QC per ROI
        roi_qc = compute_session_roi_qc(sess) if use_qc else {}

        # Determine extraction sources — always region-based
        if use_qc:
            # QC + hemisphere merging: one signal per region
            merged = merge_hemispheres(sess, qc_results=roi_qc)
            sources = {}
            for region_name, minfo in merged.items():
                sources[region_name] = (minfo['signal'], minfo['timestamps'])
        else:
            # No QC: group individual ROIs by their region
            # Each ROI contributes independently to that region's pool
            sources_by_region = defaultdict(list)
            for roi_name, trace in sess.photometry_data.items():
                region = get_roi_region(roi_name, subject_id_full)
                if region is None:
                    continue
                # Strip _L/_R suffix to get base region
                base_region = region.rsplit('_', 1)[0]
                sources_by_region[base_region].append((trace.signal, trace.timestamps))

            # For each region, average available hemispheres
            sources = {}
            for region_name, traces in sources_by_region.items():
                if len(traces) == 1:
                    sources[region_name] = traces[0]
                elif len(traces) >= 2:
                    min_len = min(len(t[0]) for t in traces)
                    avg_sig = np.mean([t[0][:min_len] for t in traces], axis=0)
                    sources[region_name] = (avg_sig, traces[0][1][:min_len])

        if not sources:
            continue

        # Extract PETHs per region
        for region_name, (signal, timestamps) in sources.items():
            for event_type in EVENT_TYPES:
                event_times = _get_event_times(sess, event_type)
                if len(event_times) == 0:
                    continue

                t_ax, peth_mat = extract_peth(
                    signal, timestamps, event_times,
                    window=WINDOW, baseline_window=BASELINE,
                )

                if time_axis is None:
                    time_axis = t_ax

                # Store each valid trial
                for row_idx in range(peth_mat.shape[0]):
                    row = peth_mat[row_idx]
                    if np.sum(np.isfinite(row)) > len(row) * 0.5:
                        data[genotype][region_name][event_type].append(
                            (subject_id_full, row)
                        )

        n_loaded += 1
        if (n_loaded % 20) == 0:
            logging.info(f"  Loaded {n_loaded} sessions...")

    logging.info(f"Total sessions loaded: {n_loaded} "
                 f"(skipped: {n_skipped_geno} unknown genotype, "
                 f"{n_skipped_behav} behavioral QC)")
    return dict(data), time_axis, summaries


# ── Aggregation helpers ───────────────────────────────────────

def aggregate_traces(trial_list):
    """From list of (subject_id, trace) tuples, compute mean/SEM over per-mouse averages.

    SEM is computed across mice (not trials) to avoid pseudo-replication.
    """
    if not trial_list:
        return None

    # Per-mouse means first
    subjects = sorted(set(s for s, _ in trial_list))
    per_mouse = {}
    for subj in subjects:
        subj_rows = np.array([r for s, r in trial_list if s == subj])
        per_mouse[subj] = np.nanmean(subj_rows, axis=0)

    # Grand mean and SEM over per-mouse averages (correct unit of replication)
    mouse_means = np.array(list(per_mouse.values()))  # (n_mice, n_timepoints)
    grand_mean = np.nanmean(mouse_means, axis=0)
    n_mice = mouse_means.shape[0]
    grand_sem = np.nanstd(mouse_means, axis=0, ddof=0) / np.sqrt(max(n_mice, 1))

    return {
        'mean': grand_mean,
        'sem': grand_sem,
        'per_mouse': per_mouse,
        'n_trials': len(trial_list),
        'n_mice': n_mice,
    }


def extract_peak(trace, time_axis, peak_window=PEAK_WINDOW):
    """Peak (abs-max, sign-preserving) value within peak_window. Delegates to the
    canonical implementation in group_statistics."""
    return extract_signed_peak(trace, time_axis, peak_window)


# ── Plotting functions ────────────────────────────────────────

def plot_peth_comparison(ax, time_axis, agg_d1, agg_d2, title='', xlabel='Time (s)'):
    """Plot D1 vs D2 mean ± SEM trace on a single axis."""
    if agg_d1 is not None:
        ax.plot(time_axis, agg_d1['mean'], color=GENOTYPE_COLORS['D1'],
                label=f"D1 (n={agg_d1['n_trials']}, {agg_d1['n_mice']} mice)", linewidth=1.5)
        ax.fill_between(time_axis,
                        agg_d1['mean'] - agg_d1['sem'],
                        agg_d1['mean'] + agg_d1['sem'],
                        color=GENOTYPE_COLORS['D1'], alpha=0.2)

    if agg_d2 is not None:
        ax.plot(time_axis, agg_d2['mean'], color=GENOTYPE_COLORS['D2'],
                label=f"D2 (n={agg_d2['n_trials']}, {agg_d2['n_mice']} mice)", linewidth=1.5)
        ax.fill_between(time_axis,
                        agg_d2['mean'] - agg_d2['sem'],
                        agg_d2['mean'] + agg_d2['sem'],
                        color=GENOTYPE_COLORS['D2'], alpha=0.2)

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('\u0394 z-dF/F', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    sns.despine(ax=ax)


def plot_peak_bars(ax, peaks_d1, peaks_d2, label, time_axis):
    """Plot peak z-dF/F bar chart for D1 vs D2 with individual mouse dots."""
    d1_vals = np.array([extract_peak(m, time_axis) for m in peaks_d1.values()])
    d2_vals = np.array([extract_peak(m, time_axis) for m in peaks_d2.values()])
    d1_vals = d1_vals[np.isfinite(d1_vals)]
    d2_vals = d2_vals[np.isfinite(d2_vals)]

    positions = [0, 1]
    means = [np.mean(d1_vals) if len(d1_vals) > 0 else 0,
             np.mean(d2_vals) if len(d2_vals) > 0 else 0]
    sems = [np.std(d1_vals) / np.sqrt(len(d1_vals)) if len(d1_vals) > 0 else 0,
            np.std(d2_vals) / np.sqrt(len(d2_vals)) if len(d2_vals) > 0 else 0]
    colors = [GENOTYPE_COLORS['D1'], GENOTYPE_COLORS['D2']]

    ax.bar(positions, means, yerr=sems, color=colors, alpha=0.6, capsize=4, width=0.6)

    # Individual mouse dots
    if len(d1_vals) > 0:
        ax.scatter(np.full_like(d1_vals, 0) + np.random.default_rng(42).uniform(-0.15, 0.15, len(d1_vals)),
                   d1_vals, color=GENOTYPE_COLORS['D1'], edgecolor='k', s=30, zorder=5, linewidth=0.5)
    if len(d2_vals) > 0:
        ax.scatter(np.full_like(d2_vals, 1) + np.random.default_rng(43).uniform(-0.15, 0.15, len(d2_vals)),
                   d2_vals, color=GENOTYPE_COLORS['D2'], edgecolor='k', s=30, zorder=5, linewidth=0.5)

    # Stats — permutation test (valid for small N; MWU has zero power with n<5)
    perm = permutation_test(d1_vals, d2_vals, n_perm=10000, seed=42)
    # Also compute rank-biserial effect size from MWU for reporting
    mwu = mannwhitney_with_effect_size(d1_vals, d2_vals)
    sig = '***' if perm['p'] < 0.001 else '**' if perm['p'] < 0.01 else '*' if perm['p'] < 0.05 else 'ns'
    ax.set_title(f"{label}\nperm p={perm['p']:.3f} {sig}, r={mwu['rank_biserial_r']:.2f}", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(['D1', 'D2'], fontsize=9)
    ax.set_ylabel('Peak \u0394 z-dF/F', fontsize=9)
    sns.despine(ax=ax)

    return {**perm, 'rank_biserial_r': mwu['rank_biserial_r']}


def plot_difference_trace(ax, time_axis, agg_hit, agg_miss, color, label):
    """Plot Hit - Miss difference trace."""
    if agg_hit is None or agg_miss is None:
        return
    diff = agg_hit['mean'] - agg_miss['mean']
    diff_sem = np.sqrt(agg_hit['sem']**2 + agg_miss['sem']**2)

    ax.plot(time_axis, diff, color=color, label=label, linewidth=1.5)
    ax.fill_between(time_axis, diff - diff_sem, diff + diff_sem,
                    color=color, alpha=0.2)


# ── Main figure builder ───────────────────────────────────────

def build_figure(data, time_axis, output_dir, region):
    """Build the multi-panel figure for a single brain region."""

    # Aggregate per genotype per event
    agg = {}
    for geno in ['D1', 'D2']:
        agg[geno] = {}
        for evt in EVENT_TYPES:
            trials = data.get(geno, {}).get(region, {}).get(evt, [])
            agg[geno][evt] = aggregate_traces(trials)

    # Check we have data
    has_d1 = any(agg['D1'][e] is not None for e in EVENT_TYPES)
    has_d2 = any(agg['D2'][e] is not None for e in EVENT_TYPES)
    if not has_d1 and not has_d2:
        logging.warning(f"No data for region {region}, skipping figure.")
        return []

    # ── Figure layout ──
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'D1 vs D2 Response Profiles \u2014 {region}', fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    # Panel A: Change-aligned Hit
    ax_a = fig.add_subplot(gs[0, 0])
    plot_peth_comparison(ax_a, time_axis, agg['D1']['change_hit'], agg['D2']['change_hit'],
                         title='A: Change Onset (Hit Trials)')

    # Panel B: Change-aligned Miss
    ax_b = fig.add_subplot(gs[0, 1])
    plot_peth_comparison(ax_b, time_axis, agg['D1']['change_miss'], agg['D2']['change_miss'],
                         title='B: Change Onset (Miss Trials)')

    # Panel C: FA-lick-aligned (all FAs)
    ax_c = fig.add_subplot(gs[0, 2])
    plot_peth_comparison(ax_c, time_axis, agg['D1']['fa_lick'], agg['D2']['fa_lick'],
                         title='C: FA Lick (All)')

    # Panel D: Peak z-dF/F bar plots (3 events)
    stats_results = []
    for col_idx, evt in enumerate(['change_hit', 'change_miss', 'fa_lick']):
        ax_d = fig.add_subplot(gs[1, col_idx])
        d1_per_mouse = agg['D1'][evt]['per_mouse'] if agg['D1'][evt] else {}
        d2_per_mouse = agg['D2'][evt]['per_mouse'] if agg['D2'][evt] else {}
        stat = plot_peak_bars(ax_d, d1_per_mouse, d2_per_mouse,
                              EVENT_LABELS[evt], time_axis)
        stat['comparison'] = f"D1_vs_D2_{evt}"
        stat['test'] = 'Permutation (10k)'
        stat['region'] = region
        stat['effect_size'] = stat.pop('rank_biserial_r')
        stats_results.append(stat)

    # Panel E: Hit-Miss difference traces
    ax_e = fig.add_subplot(gs[2, 0:2])
    plot_difference_trace(ax_e, time_axis,
                          agg['D1']['change_hit'], agg['D1']['change_miss'],
                          color=GENOTYPE_COLORS['D1'], label='D1 (Hit \u2212 Miss)')
    plot_difference_trace(ax_e, time_axis,
                          agg['D2']['change_hit'], agg['D2']['change_miss'],
                          color=GENOTYPE_COLORS['D2'], label='D2 (Hit \u2212 Miss)')
    ax_e.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_e.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
    ax_e.set_title('E: Outcome Selectivity (Hit \u2212 Miss)', fontsize=10)
    ax_e.set_xlabel('Time (s)', fontsize=9)
    ax_e.set_ylabel('\u0394 z-dF/F', fontsize=9)
    ax_e.legend(fontsize=8)
    sns.despine(ax=ax_e)

    # Panel F: Early vs Late FA comparison
    ax_f = fig.add_subplot(gs[2, 2])
    plot_peth_comparison(ax_f, time_axis, agg['D1']['fa_early'], agg['D2']['fa_early'],
                         title='F: FA Lick (Early \u22643s)')

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"C1_d1_vs_d2_profiles_{region}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved figure: {fig_path}")

    return stats_results


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C1: D1 vs D2 Population Response Profiles")
    _rr = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str,
                        default=str(_rr / "photom_data"))
    parser.add_argument("--output_dir", type=str,
                        default=str(_rr / "FIGURES" / "C1_d1_vs_d2_profiles"))
    parser.add_argument("--no-qc", action="store_true", default=False,
                        help="Disable QC filtering and hemisphere merging")
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    output_path = Path(args.output_dir)

    # Discover sessions (filter small/test CSVs for old-format subjects)
    all_sessions = io.find_all_sessions(
        str(root_path), recursive=True,
        min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    logging.info(f"Discovered {len(all_sessions)} sessions.")

    # Collect PETHs grouped by region
    use_qc = not args.no_qc
    logging.info(f"Loading sessions (QC={'ON' if use_qc else 'OFF'})...")
    data, time_axis, summaries = collect_peths_by_region(
        all_sessions, use_qc=use_qc, max_sessions=args.max_sessions,
    )

    if time_axis is None:
        logging.error("No valid data extracted. Check data paths.")
        sys.exit(1)

    # Log trial counts per group
    for geno in ['D1', 'D2']:
        for region in sorted(data.get(geno, {}).keys()):
            counts = {e: len(data[geno][region].get(e, [])) for e in EVENT_TYPES}
            nonzero = {e: c for e, c in counts.items() if c > 0}
            if nonzero:
                logging.info(f"  {geno} / {region}: {nonzero}")

    # Build figures per region
    all_regions = set()
    for geno_data in data.values():
        all_regions.update(geno_data.keys())

    all_stats = []
    for region in sorted(all_regions):
        logging.info(f"Building figure for {region}...")
        stats = build_figure(data, time_axis, output_path, region=region)
        all_stats.extend(stats)

    # Save stats table
    if all_stats:
        stats_df = format_stats_table(all_stats, save_path=str(output_path / "C1_stats_summary.csv"))
        logging.info(f"Stats summary:\n{stats_df.to_string(index=False)}")

    # Save session summaries
    if summaries:
        sum_df = pd.DataFrame(summaries)
        sum_df.to_csv(output_path / "C1_session_summaries.csv", index=False)
        logging.info(f"Session summaries saved ({len(summaries)} sessions)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
