"""
Phase 2a: DMS vs VMS Regional Response Profiles

Compares photometry responses across brain regions **within the same genotype**:
  - D1: DMS (BG_013/014/015/020) vs VMS (BG_008/009)
  - D2: DMS (BG_016/017/018/019) vs VMS (BG_010/011)

This is the headline finding enabled by adding the VMS cohort — it reveals
regional specialization of D1/D2 pathways during perceptual decision-making.

Panels per genotype:
  A: Change-aligned Hit — DMS vs VMS
  B: Change-aligned Miss — DMS vs VMS
  C: FA-lick-aligned — DMS vs VMS
  D: Peak bar plots — DMS vs VMS, 3 events, Mann-Whitney U
  E: Hit-lick-aligned — DMS vs VMS
  F: Hit - Miss difference traces — DMS vs VMS

Usage:
    py scripts/analysis/photometry/03_dms_vs_vms_profiles.py
    py scripts/analysis/photometry/03_dms_vs_vms_profiles.py --no-qc
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
    GENOTYPE_COLORS, REGION_COLORS, SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES,
    PETH_WINDOW, PETH_BASELINE, FA_RT_SPLIT, get_roi_region,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres,
)
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.group_statistics import (
    mannwhitney_with_effect_size, permutation_test, bootstrap_ci, format_stats_table,
)
from visdetect_photom.analysis.group_utils import (
    get_genotype, get_region, _get_event_times, compute_session_summary,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────
WINDOW = (-2.0, 4.0)
BASELINE = (-2.0, 0.0)
PEAK_WINDOW = (0.0, 1.5)
EVENT_TYPES = ['change_hit', 'change_miss', 'fa_lick', 'fa_early', 'fa_late', 'hit_lick']
EVENT_LABELS = {
    'change_hit': 'Change (Hit)',
    'change_miss': 'Change (Miss)',
    'fa_lick': 'FA Lick (All)',
    'fa_early': 'FA Lick (Early)',
    'fa_late': 'FA Lick (Late)',
    'hit_lick': 'Hit Lick',
}
REGIONS_TO_COMPARE = ['DMS', 'VMS']


# ── Data collection ───────────────────────────────────────────

def collect_peths_by_genotype_region(
    session_files_list: list,
    use_qc: bool = True,
    max_sessions: int = None,
) -> dict:
    """
    Load sessions, extract PETHs grouped by genotype, region, and event type.

    Returns:
        data[genotype][region][event_type] = list of (subject_id, 1D_trace)
        time_axis: 1D array
        summaries: list of per-session summary dicts
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    summaries = []
    time_axis = None
    n_loaded = 0

    for sf in session_files_list:
        if max_sessions and n_loaded >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.warning(f"Skipped session: {e}")
            continue

        genotype = get_genotype(sess.subject_id)
        if genotype == 'Unknown':
            continue

        subject_id = sess.subject_id
        if not subject_id.startswith('BG_') and subject_id.isdigit():
            subject_id_full = f'BG_{subject_id.zfill(3)}'
        else:
            subject_id_full = subject_id

        # Behavioral QC
        if use_qc:
            behav_qc = check_behavioral_engagement(sess)
            if not behav_qc['pass']:
                continue

        try:
            summaries.append(compute_session_summary(sess))
        except Exception:
            pass

        # Signal QC + hemisphere merging
        roi_qc = compute_session_roi_qc(sess) if use_qc else {}

        if use_qc:
            merged = merge_hemispheres(sess, qc_results=roi_qc)
            sources = {}
            for region_name, minfo in merged.items():
                sources[region_name] = (minfo['signal'], minfo['timestamps'])
        else:
            sources_by_region = defaultdict(list)
            for roi_name, trace in sess.photometry_data.items():
                region = get_roi_region(roi_name, subject_id_full)
                if region is None:
                    continue
                base_region = region.rsplit('_', 1)[0]
                sources_by_region[base_region].append((trace.signal, trace.timestamps))
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

                for row_idx in range(peth_mat.shape[0]):
                    row = peth_mat[row_idx]
                    if np.sum(np.isfinite(row)) > len(row) * 0.5:
                        data[genotype][region_name][event_type].append(
                            (subject_id_full, row)
                        )

        n_loaded += 1
        if n_loaded % 20 == 0:
            logging.info(f"  Loaded {n_loaded} sessions...")

    logging.info(f"Total sessions loaded: {n_loaded}")
    return dict(data), time_axis, summaries


# ── Aggregation ───────────────────────────────────────────────

def aggregate_traces(trial_list):
    """Compute mean/SEM over per-mouse averages (not individual trials).

    SEM is computed across mice to avoid pseudo-replication.
    """
    if not trial_list:
        return None

    subjects = sorted(set(s for s, _ in trial_list))
    per_mouse = {}
    for subj in subjects:
        subj_rows = np.array([r for s, r in trial_list if s == subj])
        per_mouse[subj] = np.nanmean(subj_rows, axis=0)

    mouse_means = np.array(list(per_mouse.values()))
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
    """Extract peak (abs-max) value within peak_window. Captures both activation and suppression."""
    mask = (time_axis >= peak_window[0]) & (time_axis <= peak_window[1])
    if not np.any(mask):
        return np.nan
    segment = trace[mask]
    valid = segment[np.isfinite(segment)]
    if len(valid) == 0:
        return np.nan
    idx = np.argmax(np.abs(valid))
    return float(valid[idx])


# ── Plotting ──────────────────────────────────────────────────

def plot_region_comparison(ax, time_axis, agg_dms, agg_vms, title='', xlabel='Time (s)'):
    """Plot DMS vs VMS mean +/- SEM trace."""
    for agg, region in [(agg_dms, 'DMS'), (agg_vms, 'VMS')]:
        if agg is None:
            continue
        color = REGION_COLORS[region]
        label = f"{region} (n={agg['n_trials']}, {agg['n_mice']} mice)"
        ax.plot(time_axis, agg['mean'], color=color, label=label, linewidth=1.5)
        ax.fill_between(time_axis, agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.2)

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('\u0394 z-dF/F', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    sns.despine(ax=ax)


def plot_region_peak_bars(ax, per_mouse_dms, per_mouse_vms, label, time_axis):
    """Peak z-dF/F bar chart for DMS vs VMS with mouse dots and permutation test."""
    dms_vals = np.array([extract_peak(m, time_axis) for m in per_mouse_dms.values()])
    vms_vals = np.array([extract_peak(m, time_axis) for m in per_mouse_vms.values()])
    dms_vals = dms_vals[np.isfinite(dms_vals)]
    vms_vals = vms_vals[np.isfinite(vms_vals)]

    positions = [0, 1]
    means = [np.mean(dms_vals) if len(dms_vals) > 0 else 0,
             np.mean(vms_vals) if len(vms_vals) > 0 else 0]
    sems = [np.std(dms_vals) / np.sqrt(max(len(dms_vals), 1)),
            np.std(vms_vals) / np.sqrt(max(len(vms_vals), 1))]
    colors = [REGION_COLORS['DMS'], REGION_COLORS['VMS']]

    ax.bar(positions, means, yerr=sems, color=colors, alpha=0.6, capsize=4, width=0.6)

    rng_dms = np.random.default_rng(42)
    rng_vms = np.random.default_rng(43)
    if len(dms_vals) > 0:
        ax.scatter(np.full_like(dms_vals, 0) + rng_dms.uniform(-0.15, 0.15, len(dms_vals)),
                   dms_vals, color=REGION_COLORS['DMS'], edgecolor='k', s=30, zorder=5, linewidth=0.5)
    if len(vms_vals) > 0:
        ax.scatter(np.full_like(vms_vals, 1) + rng_vms.uniform(-0.15, 0.15, len(vms_vals)),
                   vms_vals, color=REGION_COLORS['VMS'], edgecolor='k', s=30, zorder=5, linewidth=0.5)

    # Permutation test (valid for small N; MWU has zero power with n<5)
    perm = permutation_test(dms_vals, vms_vals, n_perm=10000, seed=42)
    mwu = mannwhitney_with_effect_size(dms_vals, vms_vals)
    sig = '***' if perm['p'] < 0.001 else '**' if perm['p'] < 0.01 else '*' if perm['p'] < 0.05 else 'ns'
    ax.set_title(f"{label}\nperm p={perm['p']:.3f} {sig}, r={mwu['rank_biserial_r']:.2f}", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(['DMS', 'VMS'], fontsize=9)
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
    ax.fill_between(time_axis, diff - diff_sem, diff + diff_sem, color=color, alpha=0.2)


# ── Main figure builder ───────────────────────────────────────

def build_genotype_figure(data, time_axis, output_dir, genotype):
    """Build multi-panel DMS vs VMS comparison for one genotype."""

    agg = {}
    for region in REGIONS_TO_COMPARE:
        agg[region] = {}
        for evt in EVENT_TYPES:
            trials = data.get(genotype, {}).get(region, {}).get(evt, [])
            agg[region][evt] = aggregate_traces(trials)

    has_dms = any(agg['DMS'][e] is not None for e in EVENT_TYPES)
    has_vms = any(agg['VMS'][e] is not None for e in EVENT_TYPES)
    if not has_dms and not has_vms:
        logging.warning(f"No data for {genotype}, skipping.")
        return []

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'DMS vs VMS Response Profiles \u2014 {genotype}', fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

    # Row 1: Change-aligned
    ax_a = fig.add_subplot(gs[0, 0])
    plot_region_comparison(ax_a, time_axis, agg['DMS']['change_hit'], agg['VMS']['change_hit'],
                           title='A: Change Onset (Hit)', xlabel='Time from change (s)')

    ax_b = fig.add_subplot(gs[0, 1])
    plot_region_comparison(ax_b, time_axis, agg['DMS']['change_miss'], agg['VMS']['change_miss'],
                           title='B: Change Onset (Miss)', xlabel='Time from change (s)')

    ax_c = fig.add_subplot(gs[0, 2])
    plot_region_comparison(ax_c, time_axis, agg['DMS']['fa_lick'], agg['VMS']['fa_lick'],
                           title='C: FA Lick (All)', xlabel='Time from FA lick (s)')

    # Row 2: Peak bars
    stats_results = []
    for col_idx, evt in enumerate(['change_hit', 'change_miss', 'fa_lick']):
        ax_d = fig.add_subplot(gs[1, col_idx])
        dms_pm = agg['DMS'][evt]['per_mouse'] if agg['DMS'][evt] else {}
        vms_pm = agg['VMS'][evt]['per_mouse'] if agg['VMS'][evt] else {}
        stat = plot_region_peak_bars(ax_d, dms_pm, vms_pm, EVENT_LABELS[evt], time_axis)
        stat['comparison'] = f"DMS_vs_VMS_{evt}"
        stat['test'] = 'Permutation (10k)'
        stat['genotype'] = genotype
        stat['effect_size'] = stat.pop('rank_biserial_r')
        stats_results.append(stat)

    # Row 3: Hit-lick aligned + Hit-Miss difference
    ax_e = fig.add_subplot(gs[2, 0])
    plot_region_comparison(ax_e, time_axis, agg['DMS']['hit_lick'], agg['VMS']['hit_lick'],
                           title='D: Hit Lick', xlabel='Time from lick (s)')

    ax_f = fig.add_subplot(gs[2, 1:3])
    plot_difference_trace(ax_f, time_axis,
                          agg['DMS']['change_hit'], agg['DMS']['change_miss'],
                          color=REGION_COLORS['DMS'], label='DMS (Hit \u2212 Miss)')
    plot_difference_trace(ax_f, time_axis,
                          agg['VMS']['change_hit'], agg['VMS']['change_miss'],
                          color=REGION_COLORS['VMS'], label='VMS (Hit \u2212 Miss)')
    ax_f.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_f.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
    ax_f.set_title('E: Outcome Selectivity (Hit \u2212 Miss)', fontsize=10)
    ax_f.set_xlabel('Time from change (s)', fontsize=9)
    ax_f.set_ylabel('\u0394 z-dF/F', fontsize=9)
    ax_f.legend(fontsize=8)
    sns.despine(ax=ax_f)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"dms_vs_vms_{genotype}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig_path}")

    return stats_results


# ── Phase 2b: Genotype x Region interaction summary figure ────

def build_interaction_figure(data, time_axis, output_dir):
    """
    2x2 summary: genotype (D1, D2) x region (DMS, VMS) for Change-Hit.
    All four conditions on one subplot for direct comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Genotype \u00d7 Region Interaction', fontsize=14)

    stats = []
    # Use distinct line styles for region, colors for genotype
    region_styles = {'DMS': '-', 'VMS': '--'}

    for ax_idx, evt in enumerate(['change_hit', 'change_miss', 'hit_lick']):
        ax = axes[ax_idx]
        for geno in ['D1', 'D2']:
            for region in REGIONS_TO_COMPARE:
                trials = data.get(geno, {}).get(region, {}).get(evt, [])
                agg = aggregate_traces(trials)
                if agg is None:
                    continue
                color = GENOTYPE_COLORS[geno]
                ls = region_styles[region]
                label = f"{geno}-{region} ({agg['n_mice']} mice, n={agg['n_trials']})"
                ax.plot(time_axis, agg['mean'], color=color, linestyle=ls,
                        label=label, linewidth=1.5)
                ax.fill_between(time_axis, agg['mean'] - agg['sem'],
                                agg['mean'] + agg['sem'],
                                color=color, alpha=0.1)

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
        xlabel = 'Time from lick (s)' if 'lick' in evt else 'Time from change (s)'
        ax.set_title(EVENT_LABELS[evt], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('\u0394 z-dF/F', fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        sns.despine(ax=ax)

    plt.tight_layout()
    fig_path = output_dir / "genotype_x_region_interaction.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig_path}")

    # ── Peak-based 2x2 interaction bar plot ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Peak \u0394 z-dF/F: Genotype \u00d7 Region', fontsize=14)

    for ax_idx, evt in enumerate(['change_hit', 'change_miss', 'fa_lick']):
        ax = axes2[ax_idx]
        bar_data = {}
        for geno in ['D1', 'D2']:
            for region in REGIONS_TO_COMPARE:
                trials = data.get(geno, {}).get(region, {}).get(evt, [])
                agg = aggregate_traces(trials)
                if agg is None:
                    bar_data[(geno, region)] = np.array([])
                    continue
                peaks = np.array([extract_peak(m, time_axis) for m in agg['per_mouse'].values()])
                bar_data[(geno, region)] = peaks[np.isfinite(peaks)]

        # Plot bars
        labels = ['D1-DMS', 'D1-VMS', 'D2-DMS', 'D2-VMS']
        keys = [('D1', 'DMS'), ('D1', 'VMS'), ('D2', 'DMS'), ('D2', 'VMS')]
        positions = np.arange(len(labels))
        bar_colors = [GENOTYPE_COLORS['D1'], GENOTYPE_COLORS['D1'],
                      GENOTYPE_COLORS['D2'], GENOTYPE_COLORS['D2']]
        hatches = ['', '///', '', '///']  # hatched = VMS

        for i, (key, pos) in enumerate(zip(keys, positions)):
            vals = bar_data.get(key, np.array([]))
            mean = np.mean(vals) if len(vals) > 0 else 0
            sem = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 0 else 0
            bar = ax.bar(pos, mean, yerr=sem, color=bar_colors[i], alpha=0.6,
                         capsize=3, width=0.6, hatch=hatches[i], edgecolor='k',
                         linewidth=0.5)
            if len(vals) > 0:
                rng = np.random.default_rng(40 + i)
                ax.scatter(np.full_like(vals, pos) + rng.uniform(-0.12, 0.12, len(vals)),
                           vals, color=bar_colors[i], edgecolor='k', s=25, zorder=5,
                           linewidth=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_ylabel('Peak \u0394 z-dF/F', fontsize=9)
        ax.set_title(EVENT_LABELS[evt], fontsize=10)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig2_path = output_dir / "genotype_x_region_peak_bars.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logging.info(f"Saved: {fig2_path}")

    return stats


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2a: DMS vs VMS Regional Profiles")
    _rr = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str, default=str(_rr / "photom_data"))
    parser.add_argument("--output_dir", type=str,
                        default=str(_rr / "FIGURES" / "phase2_dms_vs_vms"))
    parser.add_argument("--no-qc", action="store_true", default=False)
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    output_path = Path(args.output_dir)
    use_qc = not args.no_qc

    # Discover sessions
    all_sessions = io.find_all_sessions(
        str(root_path), recursive=True,
        min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    logging.info(f"Discovered {len(all_sessions)} sessions.")

    # Collect PETHs
    logging.info(f"Loading sessions (QC={'ON' if use_qc else 'OFF'})...")
    data, time_axis, summaries = collect_peths_by_genotype_region(
        all_sessions, use_qc=use_qc, max_sessions=args.max_sessions,
    )

    if time_axis is None:
        logging.error("No valid data extracted.")
        sys.exit(1)

    # Log trial counts
    for geno in ['D1', 'D2']:
        for region in sorted(data.get(geno, {}).keys()):
            counts = {e: len(data[geno][region].get(e, [])) for e in EVENT_TYPES}
            nonzero = {e: c for e, c in counts.items() if c > 0}
            if nonzero:
                logging.info(f"  {geno} / {region}: {nonzero}")

    # Phase 2a: Per-genotype DMS vs VMS figures
    all_stats = []
    for geno in ['D1', 'D2']:
        logging.info(f"Building DMS vs VMS figure for {geno}...")
        stats = build_genotype_figure(data, time_axis, output_path, genotype=geno)
        all_stats.extend(stats)

    # Phase 2b: Genotype x Region interaction figure
    logging.info("Building genotype x region interaction figure...")
    build_interaction_figure(data, time_axis, output_path)

    # Save stats
    if all_stats:
        stats_df = format_stats_table(all_stats, save_path=str(output_path / "dms_vs_vms_stats.csv"))
        logging.info(f"Stats:\n{stats_df.to_string(index=False)}")

    # Save summaries
    if summaries:
        pd.DataFrame(summaries).to_csv(output_path / "session_summaries.csv", index=False)
        logging.info(f"Summaries saved ({len(summaries)} sessions)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
