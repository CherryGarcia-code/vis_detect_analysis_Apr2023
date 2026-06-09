"""
Phase 2d: Response Timing Analysis

Extracts peak latency and onset latency from per-mouse averaged PETHs.
Timing is robust to fiber placement, expression level, and gain — it only
depends on WHEN the signal responds, not how large the response is.

Key comparisons:
  - DMS vs VMS within genotype (within-subject for dual-site mice)
  - D1 vs D2 within region
  - Hit vs Miss within region (sensory selectivity timing)

Outputs:
  - Timing summary CSV (per mouse, region, genotype, event)
  - Peak latency scatter per region × genotype
  - Onset latency scatter per region × genotype
  - Within-subject DMS-VMS timing difference (paired where possible)

Usage:
    py scripts/analysis/photometry/05_timing_analysis.py
    py scripts/analysis/photometry/05_timing_analysis.py --no-qc
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import (
    GENOTYPE_COLORS, REGION_COLORS, SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES,
    get_roi_region,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres,
)
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.group_statistics import (
    extract_peak_latency, extract_onset_latency,
    permutation_test, bootstrap_ci, format_stats_table,
)
from visdetect_photom.analysis.group_utils import (
    get_genotype, _get_event_times,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

WINDOW = (-2.0, 4.0)
BASELINE = (-2.0, 0.0)
PEAK_WINDOW = (0.0, 1.5)
ONSET_WINDOW = (0.0, 2.0)
EVENT_TYPES = ['change_hit', 'change_miss', 'fa_lick', 'hit_lick']
EVENT_LABELS = {
    'change_hit': 'Change (Hit)',
    'change_miss': 'Change (Miss)',
    'fa_lick': 'FA Lick',
    'hit_lick': 'Hit Lick',
}


def collect_per_mouse_traces(session_files, use_qc=True, max_sessions=None):
    """
    Collect per-trial PETHs, aggregate to per-mouse per-region per-event averages.

    Returns:
        traces[subject_id][region][event_type] = mean_trace (1D)
        time_axis: 1D array
        genotype_map: {subject_id: genotype}
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    genotype_map = {}
    time_axis = None
    n_loaded = 0

    for sf in session_files:
        if max_sessions and n_loaded >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception:
            continue

        genotype = get_genotype(sess.subject_id)
        if genotype == 'Unknown':
            continue

        subject_id = sess.subject_id
        if not subject_id.startswith('BG_') and subject_id.isdigit():
            subject_id = f'BG_{subject_id.zfill(3)}'

        genotype_map[subject_id] = genotype

        if use_qc:
            behav_qc = check_behavioral_engagement(sess)
            if not behav_qc['pass']:
                continue

        roi_qc = compute_session_roi_qc(sess) if use_qc else {}

        if use_qc:
            merged = merge_hemispheres(sess, qc_results=roi_qc)
            sources = {r: (m['signal'], m['timestamps']) for r, m in merged.items()}
        else:
            sources_by_region = defaultdict(list)
            for roi_name, trace in sess.photometry_data.items():
                region = get_roi_region(roi_name, subject_id)
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
                        raw[subject_id][region_name][event_type].append(row)

        n_loaded += 1
        if n_loaded % 20 == 0:
            logging.info(f"  Loaded {n_loaded} sessions...")

    # Average across sessions per mouse
    traces = {}
    for subj, regions in raw.items():
        traces[subj] = {}
        for region, events in regions.items():
            traces[subj][region] = {}
            for evt, rows in events.items():
                if rows:
                    traces[subj][region][evt] = np.nanmean(np.array(rows), axis=0)

    logging.info(f"Total sessions loaded: {n_loaded}, {len(traces)} mice")
    return traces, time_axis, genotype_map


def compute_timing_table(traces, time_axis, genotype_map):
    """
    Compute peak latency and onset latency for each mouse × region × event.

    Returns DataFrame with columns:
        subject_id, genotype, region, event_type, peak_latency, onset_latency, n_trials
    """
    rows = []
    for subj, regions in traces.items():
        geno = genotype_map.get(subj, 'Unknown')
        for region, events in regions.items():
            for evt, mean_trace in events.items():
                peak_lat = extract_peak_latency(mean_trace, time_axis, peak_window=PEAK_WINDOW)
                onset_lat = extract_onset_latency(
                    mean_trace, time_axis,
                    threshold_n_std=2.0,
                    baseline_window=BASELINE,
                    search_window=ONSET_WINDOW,
                    n_consecutive=3,
                )
                rows.append({
                    'subject_id': subj, 'genotype': geno, 'region': region,
                    'event_type': evt,
                    'peak_latency': peak_lat, 'onset_latency': onset_lat,
                })

    return pd.DataFrame(rows)


def build_timing_figures(timing_df, output_dir):
    """Build timing comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ['peak_latency', 'onset_latency']:
        metric_label = 'Peak Latency (s)' if metric == 'peak_latency' else 'Onset Latency (s)'

        # ── Figure 1: By region × genotype, grouped by event ──
        events_present = timing_df['event_type'].unique()
        n_events = len(events_present)
        fig, axes = plt.subplots(1, n_events, figsize=(5 * n_events, 5), sharey=True)
        if n_events == 1:
            axes = [axes]
        fig.suptitle(f'{metric_label} by Region and Genotype', fontsize=13)

        stats_rows = []
        for ax, evt in zip(axes, events_present):
            subset = timing_df[timing_df['event_type'] == evt].dropna(subset=[metric])
            if subset.empty:
                ax.set_title(EVENT_LABELS.get(evt, evt))
                continue

            # Strip plot with genotype colors
            for geno in ['D1', 'D2']:
                for region in ['DMS', 'VMS', 'VLS']:
                    vals = subset[(subset['genotype'] == geno) & (subset['region'] == region)][metric].values
                    if len(vals) == 0:
                        continue
                    x_pos = {'DMS': 0, 'VMS': 1, 'VLS': 2}[region]
                    offset = -0.15 if geno == 'D1' else 0.15
                    color = GENOTYPE_COLORS[geno]
                    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(vals))
                    ax.scatter(np.full_like(vals, x_pos + offset) + jitter, vals,
                               color=color, edgecolor='k', linewidth=0.5, s=50,
                               label=f'{geno}' if region == 'DMS' else '', zorder=5)
                    ax.plot([x_pos + offset - 0.08, x_pos + offset + 0.08],
                            [np.mean(vals)] * 2, color=color, linewidth=2.5, zorder=6)

            # Within-region D1 vs D2 permutation test
            for region in ['DMS', 'VMS']:
                d1_vals = subset[(subset['genotype'] == 'D1') & (subset['region'] == region)][metric].values
                d2_vals = subset[(subset['genotype'] == 'D2') & (subset['region'] == region)][metric].values
                if len(d1_vals) >= 2 and len(d2_vals) >= 2:
                    perm = permutation_test(d1_vals, d2_vals, n_perm=10000, seed=42)
                    stats_rows.append({
                        'metric': metric, 'event': evt,
                        'comparison': f'D1_vs_D2_{region}',
                        'test': 'Permutation (10k)',
                        'observed': perm['observed'],
                        'p': perm['p'], 'n1': perm['n1'], 'n2': perm['n2'],
                    })

            # Within-genotype DMS vs VMS permutation test
            for geno in ['D1', 'D2']:
                dms_vals = subset[(subset['genotype'] == geno) & (subset['region'] == 'DMS')][metric].values
                vms_vals = subset[(subset['genotype'] == geno) & (subset['region'] == 'VMS')][metric].values
                if len(dms_vals) >= 2 and len(vms_vals) >= 2:
                    perm = permutation_test(dms_vals, vms_vals, n_perm=10000, seed=42)
                    stats_rows.append({
                        'metric': metric, 'event': evt,
                        'comparison': f'DMS_vs_VMS_{geno}',
                        'test': 'Permutation (10k)',
                        'observed': perm['observed'],
                        'p': perm['p'], 'n1': perm['n1'], 'n2': perm['n2'],
                    })

            regions_present = sorted(subset['region'].unique())
            ax.set_xticks(range(len(regions_present)))
            ax.set_xticklabels(regions_present, fontsize=10)
            ax.set_title(EVENT_LABELS.get(evt, evt), fontsize=11)
            if ax == axes[0]:
                ax.set_ylabel(metric_label, fontsize=10)
                handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                                        color=GENOTYPE_COLORS[g], markersize=7, label=g)
                           for g in ['D1', 'D2']]
                ax.legend(handles=handles, fontsize=8, loc='upper right')
            sns.despine(ax=ax)

        fig_path = output_dir / f"timing_{metric}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved: {fig_path}")

        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            stats_df.to_csv(output_dir / f"timing_{metric}_stats.csv", index=False)

    # ── Figure 2: Within-subject DMS vs VMS timing (paired) ──
    # Find subjects with both DMS and VMS data
    paired_rows = []
    for subj in timing_df['subject_id'].unique():
        subj_data = timing_df[timing_df['subject_id'] == subj]
        regions = subj_data['region'].unique()
        if 'DMS' in regions and 'VMS' in regions:
            for evt in EVENT_TYPES:
                dms_row = subj_data[(subj_data['region'] == 'DMS') & (subj_data['event_type'] == evt)]
                vms_row = subj_data[(subj_data['region'] == 'VMS') & (subj_data['event_type'] == evt)]
                if not dms_row.empty and not vms_row.empty:
                    paired_rows.append({
                        'subject_id': subj,
                        'genotype': subj_data['genotype'].iloc[0],
                        'event_type': evt,
                        'dms_peak_latency': dms_row['peak_latency'].iloc[0],
                        'vms_peak_latency': vms_row['peak_latency'].iloc[0],
                        'dms_onset_latency': dms_row['onset_latency'].iloc[0],
                        'vms_onset_latency': vms_row['onset_latency'].iloc[0],
                    })

    if paired_rows:
        paired_df = pd.DataFrame(paired_rows)
        paired_df['peak_lat_diff'] = paired_df['dms_peak_latency'] - paired_df['vms_peak_latency']
        paired_df['onset_lat_diff'] = paired_df['dms_onset_latency'] - paired_df['vms_onset_latency']
        paired_df.to_csv(output_dir / "timing_paired_dms_vs_vms.csv", index=False)
        logging.info(f"Paired timing data: {len(paired_df)} rows from "
                     f"{paired_df['subject_id'].nunique()} mice")

        # Plot paired differences
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Within-Subject DMS vs VMS Timing Difference', fontsize=13)
        for ax, metric in zip(axes, ['peak_lat_diff', 'onset_lat_diff']):
            metric_label = 'Peak Lat (DMS - VMS) (s)' if 'peak' in metric else 'Onset Lat (DMS - VMS) (s)'
            for evt in EVENT_TYPES:
                evt_data = paired_df[paired_df['event_type'] == evt].dropna(subset=[metric])
                if evt_data.empty:
                    continue
                vals = evt_data[metric].values
                genos = evt_data['genotype'].values
                x_pos = list(EVENT_TYPES).index(evt)
                for v, g in zip(vals, genos):
                    ax.scatter(x_pos, v, color=GENOTYPE_COLORS.get(g, 'grey'),
                               edgecolor='k', linewidth=0.5, s=60, zorder=5)
                ax.plot([x_pos - 0.15, x_pos + 0.15], [np.nanmean(vals)] * 2,
                        color='k', linewidth=2.5, zorder=6)

            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_xticks(range(len(EVENT_TYPES)))
            ax.set_xticklabels([EVENT_LABELS[e] for e in EVENT_TYPES], fontsize=8, rotation=15)
            ax.set_ylabel(metric_label, fontsize=10)
            sns.despine(ax=ax)

        fig_path = output_dir / "timing_paired_dms_vms_diff.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved: {fig_path}")

    # ── Figure 3: Hit vs Miss timing within region ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Hit vs Miss Timing Comparison (per mouse)', fontsize=13)

    for ax, metric in zip(axes, ['peak_latency', 'onset_latency']):
        metric_label = 'Peak Latency (s)' if 'peak' in metric else 'Onset Latency (s)'
        hit_data = timing_df[timing_df['event_type'] == 'change_hit'].set_index(['subject_id', 'region'])
        miss_data = timing_df[timing_df['event_type'] == 'change_miss'].set_index(['subject_id', 'region'])
        common_idx = hit_data.index.intersection(miss_data.index)

        if len(common_idx) == 0:
            continue

        for subj, region in common_idx:
            hit_val = hit_data.loc[(subj, region), metric]
            miss_val = miss_data.loc[(subj, region), metric]
            if isinstance(hit_val, pd.Series):
                hit_val = hit_val.iloc[0]
            if isinstance(miss_val, pd.Series):
                miss_val = miss_val.iloc[0]
            if np.isfinite(hit_val) and np.isfinite(miss_val):
                geno = genotype_map_global.get(subj, 'Unknown')
                color = GENOTYPE_COLORS.get(geno, 'grey')
                marker = 'o' if region == 'DMS' else 's'
                ax.plot([0, 1], [hit_val, miss_val], color=color, alpha=0.4, linewidth=0.8)
                ax.scatter(0, hit_val, color=color, marker=marker, edgecolor='k',
                           linewidth=0.5, s=50, zorder=5)
                ax.scatter(1, miss_val, color=color, marker=marker, edgecolor='k',
                           linewidth=0.5, s=50, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Hit', 'Miss'], fontsize=11)
        ax.set_ylabel(metric_label, fontsize=10)
        sns.despine(ax=ax)

    fig_path = output_dir / "timing_hit_vs_miss.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig_path}")


# Module-level for access in figure builder
genotype_map_global = {}


def main():
    global genotype_map_global

    parser = argparse.ArgumentParser(description="Phase 2d: Response Timing Analysis")
    _rr = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str, default=str(_rr / "photom_data"))
    parser.add_argument("--output_dir", type=str,
                        default=str(_rr / "FIGURES" / "phase2_timing"))
    parser.add_argument("--no-qc", action="store_true", default=False)
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    use_qc = not args.no_qc
    all_sessions = io.find_all_sessions(
        args.root_dir, recursive=True,
        min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    logging.info(f"Discovered {len(all_sessions)} sessions.")

    logging.info(f"Collecting traces (QC={'ON' if use_qc else 'OFF'})...")
    traces, time_axis, genotype_map = collect_per_mouse_traces(
        all_sessions, use_qc=use_qc, max_sessions=args.max_sessions,
    )
    genotype_map_global = genotype_map

    if time_axis is None:
        logging.error("No valid data extracted.")
        sys.exit(1)

    # Compute timing table
    timing_df = compute_timing_table(traces, time_axis, genotype_map)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full timing table
    timing_csv = output_path / "timing_summary.csv"
    timing_df.to_csv(timing_csv, index=False)
    logging.info(f"Timing table saved: {timing_csv} ({len(timing_df)} rows)")

    # Log summary
    for geno in ['D1', 'D2']:
        for region in sorted(timing_df[timing_df['genotype'] == geno]['region'].unique()):
            sub = timing_df[(timing_df['genotype'] == geno) & (timing_df['region'] == region)]
            n_mice = sub['subject_id'].nunique()
            for evt in EVENT_TYPES:
                evt_sub = sub[sub['event_type'] == evt]
                if not evt_sub.empty:
                    pl = evt_sub['peak_latency'].dropna()
                    ol = evt_sub['onset_latency'].dropna()
                    logging.info(
                        f"  {geno}/{region}/{evt}: "
                        f"peak_lat={pl.mean():.3f}±{pl.std():.3f}s (n={len(pl)}), "
                        f"onset_lat={ol.mean():.3f}±{ol.std():.3f}s (n={len(ol)})"
                    )

    # Build figures
    build_timing_figures(timing_df, output_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()
