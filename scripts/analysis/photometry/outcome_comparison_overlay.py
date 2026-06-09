"""
Outcome-Comparison Overlay Figures

Overlays PETH traces from different outcomes (Hit, Miss, FA, Early/Late FA)
on the SAME subplot for direct visual comparison, with proper shared-baseline
normalization.

**Normalization rule**: All conditions within a plot share the SAME baseline
definition (pre-event window). Each trial is individually z-scored to its own
pre-event baseline [-2, 0]s, then averaged. This ensures:
  - No circular baseline inflation
  - Relative magnitudes are preserved across conditions
  - Individual trial variability is equalized

Figures produced (per region):
  A: Change-aligned — Hit vs Miss overlaid (D1), same (D2)
  B: FA lick-aligned — Early vs Late overlaid (D1), same (D2)
  C: All motor responses overlaid — Hit-lick vs FA-Early vs FA-Late (D1), same (D2)
  D: Outcome selectivity — Hit, Miss, FA together in one subplot per genotype

Usage:
    py scripts/analysis/photometry/outcome_comparison_overlay.py
    py scripts/analysis/photometry/outcome_comparison_overlay.py --qc  # with QC filtering
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
    GENOTYPE_COLORS, OUTCOME_COLORS, ROI_TO_REGION, SUBJECT_GENOTYPE,
    PETH_WINDOW, PETH_BASELINE, FA_RT_SPLIT, MIN_PHOTOM_CSV_BYTES,
    get_roi_region,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres,
    REGION_PAIRS,
)
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.group_utils import (
    get_genotype, get_region, _get_event_times,
)
from visdetect_photom.analysis.group_statistics import (
    mannwhitney_with_effect_size, permutation_test, format_stats_table,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────
WINDOW = (-2.0, 4.0)
BASELINE = (-2.0, 0.0)  # Shared baseline for ALL conditions

# Outcome colors for overlays
OVERLAY_COLORS = {
    'change_hit':  '#2ca02c',  # green
    'change_miss': '#9467bd',  # purple
    'change_cr':   '#17becf',  # cyan
    'fa_lick':     '#d62728',  # red
    'fa_early':    '#ff7f0e',  # orange
    'fa_late':     '#8c564b',  # brown
    'hit_lick':    '#1f77b4',  # blue
}
OVERLAY_LABELS = {
    'change_hit':  'Hit (change-aligned)',
    'change_miss': 'Miss (change-aligned)',
    'change_cr':   'CR (change-aligned)',
    'fa_lick':     'FA (lick-aligned)',
    'fa_early':    'Early FA (RT \u2264 3s)',
    'fa_late':     'Late FA (RT > 3s)',
    'hit_lick':    'Hit (lick-aligned)',
}

EVENT_TYPES = ['change_hit', 'change_miss', 'change_cr', 'fa_lick', 'fa_early', 'fa_late', 'hit_lick']


# ── Data collection with QC ──────────────────────────────────

def collect_peths_with_qc(
    session_files: list,
    use_qc: bool = True,
    merge_hemi: bool = True,
    max_sessions: int = None,
) -> dict:
    """
    Load sessions, apply QC, optionally merge hemispheres, extract PETHs.

    Returns:
        data[genotype][region_or_roi][event_type] = list of (subject_id, trace)
        time_axis: 1D array
        qc_summary: list of per-session QC result dicts
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    qc_summary = []
    time_axis = None
    n_loaded = 0
    n_skipped_behav = 0
    n_skipped_geno = 0
    n_roi_failed = 0

    for sf in session_files:
        if max_sessions and n_loaded >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.debug(f"Skipped loading: {e}")
            continue

        genotype = get_genotype(sess.subject_id)
        if genotype == 'Unknown':
            n_skipped_geno += 1
            continue

        # -- Behavioral QC --
        if use_qc:
            behav_qc = check_behavioral_engagement(sess)
            if not behav_qc['pass']:
                n_skipped_behav += 1
                qc_summary.append({
                    'session_id': sess.session_id, 'genotype': genotype,
                    'behav_pass': False, 'behav_reasons': behav_qc['fail_reasons'],
                })
                continue

        # -- Signal QC per ROI --
        roi_qc = compute_session_roi_qc(sess) if use_qc else {}

        # Determine what to extract from
        if merge_hemi and use_qc:
            merged = merge_hemispheres(sess, qc_results=roi_qc)
            # Build extraction sources: {label -> (signal, timestamps)}
            sources = {}
            for region_name, minfo in merged.items():
                sources[region_name] = (minfo['signal'], minfo['timestamps'])
                qc_summary.append({
                    'session_id': sess.session_id, 'genotype': genotype,
                    'region': region_name, 'source': minfo['source'],
                    'rois_used': minfo['rois_used'], 'behav_pass': True,
                })
        else:
            # Use individual ROIs, optionally filtered by QC
            # Group by subject-aware region (DMS, VMS, VLS)
            subject_id = sess.subject_id
            if not subject_id.startswith('BG_') and subject_id.isdigit():
                subject_id_full = f'BG_{subject_id.zfill(3)}'
            else:
                subject_id_full = subject_id

            region_traces = defaultdict(list)
            for roi_name, trace in sess.photometry_data.items():
                if use_qc:
                    qc_result = roi_qc.get(roi_name, {})
                    if not qc_result.get('pass', False):
                        n_roi_failed += 1
                        qc_summary.append({
                            'session_id': sess.session_id, 'genotype': genotype,
                            'roi': roi_name, 'roi_pass': False,
                            'fail_reasons': qc_result.get('fail_reasons', []),
                        })
                        continue
                region_full = get_roi_region(roi_name, subject_id_full) or roi_name
                base_region = region_full.rsplit('_', 1)[0]
                region_traces[base_region].append((trace.signal, trace.timestamps))

            sources = {}
            for region_name, traces in region_traces.items():
                if len(traces) == 1:
                    sources[region_name] = traces[0]
                elif len(traces) >= 2:
                    min_len = min(len(t[0]) for t in traces)
                    avg_sig = np.mean([t[0][:min_len] for t in traces], axis=0)
                    sources[region_name] = (avg_sig, traces[0][1][:min_len])

        if not sources:
            continue

        # -- Extract PETHs --
        for source_label, (signal, timestamps) in sources.items():
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
                        data[genotype][source_label][event_type].append(
                            (sess.subject_id, row)
                        )

        n_loaded += 1
        if n_loaded % 20 == 0:
            logging.info(f"  Loaded {n_loaded} sessions...")

    logging.info(f"Loaded {n_loaded} sessions (skipped: {n_skipped_geno} unknown genotype, "
                 f"{n_skipped_behav} behavioral QC, {n_roi_failed} ROI QC failures)")
    return dict(data), time_axis, qc_summary


# ── Aggregation ──────────────────────────────────────────────

PEAK_WINDOW = (0.0, 1.5)  # Post-event window for peak extraction


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
        'n_trials': len(trial_list),
        'n_mice': n_mice,
        'per_mouse': per_mouse,
    }


# ── Plotting helpers ─────────────────────────────────────────

def plot_overlay(ax, time_axis, traces_dict, title='', ylabel='\u0394 z-dF/F',
                 xlabel='Time (s)'):
    """
    Overlay multiple conditions on one subplot.

    Args:
        traces_dict: {event_type: aggregated_dict} — each has 'mean', 'sem', etc.
    """
    for evt, agg in traces_dict.items():
        if agg is None:
            continue
        color = OVERLAY_COLORS.get(evt, 'grey')
        label = f"{OVERLAY_LABELS.get(evt, evt)} (n={agg['n_trials']}, {agg['n_mice']} mice)"
        ax.plot(time_axis, agg['mean'], color=color, label=label, linewidth=1.5)
        ax.fill_between(time_axis, agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                        color=color, alpha=0.15)

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    sns.despine(ax=ax)


# ── Main figure builder ─────────────────────────────────────

def _extract_peak(trace, time_axis, peak_window=PEAK_WINDOW):
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


def _pairwise_stats(agg_a, agg_b, time_axis, comp_label, genotype, region):
    """Compute permutation test on per-mouse peak z-dF/F between two conditions."""
    if agg_a is None or agg_b is None:
        return None
    pm_a = agg_a.get('per_mouse', {})
    pm_b = agg_b.get('per_mouse', {})
    vals_a = np.array([_extract_peak(v, time_axis) for v in pm_a.values()])
    vals_b = np.array([_extract_peak(v, time_axis) for v in pm_b.values()])
    vals_a = vals_a[np.isfinite(vals_a)]
    vals_b = vals_b[np.isfinite(vals_b)]
    perm = permutation_test(vals_a, vals_b, n_perm=10000, seed=42)
    mwu = mannwhitney_with_effect_size(vals_a, vals_b)
    perm['comparison'] = comp_label
    perm['test'] = 'Permutation (10k)'
    perm['region'] = region
    perm['genotype'] = genotype
    perm['effect_size'] = mwu['rank_biserial_r']
    return perm


def build_outcome_comparison_figure(data, time_axis, output_dir, region_label):
    """Build multi-panel outcome comparison for a single region."""

    # Aggregate per genotype per event
    agg = {}
    for geno in ['D1', 'D2']:
        agg[geno] = {}
        for evt in EVENT_TYPES:
            trials = data.get(geno, {}).get(region_label, {}).get(evt, [])
            agg[geno][evt] = aggregate_traces(trials)

    # Check we have data
    has_d1 = any(agg['D1'][e] is not None for e in EVENT_TYPES)
    has_d2 = any(agg['D2'][e] is not None for e in EVENT_TYPES)
    if not has_d1 and not has_d2:
        logging.warning(f"No data for {region_label}, skipping figure.")
        return []

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(f'Outcome Comparison — {region_label}', fontsize=14, y=0.98)
    gs = gridspec.GridSpec(5, 2, hspace=0.45, wspace=0.3)

    # ── Row 1: Change-aligned Hit vs Miss vs CR ──
    for col, geno in enumerate(['D1', 'D2']):
        ax = fig.add_subplot(gs[0, col])
        traces = {
            'change_hit': agg[geno].get('change_hit'),
            'change_miss': agg[geno].get('change_miss'),
            'change_cr': agg[geno].get('change_cr'),
        }
        plot_overlay(ax, time_axis, traces,
                     title=f'A: Change-aligned Hit vs Miss vs CR — {geno}',
                     xlabel='Time from change onset (s)')

    # ── Row 2: FA Early vs Late ──
    for col, geno in enumerate(['D1', 'D2']):
        ax = fig.add_subplot(gs[1, col])
        traces = {
            'fa_early': agg[geno].get('fa_early'),
            'fa_late': agg[geno].get('fa_late'),
        }
        plot_overlay(ax, time_axis, traces,
                     title=f'B: FA Lick Early vs Late — {geno}',
                     xlabel='Time from FA lick (s)')

    # ── Row 3: Motor responses — Hit lick vs Early FA vs Late FA ──
    for col, geno in enumerate(['D1', 'D2']):
        ax = fig.add_subplot(gs[2, col])
        traces = {
            'hit_lick': agg[geno].get('hit_lick'),
            'fa_early': agg[geno].get('fa_early'),
            'fa_late': agg[geno].get('fa_late'),
        }
        plot_overlay(ax, time_axis, traces,
                     title=f'C: Motor Responses (lick-aligned) — {geno}',
                     xlabel='Time from lick (s)')

    # ── Row 4: Change-aligned — Hit vs Miss vs CR with Hit−Miss difference ──
    # (Same alignment event throughout; shows sensory selectivity)
    for col, geno in enumerate(['D1', 'D2']):
        ax = fig.add_subplot(gs[3, col])
        # Plot Hit and Miss traces
        hit_agg = agg[geno].get('change_hit')
        miss_agg = agg[geno].get('change_miss')
        if hit_agg is not None:
            ax.plot(time_axis, hit_agg['mean'], color=OVERLAY_COLORS['change_hit'],
                    label=f"Hit (n={hit_agg['n_trials']})", linewidth=1.5)
            ax.fill_between(time_axis, hit_agg['mean'] - hit_agg['sem'],
                            hit_agg['mean'] + hit_agg['sem'],
                            color=OVERLAY_COLORS['change_hit'], alpha=0.15)
        if miss_agg is not None:
            ax.plot(time_axis, miss_agg['mean'], color=OVERLAY_COLORS['change_miss'],
                    label=f"Miss (n={miss_agg['n_trials']})", linewidth=1.5)
            ax.fill_between(time_axis, miss_agg['mean'] - miss_agg['sem'],
                            miss_agg['mean'] + miss_agg['sem'],
                            color=OVERLAY_COLORS['change_miss'], alpha=0.15)
        # Add Hit−Miss difference trace
        if hit_agg is not None and miss_agg is not None:
            diff = hit_agg['mean'] - miss_agg['mean']
            diff_sem = np.sqrt(hit_agg['sem']**2 + miss_agg['sem']**2)
            ax.plot(time_axis, diff, color='k', linestyle='--',
                    label='Hit \u2212 Miss', linewidth=1.2, alpha=0.8)
            ax.fill_between(time_axis, diff - diff_sem, diff + diff_sem,
                            color='k', alpha=0.08)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
        ax.set_title(f'D: Sensory Selectivity (change-aligned) \u2014 {geno}', fontsize=10)
        ax.set_xlabel('Time from change onset (s)', fontsize=9)
        ax.set_ylabel('\u0394 z-dF/F', fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        sns.despine(ax=ax)

    # ── Row 5: Lick-aligned Hit lick vs FA lick ──
    for col, geno in enumerate(['D1', 'D2']):
        ax = fig.add_subplot(gs[4, col])
        traces = {
            'hit_lick': agg[geno].get('hit_lick'),
            'fa_lick': agg[geno].get('fa_lick'),
        }
        plot_overlay(ax, time_axis, traces,
                     title=f'E: Lick-aligned Hit vs FA — {geno}',
                     xlabel='Time from lick (s)')

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"outcome_comparison_{region_label}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig_path}")

    # ── Pairwise statistics ──
    stats_results = []
    comparisons = [
        ('change_hit', 'change_miss', 'Hit_vs_Miss_change'),
        ('change_hit', 'change_cr', 'Hit_vs_CR_change'),
        ('fa_early', 'fa_late', 'EarlyFA_vs_LateFA'),
        ('hit_lick', 'fa_lick', 'HitLick_vs_FALick'),
    ]
    for geno in ['D1', 'D2']:
        for evt_a, evt_b, comp_label in comparisons:
            stat = _pairwise_stats(
                agg[geno].get(evt_a), agg[geno].get(evt_b),
                time_axis, comp_label, geno, region_label,
            )
            if stat is not None:
                stats_results.append(stat)

    return stats_results


# ── Entry point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Outcome-comparison overlay figures with QC and hemisphere merging")
    _rr = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str, default=str(_rr / "photom_data"))
    parser.add_argument("--output_dir", type=str,
                        default=str(_rr / "FIGURES" / "outcome_comparison"))
    parser.add_argument("--no-qc", action="store_true", default=False,
                        help="Disable signal & behavioral QC filtering")
    parser.add_argument("--no-merge", action="store_true", default=False,
                        help="Disable hemisphere merging (uses individual ROIs grouped by region)")
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

    # Collect PETHs with QC on by default
    use_qc = not args.no_qc
    use_merge = use_qc and (not args.no_merge)
    data, time_axis, qc_summary = collect_peths_with_qc(
        all_sessions,
        use_qc=use_qc,
        merge_hemi=use_merge,
        max_sessions=args.max_sessions,
    )

    if time_axis is None:
        logging.error("No valid data extracted.")
        sys.exit(1)

    # Log trial counts
    for geno in ['D1', 'D2']:
        for region_label in sorted(data.get(geno, {}).keys()):
            counts = {e: len(data[geno][region_label].get(e, [])) for e in EVENT_TYPES}
            nonzero = {e: c for e, c in counts.items() if c > 0}
            if nonzero:
                logging.info(f"  {geno} / {region_label}: {nonzero}")

    # Build figures per region
    all_regions = set()
    for geno_data in data.values():
        all_regions.update(geno_data.keys())

    all_stats = []
    for region_label in sorted(all_regions):
        logging.info(f"Building figure for {region_label}...")
        stats = build_outcome_comparison_figure(data, time_axis, output_path, region_label)
        all_stats.extend(stats)

    # Save stats
    if all_stats:
        stats_df = format_stats_table(all_stats, save_path=str(output_path / "outcome_comparison_stats.csv"))
        logging.info(f"Stats summary:\n{stats_df.to_string(index=False)}")

    # Save QC summary
    if qc_summary:
        qc_df = pd.DataFrame(qc_summary)
        qc_path = output_path / "qc_summary.csv"
        qc_df.to_csv(qc_path, index=False)
        logging.info(f"QC summary saved: {qc_path}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
