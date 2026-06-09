"""
C2: Outcome-Comparison Overlay Figures

Per-region PETH comparisons across task outcomes with shared-baseline normalization.
QC filtering and hemisphere merging ensure one trace per region per condition.

Subplots (3 rows x 2 columns):
  Row 1: Change-aligned — Hit vs Miss vs CR (shared change-onset baseline)
  Row 2: Lick-aligned — Hit-lick vs FA-lick (shared lick-onset baseline)
  Row 3: FA subtypes — Early FA vs Late FA (shared FA-lick baseline)
  Columns: D1, D2 genotypes

One figure per region (DMS, VLS).

Usage:
    py scripts/analysis/photometry/02_outcome_comparison.py
    py scripts/analysis/photometry/02_outcome_comparison.py --max_sessions 10
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
    GENOTYPE_COLORS, OUTCOME_COLORS, PETH_WINDOW, PETH_BASELINE,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement,
    extract_merged_region_peths, MIN_TRIAL_VALID_FRACTION, REGION_PAIRS,
)
from visdetect_photom.analysis.group_statistics import (
    mannwhitney_with_effect_size, format_stats_table,
)
from visdetect_photom.analysis.group_utils import (
    get_genotype, compute_session_summary,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────
WINDOW = (-2.0, 4.0)
BASELINE = (-2.0, 0.0)
PEAK_WINDOW = (0.0, 1.5)

# Conditions grouped by alignment event (only same-alignment on same subplot)
SUBPLOT_ROWS = [
    {
        'name': 'Change-aligned',
        'conditions': {
            'Hit': 'change_hit',
            'Miss': 'change_miss',
            'CR': 'change_cr',
        },
        'xlabel': 'Time from change onset (s)',
    },
    {
        'name': 'Lick-aligned',
        'conditions': {
            'Hit lick': 'hit_lick',
            'FA lick': 'fa_lick',
        },
        'xlabel': 'Time from lick (s)',
    },
    {
        'name': 'FA subtypes',
        'conditions': {
            'Early FA': 'fa_early',
            'Late FA': 'fa_late',
        },
        'xlabel': 'Time from FA lick (s)',
    },
]

# Colors for each condition label
CONDITION_COLORS = {
    'Hit': OUTCOME_COLORS['Hit'],       # green
    'Miss': OUTCOME_COLORS['Miss'],     # purple
    'CR': OUTCOME_COLORS['CR'],         # cyan
    'Hit lick': OUTCOME_COLORS['Hit'],  # green
    'FA lick': OUTCOME_COLORS['FA'],    # red
    'Early FA': '#ff7f0e',              # orange
    'Late FA': '#8c564b',              # brown
}

# All event types we need to extract
ALL_EVENT_TYPES = set()
for row in SUBPLOT_ROWS:
    ALL_EVENT_TYPES.update(row['conditions'].values())


# ── Data collection ───────────────────────────────────────────

def collect_region_peths_across_sessions(
    session_files_list: list,
    max_sessions: int = None,
) -> tuple:
    """
    Load sessions, run QC, merge hemispheres, extract per-trial PETHs
    grouped by (genotype, region, event_type).

    Returns:
        data[genotype][region][event_type] = list of (subject_id, 1D_trace)
        time_axis: 1D array
        summaries: list of session summary dicts
        qc_records: list of per-ROI QC result dicts (for CSV export)
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    summaries = []
    qc_records = []
    time_axis = None
    n_loaded = 0
    n_skipped_genotype = 0
    n_skipped_behavior = 0
    n_skipped_no_region = 0

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
            n_skipped_genotype += 1
            continue

        # Behavioral engagement check
        behav_qc = check_behavioral_engagement(sess)
        if not behav_qc['pass']:
            n_skipped_behavior += 1
            continue

        # Run photometry QC
        roi_qc = compute_session_roi_qc(sess)

        # Record QC for export
        for roi_name, qc_result in roi_qc.items():
            qc_records.append({
                'session_id': sess.session_id,
                'subject_id': sess.subject_id,
                'genotype': genotype,
                'roi': roi_name,
                'region': qc_result.get('region', ''),
                'passed': qc_result['pass'],
                'variance': qc_result['variance'],
                'snr': qc_result['snr'],
                'nan_fraction': qc_result['nan_fraction'],
                'baseline_drift': qc_result['baseline_drift'],
                'n_valid': qc_result['n_valid'],
                'fail_reasons': '; '.join(qc_result['fail_reasons']),
            })

        # Session summary
        try:
            summaries.append(compute_session_summary(sess))
        except Exception:
            pass

        # Extract merged-region PETHs for all event types
        session_has_data = False
        for event_type in ALL_EVENT_TYPES:
            region_peths = extract_merged_region_peths(
                sess, event_type, qc_results=roi_qc,
                window=WINDOW, baseline_window=BASELINE,
            )
            for region_name, (peth_mat, t_ax, source) in region_peths.items():
                if time_axis is None:
                    time_axis = t_ax

                # Filter trials: >50% finite
                for row_idx in range(peth_mat.shape[0]):
                    row = peth_mat[row_idx]
                    frac_valid = np.sum(np.isfinite(row)) / len(row)
                    if frac_valid >= MIN_TRIAL_VALID_FRACTION:
                        data[genotype][region_name][event_type].append(
                            (sess.subject_id, row)
                        )
                        session_has_data = True

        if session_has_data:
            n_loaded += 1
            if n_loaded % 10 == 0:
                logging.info(f"  Loaded {n_loaded} sessions...")
        else:
            n_skipped_no_region += 1

    logging.info(f"Sessions loaded: {n_loaded} "
                 f"(skipped: {n_skipped_genotype} unknown genotype, "
                 f"{n_skipped_behavior} low engagement, "
                 f"{n_skipped_no_region} no QC-passing region)")
    return dict(data), time_axis, summaries, qc_records


# ── Aggregation ──────────────────────────────────────────────

def aggregate_traces(trial_list):
    """From list of (subject_id, trace) tuples, compute grand mean, SEM, per-mouse means."""
    if not trial_list:
        return None
    all_rows = np.array([r for _, r in trial_list])
    n_valid = np.sum(~np.isnan(all_rows), axis=0)
    grand_mean = np.nanmean(all_rows, axis=0)
    grand_sem = np.nanstd(all_rows, axis=0) / np.sqrt(np.maximum(n_valid, 1))

    subjects = sorted(set(s for s, _ in trial_list))
    per_mouse = {}
    for subj in subjects:
        subj_rows = np.array([r for s, r in trial_list if s == subj])
        per_mouse[subj] = np.nanmean(subj_rows, axis=0)

    return {
        'mean': grand_mean,
        'sem': grand_sem,
        'per_mouse': per_mouse,
        'n_trials': len(trial_list),
        'n_mice': len(subjects),
    }


def extract_peak(trace, time_axis, peak_window=PEAK_WINDOW):
    """Extract peak value within peak_window from a single trace."""
    mask = (time_axis >= peak_window[0]) & (time_axis <= peak_window[1])
    if not np.any(mask):
        return np.nan
    segment = trace[mask]
    valid = segment[np.isfinite(segment)]
    return float(np.nanmax(valid)) if len(valid) > 0 else np.nan


# ── Plotting ─────────────────────────────────────────────────

def plot_outcome_overlay(ax, time_axis, condition_aggs, title='', xlabel='Time (s)'):
    """
    Plot overlaid condition traces on a single axis.

    Args:
        condition_aggs: Dict[condition_label -> aggregate_traces result or None]
    """
    for label, agg in condition_aggs.items():
        if agg is None:
            continue
        color = CONDITION_COLORS.get(label, 'black')
        ax.plot(time_axis, agg['mean'], color=color,
                label=f"{label} (n={agg['n_trials']}, {agg['n_mice']}m)",
                linewidth=1.5)
        ax.fill_between(time_axis,
                        agg['mean'] - agg['sem'],
                        agg['mean'] + agg['sem'],
                        color=color, alpha=0.15)

    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('z-scored dF/F', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    sns.despine(ax=ax)


# ── Figure builder ───────────────────────────────────────────

def build_outcome_figure(data, time_axis, region, output_dir):
    """
    Build one 3×2 figure for a single region.
    Rows: Change-aligned, Lick-aligned, FA subtypes
    Columns: D1, D2

    Returns list of stats result dicts.
    """
    genotypes = ['D1', 'D2']
    n_rows = len(SUBPLOT_ROWS)
    n_cols = len(genotypes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows),
                             sharex=True, sharey='row')
    fig.suptitle(f'Outcome Comparison — {region}', fontsize=14, y=0.98)

    # Handle single-row case
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    stats_results = []

    for row_idx, row_config in enumerate(SUBPLOT_ROWS):
        for col_idx, geno in enumerate(genotypes):
            ax = axes[row_idx, col_idx]

            # Aggregate each condition
            condition_aggs = {}
            for label, event_type in row_config['conditions'].items():
                trials = data.get(geno, {}).get(region, {}).get(event_type, [])
                condition_aggs[label] = aggregate_traces(trials)

            title = f"{row_config['name']} — {geno}"
            plot_outcome_overlay(ax, time_axis, condition_aggs,
                                 title=title, xlabel=row_config['xlabel'])

            # Pairwise stats on per-mouse peaks
            labels = list(row_config['conditions'].keys())
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    l1, l2 = labels[i], labels[j]
                    agg1 = condition_aggs.get(l1)
                    agg2 = condition_aggs.get(l2)
                    if agg1 is None or agg2 is None:
                        continue
                    peaks1 = np.array([extract_peak(m, time_axis) for m in agg1['per_mouse'].values()])
                    peaks2 = np.array([extract_peak(m, time_axis) for m in agg2['per_mouse'].values()])
                    peaks1 = peaks1[np.isfinite(peaks1)]
                    peaks2 = peaks2[np.isfinite(peaks2)]
                    stat = mannwhitney_with_effect_size(peaks1, peaks2)
                    stat['comparison'] = f"{l1}_vs_{l2}"
                    stat['test'] = 'Mann-Whitney U'
                    stat['region'] = region
                    stat['genotype'] = geno
                    stat['subplot'] = row_config['name']
                    stat['effect_size'] = stat.pop('rank_biserial_r')
                    stats_results.append(stat)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"C2_outcome_{region}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved figure: {fig_path}")

    return stats_results


# ── Entry point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C2: Outcome Comparison Overlays")
    _rr = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str,
                        default=str(_rr / "photom_data"))
    parser.add_argument("--output_dir", type=str,
                        default=str(_rr / "FIGURES" / "C2_outcome_comparison"))
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest CSV for pre-filtered sessions")
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    output_path = Path(args.output_dir)

    # Discover sessions
    all_sessions = io.find_all_sessions(str(root_path), recursive=True)
    logging.info(f"Discovered {len(all_sessions)} sessions.")

    # Optional manifest filter
    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            df_m = pd.read_csv(manifest_path)
            if 'session_name' in df_m.columns:
                valid_ids = set(df_m['session_name'].astype(str))
                filtered = [s for s in all_sessions
                            if any(vid in Path(s.get('trials', '')).name
                                   for vid in valid_ids)]
                logging.info(f"Manifest filter: {len(all_sessions)} → {len(filtered)}")
                all_sessions = filtered

    # Collect data with QC + hemisphere merging
    logging.info("Loading sessions with QC filtering and hemisphere merging...")
    data, time_axis, summaries, qc_records = collect_region_peths_across_sessions(
        all_sessions, max_sessions=args.max_sessions,
    )

    if time_axis is None:
        logging.error("No valid data extracted.")
        sys.exit(1)

    # Log trial counts
    for geno in ['D1', 'D2']:
        for region in sorted(data.get(geno, {}).keys()):
            for evt in sorted(ALL_EVENT_TYPES):
                n = len(data.get(geno, {}).get(region, {}).get(evt, []))
                if n > 0:
                    logging.info(f"  {geno} / {region} / {evt}: {n} trials")

    # Determine regions with data
    all_regions = set()
    for geno_data in data.values():
        all_regions.update(geno_data.keys())

    # Build figures
    all_stats = []
    for region in sorted(all_regions):
        logging.info(f"Building figure for {region}...")
        stats = build_outcome_figure(data, time_axis, region, output_path)
        all_stats.extend(stats)

    # Save stats
    if all_stats:
        stats_df = format_stats_table(all_stats,
                                       save_path=str(output_path / "C2_stats_summary.csv"))
        logging.info(f"Stats summary:\n{stats_df.to_string(index=False)}")

    # Save QC summary
    if qc_records:
        qc_df = pd.DataFrame(qc_records)
        qc_path = output_path / "C2_qc_summary.csv"
        qc_df.to_csv(qc_path, index=False)
        # Log QC summary
        n_total_rois = len(qc_df)
        n_passed = qc_df['passed'].sum()
        logging.info(f"QC summary: {n_passed}/{n_total_rois} ROIs passed "
                     f"({100*n_passed/n_total_rois:.0f}%)")
        # Per-region pass rates
        for region in sorted(qc_df['region'].unique()):
            r_df = qc_df[qc_df['region'] == region]
            logging.info(f"  {region}: {r_df['passed'].sum()}/{len(r_df)} passed")
        logging.info(f"QC details saved: {qc_path}")

    # Save session summaries
    if summaries:
        sum_df = pd.DataFrame(summaries)
        sum_df.to_csv(output_path / "C2_session_summaries.csv", index=False)
        logging.info(f"Session summaries saved ({len(summaries)} sessions)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
