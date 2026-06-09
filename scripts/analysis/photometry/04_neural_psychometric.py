"""
Phase 2c: Neural Psychometric Functions

Peak z-dF/F as a function of stimulus change_size for Hit trials, by region
and genotype. Compares neural sensitivity to behavioral d'.

Panels:
  A: Neural psychometric curves — peak z-dF/F vs change_size, per region (D1)
  B: Same for D2
  C: Overlay D1 vs D2 within DMS
  D: Overlay D1 vs D2 within VMS
  E: Neural sensitivity (slope) vs behavioral d' scatter

Usage:
    py scripts/analysis/photometry/04_neural_psychometric.py
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
from scipy.optimize import curve_fit

# ── Path setup ────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import (
    GENOTYPE_COLORS, REGION_COLORS, SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES,
    CHANGE_SIZES, CATCH_THRESHOLD, get_roi_region,
)
from visdetect_photom.core.qc import (
    compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres,
)
from visdetect_photom.analysis.statistics import extract_peth, calculate_sdt_metrics
from visdetect_photom.analysis.group_utils import get_genotype

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

WINDOW = (-2.0, 4.0)
BASELINE = (-2.0, 0.0)
PEAK_WINDOW = (0.0, 1.5)


def _sigmoid(x, a, b, c, d):
    """4-parameter sigmoid: d + (a - d) / (1 + exp(-b * (x - c)))"""
    return d + (a - d) / (1.0 + np.exp(-b * (x - c)))


def collect_change_size_peths(session_files, use_qc=True, max_sessions=None):
    """
    Collect per-trial peak z-dF/F grouped by genotype, region, change_size.

    Returns:
        peaks[genotype][region][change_size] = list of (subject_id, peak_value)
        dprime_by_subject: {subject_id: d'}
    """
    peaks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dprime_by_subject = {}
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
            subject_id_full = f'BG_{subject_id.zfill(3)}'
        else:
            subject_id_full = subject_id

        if use_qc:
            behav_qc = check_behavioral_engagement(sess)
            if not behav_qc['pass']:
                continue

        # Compute d' for this session
        outcomes = np.array([t.outcome for t in sess.trials])
        change_sizes_arr = np.array([t.change_size if t.change_size is not None else np.nan
                                      for t in sess.trials])
        sdt = calculate_sdt_metrics(outcomes, change_sizes_arr)
        if subject_id_full not in dprime_by_subject:
            dprime_by_subject[subject_id_full] = []
        dprime_by_subject[subject_id_full].append(sdt['d_prime'])

        # QC + merge
        roi_qc = compute_session_roi_qc(sess) if use_qc else {}
        if use_qc:
            merged = merge_hemispheres(sess, qc_results=roi_qc)
            sources = {r: (m['signal'], m['timestamps']) for r, m in merged.items()}
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

        # Extract per-change_size PETHs for Hit trials
        for region_name, (signal, timestamps) in sources.items():
            for trial in sess.trials:
                if trial.outcome != 'Hit':
                    continue
                if trial.absolute_change_time is None:
                    continue
                cs = trial.change_size
                if cs is None or cs <= CATCH_THRESHOLD:
                    continue

                # Extract single-trial PETH
                t_ax, peth_mat = extract_peth(
                    signal, timestamps, np.array([trial.absolute_change_time]),
                    window=WINDOW, baseline_window=BASELINE,
                )
                if peth_mat.shape[0] == 0:
                    continue
                row = peth_mat[0]
                if np.sum(np.isfinite(row)) < len(row) * 0.5:
                    continue

                # Extract peak
                peak_mask = (t_ax >= PEAK_WINDOW[0]) & (t_ax <= PEAK_WINDOW[1])
                peak_vals = row[peak_mask]
                valid_peaks = peak_vals[np.isfinite(peak_vals)]
                if len(valid_peaks) == 0:
                    continue
                peak = float(np.max(valid_peaks))

                # Round change_size to nearest canonical value
                cs_rounded = min(CHANGE_SIZES, key=lambda x: abs(x - cs))
                peaks[genotype][region_name][cs_rounded].append((subject_id_full, peak))

        n_loaded += 1
        if n_loaded % 20 == 0:
            logging.info(f"  Loaded {n_loaded} sessions...")

    # Average d' per subject
    dprime_avg = {s: np.nanmean(vals) for s, vals in dprime_by_subject.items()}

    logging.info(f"Total sessions loaded: {n_loaded}")
    return dict(peaks), dprime_avg


def build_psychometric_figures(peaks, dprime_by_subject, output_dir):
    """Build neural psychometric curves and interaction plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Per-genotype neural psychometric by region ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Neural Psychometric Functions', fontsize=14)

    stats_rows = []

    for ax_idx, geno in enumerate(['D1', 'D2']):
        ax = axes[ax_idx]
        for region in ['DMS', 'VMS']:
            cs_data = peaks.get(geno, {}).get(region, {})
            if not cs_data:
                continue

            # Per-mouse means at each change_size
            x_vals = []
            y_means = []
            y_sems = []
            for cs in CHANGE_SIZES:
                trials = cs_data.get(cs, [])
                if not trials:
                    continue
                # Get per-mouse means
                subjects = sorted(set(s for s, _ in trials))
                mouse_means = []
                for subj in subjects:
                    subj_peaks = [p for s, p in trials if s == subj]
                    mouse_means.append(np.mean(subj_peaks))
                mouse_means = np.array(mouse_means)

                x_vals.append(cs)
                y_means.append(np.mean(mouse_means))
                y_sems.append(np.std(mouse_means) / np.sqrt(max(len(mouse_means), 1)))

                stats_rows.append({
                    'genotype': geno, 'region': region, 'change_size': cs,
                    'peak_mean': np.mean(mouse_means), 'peak_sem': y_sems[-1],
                    'n_mice': len(mouse_means),
                    'n_trials': len(trials),
                })

            if not x_vals:
                continue

            x_vals = np.array(x_vals)
            y_means = np.array(y_means)
            y_sems = np.array(y_sems)

            color = REGION_COLORS[region]
            ax.errorbar(x_vals, y_means, yerr=y_sems, color=color, marker='o',
                        markersize=6, linewidth=1.5, capsize=3, label=region)

            # Fit sigmoid if enough points
            if len(x_vals) >= 3:
                try:
                    popt, _ = curve_fit(_sigmoid, x_vals, y_means,
                                        p0=[max(y_means), 2.0, 2.0, min(y_means)],
                                        maxfev=5000)
                    x_fit = np.linspace(min(x_vals) - 0.1, max(x_vals) + 0.1, 100)
                    ax.plot(x_fit, _sigmoid(x_fit, *popt), color=color, linestyle='--',
                            linewidth=1, alpha=0.6)
                except Exception:
                    pass  # Fit failed, skip

        ax.set_title(f'{geno}', fontsize=12)
        ax.set_xlabel('Change Size (TF ratio)', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Peak \u0394 z-dF/F', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_xticks(CHANGE_SIZES)
        ax.set_xticklabels([str(c) for c in CHANGE_SIZES], fontsize=8)
        sns.despine(ax=ax)

    fig_path = output_dir / "neural_psychometric_by_genotype.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {fig_path}")

    # ── Figure 2: Per-region neural psychometric by genotype ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle('Neural Psychometric Functions (by region)', fontsize=14)

    for ax_idx, region in enumerate(['DMS', 'VMS']):
        ax = axes2[ax_idx]
        for geno in ['D1', 'D2']:
            cs_data = peaks.get(geno, {}).get(region, {})
            if not cs_data:
                continue
            x_vals, y_means, y_sems = [], [], []
            for cs in CHANGE_SIZES:
                trials = cs_data.get(cs, [])
                if not trials:
                    continue
                subjects = sorted(set(s for s, _ in trials))
                mouse_means = [np.mean([p for s2, p in trials if s2 == subj]) for subj in subjects]
                x_vals.append(cs)
                y_means.append(np.mean(mouse_means))
                y_sems.append(np.std(mouse_means) / np.sqrt(max(len(mouse_means), 1)))

            if not x_vals:
                continue
            color = GENOTYPE_COLORS[geno]
            ax.errorbar(x_vals, np.array(y_means), yerr=np.array(y_sems),
                        color=color, marker='o', markersize=6, linewidth=1.5,
                        capsize=3, label=geno)

        ax.set_title(f'{region}', fontsize=12)
        ax.set_xlabel('Change Size (TF ratio)', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Peak \u0394 z-dF/F', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xscale('log')
        ax.set_xticks(CHANGE_SIZES)
        ax.set_xticklabels([str(c) for c in CHANGE_SIZES], fontsize=8)
        sns.despine(ax=ax)

    fig2_path = output_dir / "neural_psychometric_by_region.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    logging.info(f"Saved: {fig2_path}")

    # Save stats
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(output_dir / "neural_psychometric_stats.csv", index=False)
        logging.info(f"Stats saved ({len(stats_rows)} rows)")

    return stats_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2c: Neural Psychometric Functions")
    _rr = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str, default=str(_rr / "photom_data"))
    parser.add_argument("--output_dir", type=str,
                        default=str(_rr / "FIGURES" / "phase2_neural_psychometric"))
    parser.add_argument("--no-qc", action="store_true", default=False)
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    use_qc = not args.no_qc
    all_sessions = io.find_all_sessions(
        args.root_dir, recursive=True,
        min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    logging.info(f"Discovered {len(all_sessions)} sessions.")

    logging.info(f"Collecting per-change-size PETHs (QC={'ON' if use_qc else 'OFF'})...")
    peaks, dprime_by_subject = collect_change_size_peths(
        all_sessions, use_qc=use_qc, max_sessions=args.max_sessions,
    )

    # Log counts
    for geno in ['D1', 'D2']:
        for region in sorted(peaks.get(geno, {}).keys()):
            cs_counts = {cs: len(peaks[geno][region].get(cs, []))
                         for cs in CHANGE_SIZES}
            logging.info(f"  {geno}/{region}: {cs_counts}")

    output_path = Path(args.output_dir)
    build_psychometric_figures(peaks, dprime_by_subject, output_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()
