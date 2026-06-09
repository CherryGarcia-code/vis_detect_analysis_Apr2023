"""
Group-level aggregation utilities for multi-subject photometry analysis.

Functions for loading genotype maps, aggregating PETHs by group,
and computing per-session summary metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from visdetect_photom.core.constants import (
    SUBJECT_GENOTYPE, GENOTYPE_COLORS, REGION_COLORS, OUTCOME_COLORS,
    ROI_TO_REGION, CATCH_THRESHOLD, FA_RT_SPLIT, CHANGE_SIZES,
    get_roi_region,
)
from visdetect_photom.analysis.statistics import calculate_sdt_metrics, extract_peth


def load_genotype_map() -> Dict[str, str]:
    """Return the subject-to-genotype mapping from constants."""
    return dict(SUBJECT_GENOTYPE)


def get_genotype(subject_id: str) -> str:
    """Look up genotype for a subject. Handles 'BG_016', '016', and '16' formats."""
    result = SUBJECT_GENOTYPE.get(subject_id)
    if result is not None:
        return result
    # Try with BG_ prefix
    if not subject_id.startswith('BG_'):
        result = SUBJECT_GENOTYPE.get(f'BG_{subject_id}')
        if result is not None:
            return result
        # Try zero-padded to 3 digits (e.g. '18' -> 'BG_018')
        if subject_id.isdigit():
            result = SUBJECT_GENOTYPE.get(f'BG_{subject_id.zfill(3)}')
            if result is not None:
                return result
    # Try without BG_ prefix
    if subject_id.startswith('BG_'):
        result = SUBJECT_GENOTYPE.get(subject_id.replace('BG_', ''))
        if result is not None:
            return result
    return 'Unknown'


def get_region(roi_name: str, subject_id: str = None) -> str:
    """Look up region for an ROI name, with optional subject-specific override.

    VMS subjects (BG_008–011) use G0/G2 for VMS instead of DMS.
    Returns roi_name if not found in any mapping.
    """
    region = get_roi_region(roi_name, subject_id)
    return region if region is not None else roi_name


def compute_session_summary(session) -> Dict:
    """
    Compute a summary dict for a single Session object.

    Returns dict with:
        subject_id, session_date, session_id, genotype,
        n_trials, n_hit, n_miss, n_fa, n_abort, n_cr,
        d_prime, criterion_c, sdt_hit_rate, sdt_fa_rate,
        hit_rate, fa_rate_behavioral,
        n_fa_early, n_fa_late, median_rt_hit,
        roi_names (list of available ROIs)
    """
    trials = session.trials
    outcomes = np.array([t.outcome for t in trials])
    change_sizes = np.array([t.change_size if t.change_size is not None else np.nan
                             for t in trials])
    rts = np.array([t.reaction_time if t.reaction_time is not None else np.nan
                    for t in trials])

    n_trials = len(trials)
    n_hit = int((outcomes == 'Hit').sum())
    n_miss = int((outcomes == 'Miss').sum())
    n_fa = int((outcomes == 'FA').sum())
    n_abort = int((outcomes == 'Abort').sum())
    n_cr = int((outcomes == 'CR').sum())

    # SDT metrics
    sdt = calculate_sdt_metrics(outcomes, change_sizes)

    # Behavioral hit rate (go trials only)
    n_go_behav = n_hit + n_miss
    hit_rate = n_hit / n_go_behav if n_go_behav > 0 else np.nan
    fa_rate_behavioral = n_fa / n_trials if n_trials > 0 else np.nan

    # FA split
    fa_mask = outcomes == 'FA'
    fa_rts = rts[fa_mask]
    fa_rts_valid = fa_rts[np.isfinite(fa_rts)]
    n_fa_early = int((fa_rts_valid <= FA_RT_SPLIT).sum())
    n_fa_late = int((fa_rts_valid > FA_RT_SPLIT).sum())

    # Hit RT
    hit_rts = rts[outcomes == 'Hit']
    hit_rts_valid = hit_rts[np.isfinite(hit_rts)]
    median_rt_hit = float(np.median(hit_rts_valid)) if len(hit_rts_valid) > 0 else np.nan

    # Available ROIs
    roi_names = list(session.photometry_data.keys())

    # Per-ROI photometry QC
    from visdetect_photom.core.qc import compute_session_roi_qc
    qc_results = compute_session_roi_qc(session)
    n_rois_passing = 0
    qc_cols = {}
    for roi, metrics in qc_results.items():
        qc_cols[f'qc_{roi}_passed'] = metrics['pass']
        qc_cols[f'qc_{roi}_variance'] = metrics['variance']
        qc_cols[f'qc_{roi}_snr'] = metrics['snr']
        qc_cols[f'qc_{roi}_nan_frac'] = metrics['nan_fraction']
        if metrics['pass']:
            n_rois_passing += 1

    return {
        'subject_id': session.subject_id,
        'session_date': session.session_date,
        'session_id': session.session_id,
        'genotype': get_genotype(session.subject_id),
        'n_trials': n_trials,
        'n_hit': n_hit,
        'n_miss': n_miss,
        'n_fa': n_fa,
        'n_abort': n_abort,
        'n_cr': n_cr,
        'd_prime': sdt['d_prime'],
        'criterion_c': sdt['criterion_c'],
        'sdt_hit_rate': sdt['sdt_hit_rate'],
        'sdt_fa_rate': sdt['sdt_fa_rate'],
        'hit_rate': hit_rate,
        'fa_rate_behavioral': fa_rate_behavioral,
        'n_fa_early': n_fa_early,
        'n_fa_late': n_fa_late,
        'median_rt_hit': median_rt_hit,
        'roi_names': roi_names,
        'n_rois_passing': n_rois_passing,
        **qc_cols,
    }


def aggregate_peth_by_group(
    sessions: list,
    event_type: str,
    group_key: str = 'genotype',
    roi_name: str = 'G0',
    window: Tuple[float, float] = (-2.0, 4.0),
    baseline_window: Tuple[float, float] = (-2.0, 0.0),
) -> Dict[str, Dict]:
    """
    Aggregate PETHs across sessions, grouped by a session-level attribute.

    Args:
        sessions: List of Session objects.
        event_type: One of 'change_hit', 'change_miss', 'fa_lick', 'fa_early', 'fa_late'.
        group_key: How to group sessions ('genotype', 'subject_id', 'region').
        roi_name: Which ROI to extract (e.g. 'G0', 'G2', 'G4', 'G5').
        window: PETH extraction window.
        baseline_window: Baseline for trial-based z-scoring.

    Returns:
        Dict keyed by group label, each containing:
            'time_axis': 1D array
            'mean_trace': 1D array (grand mean across all trials in group)
            'sem_trace': 1D array
            'per_mouse_means': Dict[subject_id -> 1D mean trace]
            'n_trials': int
            'n_sessions': int
    """
    from collections import defaultdict

    group_trials = defaultdict(list)  # group_label -> list of (subject_id, peth_row)
    time_axis = None

    for sess in sessions:
        trace = sess.photometry_data.get(roi_name)
        if trace is None:
            continue

        # Determine group label
        if group_key == 'genotype':
            label = get_genotype(sess.subject_id)
        elif group_key == 'subject_id':
            label = sess.subject_id
        else:
            label = str(group_key)

        # Get event times based on event_type
        event_times = _get_event_times(sess, event_type)
        if len(event_times) == 0:
            continue

        # Extract PETH
        t_ax, peth_mat = extract_peth(
            trace.signal, trace.timestamps, event_times,
            window=window, baseline_window=baseline_window
        )

        if time_axis is None:
            time_axis = t_ax

        # Store each valid trial row
        for row_idx in range(peth_mat.shape[0]):
            row = peth_mat[row_idx]
            if not np.all(np.isnan(row)):
                group_trials[label].append((sess.subject_id, row))

    # Aggregate per group
    result = {}
    for label, trial_list in group_trials.items():
        all_rows = np.array([r for _, r in trial_list])
        subjects = [s for s, _ in trial_list]

        grand_mean = np.nanmean(all_rows, axis=0)
        grand_sem = np.nanstd(all_rows, axis=0) / np.sqrt(np.sum(~np.isnan(all_rows), axis=0))

        # Per-mouse means
        unique_subjects = sorted(set(subjects))
        per_mouse = {}
        for subj in unique_subjects:
            subj_rows = np.array([r for s, r in trial_list if s == subj])
            per_mouse[subj] = np.nanmean(subj_rows, axis=0)

        result[label] = {
            'time_axis': time_axis,
            'mean_trace': grand_mean,
            'sem_trace': grand_sem,
            'per_mouse_means': per_mouse,
            'n_trials': len(trial_list),
            'n_sessions': len(set(subjects)),
        }

    return result


def _get_event_times(session, event_type: str) -> np.ndarray:
    """
    Extract absolute event timestamps from a Session based on event type.

    Valid event_types:
        'change_hit'  — change onset for Hit trials
        'change_miss' — change onset for Miss trials
        'change_cr'   — nominal change onset for CR (correct rejection) trials
        'fa_lick'     — lick time for all FA trials
        'fa_early'    — lick time for early FA (RT <= FA_RT_SPLIT)
        'fa_late'     — lick time for late FA (RT > FA_RT_SPLIT)
        'hit_lick'    — lick time for Hit trials
    """
    times = []
    for t in session.trials:
        if event_type == 'change_hit' and t.outcome == 'Hit':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)

        elif event_type == 'change_miss' and t.outcome == 'Miss':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)

        elif event_type == 'change_cr' and t.outcome == 'CR':
            if t.absolute_change_time is not None:
                times.append(t.absolute_change_time)

        elif event_type == 'fa_lick' and t.outcome == 'FA':
            if t.absolute_reaction_time is not None:
                times.append(t.absolute_reaction_time)

        elif event_type == 'fa_early' and t.outcome == 'FA':
            if (t.absolute_reaction_time is not None and
                t.reaction_time is not None and t.reaction_time <= FA_RT_SPLIT):
                times.append(t.absolute_reaction_time)

        elif event_type == 'fa_late' and t.outcome == 'FA':
            if (t.absolute_reaction_time is not None and
                t.reaction_time is not None and t.reaction_time > FA_RT_SPLIT):
                times.append(t.absolute_reaction_time)

        elif event_type == 'hit_lick' and t.outcome == 'Hit':
            if t.absolute_reaction_time is not None:
                times.append(t.absolute_reaction_time)

    return np.array(times)
