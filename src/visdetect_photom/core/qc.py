"""
Quality Control (QC) module for photometry signals.

Computes per-ROI signal quality metrics and provides:
  - Signal variance, SNR, NaN fraction, baseline drift
  - Pass/fail classification per ROI per session
  - Hemisphere merging (average L/R when both pass QC for a region)

Typical usage:
    qc = compute_trace_qc(signal, timestamps)
    if qc['pass']:
        ...  # use this ROI
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from visdetect_photom.core.constants import SAMPLING_FREQ, ROI_TO_REGION, get_roi_region


# ── QC thresholds ────────────────────────────────────────────
# These can be tuned empirically. Start conservative.
DEFAULT_QC_THRESHOLDS = {
    'min_variance': 1e-6,       # dF/F variance; below this = dead/flat fiber
    'max_nan_fraction': 0.3,    # >30% NaN = too many gaps
    'min_snr': 1.5,             # SNR < 1.5 = signal barely above noise
    'max_baseline_drift': 5.0,  # |slope| of linear fit to z-scored signal
    'min_n_samples': 500,       # Minimum usable samples (5 s at 100 Hz)
}

# Minimum fraction of valid (finite) time bins for a trial PETH to be included
MIN_TRIAL_VALID_FRACTION = 0.5

# Region pairing: which ROIs belong to same region (L/R hemisphere)
# Default for DMS/VLS subjects (BG_013+)
REGION_PAIRS = {
    'DMS': ('G0', 'G2'),   # DMS Left, DMS Right
    'VLS': ('G4', 'G5'),   # VLS Left, VLS Right
}


def get_region_pairs_for_subject(subject_id: str = None) -> Dict[str, Tuple[str, str]]:
    """Get region→(roi_L, roi_R) pairs, adjusted for subject-specific mappings.

    VMS subjects (BG_008–011) use G0/G2 for VMS instead of DMS.
    """
    if subject_id is None:
        return dict(REGION_PAIRS)

    # Build from the actual ROI→region mapping for this subject
    pairs = {}
    # Check what region each ROI maps to for this subject
    roi_regions = {}
    for roi in ['G0', 'G2', 'G4', 'G5']:
        region = get_roi_region(roi, subject_id)
        if region is not None:
            # Strip _L/_R to get the base region name
            base = region.rsplit('_', 1)[0]
            roi_regions.setdefault(base, []).append(roi)

    for base_region, rois in roi_regions.items():
        if len(rois) >= 2:
            pairs[base_region] = (rois[0], rois[1])
        elif len(rois) == 1:
            pairs[base_region] = (rois[0], rois[0])  # Single hemisphere

    return pairs


# ── Per-trace QC ─────────────────────────────────────────────

def compute_trace_qc(signal: np.ndarray,
                     timestamps: Optional[np.ndarray] = None,
                     thresholds: Optional[Dict] = None) -> Dict:
    """
    Compute QC metrics for a single photometry trace (dF/F or z-scored).

    Args:
        signal: 1D array of processed photometry signal.
        timestamps: 1D array of timestamps (used for drift calculation).
        thresholds: Override default QC thresholds.

    Returns:
        Dict with keys:
            variance, nan_fraction, snr, baseline_drift, n_valid,
            pass (bool), fail_reasons (list of str)
    """
    thresh = {**DEFAULT_QC_THRESHOLDS, **(thresholds or {})}
    signal = np.asarray(signal, dtype=float)

    n_total = len(signal)
    valid_mask = np.isfinite(signal)
    n_valid = int(valid_mask.sum())
    nan_fraction = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0

    fail_reasons = []

    # ── Variance ──
    if n_valid > 1:
        variance = float(np.var(signal[valid_mask]))
    else:
        variance = 0.0
    if variance < thresh['min_variance']:
        fail_reasons.append(f"low_variance ({variance:.2e})")

    # ── NaN fraction ──
    if nan_fraction > thresh['max_nan_fraction']:
        fail_reasons.append(f"high_nan_fraction ({nan_fraction:.2f})")

    # ── Sample count ──
    if n_valid < thresh['min_n_samples']:
        fail_reasons.append(f"too_few_samples ({n_valid})")

    # ── SNR (signal-to-noise ratio) ──
    # Estimate SNR as: std of slow component / std of fast residual
    # Slow = rolling mean (5s window), fast = signal - slow
    snr = np.nan
    if n_valid > 500:
        clean = signal.copy()
        clean[~valid_mask] = np.nanmean(signal)
        kernel_size = min(500, n_valid // 2)
        if kernel_size > 10:
            # Uniform rolling mean for speed
            kernel = np.ones(kernel_size) / kernel_size
            slow = np.convolve(clean, kernel, mode='same')
            fast = clean - slow
            std_slow = np.std(slow[valid_mask])
            std_fast = np.std(fast[valid_mask])
            if std_fast > 1e-10:
                snr = float(std_slow / std_fast)
            else:
                snr = 0.0
    if np.isfinite(snr) and snr < thresh['min_snr']:
        fail_reasons.append(f"low_snr ({snr:.2f})")

    # ── Baseline drift ──
    # Linear fit slope on z-scored signal; high |slope| = uncorrected bleaching
    baseline_drift = np.nan
    if timestamps is not None and n_valid > 100:
        t_valid = timestamps[valid_mask]
        s_valid = signal[valid_mask]
        # Normalize time to [0, 1] so slope is in signal-units per session
        t_norm = (t_valid - t_valid[0])
        t_range = t_norm[-1] - t_norm[0]
        if t_range > 0:
            t_norm = t_norm / t_range
        try:
            coef = np.polyfit(t_norm, s_valid, deg=1)
            baseline_drift = float(abs(coef[0]))
        except Exception:
            baseline_drift = np.nan
    if np.isfinite(baseline_drift) and baseline_drift > thresh['max_baseline_drift']:
        fail_reasons.append(f"high_drift ({baseline_drift:.2f})")

    passed = len(fail_reasons) == 0

    return {
        'variance': variance,
        'nan_fraction': nan_fraction,
        'snr': snr,
        'baseline_drift': baseline_drift,
        'n_valid': n_valid,
        'n_total': n_total,
        'pass': passed,
        'fail_reasons': fail_reasons,
    }


# ── Session-level QC ────────────────────────────────────────

def compute_session_roi_qc(session, thresholds: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    Run QC on all ROIs in a Session object.

    Args:
        session: A Session object with photometry_data dict of PhotometryTrace.
        thresholds: Override default QC thresholds.

    Returns:
        Dict[roi_name -> qc_result_dict], each containing 'pass', metrics, etc.
    """
    roi_qc = {}
    subject_id = getattr(session, 'subject_id', None)
    # Normalize subject_id to BG_ format for region lookup
    if subject_id and not subject_id.startswith('BG_'):
        subject_id_full = f'BG_{subject_id.zfill(3)}' if subject_id.isdigit() else subject_id
    else:
        subject_id_full = subject_id

    for roi_name, trace in session.photometry_data.items():
        qc = compute_trace_qc(
            trace.signal,
            timestamps=trace.timestamps,
            thresholds=thresholds,
        )
        qc['roi'] = roi_name
        qc['region'] = get_roi_region(roi_name, subject_id_full) or roi_name
        roi_qc[roi_name] = qc
    return roi_qc


def get_passing_rois(session, thresholds: Optional[Dict] = None) -> List[str]:
    """Return list of ROI names that pass QC for a session."""
    qc = compute_session_roi_qc(session, thresholds)
    return [roi for roi, result in qc.items() if result['pass']]


# ── Hemisphere merging ───────────────────────────────────────

def merge_hemispheres(session, qc_results: Optional[Dict] = None,
                      thresholds: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    For each brain region, determine which ROIs to use and optionally merge L/R.

    Strategy:
      - If both L and R pass QC → average their signals (merged trace)
      - If only one passes → use that one alone
      - If neither passes → skip that region

    Args:
        session: Session object.
        qc_results: Pre-computed QC results (from compute_session_roi_qc).
                     If None, computed here.
        thresholds: QC thresholds.

    Returns:
        Dict[region_name -> {'signal': ndarray, 'timestamps': ndarray,
                             'source': str, 'rois_used': list}]
    """
    if qc_results is None:
        qc_results = compute_session_roi_qc(session, thresholds)

    # Get subject-aware region pairs
    subject_id = getattr(session, 'subject_id', None)
    if subject_id and not subject_id.startswith('BG_'):
        subject_id_full = f'BG_{subject_id.zfill(3)}' if subject_id.isdigit() else subject_id
    else:
        subject_id_full = subject_id
    region_pairs = get_region_pairs_for_subject(subject_id_full)

    merged = {}

    for region_name, (roi_l, roi_r) in region_pairs.items():
        l_pass = qc_results.get(roi_l, {}).get('pass', False)
        r_pass = qc_results.get(roi_r, {}).get('pass', False)
        l_trace = session.photometry_data.get(roi_l)
        r_trace = session.photometry_data.get(roi_r)

        if l_pass and r_pass and l_trace is not None and r_trace is not None:
            # Both pass — average signals
            # Ensure same length (should be, but guard)
            min_len = min(len(l_trace.signal), len(r_trace.signal))
            avg_signal = (l_trace.signal[:min_len] + r_trace.signal[:min_len]) / 2.0
            merged[region_name] = {
                'signal': avg_signal,
                'timestamps': l_trace.timestamps[:min_len],
                'source': 'merged',
                'rois_used': [roi_l, roi_r],
            }

        elif l_pass and l_trace is not None:
            merged[region_name] = {
                'signal': l_trace.signal,
                'timestamps': l_trace.timestamps,
                'source': roi_l,
                'rois_used': [roi_l],
            }

        elif r_pass and r_trace is not None:
            merged[region_name] = {
                'signal': r_trace.signal,
                'timestamps': r_trace.timestamps,
                'source': roi_r,
                'rois_used': [roi_r],
            }
        # else: neither passes, skip

    return merged


# ── Behavioral QC ────────────────────────────────────────────

def check_behavioral_engagement(session,
                                min_go_trials: int = 10,
                                min_hit_rate: float = 0.1,
                                max_abort_fraction: float = 0.5) -> Dict:
    """
    Check whether a session has sufficient behavioral engagement.

    Args:
        session: Session object.
        min_go_trials: Minimum number of go trials (Hit + Miss).
        min_hit_rate: Minimum hit rate on go trials.
        max_abort_fraction: Maximum fraction of abort trials.

    Returns:
        Dict with 'pass', 'n_go', 'hit_rate', 'abort_fraction', 'fail_reasons'.
    """
    outcomes = [t.outcome for t in session.trials]
    n_trials = len(outcomes)
    n_hit = sum(1 for o in outcomes if o == 'Hit')
    n_miss = sum(1 for o in outcomes if o == 'Miss')
    n_abort = sum(1 for o in outcomes if o == 'Abort')
    n_go = n_hit + n_miss

    hit_rate = n_hit / n_go if n_go > 0 else 0.0
    abort_fraction = n_abort / n_trials if n_trials > 0 else 1.0

    fail_reasons = []
    if n_go < min_go_trials:
        fail_reasons.append(f"too_few_go_trials ({n_go})")
    if hit_rate < min_hit_rate:
        fail_reasons.append(f"low_hit_rate ({hit_rate:.2f})")
    if abort_fraction > max_abort_fraction:
        fail_reasons.append(f"high_abort_fraction ({abort_fraction:.2f})")

    return {
        'pass': len(fail_reasons) == 0,
        'n_go': n_go,
        'n_trials': n_trials,
        'hit_rate': hit_rate,
        'abort_fraction': abort_fraction,
        'fail_reasons': fail_reasons,
    }


# ── Hemisphere-merged PETH extraction ────────────────────────

def extract_merged_region_peths(
    session,
    event_type: str,
    qc_results: Optional[Dict[str, Dict]] = None,
    window: Tuple[float, float] = (-2.0, 4.0),
    baseline_window: Tuple[float, float] = (-2.0, 0.0),
) -> Dict[str, Tuple[Optional[np.ndarray], np.ndarray, str]]:
    """
    For each region (DMS, VLS), merge hemispheres via QC then extract PETHs.

    Uses merge_hemispheres() to get one signal per region, then calls
    extract_peth() on the merged signal.

    Args:
        session: Session object.
        event_type: Event type for _get_event_times() (e.g. 'change_hit').
        qc_results: Pre-computed QC results. Computed if None.
        window: PETH extraction window.
        baseline_window: Baseline for per-trial z-scoring.

    Returns:
        Dict[region_name -> (peth_matrix, time_axis, source_str)]
        peth_matrix is (n_events x n_timepoints) or None if no data.
        source_str is 'merged', roi_name, or absent if region skipped.
    """
    # Lazy imports to avoid circular dependency
    from visdetect_photom.analysis.statistics import extract_peth
    from visdetect_photom.analysis.group_utils import _get_event_times

    # Get event times
    event_times = _get_event_times(session, event_type)
    if len(event_times) == 0:
        return {}

    # Merge hemispheres
    merged = merge_hemispheres(session, qc_results=qc_results)

    results = {}
    for region_name, region_data in merged.items():
        signal = region_data['signal']
        timestamps = region_data['timestamps']
        source = region_data['source']

        t_ax, peth_mat = extract_peth(
            signal, timestamps, event_times,
            window=window, baseline_window=baseline_window,
        )
        results[region_name] = (peth_mat, t_ax, source)

    return results
