import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple

from visdetect_photom.core.constants import (
    PETH_WINDOW, SAMPLING_FREQ, CATCH_THRESHOLD
)

def extract_peth(signal: np.ndarray, timestamps: np.ndarray, event_times: np.ndarray,
                 window: Tuple[float, float] = PETH_WINDOW, fs: float = SAMPLING_FREQ,
                 baseline_window: Optional[Tuple[float, float]] = None,
                 normalize: str = 'subtract') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Peri-Event Time Histogram (PETH) matrix.

    Args:
        signal: 1D array of continuous signal (typically session-z-scored dF/F).
        timestamps: 1D array of timestamps corresponding to signal.
        event_times: 1D array of event timestamps to align to.
        window: (start, end) relative to event in seconds.
        fs: Sampling frequency (approximate) for output time vector.
        baseline_window: (start, end) relative to event for per-trial normalization.
                         If provided, normalization mode is controlled by `normalize`.
        normalize: How to normalize each trial relative to baseline_window.
            'subtract' — subtract baseline mean only (default, recommended).
                         Preserves session-level scaling, avoids noisy per-trial
                         std division. Y-axis = Δ(z-dF/F).
            'zscore'   — subtract baseline mean AND divide by baseline std.
                         Full per-trial z-scoring.  Y-axis = z-score.
            None       — no normalization (raw extraction).

    Returns:
        time_axis: 1D array of relative time points.
        peth_matrix: 2D array (n_events x n_timepoints) of aligned signal.
    """
    # Create relative time axis
    n_samples = int((window[1] - window[0]) * fs)
    time_axis = np.linspace(window[0], window[1], n_samples)

    peth_matrix = []

    # Using searchsorted for nearest neighbor
    dt = 1.0 / fs

    # Pre-calculate baseline mask if needed
    baseline_mask = None
    if baseline_window is not None and normalize is not None:
        baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])

    for et in event_times:
        if np.isnan(et):
            peth_matrix.append(np.full(n_samples, np.nan))
            continue

        # Define target times for this event
        target_times = et + time_axis

        # Find nearest indices (timestamps must be sorted)
        idx = np.searchsorted(timestamps, target_times)
        idx = np.clip(idx, 0, len(signal) - 1)

        # Check if the found timestamp is actually close (within 1.5/fs)
        found_times = timestamps[idx]
        valid_mask = np.abs(found_times - target_times) < (1.5 * dt)

        vals = signal[idx].astype(float)
        vals[~valid_mask] = np.nan

        # Per-trial baseline normalization
        if baseline_mask is not None:
            baseline_vals = vals[baseline_mask]
            n_valid_bl = np.sum(~np.isnan(baseline_vals))
            if n_valid_bl > 1:
                b_mean = np.nanmean(baseline_vals)
                if normalize == 'subtract':
                    vals = vals - b_mean
                elif normalize == 'zscore':
                    b_std = np.nanstd(baseline_vals)
                    if b_std > 1e-6:
                        vals = (vals - b_mean) / b_std
                    else:
                        vals = vals - b_mean
            else:
                vals[:] = np.nan

        peth_matrix.append(vals)

    return time_axis, np.array(peth_matrix)


def normalize_peths_by_condition(
    peth_target: np.ndarray,
    peth_reference: np.ndarray,
    time_axis: np.ndarray,
    baseline_window: Tuple[float, float] = (-2.0, 0.0),
) -> np.ndarray:
    """
    Normalize a PETH matrix using the baseline std from a reference condition.

    Use case: divide all conditions (Miss, FA, etc.) by **Hit trials'** baseline
    std, so the denominator is shared across conditions and reflects the
    variability of correctly-detected trials.

    Both peth_target and peth_reference should have been extracted with
    normalize='subtract' (baseline-mean-subtracted but not std-divided).

    Args:
        peth_target: (n_trials x n_timepoints) — the condition to normalize.
        peth_reference: (n_trials x n_timepoints) — the reference condition
                        whose baseline std is used as the denominator (e.g. Hit).
        time_axis: 1D array of relative time points.
        baseline_window: (start, end) in seconds for std computation.

    Returns:
        Normalized peth_target (same shape), in units of reference-baseline z-score.
    """
    bl_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    ref_bl = peth_reference[:, bl_mask]
    # Pool all reference baseline values for a robust std estimate
    ref_std = np.nanstd(ref_bl)
    if ref_std < 1e-6:
        ref_std = 1.0
    return peth_target / ref_std

def compute_peak_zdf_over_window(df: pd.DataFrame, roi_cols: List[str], start_s: float, end_s: float) -> Dict[str, float]:
    """
    Given a trial-aligned window dataframe (index is seconds relative to event),
    compute the max z-scored dF/F between [start_s, end_s] for each ROI column.
    """
    out: Dict[str, float] = {}
    # Ensure numeric index
    idx = pd.to_numeric(df.index, errors="coerce")
    mask = (idx >= start_s) & (idx <= end_s)
    for c in roi_cols:
        if c in df.columns:
            out[c] = float(pd.to_numeric(df.loc[mask, c], errors="coerce").max(skipna=True))
    return out

def calculate_session_performance(session_df: pd.DataFrame) -> Dict[float, Tuple[float, float]]:
    """
    Calculates the performance and CI95 of for a given session DataFrame.
    """
    performance_dict = {}
    for change_size in sorted(session_df['change_sizes_TF'].unique()):  # Sort the change sizes
        outcomes = session_df[session_df['change_sizes_TF'] == change_size]['outcomes']
        hits = outcomes.value_counts().get('Hit', 0)
        misses = outcomes.value_counts().get('Miss', 0)
        total = hits + misses
        if total > 0:
            performance = hits / total
            # Calculate standard error of the mean (SEM)
            ci95 = 1.96 * np.sqrt(performance * (1 - performance) / total)
        else:
            performance = np.nan
            ci95 = np.nan
        performance_dict[float(change_size)] = (performance, ci95)
    return performance_dict

def calculate_session_performance_laser(session_df: pd.DataFrame) -> Dict[str, Dict[float, Tuple[float, float]]]:
    """
    Calculates the performance and CI95 of for a given session DataFrame.
    """
    performance_dict = {'laser_on': {}, 'laser_off': {}}
    
    for laser_state in [True, False]:
        laser_state_key = 'laser_on' if laser_state else 'laser_off'
        filtered_df = session_df[session_df['laser_states'] == laser_state]
        
        for change_size in sorted(filtered_df['change_sizes_TF'].unique()):  # Sort the change sizes
            outcomes = filtered_df[filtered_df['change_sizes_TF'] == change_size]['outcomes']
            hits = outcomes.value_counts().get('Hit', 0)
            misses = outcomes.value_counts().get('Miss', 0)
            total = hits + misses
            if total > 0:
                performance = hits / total
                # Calculate standard error of the mean (SEM)
                ci95 = 1.96 * np.sqrt(performance * (1 - performance) / total)
            else:
                performance = np.nan
                ci95 = np.nan
            performance_dict[laser_state_key][float(change_size)] = (performance, ci95)
    
    return performance_dict

def _clip_rate(rate: float) -> float:
    """Clip a proportion to [0.01, 0.99] for z-transform stability (log-linear correction)."""
    return np.clip(rate, 0.01, 0.99)


def calculate_sdt_metrics(outcomes: np.ndarray, change_sizes: np.ndarray,
                          catch_threshold: float = CATCH_THRESHOLD) -> Dict[str, float]:
    """
    Correct SDT (Signal Detection Theory) d' and criterion c.

    Trial classification is based on change_size, NOT outcome labels:
      - Go trial:    change_size > catch_threshold  (stimulus actually changed)
      - Catch trial:  change_size <= catch_threshold (no real change)

    SDT categories:
      - SDT Hit:  outcome='Hit'  on a go trial
      - SDT Miss: outcome='Miss' on a go trial
      - SDT FA:   outcome='Hit'  on a catch trial  (mouse licked to no-change)
      - SDT CR:   outcome='Miss' on a catch trial  (correctly withheld)

    Note: Behavioral 'FA' (early lick) and 'Abort' trials are excluded from SDT
    because the change stimulus was never presented on those trials.

    Args:
        outcomes: 1D array of outcome labels ('Hit', 'Miss', 'FA', 'Abort', 'CR').
        change_sizes: 1D array of change_size (Stim2TF) per trial, same length.
        catch_threshold: Boundary separating go (>) from catch (<=) trials.

    Returns:
        Dict with keys: sdt_hit_rate, sdt_fa_rate, d_prime, criterion_c,
        n_go, n_catch, n_sdt_hit, n_sdt_miss, n_sdt_fa, n_sdt_cr.
    """
    outcomes = np.asarray(outcomes)
    change_sizes = np.asarray(change_sizes, dtype=float)

    # --- Go trials: change_size > threshold, outcome must be Hit or Miss ---
    go_mask = (change_sizes > catch_threshold) & np.isin(outcomes, ['Hit', 'Miss'])
    n_go = go_mask.sum()
    n_sdt_hit = ((outcomes == 'Hit') & go_mask).sum()
    n_sdt_miss = ((outcomes == 'Miss') & go_mask).sum()

    # --- Catch trials: change_size <= threshold, outcome must be Hit or Miss ---
    # (Hit on a catch = SDT false alarm; Miss on a catch = correct rejection)
    catch_mask = (change_sizes <= catch_threshold) & np.isin(outcomes, ['Hit', 'Miss'])
    n_catch = catch_mask.sum()
    n_sdt_fa = ((outcomes == 'Hit') & catch_mask).sum()
    n_sdt_cr = ((outcomes == 'Miss') & catch_mask).sum()

    # --- Rates ---
    sdt_hit_rate = n_sdt_hit / n_go if n_go > 0 else np.nan
    sdt_fa_rate = n_sdt_fa / n_catch if n_catch > 0 else np.nan

    # --- d' and criterion c ---
    if pd.notna(sdt_hit_rate) and pd.notna(sdt_fa_rate):
        hr = _clip_rate(sdt_hit_rate)
        fr = _clip_rate(sdt_fa_rate)
        d_prime = norm.ppf(hr) - norm.ppf(fr)
        criterion_c = -0.5 * (norm.ppf(hr) + norm.ppf(fr))
    else:
        d_prime = np.nan
        criterion_c = np.nan

    return {
        "sdt_hit_rate": sdt_hit_rate,
        "sdt_fa_rate": sdt_fa_rate,
        "d_prime": d_prime,
        "criterion_c": criterion_c,
        "n_go": int(n_go),
        "n_catch": int(n_catch),
        "n_sdt_hit": int(n_sdt_hit),
        "n_sdt_miss": int(n_sdt_miss),
        "n_sdt_fa": int(n_sdt_fa),
        "n_sdt_cr": int(n_sdt_cr),
    }


def calculate_behavioral_metrics(session_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate behavioral metrics for QC.

    Returns fractions of each outcome type plus correct SDT d' and criterion c
    (using change_size to classify go vs catch trials).

    Expects session_df to have columns: 'outcomes', 'change_sizes_TF'.
    """
    if session_df.empty:
        return {
            "fraction_miss": np.nan,
            "fraction_hit": np.nan,
            "fraction_fa": np.nan,
            "fraction_abort": np.nan,
            "d_prime": np.nan,
            "criterion_c": np.nan,
        }

    outcomes = session_df['outcomes']
    total_trials = len(outcomes)

    # Behavioral outcome fractions (these use ALL trials)
    fraction_hit = (outcomes == 'Hit').sum() / total_trials if total_trials > 0 else np.nan
    fraction_miss = (outcomes == 'Miss').sum() / total_trials if total_trials > 0 else np.nan
    fraction_fa = (outcomes == 'FA').sum() / total_trials if total_trials > 0 else np.nan
    fraction_abort = (outcomes == 'Abort').sum() / total_trials if total_trials > 0 else np.nan

    # SDT metrics (correct implementation using change_size)
    if 'change_sizes_TF' in session_df.columns:
        sdt = calculate_sdt_metrics(
            outcomes.values,
            session_df['change_sizes_TF'].values
        )
        d_prime = sdt['d_prime']
        criterion_c = sdt['criterion_c']
    else:
        d_prime = np.nan
        criterion_c = np.nan

    return {
        "fraction_miss": fraction_miss,
        "fraction_hit": fraction_hit,
        "fraction_fa": fraction_fa,
        "fraction_abort": fraction_abort,
        "d_prime": d_prime,
        "criterion_c": criterion_c,
    }
