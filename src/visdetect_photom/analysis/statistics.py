import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple

def extract_peth(signal: np.ndarray, timestamps: np.ndarray, event_times: np.ndarray, 
                 window: Tuple[float, float] = (-2.0, 4.0), fs: float = 100.0,
                 baseline_window: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Peri-Event Time Histogram (PETH) matrix.
    
    Args:
        signal: 1D array of continuous signal.
        timestamps: 1D array of timestamps corresponding to signal.
        event_times: 1D array of event timestamps to align to.
        window: (start, end) relative to event in seconds.
        fs: Sampling frequency (approximate) for output time vector.
        baseline_window: (start, end) relative to event for trial-based z-scoring.
                         If provided, each trial is z-scored relative to this window.
    
    Returns:
        time_axis: 1D array of relative time points.
        peth_matrix: 2D array (n_events x n_timepoints) of aligned signal.
    """
    # Create relative time axis
    n_samples = int((window[1] - window[0]) * fs)
    time_axis = np.linspace(window[0], window[1], n_samples)
    
    peth_matrix = []
    
    # Interpolation is safer than index slicing for non-uniform or jittered timestamps
    # But for speed with uniform sampling, index slicing is better.
    # Let's assume uniform sampling for now, or use searchsorted.
    
    # Using searchsorted for nearest neighbor
    dt = 1.0 / fs
    
    # Pre-calculate baseline mask if needed
    baseline_mask = None
    if baseline_window is not None:
        baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    
    for et in event_times:
        if np.isnan(et):
            peth_matrix.append(np.full(n_samples, np.nan))
            continue
            
        # Define target times for this event
        target_times = et + time_axis
        
        # Find indices (nearest)
        # This assumes timestamps are sorted
        idx = np.searchsorted(timestamps, target_times)
        
        # Clip indices
        idx = np.clip(idx, 0, len(signal) - 1)
        
        # Extract values
        # Check if the found timestamp is actually close (within 1/fs)
        # If not, fill with NaN (gap in recording)
        found_times = timestamps[idx]
        valid_mask = np.abs(found_times - target_times) < (1.5 * dt)
        
        vals = signal[idx].astype(float)
        vals[~valid_mask] = np.nan
        
        # Apply trial-based z-scoring if requested
        if baseline_mask is not None:
            baseline_vals = vals[baseline_mask]
            # Check if we have enough valid data points in baseline
            if np.sum(~np.isnan(baseline_vals)) > 1:
                b_mean = np.nanmean(baseline_vals)
                b_std = np.nanstd(baseline_vals)
                if b_std != 0:
                    vals = (vals - b_mean) / b_std
                else:
                    # If std is 0 (flat line), just subtract mean (center at 0)
                    vals = vals - b_mean
            else:
                # Not enough baseline data, fill with NaNs or keep as is?
                # Usually safer to fill with NaNs if we can't normalize properly
                vals[:] = np.nan

        peth_matrix.append(vals)
        
    return time_axis, np.array(peth_matrix)

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

def calculate_behavioral_metrics(session_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate behavioral metrics for QC: fraction_miss, fraction_hit, fraction_fa, d_prime.
    """
    if session_df.empty:
        return {
            "fraction_miss": np.nan,
            "fraction_hit": np.nan,
            "fraction_fa": np.nan,
            "d_prime": np.nan
        }

    # Outcomes
    outcomes = session_df['outcomes']
    total_trials = len(outcomes)
    
    # Counts
    n_hit = (outcomes == 'Hit').sum()
    n_miss = (outcomes == 'Miss').sum()
    n_fa = (outcomes == 'FA').sum()
    n_cr = (outcomes == 'CR').sum() # Assuming CR exists or is inferred
    
    # Fractions
    fraction_miss = n_miss / total_trials if total_trials > 0 else np.nan
    fraction_hit = n_hit / total_trials if total_trials > 0 else np.nan
    fraction_fa = n_fa / total_trials if total_trials > 0 else np.nan
    
    # d-prime calculation
    # Hit Rate = Hits / (Hits + Misses)
    # FA Rate = FAs / (FAs + CRs)
    # If CR is not explicitly tracked, we might need to infer it or use a different denominator depending on task structure.
    # In many Go/NoGo tasks, "Catch" trials are where FA/CR happen.
    
    # Let's assume standard signal detection theory:
    # Signal trials: Hit + Miss
    # Noise trials: FA + CR
    
    n_signal = n_hit + n_miss
    n_noise = n_fa + n_cr
    
    hit_rate = n_hit / n_signal if n_signal > 0 else np.nan
    fa_rate = n_fa / n_noise if n_noise > 0 else np.nan
    
    # Correction for 0 or 1 rates to avoid infinity
    def correct_rate(r, n):
        if pd.isna(r) or n == 0: return r
        if r == 0: return 0.5 / n
        if r == 1: return (n - 0.5) / n
        return r

    hit_rate_c = correct_rate(hit_rate, n_signal)
    fa_rate_c = correct_rate(fa_rate, n_noise)
    
    if pd.notna(hit_rate_c) and pd.notna(fa_rate_c):
        d_prime = norm.ppf(hit_rate_c) - norm.ppf(fa_rate_c)
    else:
        d_prime = np.nan
        
    return {
        "fraction_miss": fraction_miss,
        "fraction_hit": fraction_hit,
        "fraction_fa": fraction_fa,
        "d_prime": d_prime
    }
