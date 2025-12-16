import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple

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
