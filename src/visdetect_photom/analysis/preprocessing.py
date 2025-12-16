import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple, Optional

def calculate_dff(trial_df: pd.DataFrame, baseline_timestamp: float, session_zscored: bool = True) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Calculate dF/F for photometry signals.
    """
    # Select the baseline period
    baseline_period = trial_df[trial_df['SystemTimestamp'] <= baseline_timestamp]
    
    # Calculate the mean signal during the baseline period for each photometry signal
    baseline_means = {
        column: baseline_period[column].mean() for column in trial_df.columns if 'clean_signal_dff' in column
    }
    baseline_stds = {
        column: baseline_period[column].std() for column in trial_df.columns if 'clean_signal_dff' in column
    }
    
    # Calculate ΔF/F for each signal
    trial_dff = pd.DataFrame()
    trial_df_copy = trial_df.copy()  # Create a copy to avoid modifying the original DataFrame
    for (signal, baseline_mean), (_, baseline_std) in zip(baseline_means.items(), baseline_stds.items()):
        
        dff_column_name = f'{signal}'
        trial_dff['SystemTimestamp'] = trial_df_copy['SystemTimestamp']
        # Calculate the z-score relative to the baseline
        if not session_zscored:
            if baseline_std != 0:  # To avoid division by zero
                trial_dff[f'zscored_{dff_column_name}'] = (trial_df_copy[signal] - baseline_mean) / baseline_std
            else:
                trial_dff[f'zscored_{dff_column_name}'] = 0
        else:
             trial_dff[f'{dff_column_name}'] = trial_df_copy[f'{dff_column_name}']
        
    return trial_dff, baseline_means, baseline_stds

def z_score(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / np.std(values)

def compute_zscores(session_df: pd.DataFrame) -> pd.DataFrame:
    baseline_means_df = pd.DataFrame(session_df['baseline_means'].tolist())
    zscores_df = baseline_means_df.apply(z_score)
    zscores_df['outcome'] = session_df['outcomes']
    return zscores_df
