import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple, Optional, Union

def process_photometry_signals(df: pd.DataFrame, smooth_poly: int = 4, session_zscored: bool = True) -> pd.DataFrame:
    """
    Process raw photometry data: de-interleave, fit isosbestic, smooth, and calculate dF/F.
    Replicates logic from vis_detect_helpers_v9.py get_signal().
    
    Args:
        df: Raw dataframe with columns [SystemTimestamp, LedState, G0, G2, etc.]
        smooth_poly: Polynomial order for Savitzky-Golay filter.
        session_zscored: Whether to z-score the final signal.
        
    Returns:
        clean_signal_df: DataFrame with timestamps and processed signals.
    """
    clean_signal_df = pd.DataFrame()
    data = df.copy()
    
    # Determine ROIs
    dms_rois = ['G0', 'G2']
    vls_rois = ['G4', 'G5']
    rois = dms_rois + vls_rois if 'G4' in data.columns and 'G5' in data.columns else dms_rois
    
    # Filter valid ROIs that actually exist in the dataframe
    rois = [r for r in rois if r in data.columns]
    
    if not rois:
        return clean_signal_df

    # Constants
    Hz = 1
    sampling_freq = 100 * Hz
    second = 100
    trim_samples = 10 * second # 10 seconds trim

    for roi in rois:
        # Extract Isosbestic (LedState=1) and Signal (LedState=2)
        iso_mask = data['LedState'] == 1
        sig_mask = data['LedState'] == 2
        
        iso_timestamps = data.loc[iso_mask, 'SystemTimestamp'].to_numpy()
        sig_timestamps = data.loc[sig_mask, 'SystemTimestamp'].to_numpy()
        
        iso_data = data.loc[iso_mask, roi].to_numpy()
        sig_data = data.loc[sig_mask, roi].to_numpy()
        
        # Trim beginning
        if len(iso_data) > trim_samples:
            iso_data = iso_data[trim_samples:]
            iso_timestamps = iso_timestamps[trim_samples:]
        if len(sig_data) > trim_samples:
            sig_data = sig_data[trim_samples:]
            sig_timestamps = sig_timestamps[trim_samples:]
            
        # Ensure equal length
        min_length = min(len(iso_data), len(sig_data))
        iso_data = iso_data[:min_length]
        sig_data = sig_data[:min_length]
        iso_timestamps = iso_timestamps[:min_length]
        sig_timestamps = sig_timestamps[:min_length]
        
        if min_length == 0:
            continue

        # Fit Isosbestic to Signal (Linear Fit)
        try:
            iso_coef = np.polyfit(iso_data, sig_data, deg=1)
            iso_fitted = np.polyval(iso_coef, iso_data)
        except Exception as e:
            print(f"Error fitting isosbestic for {roi}: {e}")
            continue

        # Smooth
        # Logic from helper: window_length=90/40, polyorder=smooth_poly-1 / smooth_poly-2
        # Ensure window_length is odd and <= length of data
        win_iso = min(91, len(iso_data) if len(iso_data) % 2 != 0 else len(iso_data)-1)
        win_sig = min(41, len(sig_data) if len(sig_data) % 2 != 0 else len(sig_data)-1)
        
        if win_iso > smooth_poly and win_sig > smooth_poly:
            iso_smooth = savgol_filter(iso_fitted, window_length=win_iso, polyorder=smooth_poly-1)
            sig_smooth = savgol_filter(sig_data, window_length=win_sig, polyorder=smooth_poly-2)
        else:
            iso_smooth = iso_fitted
            sig_smooth = sig_data

        # Calculate dF/F
        # sig_smooth_clean = (sig_smooth - iso_smooth)
        # sig_smooth_clean_dff = (sig_smooth_clean/iso_smooth)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            sig_smooth_clean = (sig_smooth - iso_smooth)
            sig_smooth_clean_dff = (sig_smooth_clean / iso_smooth)
            
        # Handle NaNs/Infs
        sig_smooth_clean_dff = np.nan_to_num(sig_smooth_clean_dff)

        # Baseline correction (min subtraction)
        # NOTE: This was in the original helper, but it might be problematic if we want true dF/F centered around 0.
        # However, for z-scoring later, the mean subtraction handles centering.
        # The min subtraction ensures non-negative values for dF/F calculation if we were dividing by baseline,
        # but here we already calculated dF/F.
        # Let's keep it to match legacy behavior for now, but be aware.
        # Actually, looking at the helper:
        # sig_smooth_clean = (sig_smooth - iso_smooth)
        # sig_smooth_clean_dff = (sig_smooth_clean/iso_smooth)
        # Then:
        # if sig_smooth_clean.min() < 0: sig_smooth_clean = sig_smooth_clean - sig_smooth_clean.min()
        # if sig_smooth_clean_dff.min() < 0: sig_smooth_clean_dff = sig_smooth_clean_dff - sig_smooth_clean_dff.min()
        
        # if sig_smooth_clean.min() < 0:
        #     sig_smooth_clean = sig_smooth_clean - sig_smooth_clean.min()
        # if sig_smooth_clean_dff.min() < 0:
        #     sig_smooth_clean_dff = sig_smooth_clean_dff - sig_smooth_clean_dff.min()

        # Store results
        # We use sig_timestamps for the processed signal
        if 'SystemTimestamp' not in clean_signal_df.columns:
            clean_signal_df['SystemTimestamp'] = sig_timestamps
        
        # Align timestamps if needed (assuming they match since we trimmed same amount)
        # If multiple ROIs have slightly different timestamps due to some reason, this might be an issue,
        # but usually they are synchronized.
        
        clean_signal_df[f'{roi}_clean_signal'] = sig_smooth_clean
        clean_signal_df[f'{roi}_clean_signal_dff'] = sig_smooth_clean_dff
        
        if session_zscored:
            std = sig_smooth_clean_dff.std()
            if std != 0:
                clean_signal_df[f'zscored_{roi}_clean_signal_dff'] = (sig_smooth_clean_dff - sig_smooth_clean_dff.mean()) / std
            else:
                clean_signal_df[f'zscored_{roi}_clean_signal_dff'] = 0.0

    return clean_signal_df

def calculate_dff_trace(signal: np.ndarray, timestamps: np.ndarray, baseline_window: Tuple[float, float] = None) -> np.ndarray:
    """
    Calculate dF/F for a single trace using a baseline window or whole session.
    
    Args:
        signal: 1D array of raw signal.
        timestamps: 1D array of timestamps.
        baseline_window: (start, end) tuple defining the baseline period. 
                         If None, uses the whole trace mean/std (session z-score).
    
    Returns:
        dff: 1D array of dF/F (or z-scored) values.
    """
    if baseline_window:
        mask = (timestamps >= baseline_window[0]) & (timestamps <= baseline_window[1])
        if not np.any(mask):
            # Fallback if baseline window is invalid
            baseline_mean = np.mean(signal)
            baseline_std = np.std(signal)
        else:
            baseline_mean = np.mean(signal[mask])
            baseline_std = np.std(signal[mask])
    else:
        baseline_mean = np.mean(signal)
        baseline_std = np.std(signal)
        
    if baseline_std == 0:
        return np.zeros_like(signal)
        
    # Z-score calculation: (F - mean) / std
    return (signal - baseline_mean) / baseline_std

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
