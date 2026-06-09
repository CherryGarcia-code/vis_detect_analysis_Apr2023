"""
Unified session dataclasses and I/O for visdetect_photom.

This module provides canonical dataclasses (Trial, PhotometryTrace, Session) 
to standardize data access across analysis scripts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class Trial:
    """Represents a single behavioral trial."""
    trial_index: int
    outcome: str  # 'Hit', 'Miss', 'FA', 'CR', 'Abort'
    
    # Relative times and parameters
    change_time: Optional[float] = None # 'stimT'
    change_size: Optional[float] = None # 'Stim2TF'
    reaction_time: Optional[float] = None
    iti_duration: Optional[float] = None # 'stimD'
    
    # Absolute timestamps (aligned with photometry SystemTimestamp)
    absolute_start_time: Optional[float] = None # Calculated from IO Input0 - ITI
    absolute_change_time: Optional[float] = None # Calculated from IO Input0 + change_time
    absolute_reaction_time: Optional[float] = None 
    
    # Legacy/Generic fields (optional)
    start_time: float = 0.0
    stop_time: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PhotometryTrace:
    """Represents a continuous photometry signal."""
    roi_name: str
    timestamps: np.ndarray  # 1D array of timestamps
    signal: np.ndarray      # 1D array of signal values (raw or dff)
    signal_type: str = "raw" # 'raw', 'dff', 'zscored'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """Represents a complete recording session."""
    subject_id: str
    session_date: str # YYYY-MM-DD
    session_id: str   # Unique identifier
    trials: List[Trial] = field(default_factory=list)
    photometry_data: Dict[str, PhotometryTrace] = field(default_factory=dict)
    behavior_data: Optional[pd.DataFrame] = None # Full behavioral dataframe if needed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def trial_outcomes(self) -> List[str]:
        return [t.outcome for t in self.trials]

    def get_trace(self, roi_name: str) -> Optional[PhotometryTrace]:
        return self.photometry_data.get(roi_name)

def _extract_io_from_embedded_columns(photom_df: pd.DataFrame) -> list:
    """Extract baseline-onset timestamps from embedded Input0 column.

    Old-format photometry CSVs (BG_008–011) have Input0/Input1 columns
    with 0/1 values. Baseline onset = rising edge (0→1) of Input0.
    The timestamp column may be 'Timestamp' (old) or 'SystemTimestamp' (new).
    """
    from visdetect_photom.core.constants import OLD_FORMAT_COLUMN_MAP

    # Determine timestamp column name
    if 'SystemTimestamp' in photom_df.columns:
        ts_col = 'SystemTimestamp'
    elif 'Timestamp' in photom_df.columns:
        ts_col = 'Timestamp'
    else:
        return []

    if 'Input0' not in photom_df.columns:
        return []

    input0 = photom_df['Input0'].values
    timestamps = photom_df[ts_col].values

    # Detect rising edges: previous sample is 0, current sample is 1
    rising = np.where((input0[1:] == 1) & (input0[:-1] == 0))[0] + 1
    return timestamps[rising].tolist()


def load_session_from_files(file_paths: Dict[str, str]) -> Session:
    """
    Factory function to create a Session object from a dictionary of file paths.
    
    Args:
        file_paths: Dict with keys 'trials', 'session_settings', 'photom', 'photom_io'
                    (as returned by core.io.pair_session_files)
    
    Returns:
        Session object populated with data.
    """
    from visdetect_photom.core import io
    from visdetect_photom.analysis import preprocessing
    
    # Extract paths
    trials_path = file_paths.get('trials')
    photom_path = file_paths.get('photom')
    photom_io_path = file_paths.get('photom_io')
    
    if not trials_path:
        raise ValueError("Trials file path is missing.")
        
    # Infer subject and date
    subject_id, session_date_str = io.infer_session_keys_from_paths(trials_path)
    if not subject_id: subject_id = "Unknown"
    if not session_date_str: session_date_str = "Unknown"
    session_id = f"{subject_id}_{session_date_str}"

    # Load Data
    trials_data = io.load_json_data(trials_path)
    
    # Handle trials data structure (list vs dict)
    if isinstance(trials_data, list):
        raw_trials = trials_data
    elif isinstance(trials_data, dict):
        raw_trials = trials_data.get('trials', [])
    else:
        raw_trials = []

    # Load Photometry IO for synchronization
    # For old-format subjects (no separate IO file), we extract IO events
    # from embedded Input0/Input1 columns in the photometry CSV itself.
    # When multiple behavioral sessions share one photom CSV, io_event_offset
    # tells us which slice of IO events belongs to this session.
    io_event_offset = file_paths.get('io_event_offset', 0)
    baseline_on_timestamps = []
    _photom_df_preloaded = None  # Cache to avoid double-loading
    if photom_io_path:
        try:
            photom_io_df = io.load_csv_data(photom_io_path)
            # Logic from helpers.py: baseline_on_df = photom_io_df[photom_io_df['DigitalIOName'] == 'Input0']
            if 'DigitalIOName' in photom_io_df.columns and 'SystemTimestamp' in photom_io_df.columns:
                baseline_on_df = photom_io_df[photom_io_df['DigitalIOName'] == 'Input0']
                baseline_on_timestamps = baseline_on_df['SystemTimestamp'].tolist()
        except Exception as e:
            print(f"Warning: Failed to load or parse photom_io file: {e}")
    elif photom_path:
        # Old format: extract IO events from embedded columns
        try:
            _photom_df_preloaded = io.load_csv_data(photom_path)
            all_baseline_ts = _extract_io_from_embedded_columns(_photom_df_preloaded)
            # Slice to this session's portion using offset and trial count
            n_trials = len(raw_trials)
            start = io_event_offset
            end = start + n_trials
            if end <= len(all_baseline_ts):
                baseline_on_timestamps = all_baseline_ts[start:end]
            else:
                # Fallback: take what's available from offset
                baseline_on_timestamps = all_baseline_ts[start:]
                if len(baseline_on_timestamps) != n_trials:
                    print(f"Warning: Expected {n_trials} IO events at offset {start}, "
                          f"got {len(baseline_on_timestamps)} (total available: {len(all_baseline_ts)})")
        except Exception as e:
            print(f"Warning: Failed to extract embedded IO events: {e}")

    trials_list = []
    
    # Check alignment
    if len(baseline_on_timestamps) > 0 and len(raw_trials) != len(baseline_on_timestamps):
        print(f"Warning: Mismatch between trials ({len(raw_trials)}) and baseline timestamps ({len(baseline_on_timestamps)}). Synchronization may be partial.")
    
    # Outcome label normalization (JSON has 'abort', 'Ref' — see constants.py)
    from visdetect_photom.core.constants import OUTCOME_NORMALIZATION

    for idx, t_data in enumerate(raw_trials):
        # Extract basic fields
        raw_outcome = t_data.get('trialoutcome', 'Unknown')
        outcome = OUTCOME_NORMALIZATION.get(raw_outcome, raw_outcome)
        stim_d = t_data.get('stimD', 0.0) # ITI
        stim_t = t_data.get('stimT', 0.0) # Change time
        stim_2tf = t_data.get('Stim2TF')  # Change size
        
        # Extract Reaction Time
        # Logic from helpers.py:
        # if trial_outcome == 'Hit': trial_outcome = 'RT'
        # reaction_time = trial['reactiontimes'][trial_outcome]
        rt_key = 'RT' if outcome == 'Hit' else outcome
        reaction_times_dict = t_data.get('reactiontimes', {})
        reaction_time = reaction_times_dict.get(rt_key) if isinstance(reaction_times_dict, dict) else None

        # Calculate Absolute Timestamps
        abs_start = None
        abs_change = None
        abs_rt = None
        
        if idx < len(baseline_on_timestamps):
            baseline_ts = baseline_on_timestamps[idx]
            
            # Logic from helpers.py:
            # reference_start_actual = baseline_on_timestamp - iti (stimD)
            # change_times_actual = baseline_on_timestamp + change_time (stimT)
            
            abs_start = baseline_ts - stim_d
            abs_change = baseline_ts + stim_t
            
            if reaction_time is not None:
                # Logic from helpers.py:
                # reaction_time_from_reference_start = reaction_time + iti + change_time if (trial_outcome == 'RT' or trial_outcome == 'Miss') else reaction_time+iti
                
                if outcome in ['Hit', 'Miss', 'RT']:
                     abs_rt = baseline_ts + stim_t + reaction_time
                else:
                     abs_rt = baseline_ts + reaction_time

        trial = Trial(
            trial_index=idx,
            outcome=outcome,
            change_time=stim_t,
            change_size=stim_2tf,
            reaction_time=reaction_time,
            iti_duration=stim_d,
            absolute_start_time=abs_start,
            absolute_change_time=abs_change,
            absolute_reaction_time=abs_rt,
            metadata=t_data
        )
        trials_list.append(trial)

    # Load Photometry
    photom_traces = {}
    if photom_path:
        try:
            # Reuse preloaded DataFrame if available (avoids double I/O for old-format)
            photom_df = _photom_df_preloaded if _photom_df_preloaded is not None else io.load_csv_data(photom_path)
            
            # Process Photometry (De-interleave, Isosbestic Fit, dF/F)
            # This handles the complex logic of LedState 1 vs 2
            processed_df = preprocessing.process_photometry_signals(photom_df, session_zscored=True)
            
            if not processed_df.empty and 'SystemTimestamp' in processed_df.columns:
                timestamps = processed_df['SystemTimestamp'].values
                
                for col in processed_df.columns:
                    # We only want the final z-scored traces for the Session object
                    # or maybe we want all? Let's store the zscored ones as primary.
                    if 'zscored' in col and 'clean_signal_dff' in col:
                        # Extract ROI name: zscored_G0_clean_signal_dff -> G0
                        roi_name = col.replace('zscored_', '').replace('_clean_signal_dff', '')
                        
                        trace = PhotometryTrace(
                            roi_name=roi_name,
                            timestamps=timestamps,
                            signal=processed_df[col].values,
                            signal_type="zscored_dff"
                        )
                        photom_traces[roi_name] = trace
            else:
                print(f"Warning: Photometry processing returned empty or invalid dataframe for {session_id}")

        except Exception as e:
            print(f"Warning: Failed to load or process photom file: {e}")

    return Session(
        subject_id=subject_id,
        session_date=session_date_str,
        session_id=session_id,
        trials=trials_list,
        photometry_data=photom_traces,
        metadata={"file_paths": file_paths}
    )
