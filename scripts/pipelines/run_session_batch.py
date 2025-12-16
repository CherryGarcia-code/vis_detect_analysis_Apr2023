import argparse
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional

# Add the src directory to the path so we can import the package
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from visdetect_photom.core.io import find_all_sessions, infer_session_keys_from_paths
from visdetect_photom.analysis.statistics import compute_peak_zdf_over_window, calculate_behavioral_metrics
from visdetect_photom.viz.plotting import plot_melted_and_save

# Import legacy helpers for now until fully migrated
# We need to make sure scripts/ is in path to import vis_detect_helpers_v9
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    from vis_detect_helpers_v9 import (
        process_session_data,
        extract_photom_windows_from_session_s,
        extract_signal_window_from_trial_df,
    )
except ImportError:
    print("Warning: Could not import vis_detect_helpers_v9. Some functionality may be missing.")

def summarize_session(session_df: pd.DataFrame) -> dict:
    """
    Produce a small set of "interesting messages":
    - hit vs miss peak response after reaction/change
    - big vs small vs no_change peak after change
    """
    # Windows in seconds relative to event
    peak_window = (0.0, 1.0)

    def safe_extract(event: str):
        """Robustly return a list of trial windows for an event using helper or manual fallback."""
        try:
            if event == "hit":
                hit_windows, change_windows, base_windows = extract_photom_windows_from_session_s(session_df, "hit")
                return hit_windows
            elif event == "miss":
                miss_windows, _, _ = extract_photom_windows_from_session_s(session_df, "miss")
                return miss_windows
            elif event == "change":
                change_only_windows, _ = extract_photom_windows_from_session_s(session_df, "change")
                return change_only_windows
        except Exception:
            pass
        # Fallback manual extraction
        dfs = []
        try:
            if event == "hit":
                sel = session_df[session_df['outcomes'] == 'Hit']
                for _, row in sel.iterrows():
                    dfs.append(extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']))
            elif event == "miss":
                sel = session_df[session_df['outcomes'] == 'Miss']
                for _, row in sel.iterrows():
                    dfs.append(extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']))
            elif event == "change":
                sel = session_df[~session_df['outcomes'].isin(['FA','abort'])]
                for _, row in sel.iterrows():
                    dfs.append(extract_signal_window_from_trial_df(row, row['change_time_Timestamps']))
        except Exception:
            return []
        return dfs

    # Extract windows
    hit_windows = safe_extract("hit")
    miss_windows = safe_extract("miss")
    change_only_windows = safe_extract("change")

    # Collect per-trial peaks
    def roi_cols(df: pd.DataFrame) -> list:
        return [c for c in df.columns if c.startswith("zscored_") and c.endswith("_clean_signal_dff")]

    hit_peaks = [compute_peak_zdf_over_window(w, roi_cols(w), *peak_window) for w in hit_windows]
    miss_peaks = [compute_peak_zdf_over_window(w, roi_cols(w), *peak_window) for w in miss_windows]
    change_peaks = [compute_peak_zdf_over_window(w, roi_cols(w), *peak_window) for w in change_only_windows]

    # Aggregate
    def agg(peaks: list) -> dict:
        keys = set().union(*[p.keys() for p in peaks]) if peaks else set()
        return {k: float(np.nanmean([p.get(k, np.nan) for p in peaks])) for k in keys}

    hit_mean = agg(hit_peaks)
    miss_mean = agg(miss_peaks)
    change_mean = agg(change_peaks)

    # Behavioral Metrics
    beh_metrics = calculate_behavioral_metrics(session_df)

    summary = {
        "n_trials": len(session_df),
        "hit_peak_0to1s": hit_mean,
        "miss_peak_0to1s": miss_mean,
        "change_peak_0to1s": change_mean,
    }
    summary.update(beh_metrics)
    
    return summary

def run_batch_analysis(mouse_dir: str, out_dir: Optional[str] = None, limit: Optional[int] = None) -> None:
    """
    Run the batch analysis pipeline.
    """
    if out_dir is None:
        out_dir = os.path.join(REPO_ROOT, "pdf_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {mouse_dir} for sessions...")
    sessions = find_all_sessions(mouse_dir, recursive=True)
    print(f"Found {len(sessions)} sessions.")

    if limit:
        sessions = sessions[:limit]
        print(f"Limiting to first {limit} sessions.")

    all_summaries = []

    for sess in sessions:
        try:
            mouse_id, date = infer_session_keys_from_paths(sess['trials'])
            print(f"Processing {mouse_id} {date}...")
            
            # Load and process data using the legacy helper for now (it does a lot of heavy lifting)
            # In the future, we should migrate process_session_data to src/visdetect_photom/core/session.py
            # process_session_data expects lists of files and specific order: photom, photom_io, settings, trials
            session_df = process_session_data(
                [sess['photom']], 
                [sess['photom_io']], 
                [sess['session_settings']], 
                [sess['trials']]
            )

            if session_df is None or session_df.empty:
                print(f"Skipping {mouse_id} {date} - empty dataframe")
                continue

            summary = summarize_session(session_df)
            summary['mouse_id'] = mouse_id
            summary['date'] = date
            summary['session_name'] = f"{mouse_id}_{date}" # For compatibility with filter script
            
            # Save summary
            out_file = os.path.join(out_dir, f"photom_summary_{mouse_id}_{date}.csv")
            pd.DataFrame([summary]).to_csv(out_file, index=False)
            print(f"Saved summary to {out_file}")
            
            all_summaries.append(summary)

        except Exception as e:
            print(f"Error processing session {sess}: {e}")
            import traceback
            traceback.print_exc()

    # Save aggregate manifest
    if all_summaries:
        manifest_path = os.path.join(out_dir, "all_sessions_manifest.csv")
        pd.DataFrame(all_summaries).to_csv(manifest_path, index=False)
        print(f"Saved aggregate manifest to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch photometry analysis")
    parser.add_argument("mouse_dir", help="Directory containing mouse data")
    parser.add_argument("--out", help="Output directory", default=None)
    parser.add_argument("--limit", type=int, help="Limit number of sessions", default=None)
    args = parser.parse_args()

    run_batch_analysis(args.mouse_dir, args.out, args.limit)
