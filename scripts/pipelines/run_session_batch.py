"""
Batch Processing Pipeline for Photometry Sessions.

This script scans a directory for mouse sessions, processes them to extract photometry signals 
aligned to behavioral events, calculates behavioral metrics (d-prime, hit rates), and 
generates a summary manifest.

Usage:
    python scripts/pipelines/run_session_batch.py <mouse_dir> [--out <out_dir>] [--limit <N>]

Arguments:
    mouse_dir   : Path to the directory containing mouse session data (recursive search).
    --out       : Output directory for summary CSVs and manifest (default: FIGURES).
    --limit     : Limit the number of sessions to process (useful for testing).

Example:
    python scripts/pipelines/run_session_batch.py photom_data --out FIGURES --limit 5
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def process_single_session(sess: dict, out_dir: str) -> Optional[Dict[str, Any]]:
    """
    Process a single session: load data, calculate metrics, and save summary.
    Returns the summary dictionary or None if failed.
    """
    try:
        mouse_id, date = infer_session_keys_from_paths(sess['trials'])
        
        # Format subject and session names to match reference repo (BG_XXX)
        # If mouse_id is numeric (e.g. "013"), prepend "BG_"
        if mouse_id and mouse_id.isdigit():
            subject_id = f"BG_{mouse_id}"
        else:
            subject_id = mouse_id

        # Create session-specific output folder with hierarchy: Subject/Session
        session_name = f"{subject_id}_{date}"
        session_out_dir = os.path.join(out_dir, subject_id, session_name)
        os.makedirs(session_out_dir, exist_ok=True)
        
        # Load and process data using the legacy helper for now
        session_df = process_session_data(
            [sess['photom']], 
            [sess['photom_io']], 
            [sess['session_settings']], 
            [sess['trials']]
        )

        if session_df is None or session_df.empty:
            return None

        summary = summarize_session(session_df)
        summary['mouse_id'] = subject_id
        summary['date'] = date
        summary['session_name'] = session_name
        
        # Save summary to session folder
        out_file = os.path.join(session_out_dir, f"photom_summary_{session_name}.csv")
        pd.DataFrame([summary]).to_csv(out_file, index=False)
        
        return summary

    except Exception as e:
        print(f"Error processing session {sess.get('trials', 'unknown')}: {e}")
        # import traceback
        # traceback.print_exc()
        return None

def run_batch_analysis(mouse_dir: str, out_dir: Optional[str] = None, limit: Optional[int] = None, workers: int = 1) -> None:
    """
    Run the batch analysis pipeline.
    """
    if out_dir is None:
        out_dir = os.path.join(REPO_ROOT, "FIGURES")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {mouse_dir} for sessions...")
    sessions = find_all_sessions(mouse_dir, recursive=True)
    print(f"Found {len(sessions)} sessions.")

    if limit:
        sessions = sessions[:limit]
        print(f"Limiting to first {limit} sessions.")

    print(f"Processing sessions with workers={workers}...")
    
    all_summaries = []
    
    # Run processing in parallel or serial
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_session = {executor.submit(process_single_session, sess, out_dir): sess for sess in sessions}
            
            for future in tqdm(as_completed(future_to_session), total=len(sessions), desc="Processing Sessions"):
                try:
                    result = future.result()
                    if result is not None:
                        all_summaries.append(result)
                except Exception as exc:
                    print(f"Session generated an exception: {exc}")
    else:
        for sess in tqdm(sessions, desc="Processing Sessions"):
            result = process_single_session(sess, out_dir)
            if result is not None:
                all_summaries.append(result)

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
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: 1)", default=1)
    args = parser.parse_args()

    run_batch_analysis(args.mouse_dir, args.out, args.limit, args.workers)
