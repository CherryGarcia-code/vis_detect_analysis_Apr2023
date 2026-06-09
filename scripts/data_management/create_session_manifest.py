"""
Create a session manifest CSV from a directory of session data (photometry/behavior).
Matches the structure of the 2025 ephys analysis repo manifest.

Usage:
    python scripts/data_management/create_session_manifest.py --root-dir photom_data --out FIGURES/all_sessions_manifest.csv
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import logging
import concurrent.futures

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

# Add scripts to path for behavior imports
if str(repo_root / 'scripts') not in sys.path:
    sys.path.insert(0, str(repo_root / 'scripts'))

from visdetect_photom.core import io, session
try:
    from analysis.behavior.plot_session_behavior import compute_session_performance
except ImportError:
    # If standard import fails, try direct module path
    sys.path.append(str(repo_root / "scripts" / "analysis" / "behavior"))
    from plot_session_behavior import compute_session_performance

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def get_session_metrics(sess):
    try:
        # Use the centralized behavior analysis function from plot_session_behavior.py
        perf = compute_session_performance(sess)
        if not perf:
            return None

        return {
            'subject': sess.subject_id,
            'session_name': sess.session_id, # In this repo session_id is roughly equivalent to session_name
            'date': sess.session_date,
            'n_trials': perf['n_trials'],
            'n_hits': perf['n_hits'],
            'n_miss': perf['n_miss'],
            'n_fa': perf['n_fa'],
            'n_fa_early': perf['n_fa_early'],
            'n_fa_late': perf['n_fa_late'],
            'n_abort': perf['n_abort'],
            'hit_rate': perf['hit_rate'],
            'miss_rate': perf['miss_rate'],
            'fa_rate': perf['fa_rate'],
            'abort_rate': perf['abort_rate'],
            'fraction_hit': perf['fraction_hit'],
            'fraction_miss': perf['fraction_miss'],
            'fraction_fa': perf['fraction_fa'],
            'fraction_abort': perf['fraction_abort'],
            'median_rt': perf['median_rt_hit'],
            'mean_rt': perf['mean_rt_hit'],
            'sem_rt_hit': perf['sem_rt_hit'],
            'd_prime': perf['d_prime']
        }
    except Exception as e:
        logging.error(f"Error computing metrics for {sess.session_id}: {e}")
        return None

def process_single_session(sess_files):
    """Worker function for parallel processing."""
    try:
        if 'trials' not in sess_files:
            return None
            
        sess = session.load_session_from_files(sess_files)
        return get_session_metrics(sess)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Create session manifest from root directory.")
    parser.add_argument('--root-dir', required=True, help='Root directory containing session data')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--recursive', action='store_true', default=True, help='Search recursively')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover sessions
    logging.info("Searching for sessions...")
    sessions_files = io.find_all_sessions(str(root_dir), recursive=args.recursive)
    logging.info(f"Found {len(sessions_files)} sessions. Processing with {args.n_workers} workers...")

    rows = []
    
    if args.n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(process_single_session, sf) for sf in sessions_files]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(sessions_files), desc="Processing"):
                metrics = future.result()
                if metrics:
                    rows.append(metrics)
    else:
        # Serial fallback
        for sess_files in tqdm(sessions_files):
            metrics = process_single_session(sess_files)
            if metrics:
                rows.append(metrics)

    if not rows:
        logging.warning("No valid session data extracted.")
        return

    df = pd.DataFrame(rows)
    
    # Format
    if 'session_name' in df.columns:
        df['session_name'] = df['session_name'].astype(str)
    
    # Sort
    # Try to sort by date if available
    # Assuming date format YYYYMMDD or similar
    try:
        df = df.sort_values('date')
    except:
        pass

    df.to_csv(out_path, index=False)
    
    logging.info(f"Manifest saved to {out_path}")
    print(df[['session_name', 'n_trials', 'hit_rate']].head())

if __name__ == "__main__":
    main()
