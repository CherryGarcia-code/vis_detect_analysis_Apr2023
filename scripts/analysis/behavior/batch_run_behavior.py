"""
Batch runner for behavior analysis pipeline.

Usage:
    python scripts/analysis/behavior/batch_run_behavior.py --root_dir photom_data --output_dir FIGURES/behavior --n_workers 4
"""
import argparse
import sys
from pathlib import Path
import logging
import concurrent.futures
from tqdm import tqdm

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from visdetect_photom.core import io, session
# Reuse the plotting function directly instead of shelling out to python script
# This is more efficient and robust
try:
    from scripts.analysis.behavior.plot_session_behavior import plot_session_behavior
except ImportError:
    # If running as script, directory is in path, import directly
    from plot_session_behavior import plot_session_behavior

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def process_session_behavior(session_files, output_dir):
    try:
        # Load Session
        sess = session.load_session_from_files(session_files)
        
        # Create Output Dir
        subject_folder = sess.subject_id if sess.subject_id.startswith("BG_") else f"BG_{sess.subject_id}"
        session_out_dir = output_dir / subject_folder / sess.session_id
        session_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Run Plots
        plot_session_behavior(sess, session_out_dir)
        
        return True, sess.session_id
    except Exception as e:
        # logging.error(f"Failed behavior analysis for {session_files.get('trials')}: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Batch process behavior analysis.")
    _repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--root_dir", type=str, default=str(_repo_root / "photom_data"), help="Root directory.")
    parser.add_argument("--output_dir", type=str, default=str(_repo_root / "FIGURES" / "behavior"), help="Output directory.")
    parser.add_argument("--pattern", type=str, default="", help="Filter sessions by string pattern.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV for filtering sessions (e.g. all_sessions_manifest_clean.csv).")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover Sessions (reuse existing IO tool)
    logging.info("Searching for sessions...")
    sessions = io.find_all_sessions(str(root_path), recursive=True)
    
    if args.pattern:
        logging.info(f"Filtering sessions with pattern: {args.pattern}")
        # Only keep sessions where 'trials' path matches pattern
        sessions = [s for s in sessions if args.pattern in str(s['trials'])]
        
    if args.manifest:
        logging.info(f"Filtering sessions using manifest: {args.manifest}")
        import pandas as pd
        if not Path(args.manifest).exists():
             logging.error(f"Manifest file not found: {args.manifest}")
             return

        df = pd.read_csv(args.manifest)
        
        # Support both 'session_id' and 'session_name'
        id_col = 'session_id' if 'session_id' in df.columns else 'session_name'
        
        if id_col not in df.columns:
             logging.warning(f"Manifest missing 'session_id' or 'session_name' column. Skipping filter.")
        else:
            valid_ids = set(df[id_col].astype(str))
            
            # Filter logic: Keep session if its ID matches any in valid_ids

            # Since we don't have the object yet, we match the ID string against the trials file path
            filtered_sessions = []
            for s in sessions:
                trials_path = str(s['trials'])
                # Check if any valid_id corresponds to this file
                # A robust way is to check if the valid_id represents the file correctly
                # Often session_id is "Subject_Date_Time" or similar.
                # Let's check if the ID is contained in the path
                if any(vid in trials_path for vid in valid_ids):
                    filtered_sessions.append(s)
            
            logging.info(f"Manifest filtering: {len(sessions)} -> {len(filtered_sessions)} sessions.")
            sessions = filtered_sessions
    
    logging.info(f"Found {len(sessions)} sessions.")
    
    # 2. Parallel Processing
    failed = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(process_session_behavior, sess_files, output_path) for sess_files in sessions]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sessions), desc="Analyzing Behavior"):
            success, msg = future.result()
            if not success:
                failed.append(msg)
    
    if failed:
        logging.warning(f"Failed sessions ({len(failed)}): {failed}")
    else:
        logging.info("All behavior analyses completed successfully.")

if __name__ == "__main__":
    main()
