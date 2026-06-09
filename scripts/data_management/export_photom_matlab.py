"""
Export photometry data for MATLAB (CSV format).
Filters sessions by:
1. Protocol 4 (pprobe0 == 0.2, hazardtype != 'split block')
2. Performance on largest change size (4Hz) > 0.7

Output columns:
- subject_id
- cell_type (D1/D2)
- region
- session_id
- roi_key
- performance_on_change_size_4
- late_fa_mean_trace ([-2, +1.5]s around Lick)
- late_fa_ci95_trace
- all_fa_mean_trace ([-2, +1.5]s around Lick)
- all_fa_ci95_trace
- hit_mean_trace ([-2, +1.5]s around Lick)
- hit_ci95_trace
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from visdetect_photom.core import io, session

def get_protocol(settings):
    """Infer protocol from session settings."""
    if settings.get('hazardtype') == 'split block':
        return 5
    elif settings.get('pprobe0') == 0.2:
        return 4
    elif settings.get('pprobe0') == 0.5:
        return 3
    elif settings.get('Trewdavailable') == 0.5:
        return 1
    else:
        return 2

def load_metadata_map(root_dir):
    """Parse mouse_genotypes_and_procedeures.txt"""
    possible_paths = [
        Path(root_dir) / "mouse_genotypes_and_procedeures.txt",
        Path(root_dir).parent / "photom_data" / "mouse_genotypes_and_procedeures.txt"
    ]
    
    meta_path = None
    for p in possible_paths:
        if p.exists():
            meta_path = p
            break
            
    if not meta_path:
        logging.error("Metadata file not found in photom_data")
        return {}
    
    mapping = {}
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("BG_"):
                    continue
                parts = line.split()
                subject = parts[0]
                
                if "A2a" in line:
                    ctype = "D2"
                elif "Drd1" in line:
                    ctype = "D1"
                else:
                    ctype = "Unknown"
                
                regions = []
                possible_regions = ["DMS", "VLS", "VMS", "DLS"]
                if "DMS&VLS" in line:
                    regions = ["DMS", "VLS"]
                else:
                    for r in possible_regions:
                         if r in line and "DMS&VLS" not in line:
                             regions.append(r)
                if not regions:
                    regions = ["Unknown"]
                    
                mapping[subject] = {'cell_type': ctype, 'regions': regions}
    except Exception as e:
        logging.error(f"Failed to read metadata: {e}")
    return mapping

def extract_peth_window_vector(trace, timestamps, event_time, win_start=-2.0, win_end=-1.5, n_points=50):
    """
    Extracts the trace segment in [win_start, win_end] relative to event_time.
    Resamples to a fixed number of points.
    """
    from scipy.interpolate import interp1d

    if event_time is None or np.isnan(event_time):
        return None
        
    t_start = event_time + win_start
    t_end = event_time + win_end
    
    if len(timestamps) == 0:
        return None
    if t_start < timestamps[0] or t_end > timestamps[-1]:
        return None
        
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    if not np.any(mask):
        return None
        
    seg_time = timestamps[mask]
    seg_sig = trace[mask]
    
    if len(seg_time) < 2:
        return None
        
    target_time = np.linspace(t_start, t_end, n_points)
    
    try:
        f = interp1d(seg_time, seg_sig, kind='linear', fill_value="extrapolate")
        return f(target_time)
    except Exception as e:
        return None

def compute_mean_vector_from_list(vector_list):
    """Computes mean vector and CI95 vector for a list of vectors."""
    if not vector_list:
        return None, None, 0
        
    stack = np.vstack(vector_list) # shape (n_trials, n_points)
    
    mean_vec = np.mean(stack, axis=0)
    sem_vec = np.std(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
    ci95_vec = 1.96 * sem_vec
            
    return mean_vec, ci95_vec, len(vector_list)

def group_sessions_by_subject_date(all_sessions):
    """
    Groups sessions by Subject and Date to handle multiple files per day.
    Returns: dict { (subject, date_str): [sess_dict_1, sess_dict_2] }
    """
    groups = defaultdict(list)
    date_pattern = re.compile(r"(\d{8})")
    
    for sess in all_sessions:
        settings_path = sess.get('session_settings')
        if not settings_path:
            continue
            
        filename = Path(settings_path).name
        # Filename usually: BG_020_20240318_162054__session_settings.json
        # Or: BG_020_20240318_162054__session_settings.json
        
        # Extract Subject
        # Assumption: Subject is always first part OR we use the subject inside JSON?
        # Parsing filename is faster than loading all JSONs.
        
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)
            # Subject is everything before the date_str (minus potential separator)
            # Find the position of the date
            idx = filename.find(date_str)
            subject_part = filename[:idx].rstrip('_')
            
            # Additional cleanup for subject
            # If subject appears multiple times or has weird chars
            groups[(subject_part, date_str)].append(sess)
        else:
            # Fallback if no date found (shouldn't happen with standard names)
            logging.warning(f"Could not parse date from {filename}")
            
    # Sort each group by timestamp in filename
    for key in groups:
        groups[key].sort(key=lambda x: Path(x['session_settings']).name)
        
    return groups

def process_session_group(sess_group_list, meta_map, vector_n_points=50, exclude_subjects=None):
    """
    Process a list of session files (Grouped by Subject+Date).
    Aggregates metrics and vectors across all files in the group.
    """
    try:
        # Accumulators
        collected_hits_4hz = 0
        collected_misses_4hz = 0
        
        roi_vectors = defaultdict(lambda: {'hits': [], 'late_fas': [], 'all_fas': []})
        
        # Metadata check using the first session in group
        first_sess_files = sess_group_list[0]
        first_settings_path = first_sess_files.get('session_settings')
        first_settings = io.load_json_data(first_settings_path)
        
        # Basic Protocol Check (must match for the group to be valid?)
        # We'll assume the first session defines the protocol for the day
        protocol = get_protocol(first_settings)
        if protocol != 4:
            return []

        # Load Subject Info
        subj = first_settings.get('subject_id') or "Unknown"
        # Try cleaning from filename if json is weird
        if subj == "Unknown":
             subj = Path(first_settings_path).name.split('_')[0] + "_" + Path(first_settings_path).name.split('_')[1]

        # Metadata Validation
        if subj not in meta_map:
            if f"BG_{subj}" in meta_map:
                subj = f"BG_{subj}"
        
        # Check Exclusion
        if exclude_subjects:
             if subj in exclude_subjects: return []
             if subj.replace("BG_", "") in exclude_subjects: return []
             if f"BG_{subj}" in exclude_subjects: return []

        subj_meta = meta_map.get(subj, {'cell_type': 'Unknown', 'regions': ['Unknown']})
        regions_meta = subj_meta['regions']

        # Determine Output Session ID (From the Earliest Session)
        settings_filename = Path(first_settings_path).name
        base = settings_filename.split('__')[0]
        if base.startswith(f"{subj}_"):
            clean_sess_id = base[len(subj)+1:]
        elif base.startswith(subj): # Case where subj has no underscore or weird format
             clean_sess_id = base[len(subj):].lstrip('_')
        else:
             clean_sess_id = base

        # Region Map
        roi_map_static = {
            'G0': 'DMS_Left',
            'G2': 'DMS_Right',
            'G4': 'VLS_Left',
            'G5': 'VLS_Right',
        }

        # Iterate through all sessions in the group
        valid_activity_found = False

        for sess_files in sess_group_list:
            
            # Load Session
            try:
                sess_obj = session.load_session_from_files(sess_files)
            except Exception as e:
                logging.warning(f"Failed to load sub-session {sess_files.get('session_settings')}: {e}")
                continue

            # Accumulate Performance Stats
            trials_4hz = [t for t in sess_obj.trials if t.change_size is not None and abs(t.change_size - 4) < 0.1]
            hits_4hz = sum(1 for t in trials_4hz if t.outcome == 'Hit')
            misses_4hz = sum(1 for t in trials_4hz if t.outcome == 'Miss')
            
            collected_hits_4hz += hits_4hz
            collected_misses_4hz += misses_4hz
            
            # Events
            sess_late_fas = [t for t in sess_obj.trials 
                        if t.outcome == 'FA' 
                        and t.reaction_time is not None 
                        and t.reaction_time > 3.0
                        and t.absolute_reaction_time is not None]

            sess_all_fas = [t for t in sess_obj.trials 
                        if t.outcome == 'FA' 
                        and t.absolute_reaction_time is not None]
                        
            sess_hits = [t for t in sess_obj.trials 
                        if t.outcome == 'Hit'
                        and t.absolute_reaction_time is not None]
            
            # Extract Vectors for each ROI present in this session
            roi_keys = [k for k in sess_obj.photometry_data.keys() if k.startswith('G')]
            if not roi_keys:
                roi_keys = list(sess_obj.photometry_data.keys()) # Fallback
            
            for roi_key in roi_keys:
                trace_obj = sess_obj.photometry_data[roi_key]
                trace = trace_obj.signal
                timestamps = trace_obj.timestamps
                
                # Hits
                for t in sess_hits:
                    v = extract_peth_window_vector(trace, timestamps, t.absolute_reaction_time, 
                                                 win_start=-2.0, win_end=1.5, n_points=vector_n_points)
                    if v is not None:
                        roi_vectors[roi_key]['hits'].append(v)
                
                # Late FAs (reaction_time > 3s)
                for t in sess_late_fas:
                    v = extract_peth_window_vector(trace, timestamps, t.absolute_reaction_time, 
                                                 win_start=-2.0, win_end=1.5, n_points=vector_n_points)
                    if v is not None:
                        roi_vectors[roi_key]['late_fas'].append(v)

                # All FAs
                for t in sess_all_fas:
                    v = extract_peth_window_vector(trace, timestamps, t.absolute_reaction_time, 
                                                 win_start=-2.0, win_end=1.5, n_points=vector_n_points)
                    if v is not None:
                        roi_vectors[roi_key]['all_fas'].append(v)

        # -- Protocol / Performance Filtering on Aggregated Data --
        
        total_go_4hz = collected_hits_4hz + collected_misses_4hz
        if total_go_4hz == 0:
            return []
            
        hr_4hz = collected_hits_4hz / total_go_4hz
        if hr_4hz <= 0.7:
            return []
            
        # -- Generate Rows --
        rows = []
        all_roi_keys = sorted(roi_vectors.keys())
        
        for idx, roi_key in enumerate(all_roi_keys):
            # Resolve Region
            if roi_key in roi_map_static:
                assigned_region = roi_map_static[roi_key]
            else:
                # Heuristic: try to map based on index if multiple ROIs exist
                # This is tricky across merged sessions if ROIs appear/disappear, but for same day it should be consistent.
                # Just use the index in the sorted list of all keys found.
                if idx < len(regions_meta):
                    assigned_region = regions_meta[idx]
                else:
                    assigned_region = "Unknown"
            
            hits_vecs = roi_vectors[roi_key]['hits']
            late_fas_vecs = roi_vectors[roi_key]['late_fas']
            all_fas_vecs = roi_vectors[roi_key]['all_fas']
            
            # Filter condition: n_hit < 5
            if len(hits_vecs) < 5:
                continue
                
            hit_mean, hit_ci, n_hit = compute_mean_vector_from_list(hits_vecs)
            lfa_mean, lfa_ci, n_lfa = compute_mean_vector_from_list(late_fas_vecs)
            afa_mean, afa_ci, n_afa = compute_mean_vector_from_list(all_fas_vecs)
            
            row_dict = {
                'subject_id': subj,
                'cell_type': subj_meta['cell_type'],
                'region': assigned_region,
                'session_id': clean_sess_id,
                'roi_key': roi_key,
                'performance_on_change_size_4': hr_4hz,
                'n_late_fa_trials_used': n_lfa,
                'n_all_fa_trials_used': n_afa,
                'n_hit_trials_used': n_hit
            }
            
            # Helper to add vector columns
            def add_vec_cols(prefix, vec):
                if vec is not None:
                    for i in range(vector_n_points):
                        row_dict[f'{prefix}_{i+1}'] = vec[i]
                else:
                    for i in range(vector_n_points):
                        row_dict[f'{prefix}_{i+1}'] = np.nan

            add_vec_cols('late_fa_mean_trace', lfa_mean)
            add_vec_cols('late_fa_ci95_trace', lfa_ci)
            add_vec_cols('all_fa_mean_trace', afa_mean)
            add_vec_cols('all_fa_ci95_trace', afa_ci)
            add_vec_cols('hit_mean_trace', hit_mean)
            add_vec_cols('hit_ci95_trace', hit_ci)

            rows.append(row_dict)
            
        return rows

    except Exception as e:
        logging.error(f"Error processing group {sess_group_list[0].get('session_settings')}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Export photometry data for MATLAB.")
    parser.add_argument("--exclude", nargs='+', default=[], help="List of subjects to exclude (e.g. BG_014 BG_027)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the CSV file")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    root_dir = "photom_data"
    output_filename = "photometry_export_matlab.csv"
    output_path = Path(args.output_dir) / output_filename
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load Metadata
    meta_map = load_metadata_map(root_dir)
    logging.info(f"Loaded metadata for subjects: {list(meta_map.keys())}")
    
    if args.exclude:
        logging.info(f"Excluding subjects: {args.exclude}")
    
    # Find Sessions
    raw_sessions = io.find_all_sessions(root_dir, recursive=True)
    logging.info(f"Found {len(raw_sessions)} raw session files.")
    
    # Group Sessions
    session_groups = group_sessions_by_subject_date(raw_sessions)
    logging.info(f"grouped into {len(session_groups)} unique subject-days.")
    
    sorted_groups = list(session_groups.values())
    
    data_rows = []
    vector_n_points = 350  # 100 pts/sec * 3.5s window [-2, +1.5]
    
    # Parallel Processing
    if args.workers > 1:
        logging.info(f"Starting parallel processing with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_group = {
                executor.submit(process_session_group, grp, meta_map, vector_n_points, args.exclude): grp 
                for grp in sorted_groups
            }
            
            for future in tqdm(as_completed(future_to_group), total=len(sorted_groups), desc="Processing Groups"):
                try:
                    rows = future.result()
                    if rows:
                        data_rows.extend(rows)
                except Exception as exc:
                    logging.error(f"Generated an exception: {exc}")
    else:
        logging.info(f"Starting sequential processing (1 worker)...")
        for grp in tqdm(sorted_groups, desc="Processing Groups"):
            rows = process_session_group(grp, meta_map, vector_n_points, args.exclude)
            if rows:
                data_rows.extend(rows)

    # Save
    if data_rows:
        df_out = pd.DataFrame(data_rows)
        # Sort
        df_out.sort_values(by=['subject_id', 'session_id', 'roi_key'], inplace=True)
        
        # Ensure column order is nice
        # Get base columns first
        base_cols = ['subject_id', 'cell_type', 'region', 'session_id', 'roi_key', 
                     'performance_on_change_size_4', 'n_late_fa_trials_used', 'n_all_fa_trials_used', 'n_hit_trials_used']
        
        # Then the vector columns in order
        vec_prefixes = ['late_fa_mean_trace', 'late_fa_ci95_trace', 'all_fa_mean_trace', 'all_fa_ci95_trace', 'hit_mean_trace', 'hit_ci95_trace']
        vec_cols = []
        for p in vec_prefixes:
            for i in range(1, vector_n_points + 1):
                vec_cols.append(f'{p}_{i}')
                
        final_cols = base_cols + vec_cols
        # Reorder if all exist, otherwise let pandas handle it (but we built them so they should exist)
        # Use simple reindex to ignore missing if any (unlikely)
        df_out = df_out.reindex(columns=final_cols)

        try:
            df_out.to_csv(output_path, index=False)
            logging.info(f"Exported {len(df_out)} rows to {output_path}")
        except PermissionError:
            logging.error(f"PERMISSION DENIED: Could not write to {output_path}. Is the file open?")
    else:
        logging.warning("No rows extracted.")

if __name__ == "__main__":
    main()
