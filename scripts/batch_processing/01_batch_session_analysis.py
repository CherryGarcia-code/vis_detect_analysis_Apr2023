"""
Batch processing script for session-level photometry analysis.

This script iterates over all discovered sessions, loads the data, performs
standard analyses (QC, PETH, performance metrics), and saves summary figures and tables.

Usage:
    python 01_batch_session_analysis.py --root_dir /path/to/data --output_dir /path/to/results
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm
import concurrent.futures

# Ensure src is in path if running as script
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from visdetect_photom.core import io, session
from visdetect_photom.analysis import preprocessing, statistics
from visdetect_photom.viz import plotting

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def process_session(session_files: dict, output_dir: Path):
    """
    Process a single session.
    
    Args:
        session_files: Dict containing paths to session files.
        output_dir: Directory to save results.
    """
    try:
        # logging.info(f"Processing session: {session_files.get('trials', 'Unknown')}")
        
        # 1. Load Data
        sess = session.load_session_from_files(session_files)
        # logging.info(f"Loaded session {sess.session_id}: {len(sess.trials)} trials, {len(sess.photometry_data)} traces.")
        
        # Create session output directory
        # Structure: FIGURES/<SubjectID>/<SessionID>
        # Ensure subject_id has BG_ prefix if missing
        subject_folder = sess.subject_id if sess.subject_id.startswith("BG_") else f"BG_{sess.subject_id}"
        session_out_dir = output_dir / subject_folder / sess.session_id
        session_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Analysis Pipeline
        
        # A. Compute dF/F (Z-score)
        # NOTE: session.load_session_from_files now handles preprocessing (de-interleaving, isosbestic fit, dF/F)
        # and returns z-scored traces in sess.photometry_data.
        # So we skip manual calculation here.

        # B. Align to Events (PETH)
        # Events of interest: 'change_time' (for Hits/Misses), 'start_time'
        
        # Extract event times (Use ABSOLUTE timestamps for alignment with photometry)
        # Hit Times: Use absolute_reaction_time (Time of Lick)
        hit_times = [t.absolute_reaction_time for t in sess.trials if t.outcome == 'Hit' and t.absolute_reaction_time is not None]
        
        # Miss Times: Use absolute_change_time (Time of Stimulus Change)
        miss_times = [t.absolute_change_time for t in sess.trials if t.outcome == 'Miss' and t.absolute_change_time is not None]
        
        # Split Hits by Change Size (using Lick Time)
        small_sizes = [1.25, 1.35]
        big_sizes = [1.5, 2, 4]
        
        hit_times_small = [t.absolute_reaction_time for t in sess.trials 
                           if t.outcome == 'Hit' and t.absolute_reaction_time is not None 
                           and t.change_size in small_sizes]
        
        hit_times_big = [t.absolute_reaction_time for t in sess.trials 
                         if t.outcome == 'Hit' and t.absolute_reaction_time is not None 
                         and t.change_size in big_sizes]

        # Change Onset (Hit + Miss) - Aligned to Stimulus Change
        change_times_small = [t.absolute_change_time for t in sess.trials 
                              if t.outcome in ['Hit', 'Miss'] and t.absolute_change_time is not None 
                              and t.change_size in small_sizes]
        
        change_times_big = [t.absolute_change_time for t in sess.trials 
                            if t.outcome in ['Hit', 'Miss'] and t.absolute_change_time is not None 
                            and t.change_size in big_sizes]

        # Baseline Onset (Start of Stimulus = Input0 = absolute_start_time + iti_duration)
        def get_baseline_onset(t):
            if t.absolute_start_time is not None and t.iti_duration is not None:
                return t.absolute_start_time + t.iti_duration
            return None

        baseline_hit = [get_baseline_onset(t) for t in sess.trials if t.outcome == 'Hit' and get_baseline_onset(t) is not None]
        baseline_miss = [get_baseline_onset(t) for t in sess.trials if t.outcome == 'Miss' and get_baseline_onset(t) is not None]
        
        # Filter FA and Abort baselines: only include trials where reaction time > 1s
        baseline_fa = [get_baseline_onset(t) for t in sess.trials 
                       if t.outcome == 'FA' and get_baseline_onset(t) is not None 
                       and t.reaction_time is not None and t.reaction_time > 1.0]
        
        baseline_fa_early = [get_baseline_onset(t) for t in sess.trials 
                       if t.outcome == 'FA' and get_baseline_onset(t) is not None 
                       and t.reaction_time is not None and t.reaction_time > 1.0 and t.reaction_time <= 3.0]

        baseline_fa_late = [get_baseline_onset(t) for t in sess.trials 
                       if t.outcome == 'FA' and get_baseline_onset(t) is not None 
                       and t.reaction_time is not None and t.reaction_time > 1.0 and t.reaction_time > 3.0]
        
        baseline_abort = [get_baseline_onset(t) for t in sess.trials 
                          if t.outcome == 'Abort' and get_baseline_onset(t) is not None 
                          and t.reaction_time is not None and t.reaction_time > 1.0]

        # FA times: Use absolute reaction time if available, else try to calculate
        fa_times = []
        fa_times_early = []
        fa_times_late = []
        
        for t in sess.trials:
            if t.outcome == 'FA':
                rt = t.reaction_time
                abs_time = None
                if t.absolute_reaction_time is not None:
                    abs_time = t.absolute_reaction_time
                elif t.absolute_start_time is not None and rt is not None:
                    abs_time = t.absolute_start_time + rt
                
                if abs_time is not None:
                    fa_times.append(abs_time)
                    if rt is not None:
                        if rt <= 3.0:
                            fa_times_early.append(abs_time)
                        else:
                            fa_times_late.append(abs_time)

        # Abort times: Use absolute reaction time (time of abort)
        abort_times = []
        for t in sess.trials:
            if t.outcome == 'Abort':
                if t.absolute_reaction_time is not None:
                    abort_times.append(t.absolute_reaction_time)
                elif t.absolute_start_time is not None and t.reaction_time is not None:
                    abort_times.append(t.absolute_start_time + t.reaction_time)
        
        # Split Change Times by Outcome for Comparison
        change_times_small_hit = [t.absolute_change_time for t in sess.trials 
                                  if t.outcome == 'Hit' and t.absolute_change_time is not None 
                                  and t.change_size in small_sizes]
        change_times_small_miss = [t.absolute_change_time for t in sess.trials 
                                   if t.outcome == 'Miss' and t.absolute_change_time is not None 
                                   and t.change_size in small_sizes]
        
        change_times_big_hit = [t.absolute_change_time for t in sess.trials 
                                if t.outcome == 'Hit' and t.absolute_change_time is not None 
                                and t.change_size in big_sizes]
        change_times_big_miss = [t.absolute_change_time for t in sess.trials 
                                 if t.outcome == 'Miss' and t.absolute_change_time is not None 
                                 and t.change_size in big_sizes]

        # Convert to numpy
        hit_times = np.array(hit_times)
        hit_times_small = np.array(hit_times_small)
        hit_times_big = np.array(hit_times_big)
        miss_times = np.array(miss_times)
        change_times_small = np.array(change_times_small)
        change_times_big = np.array(change_times_big)
        baseline_hit = np.array(baseline_hit)
        baseline_miss = np.array(baseline_miss)
        baseline_fa = np.array(baseline_fa)
        baseline_fa_early = np.array(baseline_fa_early)
        baseline_fa_late = np.array(baseline_fa_late)
        baseline_abort = np.array(baseline_abort)
        fa_times = np.array(fa_times)
        fa_times_early = np.array(fa_times_early)
        fa_times_late = np.array(fa_times_late)
        abort_times = np.array(abort_times)
        change_times_small_hit = np.array(change_times_small_hit)
        change_times_small_miss = np.array(change_times_small_miss)
        change_times_big_hit = np.array(change_times_big_hit)
        change_times_big_miss = np.array(change_times_big_miss)
        
        # Debug logging for event times
        # logging.info(f"Session {sess.session_id}: Found {len(hit_times)} hits, {len(miss_times)} misses, {len(fa_times)} FAs, {len(abort_times)} aborts.")

        # C. Generate Plots
        # Define events to plot
        events = [
            (hit_times, "Hit", "green"),
            (hit_times_small, "Hit_Small", "limegreen"),
            (hit_times_big, "Hit_Big", "darkgreen"),
            (miss_times, "Miss", "purple"),
            (fa_times, "FA", "red"),
            # (fa_times_early, "FA_Early", "lightcoral"), # Combined in separate plot
            # (fa_times_late, "FA_Late", "darkred"),     # Combined in separate plot
            (abort_times, "Abort", "darkgrey"),
            (change_times_small, "Change_Small", "cornflowerblue"),
            (change_times_big, "Change_Big", "navy"),
            (baseline_hit, "Baseline_Hit", "green"),
            (baseline_miss, "Baseline_Miss", "purple"),
            (baseline_fa, "Baseline_FA", "red"),
            (baseline_abort, "Baseline_Abort", "darkgrey")
        ]

        # 1. Plot Individual Events (Trace + Heatmap)
        for times, event_name, color in events:
            if len(times) == 0:
                continue
                
            # --- Trace Plot (Side-by-Side) ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            
            for i, roi_key in enumerate(['G0', 'G2']):
                ax = axes[i]
                if roi_key in sess.photometry_data:
                    trace = sess.photometry_data[roi_key]
                    if len(trace.signal) == 0:
                        ax.axis('off')
                        continue
                        
                    time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, times, 
                                                            window=(-2, 4), 
                                                            baseline_window=(-2, 0))
                    
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    
                    ax.plot(time_axis, mean_trace, color=color, label=f"{event_name} (n={len(times)})")
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.3, color=color)
                    ax.axvline(0, linestyle='--', color='k')
                    ax.set_title(f"{roi_key} - {event_name}")
                    ax.set_xlabel("Time (s)")
                    if i == 0:
                        ax.set_ylabel("Z-Score")
                    
                    if i == 1:
                        ax.legend(loc='upper right')
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(session_out_dir / f"Trace_{event_name}.png")
            plt.close()

            # --- Heatmap Plot (Side-by-Side) ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
            
            peths = {}
            for roi_key in ['G0', 'G2']:
                if roi_key in sess.photometry_data:
                    trace = sess.photometry_data[roi_key]
                    if len(trace.signal) > 0:
                        _, peth = statistics.extract_peth(trace.signal, trace.timestamps, times, 
                                                        window=(-2, 4), baseline_window=(-2, 0))
                        peths[roi_key] = peth
            
            if peths:
                all_vals = np.concatenate([p.flatten() for p in peths.values()])
                max_val = np.nanmax(np.abs(all_vals)) if len(all_vals) > 0 else 1
                vmin, vmax = -max_val, max_val

                im = None
                for i, roi_key in enumerate(['G0', 'G2']):
                    ax = axes[i]
                    if roi_key in peths:
                        peth = peths[roi_key]
                        im = ax.imshow(peth, aspect='auto', cmap='RdBu_r', 
                                     extent=[-2, 4, len(peth), 0], 
                                     interpolation='nearest', vmin=vmin, vmax=vmax)
                        ax.set_title(f"{roi_key} - {event_name}")
                        ax.axvline(0, color='k', linestyle='--')
                        ax.set_xlabel("Time (s)")
                        if i == 0:
                            ax.set_ylabel("Trials")
                    else:
                        ax.axis('off')

                if im:
                    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label='Z-Score')
                
                plt.savefig(session_out_dir / f"Heatmap_{event_name}.png")
                plt.close()

        # 2. Combined Baseline Plot (Side-by-Side)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        baselines_to_plot = [
            (baseline_hit, "Hit", "green"),
            (baseline_miss, "Miss", "purple"),
            (baseline_fa, "FA", "red"),
            (baseline_abort, "Abort", "darkgrey")
        ]
        
        has_data = False
        for i, roi_key in enumerate(['G0', 'G2']):
            ax = axes[i]
            if roi_key in sess.photometry_data:
                trace = sess.photometry_data[roi_key]
                if len(trace.signal) == 0:
                    ax.axis('off')
                    continue

                for times, label, color in baselines_to_plot:
                    if len(times) > 0:
                        time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, times, 
                                                                window=(-2, 4), 
                                                                baseline_window=(-2, 0))
                        
                        mean_trace = np.nanmean(peth, axis=0)
                        sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                        
                        ax.plot(time_axis, mean_trace, color=color, label=f"{label} (n={len(times)})", linewidth=2)
                        plt.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color=color)
                        has_data = True
                
                ax.axvline(0, linestyle='--', color='k', alpha=0.7)
                ax.set_title(f"{roi_key} - Baseline Comparison")
                ax.set_xlabel("Time from Stimulus Onset (s)")
                if i == 0:
                    ax.set_ylabel("Z-Score")
                
                if i == 1:
                    ax.legend(loc='upper right')
            else:
                ax.axis('off')
                
        if has_data:
            plt.tight_layout()
            plt.savefig(session_out_dir / "Combined_Baseline.png")
            plt.close()

        # 3. Change Size Comparison (Hit vs Miss)
        change_comparisons = [
            ("Change_Small_Comparison", change_times_small_hit, change_times_small_miss),
            ("Change_Big_Comparison", change_times_big_hit, change_times_big_miss)
        ]

        for comp_name, hit_t, miss_t in change_comparisons:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            
            for i, roi_key in enumerate(['G0', 'G2']):
                ax = axes[i]
                if roi_key in sess.photometry_data:
                    trace = sess.photometry_data[roi_key]
                    if len(trace.signal) == 0:
                        ax.axis('off')
                        continue
                    
                    # Plot Hit
                    if len(hit_t) > 0:
                        time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, hit_t, 
                                                                window=(-2, 4), baseline_window=(-2, 0))
                        mean_trace = np.nanmean(peth, axis=0)
                        sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                        ax.plot(time_axis, mean_trace, color='green', label=f"Hit (n={len(hit_t)})", linewidth=2)
                        ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='green')

                    # Plot Miss
                    if len(miss_t) > 0:
                        time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, miss_t, 
                                                                window=(-2, 4), baseline_window=(-2, 0))
                        mean_trace = np.nanmean(peth, axis=0)
                        sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                        ax.plot(time_axis, mean_trace, color='purple', label=f"Miss (n={len(miss_t)})", linewidth=2)
                        ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='purple')
                    
                    ax.axvline(0, linestyle='--', color='k', alpha=0.7)
                    ax.set_title(f"{roi_key} - {comp_name.replace('_', ' ')}")
                    ax.set_xlabel("Time from Change (s)")
                    if i == 0:
                        ax.set_ylabel("Z-Score")
                    if i == 1:
                        ax.legend(loc='upper right')
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(session_out_dir / f"{comp_name}.png")
            plt.close()

        # 4. Baseline FA Comparison (Early vs Late)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        for i, roi_key in enumerate(['G0', 'G2']):
            ax = axes[i]
            if roi_key in sess.photometry_data:
                trace = sess.photometry_data[roi_key]
                if len(trace.signal) == 0:
                    ax.axis('off')
                    continue
                
                # Plot Early FA
                if len(baseline_fa_early) > 0:
                    time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, baseline_fa_early, 
                                                            window=(-2, 4), baseline_window=(-2, 0))
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    ax.plot(time_axis, mean_trace, color='lightcoral', label=f"Early FA (n={len(baseline_fa_early)})", linewidth=2)
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='lightcoral')

                # Plot Late FA
                if len(baseline_fa_late) > 0:
                    time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, baseline_fa_late, 
                                                            window=(-2, 4), baseline_window=(-2, 0))
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    ax.plot(time_axis, mean_trace, color='darkred', label=f"Late FA (n={len(baseline_fa_late)})", linewidth=2)
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='darkred')
                
                ax.axvline(0, linestyle='--', color='k', alpha=0.7)
                ax.set_title(f"{roi_key} - Baseline FA Comparison")
                ax.set_xlabel("Time from Stimulus Onset (s)")
                if i == 0:
                    ax.set_ylabel("Z-Score")
                if i == 1:
                    ax.legend(loc='upper right')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(session_out_dir / "Baseline_FA_Comparison.png")
        plt.close()

        # 5. FA Response Comparison (Early vs Late)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        for i, roi_key in enumerate(['G0', 'G2']):
            ax = axes[i]
            if roi_key in sess.photometry_data:
                trace = sess.photometry_data[roi_key]
                if len(trace.signal) == 0:
                    ax.axis('off')
                    continue
                
                # Plot Early FA (Response Aligned)
                if len(fa_times_early) > 0:
                    time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, fa_times_early, 
                                                            window=(-2, 4), baseline_window=(-2, 0))
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    ax.plot(time_axis, mean_trace, color='lightcoral', label=f"Early FA (n={len(fa_times_early)})", linewidth=2)
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='lightcoral')

                # Plot Late FA (Response Aligned)
                if len(fa_times_late) > 0:
                    time_axis, peth = statistics.extract_peth(trace.signal, trace.timestamps, fa_times_late, 
                                                            window=(-2, 4), baseline_window=(-2, 0))
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    ax.plot(time_axis, mean_trace, color='darkred', label=f"Late FA (n={len(fa_times_late)})", linewidth=2)
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='darkred')
                
                ax.axvline(0, linestyle='--', color='k', alpha=0.7)
                ax.set_title(f"{roi_key} - FA Response Comparison")
                ax.set_xlabel("Time from Lick (s)")
                if i == 0:
                    ax.set_ylabel("Z-Score")
                if i == 1:
                    ax.legend(loc='upper right')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(session_out_dir / "FA_Response_Comparison.png")
        plt.close()

        # 3. Save Summary Stats
        # Calculate performance
        n_hits = len(hit_times)
        n_misses = len(miss_times)
        n_fas = len(fa_times)
        n_aborts = len(abort_times)
        n_trials = len(sess.trials)
        
        summary_data = {
            "session_id": sess.session_id,
            "subject_id": sess.subject_id,
            "date": sess.session_date,
            "n_trials": n_trials,
            "n_hits": n_hits,
            "n_misses": n_misses,
            "n_fas": n_fas,
            "n_aborts": n_aborts,
            "hit_rate": n_hits / (n_hits + n_misses) if (n_hits + n_misses) > 0 else 0,
            "n_traces": len(sess.photometry_data)
        }
        pd.DataFrame([summary_data]).to_csv(session_out_dir / "session_summary.csv", index=False)
        
        # logging.info(f"Finished processing {sess.session_id}")
        return True

    except Exception as e:
        logging.error(f"Failed to process session {session_files.get('trials')}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process photometry sessions.")
    parser.add_argument("--root_dir", type=str, default="E:/python_analysis/git_repos/vis_detect_analysis_Apr2023/photom_data", help="Root directory containing session data.")
    parser.add_argument("--output_dir", type=str, default="E:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/batch_output", help="Directory to save outputs.")
    parser.add_argument("--max_sessions", type=int, default=None, help="Maximum number of sessions to process (for testing).")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Discover sessions
    sessions = io.find_all_sessions(str(root_path), recursive=True)
    logging.info(f"Found {len(sessions)} sessions.")
    
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
        logging.info(f"Processing subset of {len(sessions)} sessions.")
    
    # Parallel processing with tqdm
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_session, sess_files, output_path) for sess_files in sessions]
        
        # Iterate over completed futures with progress bar
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(sessions), desc="Processing Sessions"):
            pass

if __name__ == "__main__":
    main()
