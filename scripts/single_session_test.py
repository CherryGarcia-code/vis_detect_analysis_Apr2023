import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from visdetect_photom.core import session, io
from visdetect_photom.analysis import preprocessing, statistics

def main():
    # Define paths
    root_dir = Path("photom_data")
    output_root = Path("FIGURES")
    
    # Test Session: BG_016 2024-01-11
    # Note: Using absolute paths resolved from current working directory
    cwd = Path.cwd()
    session_files = {
        'trials': str(cwd / root_dir / "BG_016/BG_016_20240228_133749__trials.json"),
        'photom': str(cwd / root_dir / "BG_016/BG_016__photom_2024-02-28T13_37_20.csv"),
        'photom_io': str(cwd / root_dir / "BG_016/BG_016__photom_IO_2024-02-28T13_37_20.csv"),
        'session_settings': str(cwd / root_dir / "BG_016/BG_016_20240228_133749__session_settings.json")
    }
    
    print("Loading session...")
    try:
        sess = session.load_session_from_files(session_files)
        print(f"Session loaded: {sess.session_id}")
    except Exception as e:
        print(f"Failed to load session: {e}")
        return
    
    # Create output directory
    # Ensure subject_id has BG_ prefix if missing
    subject_folder = sess.subject_id if sess.subject_id.startswith("BG_") else f"BG_{sess.subject_id}"
    output_dir = output_root / subject_folder / sess.session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process Photometry (De-interleave and dF/F)
    print("Processing photometry signals...")
    try:
        raw_photom = io.load_csv_data(session_files['photom'])
        print(f"Raw photometry loaded. Shape: {raw_photom.shape}")
        print(f"Columns: {raw_photom.columns.tolist()}")
        print(f"LedStates found: {raw_photom['LedState'].unique()}")
        
        processed_df = preprocessing.process_photometry_signals(raw_photom, session_zscored=True)
        print(f"Processed DF Shape: {processed_df.shape}")
        print(f"Processed Columns: {processed_df.columns.tolist()}")
        
    except Exception as e:
        print(f"Failed to process photometry: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if processed_df.empty:
        print("Error: No processed signals found.")
        return

    # Extract Signals for Plotting
    # We need to align 'zscored_G0_clean_signal_dff' etc. to events
    
    # Events
    # Hit Times: Use absolute_reaction_time (Time of Lick)
    hit_times = [t.absolute_reaction_time for t in sess.trials if t.outcome == 'Hit' and t.absolute_reaction_time is not None]
    
    # Miss Times: Use absolute_change_time (Time of Stimulus Change)
    miss_times = [t.absolute_change_time for t in sess.trials if t.outcome == 'Miss' and t.absolute_change_time is not None]
    
    # Split Hits by Change Size (using Lick Time)
    # Small: 1.25, 1.35
    # Big: 1.5, 2, 4
    small_sizes = [1.25, 1.35]
    big_sizes = [1.5, 2, 4]
    
    # Debug: Print unique change sizes found
    unique_sizes = set(t.change_size for t in sess.trials if t.change_size is not None)
    print(f"Unique change sizes found in session: {unique_sizes}")
    
    hit_times_small = [t.absolute_reaction_time for t in sess.trials 
                       if t.outcome == 'Hit' and t.absolute_reaction_time is not None 
                       and t.change_size in small_sizes]
    
    hit_times_big = [t.absolute_reaction_time for t in sess.trials 
                     if t.outcome == 'Hit' and t.absolute_reaction_time is not None 
                     and t.change_size in big_sizes]
    
    print(f"Found {len(hit_times)} Hits ({len(hit_times_small)} Small, {len(hit_times_big)} Big)")

    # Change Onset (Hit + Miss) - Aligned to Stimulus Change
    # Small: 1.25, 1.35
    # Big: 1.5, 2, 4
    change_times_small = [t.absolute_change_time for t in sess.trials 
                          if t.outcome in ['Hit', 'Miss'] and t.absolute_change_time is not None 
                          and t.change_size in small_sizes]
    
    change_times_big = [t.absolute_change_time for t in sess.trials 
                        if t.outcome in ['Hit', 'Miss'] and t.absolute_change_time is not None 
                        and t.change_size in big_sizes]
    
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
    
    print(f"Found {len(change_times_small)} Small Changes, {len(change_times_big)} Big Changes (Hit+Miss)")

    # Baseline Onset (Start of Stimulus = Input0)
    # Note: absolute_start_time in session.py is (Input0 - ITI).
    # So Baseline Onset = absolute_start_time + iti_duration
    
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

    # FA and Abort: Use absolute_reaction_time (time of the lick)
    fa_times = [t.absolute_reaction_time for t in sess.trials if t.outcome == 'FA' and t.absolute_reaction_time is not None]
    
    # Split FAs by Reaction Time (Early <= 3s, Late > 3s)
    fa_times_early = [t.absolute_reaction_time for t in sess.trials 
                      if t.outcome == 'FA' and t.absolute_reaction_time is not None 
                      and t.reaction_time is not None and t.reaction_time <= 3.0]
    
    fa_times_late = [t.absolute_reaction_time for t in sess.trials 
                     if t.outcome == 'FA' and t.absolute_reaction_time is not None 
                     and t.reaction_time is not None and t.reaction_time > 3.0]
    
    abort_times = [t.absolute_reaction_time for t in sess.trials if t.outcome == 'Abort' and t.absolute_reaction_time is not None]
    
    print(f"Found {len(hit_times)} Hits, {len(miss_times)} Misses, {len(fa_times)} FAs ({len(fa_times_early)} Early, {len(fa_times_late)} Late), {len(abort_times)} Aborts.")
    if len(hit_times) > 0:
        print(f"Sample Hit Times: {hit_times[:5]}")
    
    # Plotting
    timestamps = processed_df['SystemTimestamp'].values
    print(f"Timestamp range: {timestamps.min()} to {timestamps.max()}")
    
    # Identify ROIs
    rois = {}
    for col in processed_df.columns:
        if 'zscored' in col and 'clean_signal_dff' in col:
            roi_name = col.replace('zscored_', '').replace('_clean_signal_dff', '')
            rois[roi_name] = processed_df[col].values
            
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
        if not times or len(times) == 0:
            continue
            
        # --- Trace Plot (Side-by-Side) ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        for i, roi_key in enumerate(['G0', 'G2']):
            ax = axes[i]
            if roi_key in rois:
                signal = rois[roi_key]
                # Use trial-based baseline z-scoring (Option 1)
                time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(times), 
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
                
                # Add legend to the right subplot
                if i == 1:
                    ax.legend(loc='upper right')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"Trace_{event_name}.png")
        plt.close()

        # --- Heatmap Plot (Side-by-Side) ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        
        # Extract PETHs first to find global min/max for consistent color scale
        peths = {}
        for roi_key in ['G0', 'G2']:
            if roi_key in rois:
                _, peth = statistics.extract_peth(rois[roi_key], timestamps, np.array(times), 
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

            # Colorbar
            if im:
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), label='Z-Score')
            
            plt.savefig(output_dir / f"Heatmap_{event_name}.png")
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
        if roi_key in rois:
            signal = rois[roi_key]
            
            for times, label, color in baselines_to_plot:
                if times and len(times) > 0:
                    time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(times), 
                                                            window=(-2, 4), 
                                                            baseline_window=(-2, 0))
                    
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    
                    ax.plot(time_axis, mean_trace, color=color, label=f"{label} (n={len(times)})", linewidth=2)
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color=color)
                    has_data = True
            
            ax.axvline(0, linestyle='--', color='k', alpha=0.7)
            ax.set_title(f"{roi_key} - Baseline Comparison")
            ax.set_xlabel("Time from Stimulus Onset (s)")
            if i == 0:
                ax.set_ylabel("Z-Score")
            
            # Legend on the right subplot
            if i == 1:
                ax.legend(loc='upper right')
        else:
            ax.axis('off')
            
    if has_data:
        plt.tight_layout()
        plt.savefig(output_dir / "Combined_Baseline.png")
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
            if roi_key in rois:
                signal = rois[roi_key]
                
                # Plot Hit
                if hit_t and len(hit_t) > 0:
                    time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(hit_t), 
                                                            window=(-2, 4), baseline_window=(-2, 0))
                    mean_trace = np.nanmean(peth, axis=0)
                    sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                    ax.plot(time_axis, mean_trace, color='green', label=f"Hit (n={len(hit_t)})", linewidth=2)
                    ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='green')

                # Plot Miss
                if miss_t and len(miss_t) > 0:
                    time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(miss_t), 
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
        plt.savefig(output_dir / f"{comp_name}.png")
        plt.close()

    # 4. Baseline FA Comparison (Early vs Late)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, roi_key in enumerate(['G0', 'G2']):
        ax = axes[i]
        if roi_key in rois:
            signal = rois[roi_key]
            
            # Plot Early FA
            if baseline_fa_early and len(baseline_fa_early) > 0:
                time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(baseline_fa_early), 
                                                        window=(-2, 4), baseline_window=(-2, 0))
                mean_trace = np.nanmean(peth, axis=0)
                sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                ax.plot(time_axis, mean_trace, color='lightcoral', label=f"Early FA (n={len(baseline_fa_early)})", linewidth=2)
                ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='lightcoral')

            # Plot Late FA
            if baseline_fa_late and len(baseline_fa_late) > 0:
                time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(baseline_fa_late), 
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
    plt.savefig(output_dir / "Baseline_FA_Comparison.png")
    plt.close()

    # 5. FA Response Comparison (Early vs Late)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    for i, roi_key in enumerate(['G0', 'G2']):
        ax = axes[i]
        if roi_key in rois:
            signal = rois[roi_key]
            
            # Plot Early FA (Response Aligned)
            if fa_times_early and len(fa_times_early) > 0:
                time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(fa_times_early), 
                                                        window=(-2, 4), baseline_window=(-2, 0))
                mean_trace = np.nanmean(peth, axis=0)
                sem_trace = np.nanstd(peth, axis=0) / np.sqrt(peth.shape[0])
                ax.plot(time_axis, mean_trace, color='lightcoral', label=f"Early FA (n={len(fa_times_early)})", linewidth=2)
                ax.fill_between(time_axis, mean_trace-sem_trace, mean_trace+sem_trace, alpha=0.1, color='lightcoral')

            # Plot Late FA (Response Aligned)
            if fa_times_late and len(fa_times_late) > 0:
                time_axis, peth = statistics.extract_peth(signal, timestamps, np.array(fa_times_late), 
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
    plt.savefig(output_dir / "FA_Response_Comparison.png")
    plt.close()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
