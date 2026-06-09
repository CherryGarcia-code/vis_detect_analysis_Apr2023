"""
Plot single-session traces (Hit vs Late FA) for each subject in a grid.
Generates one figure per Subject.
Each figure contains a grid of subplots, one per Session.
Each subplot shows traces for all ROIs in that session.

Usage:
    python scripts/data_management/plot_subject_sessions_grid.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import math

# --- Configuration ---
CSV_PATH = Path("photometry_export_matlab.csv")
OUTPUT_DIR = Path("FIGURES/session_grids")
N_POINTS = 350
TIME_WINDOW = np.linspace(-2.0, 1.5, N_POINTS)
# ---------------------

def extract_trace_from_row(row, prefix, n_points=N_POINTS):
    """Extract a trace vector from numbered columns (e.g. hit_mean_trace_1 ... _350)."""
    cols = [f'{prefix}_{i}' for i in range(1, n_points + 1)]
    try:
        vals = row[cols].values.astype(float)
        return vals
    except Exception:
        return np.full(n_points, np.nan)

def main():
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = sorted(df['subject_id'].unique())
    print(f"Processing {len(subjects)} subjects...")

    for subj in subjects:
        subj_df = df[df['subject_id'] == subj].copy()
        
        # Get unique sessions (ensure sorting)
        sessions = sorted(subj_df['session_id'].unique())
        n_sessions = len(sessions)
        
        if n_sessions == 0:
            continue
            
        print(f"  Plotting {subj} ({n_sessions} sessions)...")

        # Layout: ~5 columns
        cols = 5
        rows = math.ceil(n_sessions / cols)
        
        # Use constrained_layout for better spacing
        fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows), 
                                 sharex=True, sharey=True, constrained_layout=True)
        # Verify axes Type
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        for i, sess_id in enumerate(sessions):
            ax = axes_flat[i]
            sess_data = subj_df[subj_df['session_id'] == sess_id]
            
            # Extract trials counts
            n_hit = sess_data['n_hit_trials_used'].max()
            n_fa = sess_data['n_late_fa_trials_used'].max()
            perf = sess_data.iloc[0]['performance_on_change_size_4']
            
            # Title: Session ID + Params
            ax.set_title(f"{sess_id}\nH:{n_hit} F:{n_fa} p:{perf:.2f}", fontsize=9)
            
            # Plot each ROI
            rois = sess_data['roi_key'].unique()
            
            for roi in rois:
                row = sess_data[sess_data['roi_key'] == roi].iloc[0]
                region = row['region'] 
                
                hit_trace = extract_trace_from_row(row, 'hit_mean_trace')
                fa_trace = extract_trace_from_row(row, 'late_fa_mean_trace')
                all_fa_trace = extract_trace_from_row(row, 'all_fa_mean_trace')
                
                # Style Mapping: Left=Solid, Right=Dashed
                if 'Left' in region or 'G0' in roi or 'G4' in roi:
                    ls = '-'
                else: 
                    ls = '--' # Right
                
                # Plot Hit (Green)
                if not np.all(np.isnan(hit_trace)):
                    lbl = f"Hit {region}"
                    ax.plot(TIME_WINDOW, hit_trace, color='green', linestyle=ls, linewidth=1.5, label=lbl, alpha=0.8)

                # Plot Late FA (Red)
                if not np.all(np.isnan(fa_trace)):
                    lbl = f"Late FA {region}"
                    ax.plot(TIME_WINDOW, fa_trace, color='red', linestyle=ls, linewidth=1.5, label=lbl, alpha=0.8)

                # Plot All FA (Orange)
                if not np.all(np.isnan(all_fa_trace)):
                    lbl = f"All FA {region}"
                    ax.plot(TIME_WINDOW, all_fa_trace, color='orange', linestyle=ls, linewidth=1.2, label=lbl, alpha=0.6)

            ax.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
            ax.grid(True, alpha=0.3)
            if i >= n_sessions - cols:
                 ax.set_xlabel("Time from Lick (s)")

        # Hide empty axes
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        # Global Legend (Deduplicated)
        handles, labels = [], []
        seen = set()
        for ax in axes_flat:
            h, l = ax.get_legend_handles_labels()
            for h_test, l_test in zip(h, l):
                if l_test not in seen:
                    handles.append(h_test)
                    labels.append(l_test)
                    seen.add(l_test)
        
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')

        cell_type = subj_df.iloc[0]['cell_type']
        fig.suptitle(f"Subject: {subj} ({cell_type}) - Hit vs Late FA vs All FA", fontsize=16)
        
        out_file = OUTPUT_DIR / f"{subj}_session_grid.png"
        
        try:
            plt.savefig(out_file, dpi=100)
            print(f"  Saved {out_file}")
        except Exception as e:
            print(f"  Error saving {subj}: {e}")
            
        plt.close(fig)

    print("All done.")

if __name__ == "__main__":
    main()
