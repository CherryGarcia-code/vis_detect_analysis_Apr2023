"""
Plot summary of exported MATLAB data to verify contents.
Aggregates traces across sessions/mice for Hits vs Late FAs, split by Cell Type and Region.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_trace_string(s):
    """Convert space-separated string to numpy array."""
    if pd.isna(s):
        return np.full(50, np.nan)
    return np.fromstring(s, sep=' ')

def main():
    csv_path = Path("photometry_export_matlab.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    # 1. Parse Vectors
    # Current shape: One row per session-ROI
    # We want to stack all session traces to compute Grand Mean & Error across sessions
    
    # Expand the dataframe: We need to allow grouping by Region/CellType
    # Simplified Region: "DMS_Left" -> "DMS"
    df['region_broad'] = df['region'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    # Remove Unknowns if likely bad mappings
    df = df[df['region_broad'].isin(['DMS', 'VLS'])]

    # Data structure for plotting
    # List of dictionaries? Or just iterate groups
    plot_groups = df.groupby(['cell_type', 'region_broad'])
    
    # Setup Plot
    # Rows: Cell Type (D1, D2)
    # Cols: Region (DMS, VLS)
    # But we might have different sets. Let's find unique combos.
    cell_types = sorted(df['cell_type'].unique())
    regions = sorted(df['region_broad'].unique())
    
    fig, axes = plt.subplots(len(cell_types), len(regions), 
                             figsize=(5*len(regions), 4*len(cell_types)), 
                             sharex=True, sharey=True)
    
    # Handle single row/col cases for axes indexing
    if len(cell_types) == 1 and len(regions) == 1:
        axes = np.array([[axes]])
    elif len(cell_types) == 1:
        axes = axes.reshape(1, -1)
    elif len(regions) == 1:
        axes = axes.reshape(-1, 1)

    time_vec = np.linspace(-2.0, -1.5, 50)

    for i, ctype in enumerate(cell_types):
        for j, reg in enumerate(regions):
            ax = axes[i, j]
            
            # Filter Data
            subset = df[(df['cell_type'] == ctype) & (df['region_broad'] == reg)]
            n_sessions = len(subset)
            
            if n_sessions == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                continue
                
            # Collect Traces
            hit_traces = np.vstack(subset['hit_mean_trace'].apply(parse_trace_string).values)
            fa_traces = np.vstack(subset['late_fa_mean_trace'].apply(parse_trace_string).values)
            
            # Simple Cleaning of NaNs (e.g. sessions with no Late FAs)
            # Mask rows where all are NaN
            valid_fa = ~np.isnan(fa_traces).all(axis=1)
            valid_hit = ~np.isnan(hit_traces).all(axis=1)
            
            fa_traces = fa_traces[valid_fa]
            hit_traces = hit_traces[valid_hit]
            
            # Compute total trials used
            # We align valid mask with the subset to sum the correct rows
            # valid_hit matches the subset index order? Yes, .values
            total_hit_trials = subset.iloc[valid_hit]['n_hit_trials_used'].sum()
            total_fa_trials = subset.iloc[valid_fa]['n_late_fa_trials_used'].sum()
            
            # Compute Grand Means and SEM across sessions
            hit_mean = np.nanmean(hit_traces, axis=0)
            hit_sem = np.nanstd(hit_traces, axis=0, ddof=1) / np.sqrt(hit_traces.shape[0])
            
            fa_mean = np.nanmean(fa_traces, axis=0)
            fa_sem = np.nanstd(fa_traces, axis=0, ddof=1) / np.sqrt(fa_traces.shape[0])
            
            # Plot Hits
            ax.plot(time_vec, hit_mean, color='green', label=f'Hit\n(N={len(hit_traces)} sess, k={total_hit_trials} trials)', lw=2)
            ax.fill_between(time_vec, hit_mean - hit_sem, hit_mean + hit_sem, color='green', alpha=0.2)
            
            # Plot FAs
            ax.plot(time_vec, fa_mean, color='red', label=f'Late FA\n(N={len(fa_traces)} sess, k={total_fa_trials} trials)', lw=2)
            ax.fill_between(time_vec, fa_mean - fa_sem, fa_mean + fa_sem, color='red', alpha=0.2)
            
            ax.set_title(f"{ctype} - {reg}")
            if i == len(cell_types) - 1:
                ax.set_xlabel("Time from Lick (s)")
            if j == 0:
                ax.set_ylabel("dF/F (z-score)")
                
            ax.legend(fontsize='small')
            ax.axvline(0, color='k', linestyle='--', alpha=0.3) # Lick time (though we are at -2 to -1.5)
            # Note: 0 is not in view window (-2 to -1.5), so axvline 0 won't show.
            # Maybe show window edges? 
    
    plt.suptitle("Photometry Activity: Late Pre-Lick Window (-2.0s to -1.5s)\nMean ± SEM across sessions", y=1.02)
    plt.tight_layout()
    
    out_file = "photometry_export_check_plot.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
