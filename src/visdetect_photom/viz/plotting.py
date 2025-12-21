import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

def plot_peth_heatmap(peth_matrix: np.ndarray, time_axis: np.ndarray, 
                      event_name: str, roi_name: str, 
                      sort_idx: Optional[np.ndarray] = None,
                      output_path: Optional[str] = None,
                      trace_color: str = 'k') -> None:
    """
    Plot PETH heatmap and average trace.
    """
    if peth_matrix.shape[0] == 0:
        return

    # Sort if requested
    if sort_idx is not None:
        # Ensure sort_idx is valid
        valid_sort = sort_idx[sort_idx < peth_matrix.shape[0]]
        matrix_to_plot = peth_matrix[valid_sort]
    else:
        matrix_to_plot = peth_matrix

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Heatmap
    # Calculate symmetric limits for the heatmap to ensure 0 is in the middle
    max_val = np.nanmax(np.abs(matrix_to_plot)) if np.any(matrix_to_plot) else 1
    vmin, vmax = -max_val, max_val
    
    # Extent: [left, right, bottom, top]
    extent = [time_axis[0], time_axis[-1], matrix_to_plot.shape[0], 0]
    im = ax1.imshow(matrix_to_plot, aspect='auto', cmap='RdBu_r', extent=extent, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_title(f'{roi_name} aligned to {event_name}')
    ax1.set_ylabel('Trials')
    ax1.axvline(0, color='k', linestyle='--')
    plt.colorbar(im, ax=ax1, label='z-score')
    
    # Average Trace
    mean_trace = np.nanmean(peth_matrix, axis=0)
    sem_trace = np.nanstd(peth_matrix, axis=0) / np.sqrt(peth_matrix.shape[0])
    
    ax2.plot(time_axis, mean_trace, color=trace_color)
    ax2.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=trace_color, alpha=0.3)
    ax2.axvline(0, color='k', linestyle='--')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mean z-score')
    ax2.set_xlim(time_axis[0], time_axis[-1])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_melted_and_save(melted: pd.DataFrame, behave_event: str, out_png: str, title: Optional[str] = None) -> None:
    """
    Plot melted photometry data and save to file.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted, x='time', y='signal', hue='roi')
    if title:
        plt.title(title)
    else:
        plt.title(f'Photometry Signal Aligned to {behave_event}')
    plt.xlabel('Time (s)')
    plt.ylabel('z-scored dF/F')
    plt.savefig(out_png)
    plt.close()
