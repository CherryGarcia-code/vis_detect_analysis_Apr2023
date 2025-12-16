import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

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
