"""
Plot single-session behavior analysis.

Generates:
1. Performance Summary (Rolling Hit/Response rates, RTs).
2. Psychometric Curve (Hit Rate vs Change Size).
3. RT Distribution.

Usage:
    python scripts/analysis/behavior/plot_session_behavior.py --session_dir <path_to_session_data> --out <output_dir>
"""
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root / 'src') not in sys.path:
    sys.path.insert(0, str(repo_root / 'src'))

from visdetect_photom.core import session
from visdetect_photom.analysis.statistics import calculate_sdt_metrics
from visdetect_photom.viz import plotting

def compute_rolling_performance(sess):
    """
    Computes rolling performance metrics (Hit Rate, FA Rate, Miss Rate).
    Returns a DataFrame with trial-by-trial metrics.
    """
    # Create a DataFrame from trials
    trials = sess.trials
    data = []
    
    for i, t in enumerate(trials):
        row = {
            'trial_idx': i,
            'is_hit': t.outcome == 'Hit',
            'is_miss': t.outcome == 'Miss',
            'is_fa': t.outcome == 'FA',
            'is_abort': t.outcome == 'Abort',
            'change_size': t.change_size if hasattr(t, 'change_size') else np.nan,
            'rt': t.reaction_time if hasattr(t, 'reaction_time') else np.nan
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Rolling window parameters
    window = 30
    min_periods = 5
    
    df['rolling_hit_rate'] = df['is_hit'].rolling(window=window, min_periods=min_periods).mean()
    df['rolling_miss_rate'] = df['is_miss'].rolling(window=window, min_periods=min_periods).mean()
    df['rolling_fa_rate'] = df['is_fa'].rolling(window=window, min_periods=min_periods).mean()
    
    # Infer State:
    # Impulsive: FA Rate > 0.48 (matches 2025 repo)
    # Disengaged: Miss Rate > 0.35 (matches 2025 repo)
    # Balanced: Otherwise
    conditions = [
        (df['rolling_fa_rate'] > 0.48), # Impulsive
        (df['rolling_miss_rate'] > 0.35) # Disengaged
    ]
    choices = ['impulsive', 'disengaged']
    df['state'] = np.select(conditions, choices, default='balanced')
    
    return df

def compute_psychometric_data(sess):
    """
    Computes Hit Rate per Change Size.
    """
    trials = [t for t in sess.trials if t.outcome in ['Hit', 'Miss']]
    
    if not trials:
        return pd.DataFrame()
        
    data = []
    for t in trials:
        data.append({
            'change_size': t.change_size,
            'is_hit': 1 if t.outcome == 'Hit' else 0
        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby('change_size')['is_hit'].agg(['mean', 'sem', 'count']).reset_index()
    summary = summary.rename(columns={'mean': 'hit_rate', 'sem': 'sem_hit_rate', 'count': 'n_trials'})
    
    return summary

def get_trial_dataframe(sess):
    """
    Returns a DataFrame of all trials for RT analysis.
    """
    data = []
    for i, t in enumerate(sess.trials):
        rt = t.reaction_time if hasattr(t, 'reaction_time') else np.nan
        
        # Calculate response_time relative to trial start for Hits (as requested)
        # Hit RT = Time of Lick (Absolute) - Time of Trial Start (Absolute)
        response_time = rt
        
        if t.outcome == 'Hit':
             if hasattr(t, 'absolute_reaction_time') and hasattr(t, 'absolute_start_time'):
                 if t.absolute_reaction_time is not None and t.absolute_start_time is not None:
                    # Correct to Stimulus Onset (Start + ITI)
                    stimulus_onset = t.absolute_start_time
                    if hasattr(t, 'iti_duration') and t.iti_duration is not None:
                        stimulus_onset += t.iti_duration
                        
                    response_time = t.absolute_reaction_time - stimulus_onset
        
        row = {
            'trial_idx': i,
            'outcome': t.outcome,
            'is_hit': t.outcome == 'Hit',
            'is_miss': t.outcome == 'Miss',
            'is_fa': t.outcome == 'FA',
            'is_abort': t.outcome == 'Abort',
            'change_size': t.change_size if hasattr(t, 'change_size') else 0,
            'rt': rt,
            'response_time': response_time 
        }
        data.append(row)
        
    return pd.DataFrame(data)

def compute_session_performance(sess):
    """Compute aggregate performance metrics for a session (returns Dict)."""
    df = get_trial_dataframe(sess)
    if df.empty:
        return {}

    n_trials = len(df)
    n_hits = df['is_hit'].sum()
    n_miss = df['is_miss'].sum()
    n_fa = df['is_fa'].sum()
    n_abort = df['is_abort'].sum()

    # Hit Rate (Go trials)
    n_go = n_hits + n_miss
    hit_rate = n_hits / n_go if n_go > 0 else 0.0
    miss_rate = n_miss / n_go if n_go > 0 else 0.0

    fa_rate_total = n_fa / n_trials if n_trials > 0 else 0.0
    abort_rate = n_abort / n_trials if n_trials > 0 else 0.0

    fraction_hit = n_hits / n_trials if n_trials > 0 else 0.0
    fraction_miss = n_miss / n_trials if n_trials > 0 else 0.0
    fraction_fa = n_fa / n_trials if n_trials > 0 else 0.0
    fraction_abort = n_abort / n_trials if n_trials > 0 else 0.0

    # FA Split (Early <= 3s, Late > 3s)
    fas = df[df['is_fa']].copy()
    n_fa_early = fas[fas['rt'] <= 3.0].shape[0]
    n_fa_late = fas[fas['rt'] > 3.0].shape[0]

    mean_rt_fa_early = fas[fas['rt'] <= 3.0]['rt'].mean() if not fas.empty else np.nan
    mean_rt_fa_late = fas[fas['rt'] > 3.0]['rt'].mean() if not fas.empty else np.nan

    # Hit RT
    hits = df[df['is_hit']]
    mean_rt_hit = hits['response_time'].mean() if not hits.empty else np.nan
    median_rt_hit = hits['response_time'].median() if not hits.empty else np.nan
    sem_rt_hit = hits['response_time'].sem() if not hits.empty else np.nan

    # Correct SDT d' using change_size to classify go vs catch trials
    outcomes_arr = np.array([t.outcome for t in sess.trials])
    change_sizes_arr = np.array([t.change_size if t.change_size is not None else np.nan
                                  for t in sess.trials])
    sdt = calculate_sdt_metrics(outcomes_arr, change_sizes_arr)
    d_prime = sdt['d_prime']
    criterion_c = sdt['criterion_c']

    return {
        'n_trials': n_trials,
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'fa_rate': fa_rate_total,
        'abort_rate': abort_rate,
        'fraction_hit': fraction_hit,
        'fraction_miss': fraction_miss,
        'fraction_fa': fraction_fa,
        'fraction_abort': fraction_abort,
        'd_prime': d_prime,
        'criterion_c': criterion_c,
        'sdt_hit_rate': sdt['sdt_hit_rate'],
        'sdt_fa_rate': sdt['sdt_fa_rate'],
        'n_hits': n_hits,
        'n_miss': n_miss,
        'n_fa': n_fa,
        'n_fa_early': n_fa_early,
        'n_fa_late': n_fa_late,
        'n_abort': n_abort,
        'mean_rt_hit': mean_rt_hit,
        'median_rt_hit': median_rt_hit,
        'sem_rt_hit': sem_rt_hit,
    }

def despine(ax):
    sns.despine(ax=ax, top=True, right=True)

def set_style(context='talk'):
    sns.set_context(context)
    sns.set_style("ticks")

def plot_session_behavior(sess, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    set_style(context='talk')
    
    # 1. Rolling Performance (Engagement)
    df_rolling = compute_rolling_performance(sess)
    if not df_rolling.empty:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Background Shading for States
        state_colors = {'impulsive': '#ffcccc', 'disengaged': '#e6e6fa', 'balanced': 'white'}
        
        # Identify state changes to group contiguous blocks
        df_rolling['state_group'] = (df_rolling['state'] != df_rolling['state'].shift()).cumsum()
        
        for _, group in df_rolling.groupby('state_group'):
            state = group['state'].iloc[0]
            if state in state_colors and state != 'balanced':
                ax1.axvspan(group['trial_idx'].min(), group['trial_idx'].max(), 
                            color=state_colors[state], alpha=0.3, lw=0)

        # Plot Rolling Rates
        df_rolling['rolling_abort'] = df_rolling['is_abort'].rolling(window=30, min_periods=5).mean()
        
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_hit_rate'], color='green', label='Hit Rate', linewidth=2)
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_miss_rate'], color='purple', label='Miss Rate', linewidth=2)
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_fa_rate'], color='red', label='FA Rate', linewidth=2)
        ax1.plot(df_rolling['trial_idx'], df_rolling['rolling_abort'], color='darkgrey', label='Abort Rate', linewidth=2)
        
        ax1.set_xlabel('Trial Index')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        plt.title(f"Session Dynamics: {sess.session_id}")
        
        despine(ax1)
        plt.tight_layout()
        plt.savefig(out_dir / "session_dynamics.png", dpi=150)
        plt.close()

    # 2. Psychometric Curve
    psy_df = compute_psychometric_data(sess)
    if not psy_df.empty and len(psy_df) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(psy_df))
        
        ax.errorbar(x_pos, psy_df['hit_rate'], yerr=psy_df['sem_hit_rate'], 
                    fmt='o-', color='black', capsize=5, linewidth=2, markersize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(psy_df['change_size'])
        ax.set_xlabel('Change Size (deg)')
        ax.set_ylabel('Performance (Hit Rate)')
        ax.set_ylim(0, 1.05)
        
        plt.title(f"Psychometric Curve: {sess.session_id}")
        despine(ax)
        plt.tight_layout()
        plt.savefig(out_dir / "psychometric_curve.png", dpi=150)
        plt.close()

    # 3. RT Distribution
    df_trials = get_trial_dataframe(sess)
    if not df_trials.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use 'response_time' (calculated from trial start) for hits
        hits = df_trials[df_trials['is_hit']]['response_time'].dropna()
        
        fas = df_trials[df_trials['is_fa']]
        # Use standard RT for FAs (usually already relative to trial start)
        fas_early = fas[fas['rt'] <= 3.0]['rt'].dropna()
        fas_late = fas[fas['rt'] > 3.0]['rt'].dropna()
        
        if len(hits) > 0:
            sns.histplot(hits, color='green', label=f'Hits (n={len(hits)})', kde=False, ax=ax, alpha=0.6, bins=20, element="step")
        if len(fas_early) > 0:
            sns.histplot(fas_early, color='lightcoral', label=f'FA <= 3s (n={len(fas_early)})', kde=False, ax=ax, alpha=0.6, bins=20, element="step")
        if len(fas_late) > 0:
            sns.histplot(fas_late, color='darkred', label=f'FA > 3s (n={len(fas_late)})', kde=False, ax=ax, alpha=0.6, bins=20, element="step")
            
        ax.set_xlabel('Response Time (from Stimulus Onset) (s)')
        ax.set_ylabel('Count')
        ax.set_title(f"Response Time Distribution: {sess.session_id}")
        ax.legend()
        
        despine(ax)
        plt.tight_layout()
        plt.savefig(out_dir / "rt_distribution.png", dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot single session behavior.")
    # Changed arguments to match our workflow: accept directory/files to load session
    parser.add_argument('--session_dir', required=True, help='Path to session directory (containing trials.json etc)')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    try:
        # Load session using our existing infrastructure
        # We need to construct a dict of files for io.load_session_from_files
        # Assuming args.session_dir is the folder containing the json files
        session_dir = Path(args.session_dir)
        
        # Find trials.json
        trials_files = list(session_dir.glob("*_trials.json"))
        if not trials_files:
            raise FileNotFoundError(f"No trials.json found in {session_dir}")
        trials_file = trials_files[0]
        
        # Find settings (optional)
        settings_files = list(session_dir.glob("*_session_settings.json"))
        settings_file = settings_files[0] if settings_files else None
        
        # Create file dict
        session_files = {'trials': trials_file}
        if settings_file:
            session_files['session_settings'] = settings_file
            
        # Load
        sess = session.load_session_from_files(session_files)
        
        plot_session_behavior(sess, args.out)
        print(f"Behavior plots saved to {args.out}")
        
    except Exception as e:
        print(f"Error processing {args.session_dir}: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
