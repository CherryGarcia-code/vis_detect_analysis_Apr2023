import pandas as pd
import os

# Define file path
file_path = 'photometry_export_matlab.csv'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit(1)

# Load data
try:
    df = pd.read_csv(file_path)
    print("Successfully loaded data.")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# 1. Identify sessions with inconsistent performance_on_change_size_4
print("\n--- Check 1: Inconsistent performance_on_change_size_4 ---")
inconsistent_sessions = []
grouped = df.groupby(['subject_id', 'session_id'])

for (subject, session), group in grouped:
    unique_perfs = group['performance_on_change_size_4'].unique()
    if len(unique_perfs) > 1:
        inconsistent_sessions.append({
            'subject_id': subject,
            'session_id': session,
            'values': unique_perfs
        })

if inconsistent_sessions:
    print(f"Found {len(inconsistent_sessions)} sessions with inconsistent performance values:")
    for item in inconsistent_sessions:
        print(f"Subject: {item['subject_id']}, Session: {item['session_id']}, Values: {item['values']}")
        
        # Check 3: Analyze inconsistencies pattern
        values = item['values']
        # Filter out NaN if any, though unique() keeps them
        numeric_values = [v for v in values if pd.notna(v)]
        if len(numeric_values) > 1:
            diff = max(numeric_values) - min(numeric_values)
            print(f"  -> Max difference: {diff}")
else:
    print("No sessions with inconsistent performance_on_change_size_4 found.")

# 2. Identify sessions/rows where trials used is 0
print("\n--- Check 2: Zero trials used ---")
zero_trials_df = df[(df['n_hit_trials_used'] == 0) | (df['n_late_fa_trials_used'] == 0)]

if not zero_trials_df.empty:
    print(f"Found {len(zero_trials_df)} rows with 0 trials used.")
    summary = zero_trials_df[['subject_id', 'session_id', 'n_hit_trials_used', 'n_late_fa_trials_used', 'performance_on_change_size_4']]
    print(summary.to_string())
    
    unique_subjects = zero_trials_df['subject_id'].unique()
    print(f"Subjects affected: {unique_subjects}")
else:
    print("No rows with 0 trials used found.")
