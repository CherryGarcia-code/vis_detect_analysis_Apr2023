"""Generate a staging manifest for all subjects.

Ports the ephys staging pipeline (stage_sessions.py from Sep2025 repo) to
photometry. For each subject, applies QC gates and a chronological
sliding-window algorithm to assign learning stages.

QC Gates (sessions failing any gate are marked 'Excluded'):
  Gate 1 — Minimum trial counts  : n_go >= 20, n_catch >= 10
  Gate 2 — Minimum total trials  : n_go + n_catch >= 100
  Gate 3 — Minimum engagement    : hit_rate >= 0.10 OR sdt_fa_rate >= 0.10
  Gate 4 — Minimum performance   : d' >= threshold (default 0.8, optional)

Stage assignment (one-way, chronological, per subject):
  Naive    → Learning : 3 of last 4 valid sessions have d' > 1.0
  Learning → Expert   : 3 of last 4 valid sessions have d' > 1.5
  Expert sessions with d' < 0.5 are labelled 'Disengaged'.

Usage:
    py scripts/data_management/stage_sessions.py
    py scripts/data_management/stage_sessions.py --subject BG_013
    py scripts/data_management/stage_sessions.py --skip-dprime-gate
    py scripts/data_management/stage_sessions.py --dprime-threshold 1.0
"""
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from visdetect_photom.core import io as io_mod
from visdetect_photom.core.io import find_all_sessions
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import (
    SUBJECT_GENOTYPE, MIN_PHOTOM_CSV_BYTES, CATCH_THRESHOLD, FA_RT_SPLIT,
)
from visdetect_photom.analysis.statistics import calculate_sdt_metrics

# ── Configuration ─────────────────────────────────────────────
DATA_ROOT = "photom_data"
OUTPUT_PATH = "results/staging_manifest.csv"

# QC gate thresholds
MIN_GO = 20
MIN_CATCH = 10
MIN_TOTAL = 100
MIN_LICK_RATE = 0.10

# Staging thresholds (matching ephys repo)
NAIVE_CEILING = 1.0
EXPERT_FLOOR = 1.5
DISENGAGE_CUTOFF = 0.5
WINDOW_SIZE = 4
REQUIRED_ABOVE = 3


# ── Per-session metrics ──────────────────────────────────────

def compute_session_row(session, subject_id):
    """Compute behavioral + SDT metrics for one session, return a dict row."""
    trials = session.trials
    if not trials:
        return None

    outcomes = np.array([t.outcome for t in trials])
    change_sizes = np.array([
        t.change_size if t.change_size is not None else np.nan
        for t in trials
    ])

    # SDT metrics (correct: uses change_size to classify go vs catch)
    sdt = calculate_sdt_metrics(outcomes, change_sizes)

    # Behavioral counts
    n_trials = len(trials)
    n_hits = int((outcomes == 'Hit').sum())
    n_miss = int((outcomes == 'Miss').sum())
    n_fa = int((outcomes == 'FA').sum())
    n_abort = int((outcomes == 'Abort').sum())
    n_cr = int((outcomes == 'CR').sum())

    # FA split
    fa_rts = [t.reaction_time for t in trials
              if t.outcome == 'FA' and t.reaction_time is not None]
    n_fa_early = sum(1 for rt in fa_rts if rt <= FA_RT_SPLIT)
    n_fa_late = sum(1 for rt in fa_rts if rt > FA_RT_SPLIT)

    # Hit RT
    hit_rts = [t.reaction_time for t in trials
               if t.outcome == 'Hit' and t.reaction_time is not None]
    mean_rt_hit = float(np.mean(hit_rts)) if hit_rts else np.nan
    median_rt_hit = float(np.median(hit_rts)) if hit_rts else np.nan

    # Rates
    n_go = sdt['n_go']
    n_catch = sdt['n_catch']
    hit_rate = sdt['sdt_hit_rate']
    sdt_fa_rate = sdt['sdt_fa_rate']
    fa_rate_behavioral = n_fa / n_trials if n_trials > 0 else 0.0

    return {
        'subject_id': subject_id,
        'session_name': session.session_id,
        'date': session.session_date,
        'genotype': SUBJECT_GENOTYPE.get(subject_id, '?'),
        'n_trials': n_trials,
        'n_go': n_go,
        'n_catch': n_catch,
        'n_hits': n_hits,
        'n_miss': n_miss,
        'n_fa': n_fa,
        'n_fa_early': n_fa_early,
        'n_fa_late': n_fa_late,
        'n_abort': n_abort,
        'n_cr': n_cr,
        'hit_rate': hit_rate,
        'sdt_fa_rate': sdt_fa_rate,
        'fa_rate_behavioral': fa_rate_behavioral,
        'd_prime': sdt['d_prime'],
        'criterion_c': sdt['criterion_c'],
        'mean_rt_hit': mean_rt_hit,
        'median_rt_hit': median_rt_hit,
    }


# ── QC gates ─────────────────────────────────────────────────

def apply_qc_gates(row, dprime_threshold=0.8, skip_dprime_gate=False):
    """Apply QC gates to a session row. Returns (qc_fail: bool, reasons: list)."""
    reasons = []

    # Gate 1: Minimum trial counts
    if row['n_go'] < MIN_GO or row['n_catch'] < MIN_CATCH:
        reasons.append(f"low trial count (n_go={row['n_go']}, n_catch={row['n_catch']})")

    # Gate 2: Minimum total trials
    if (row['n_go'] + row['n_catch']) < MIN_TOTAL:
        reasons.append(f"insufficient total (n_go+n_catch={row['n_go']+row['n_catch']})")

    # Gate 3: Behavioural engagement
    hr = row['hit_rate'] if pd.notna(row['hit_rate']) else 0.0
    fr = row['sdt_fa_rate'] if pd.notna(row['sdt_fa_rate']) else 0.0
    if hr < MIN_LICK_RATE and fr < MIN_LICK_RATE:
        reasons.append(f"low engagement (hit={hr:.3f}, fa={fr:.3f})")

    # Gate 4: Minimum d' (optional)
    if not skip_dprime_gate:
        dp = row['d_prime']
        if pd.isna(dp) or dp < dprime_threshold:
            reasons.append(f"d' below threshold ({dp:.3f} < {dprime_threshold})" if pd.notna(dp) else "d' is NaN")

    return len(reasons) > 0, reasons


# ── Staging algorithm ────────────────────────────────────────

def assign_stages(df_subject):
    """Assign learning stages to a single subject's sessions (chronological).

    Modifies df_subject in-place, adding 'stage' column.
    """
    df = df_subject.sort_values('date_parsed').reset_index(drop=True)

    # Mark QC failures as Excluded
    df['stage'] = ''
    df.loc[df['qc_fail'] == True, 'stage'] = 'Excluded'

    # Walk through valid sessions chronologically
    valid_idx = df.index[df['qc_fail'] == False].tolist()
    current_stage = 'Naive'

    for pos, i in enumerate(valid_idx):
        d = df.loc[i, 'd_prime']

        # Naive → Learning
        if current_stage == 'Naive' and pos >= WINDOW_SIZE - 1:
            window_positions = valid_idx[pos - WINDOW_SIZE + 1: pos + 1]
            window_vals = df.loc[window_positions, 'd_prime'].values
            if (window_vals > NAIVE_CEILING).sum() >= REQUIRED_ABOVE:
                current_stage = 'Learning'

        # Learning → Expert
        elif current_stage == 'Learning' and pos >= WINDOW_SIZE - 1:
            window_positions = valid_idx[pos - WINDOW_SIZE + 1: pos + 1]
            window_vals = df.loc[window_positions, 'd_prime'].values
            if (window_vals > EXPERT_FLOOR).sum() >= REQUIRED_ABOVE:
                current_stage = 'Expert'

        # Assign (with Disengaged check in Expert)
        if current_stage == 'Expert' and d < DISENGAGE_CUTOFF:
            df.loc[i, 'stage'] = 'Disengaged'
        else:
            df.loc[i, 'stage'] = current_stage

    return df


# ── Date parsing ─────────────────────────────────────────────

def parse_date(date_str):
    """Parse session date string to datetime for chronological sorting.

    Handles YYYYMMDD (new format) and DDMMYYYY (old format).
    """
    s = str(date_str).replace('-', '')
    if len(s) == 8 and s.isdigit():
        # Try YYYYMMDD first
        try:
            dt = datetime.strptime(s, '%Y%m%d')
            if 2020 <= dt.year <= 2030:
                return dt
        except ValueError:
            pass
        # Try DDMMYYYY
        try:
            dt = datetime.strptime(s, '%d%m%Y')
            if 2020 <= dt.year <= 2030:
                return dt
        except ValueError:
            pass
    return datetime.min


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate staging manifest")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process only one subject (e.g. BG_013)")
    parser.add_argument("--dprime-threshold", type=float, default=0.8,
                        help="d' threshold for Gate 4 (default: 0.8)")
    parser.add_argument("--skip-dprime-gate", action="store_true",
                        help="Skip Gate 4 (d' threshold)")
    parser.add_argument("--out", type=str, default=OUTPUT_PATH,
                        help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Discover all sessions
    print("[INFO] Discovering sessions...")
    all_sessions = find_all_sessions(
        DATA_ROOT, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES,
    )
    print(f"[INFO] Found {len(all_sessions)} sessions")

    # Group by subject
    subject_sessions = defaultdict(list)
    for fp in all_sessions:
        trials_path = fp.get('trials', '')
        sid_num, _ = io_mod.infer_session_keys_from_paths(trials_path)
        if sid_num is None:
            continue
        sid = f"BG_{sid_num.zfill(3)}"
        if sid in SUBJECT_GENOTYPE:
            subject_sessions[sid].append(fp)

    if args.subject:
        if args.subject not in subject_sessions:
            print(f"[ERROR] {args.subject} not found. "
                  f"Available: {sorted(subject_sessions.keys())}")
            return
        subject_sessions = {args.subject: subject_sessions[args.subject]}

    print(f"[INFO] Processing {len(subject_sessions)} subjects")

    # Process all subjects
    all_rows = []
    for sid in sorted(subject_sessions.keys()):
        sessions = subject_sessions[sid]
        print(f"\n  {sid} ({SUBJECT_GENOTYPE.get(sid, '?')}): "
              f"{len(sessions)} sessions")

        subject_rows = []
        for file_paths in tqdm(sessions, desc=f"  {sid}", leave=False):
            try:
                session = load_session_from_files(file_paths)
                row = compute_session_row(session, sid)
                if row is not None:
                    subject_rows.append(row)
            except Exception as e:
                print(f"    [WARN] Failed: {e}")
                continue

        if not subject_rows:
            continue

        df_subj = pd.DataFrame(subject_rows)

        # Apply QC gates
        qc_results = df_subj.apply(
            lambda r: apply_qc_gates(r, args.dprime_threshold, args.skip_dprime_gate),
            axis=1,
        )
        df_subj['qc_fail'] = [r[0] for r in qc_results]
        df_subj['qc_reasons'] = ['; '.join(r[1]) for r in qc_results]

        # Set d' to NaN for excluded sessions (prevents staging on bad data)
        df_subj.loc[df_subj['qc_fail'], 'd_prime'] = np.nan

        # Parse dates and assign stages
        df_subj['date_parsed'] = df_subj['date'].apply(parse_date)
        df_subj = assign_stages(df_subj)
        df_subj = df_subj.drop(columns=['date_parsed'])

        # Print summary
        stage_counts = df_subj['stage'].value_counts()
        print(f"    Stages: {dict(stage_counts)}")

        all_rows.append(df_subj)

    # Combine and save
    if not all_rows:
        print("[ERROR] No sessions processed.")
        return

    manifest = pd.concat(all_rows, ignore_index=True)
    manifest.to_csv(args.out, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("STAGING MANIFEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total sessions: {len(manifest)}")
    print(f"Subjects: {manifest['subject_id'].nunique()}")
    print(f"\nStage counts:")
    for stage in ['Naive', 'Learning', 'Expert', 'Disengaged', 'Excluded']:
        n = (manifest['stage'] == stage).sum()
        if n > 0:
            print(f"  {stage:>12}: {n}")

    print(f"\nPer-subject breakdown:")
    for sid in sorted(manifest['subject_id'].unique()):
        sub = manifest[manifest['subject_id'] == sid]
        n_valid = (sub['qc_fail'] == False).sum()
        stages = sub[sub['qc_fail'] == False]['stage'].value_counts().to_dict()
        dp_vals = sub[sub['qc_fail'] == False]['d_prime'].dropna()
        dp_str = f"d'={dp_vals.mean():.2f}±{dp_vals.std():.2f}" if len(dp_vals) > 0 else "no valid d'"
        print(f"  {sid} ({SUBJECT_GENOTYPE.get(sid, '?')}): "
              f"{len(sub)} total, {n_valid} valid, {dp_str}, stages={stages}")

    print(f"\nManifest saved to {args.out}")

    # Also print sessions suitable for HMM (Learning + Expert)
    hmm_mask = manifest['stage'].isin(['Learning', 'Expert'])
    print(f"\nSessions suitable for HMM (Learning+Expert): {hmm_mask.sum()}")
    for sid in sorted(manifest[hmm_mask]['subject_id'].unique()):
        n = (manifest[hmm_mask]['subject_id'] == sid).sum()
        print(f"  {sid}: {n} sessions")


if __name__ == "__main__":
    main()
