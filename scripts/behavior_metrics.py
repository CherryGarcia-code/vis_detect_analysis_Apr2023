"""
Behavioral Metrics Calculator.

This script calculates behavioral performance metrics (d-prime, hit rates, false alarm rates) 
from raw session data.

Usage:
    python -m scripts.behavior_metrics <mouse_dir> [--out <out_csv>]

Arguments:
    mouse_dir   : Path to the directory containing mouse session data.
    --out       : Path to the output CSV file for behavioral metrics.

Example:
    python -m scripts.behavior_metrics photom_data --out behavior_summary.csv
"""
import os
import argparse
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm
import pandas as pd

# Reuse discovery/utilities from photometry_analysis
from .photometry_analysis import find_all_sessions, infer_session_keys_from_paths

try:
    from .vis_detect_helpers_v9 import load_json_data, process_session_data
except Exception:
    # Fallback: we can still parse trials JSON via pandas
    load_json_data = None  # type: ignore
    process_session_data = None  # type: ignore


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def _get_outcome(trial: Dict) -> Optional[str]:
    for k in ("outcomes", "outcome", "Outcome"):
        v = trial.get(k)
        if isinstance(v, str):
            return v
    return None


def _get_change_category(trial: Dict) -> Optional[str]:
    # Many repos use labels like 'no_change','small','big'
    for k in ("change_category", "changeCategory"):
        v = trial.get(k)
        if isinstance(v, str):
            return v
    # Fallback from boolean or size fields
    if trial.get("change_sizes_TF") is True:
        # Can't tell big/small reliably without magnitude; mark as 'change'
        return "change"
    if trial.get("change_sizes_TF") is False:
        return "no_change"
    return None


def _get_rt_seconds(trial: Dict) -> Optional[float]:
    # Try direct reaction time in seconds
    for k in ("reaction_time_s", "reaction_time_sec", "rt_s", "rt"):
        v = trial.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # As a fallback, compute from timestamps if both are present
    try:
        rt_ts = trial.get("reaction_time_Timestamps")
        ch_ts = trial.get("change_time_Timestamps")
        if rt_ts is not None and ch_ts is not None:
            return float(rt_ts) - float(ch_ts)
    except Exception:
        pass
    return None


def compute_behavior_metrics_from_trials(trials_path: str) -> Optional[Dict[str, object]]:
    # Load trials JSON
    try:
        trials_json = None
        if load_json_data is not None:
            trials_json = load_json_data(trials_path)
        if trials_json is None:
            # Fallback generic loader
            import json
            with open(trials_path, 'r', encoding='utf-8') as f:
                trials_json = json.load(f)
    except Exception:
        return None

    # Trials are typically a list under a key or the root itself
    trials_list = trials_json.get("trials") if isinstance(trials_json, dict) else trials_json
    if not isinstance(trials_list, list):
        return None

    n_total = len(trials_list)

    # Normalize outcome labels (JSON has 'abort', 'Ref')
    _OUTCOME_MAP = {'abort': 'Abort', 'Ref': 'CR', 'ref': 'CR'}
    outcomes = [_OUTCOME_MAP.get(o, o) for o in (_get_outcome(t) for t in trials_list) if o is not None]

    # Counts
    n_hit = sum(1 for o in outcomes if o == 'Hit')
    n_miss = sum(1 for o in outcomes if o == 'Miss')
    n_fa = sum(1 for o in outcomes if o == 'FA')
    n_cr = sum(1 for o in outcomes if o == 'CR')
    n_abort = sum(1 for o in outcomes if o == 'Abort')

    # Reaction time
    rts = [v for v in (_get_rt_seconds(t) for t in trials_list) if v is not None and np.isfinite(v)]
    median_rt = float(np.median(rts)) if rts else None

    # --- Correct SDT d' using change_size (Stim2TF) ---
    # Extract change_size per trial
    change_sizes = []
    for t in trials_list:
        cs = t.get('Stim2TF')
        if cs is None:
            cs = t.get('change_sizes_TF')
        change_sizes.append(float(cs) if cs is not None else np.nan)

    outcomes_arr = np.array(outcomes)
    change_sizes_arr = np.array(change_sizes, dtype=float)

    # Go trials: change_size > 1.01, outcome in [Hit, Miss]
    go_mask = (change_sizes_arr > 1.01) & np.isin(outcomes_arr, ['Hit', 'Miss'])
    n_go = go_mask.sum()
    n_sdt_hit = ((outcomes_arr == 'Hit') & go_mask).sum()

    # Catch trials: change_size <= 1.01, outcome in [Hit, Miss] (or CR)
    catch_mask = (change_sizes_arr <= 1.01) & np.isin(outcomes_arr, ['Hit', 'Miss', 'CR'])
    n_catch = catch_mask.sum()
    n_sdt_fa = ((outcomes_arr == 'Hit') & catch_mask).sum()

    sdt_hit_rate = n_sdt_hit / n_go if n_go > 0 else np.nan
    sdt_fa_rate = n_sdt_fa / n_catch if n_catch > 0 else np.nan

    try:
        hr = np.clip(sdt_hit_rate, 0.01, 0.99) if np.isfinite(sdt_hit_rate) else np.nan
        fr = np.clip(sdt_fa_rate, 0.01, 0.99) if np.isfinite(sdt_fa_rate) else np.nan
        d_prime = float(norm.ppf(hr) - norm.ppf(fr)) if np.isfinite(hr) and np.isfinite(fr) else np.nan
        criterion_c = float(-0.5 * (norm.ppf(hr) + norm.ppf(fr))) if np.isfinite(hr) and np.isfinite(fr) else np.nan
    except Exception:
        d_prime = np.nan
        criterion_c = np.nan

    # Behavioral rates (all trials)
    hit_rate = n_hit / max(1, n_hit + n_miss)
    fa_rate_behavioral = n_fa / max(1, n_total)

    return {
        "n_trials": n_total,
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_FA": n_fa,
        "n_cr": n_cr,
        "n_abort": n_abort,
        "hit_rate": float(hit_rate),
        "fa_rate_behavioral": float(fa_rate_behavioral),
        "sdt_hit_rate": float(sdt_hit_rate) if np.isfinite(sdt_hit_rate) else None,
        "sdt_fa_rate": float(sdt_fa_rate) if np.isfinite(sdt_fa_rate) else None,
        "median_rt_s": median_rt,
        "d_prime": d_prime,
        "criterion_c": criterion_c,
    }


def run_behavior_export(root_dir: str, out_csv: Optional[str] = None, recursive: bool = True) -> str:
    sessions = find_all_sessions(root_dir, recursive=recursive)
    rows: List[Dict[str, object]] = []
    for s in sessions:
        trials_path = s.get("trials")
        if not trials_path:
            continue
        mouse, date = infer_session_keys_from_paths(trials_path)
        m = compute_behavior_metrics_from_trials(trials_path)
        if not m:
            # Best effort fallback via process_session_data to try to recover counts
            try:
                if process_session_data is not None:
                    df = process_session_data([s["photom"]], [s["photom_io"]], [s["session_settings"]], [s["trials"]])
                    if df is not None and not df.empty:
                        # Use outcomes column
                        vc = df["outcomes"].value_counts(dropna=False)
                        n_hit = int(vc.get('Hit', 0))
                        n_miss = int(vc.get('Miss', 0))
                        n_fa = int(vc.get('FA', 0))
                        n_cr = int(vc.get('CR', 0) + vc.get('Ref', 0))
                        n_abort = int(vc.get('Abort', 0) + vc.get('abort', 0))
                        hit_rate = n_hit / max(1, (n_hit + n_miss))
                        fa_rate_behavioral = n_fa / max(1, int(df.shape[0]))
                        m = {
                            "n_trials": int(df.shape[0]),
                            "n_hit": n_hit,
                            "n_miss": n_miss,
                            "n_FA": n_fa,
                            "n_cr": n_cr,
                            "n_abort": n_abort,
                            "hit_rate": float(hit_rate),
                            "fa_rate_behavioral": float(fa_rate_behavioral),
                            "median_rt_s": None,
                            "d_prime": np.nan,
                        }
            except Exception:
                m = None
        if m:
            m.update({"mouse_id": mouse, "session_date": date})
            rows.append(m)

    out_dir = os.path.join(REPO_ROOT, "pdf_output")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = out_csv or os.path.join(out_dir, "behavior_summary_all.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Export per-session behavioral metrics from trials JSON")
    parser.add_argument("root_dir", help="Root directory (e.g., photom_data)")
    parser.add_argument("--out", dest="out_csv", default=None, help="Output CSV path (default: pdf_output/behavior_summary_all.csv)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not search subfolders recursively")
    args = parser.parse_args()

    run_behavior_export(args.root_dir, out_csv=args.out_csv, recursive=args.recursive)


if __name__ == "__main__":
    main()
