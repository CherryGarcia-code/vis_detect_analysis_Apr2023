"""
(Legacy) Photometry Analysis Script.

This script processes photometry sessions to generate per-session summaries of peak dF/F 
responses aligned to behavioral events. It is the predecessor to the `run_session_batch.py` pipeline.

Usage:
    python -m scripts.photometry_analysis <mouse_dir> [--out <out_dir>] [--limit <N>]

Arguments:
    mouse_dir   : Path to the directory containing mouse session data.
    --out       : Output directory for summary CSVs (default: pdf_output).
    --limit     : Limit the number of sessions to process.

Example:
    python -m scripts.photometry_analysis photom_data --out pdf_output
"""
import argparse
import json
import os
import re
import sys
from glob import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure we can import helpers from this repo regardless of how the script is invoked
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Helper imports (existing in this repo)
try:
    from vis_detect_helpers_v9 import (
        load_csv_data,
        load_json_data,
        process_session_data,
        get_signal,
        melt_signals,
        extract_photom_windows_from_session_s,
        extract_signal_window_from_trial_df,
    )
except Exception as e:
    print("Failed to import analysis helpers from scripts/vis_detect_helpers_v9.py:\n ", e)
    print("Make sure you're running this from the repo root or as a module: python -m scripts.photometry_analysis ...")
    raise


PHOTOM_GLOB = "*__photom_*.csv"
PHOTOM_IO_GLOB = "*__photom_IO_*.csv"
TRIALS_GLOB = "*__trials.json"
SESSION_SETTINGS_GLOB = "*__session_settings.json"

DATE_IN_TRIALS_RE = re.compile(r"_(\d{8})_(\d{6})__trials\.json$")
# Generic JSON timestamp for files like *__trials.json and *__session_settings.json
DATE_IN_JSON_RE = re.compile(r"_(\d{8})_(\d{6})__[^/\\]+\.json$")
DATE_IN_PHOTOM_RE = re.compile(r"__(?:photom|photom_IO)_(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})\.csv$")


def parse_trials_timestamp(path: str) -> Optional[datetime]:
    m = DATE_IN_TRIALS_RE.search(os.path.basename(path))
    if not m:
        return None
    date, time = m.groups()  # YYYYMMDD, HHMMSS
    try:
        return datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S")
    except Exception:
        return None


def parse_session_json_timestamp(path: str) -> Optional[datetime]:
    """Parse timestamps from either trials or session_settings JSON names."""
    m = DATE_IN_JSON_RE.search(os.path.basename(path))
    if not m:
        return None
    date, time = m.groups()
    try:
        return datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S")
    except Exception:
        return None


def parse_photom_timestamp(path: str) -> Optional[datetime]:
    # photom CSV uses: YYYY-MM-DDTHH_MM_SS
    m = DATE_IN_PHOTOM_RE.search(os.path.basename(path))
    if not m:
        return None
    yyyy_mm_dd, hh, mm, ss = m.groups()
    try:
        return datetime.strptime(f"{yyyy_mm_dd} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def pair_session_files(mouse_dir: str) -> List[Dict[str, str]]:
    """
    Build best-effort pairs of photom, photom_io, trials, session_settings per session.
    The photometry CSV timestamps rarely match trials to-the-second, so we match by date
    and choose the closest photom timestamps on the same day.
    """
    trials = sorted(glob(os.path.join(mouse_dir, TRIALS_GLOB)))
    sess = sorted(glob(os.path.join(mouse_dir, SESSION_SETTINGS_GLOB)))
    phot = sorted([p for p in glob(os.path.join(mouse_dir, PHOTOM_GLOB)) if "__photom_IO_" not in p])
    phot_io = sorted(glob(os.path.join(mouse_dir, PHOTOM_IO_GLOB)))

    # Index trials and session settings by their exact timestamp
    trial_ts = {t: parse_trials_timestamp(t) for t in trials}
    sess_by_date: Dict[str, List[str]] = {}
    for s in sess:
        ts = parse_session_json_timestamp(s)
        if not ts:
            continue
        sess_by_date.setdefault(ts.strftime("%Y-%m-%d"), []).append(s)

    phot_by_date: Dict[str, List[str]] = {}
    for p in phot:
        ts = parse_photom_timestamp(p)
        if not ts:
            continue
        phot_by_date.setdefault(ts.strftime("%Y-%m-%d"), []).append(p)

    photio_by_date: Dict[str, List[str]] = {}
    for p in phot_io:
        ts = parse_photom_timestamp(p)
        if not ts:
            continue
        photio_by_date.setdefault(ts.strftime("%Y-%m-%d"), []).append(p)

    sessions: List[Dict[str, str]] = []
    for t_path, t_ts in trial_ts.items():
        if not t_ts:
            continue
        session_date = t_ts.strftime("%Y-%m-%d")
        # pick session_settings with same date and closest time
        candidate_sess = sess_by_date.get(session_date, [])
        s_best = None
        if candidate_sess:
            s_best = min(candidate_sess, key=lambda s: abs(parse_session_json_timestamp(s) - t_ts))

        # pick phot and phot_io with same date and closest time
        candidate_phot = phot_by_date.get(session_date, [])
        candidate_photio = photio_by_date.get(session_date, [])
        def closest(lst: List[str]) -> Optional[str]:
            return min(lst, key=lambda p: abs(parse_photom_timestamp(p) - t_ts)) if lst else None
        p_best = closest(candidate_phot)
        pio_best = closest(candidate_photio)

        if p_best and pio_best and s_best:
            sessions.append({
                "trials": t_path,
                "session_settings": s_best,
                "photom": p_best,
                "photom_io": pio_best,
            })

    return sessions


def find_all_sessions(root_dir: str, recursive: bool = False) -> List[Dict[str, str]]:
    """Discover sessions under root_dir. If recursive, search subfolders per mouse."""
    if not recursive:
        return pair_session_files(root_dir)
    all_sessions: List[Dict[str, str]] = []
    # Include root itself
    all_sessions.extend(pair_session_files(root_dir))
    # Walk subdirectories one level deep (mouse folders), then deeper if needed
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root in first iteration since already handled
        if os.path.abspath(dirpath) == os.path.abspath(root_dir):
            continue
        # Only consider directories that contain at least one matching file to keep it fast
        has_any = (
            glob(os.path.join(dirpath, TRIALS_GLOB)) or
            glob(os.path.join(dirpath, SESSION_SETTINGS_GLOB)) or
            glob(os.path.join(dirpath, PHOTOM_GLOB)) or
            glob(os.path.join(dirpath, PHOTOM_IO_GLOB))
        )
        if has_any:
            try:
                all_sessions.extend(pair_session_files(dirpath))
            except Exception:
                continue
    return all_sessions


MOUSE_ID_FROM_BASENAME_RE = re.compile(r"^BG_(\d+)")

def infer_session_keys_from_paths(trials_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer (mouse_id, session_date yyyymmdd) from a trials filename.
    Returns (None, None) if not parseable."""
    base = os.path.basename(trials_path)
    # Expect BG_XXX_YYYYMMDD_HHMMSS__trials.json
    parts = base.split('_')
    if len(parts) >= 3 and parts[0].startswith('BG_'):
        mouse = parts[0].replace('BG_', '')
        date = parts[1]
        if mouse.isdigit() and len(date) == 8 and date.isdigit():
            return mouse, date
    # Fallback regex
    m = MOUSE_ID_FROM_BASENAME_RE.match(base)
    if m:
        mouse = m.group(1)
    else:
        mouse = None
    ts = parse_trials_timestamp(trials_path)
    date = ts.strftime('%Y%m%d') if ts else None
    return mouse, date


def compute_peak_zdf_over_window(df: pd.DataFrame, roi_cols: List[str], start_s: float, end_s: float) -> Dict[str, float]:
    """
    Given a trial-aligned window dataframe (index is seconds relative to event),
    compute the max z-scored dF/F between [start_s, end_s] for each ROI column.
    """
    out: Dict[str, float] = {}
    # Ensure numeric index
    idx = pd.to_numeric(df.index, errors="coerce")
    mask = (idx >= start_s) & (idx <= end_s)
    for c in roi_cols:
        if c in df.columns:
            out[c] = float(pd.to_numeric(df.loc[mask, c], errors="coerce").max(skipna=True))
    return out


def summarize_session(session_df: pd.DataFrame) -> Dict[str, object]:
    """
    Produce a small set of "interesting messages":
    - hit vs miss peak response after reaction/change
    - big vs small vs no_change peak after change
    """
    # Windows in seconds relative to event
    peak_window = (0.0, 1.0)

    def safe_extract(event: str):
        """Robustly return a list of trial windows for an event using helper or manual fallback."""
        try:
            if event == "hit":
                hit_windows, change_windows, base_windows = extract_photom_windows_from_session_s(session_df, "hit")
                return hit_windows
            elif event == "miss":
                miss_windows, _, _ = extract_photom_windows_from_session_s(session_df, "miss")
                return miss_windows
            elif event == "change":
                change_only_windows, _ = extract_photom_windows_from_session_s(session_df, "change")
                return change_only_windows
        except Exception:
            pass
        # Fallback manual extraction
        dfs = []
        try:
            if event == "hit":
                sel = session_df[session_df['outcomes'] == 'Hit']
                for _, row in sel.iterrows():
                    dfs.append(extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']))
            elif event == "miss":
                sel = session_df[session_df['outcomes'] == 'Miss']
                for _, row in sel.iterrows():
                    dfs.append(extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']))
            elif event == "change":
                sel = session_df[~session_df['outcomes'].isin(['FA','abort'])]
                for _, row in sel.iterrows():
                    dfs.append(extract_signal_window_from_trial_df(row, row['change_time_Timestamps']))
        except Exception:
            return []
        return dfs

    # Extract windows
    hit_windows = safe_extract("hit")
    miss_windows = safe_extract("miss")
    change_only_windows = safe_extract("change")

    # Collect per-trial peaks
    def roi_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("zscored_") and c.endswith("_clean_signal_dff")]

    hit_peaks = [compute_peak_zdf_over_window(w, roi_cols(w), *peak_window) for w in hit_windows]
    miss_peaks = [compute_peak_zdf_over_window(w, roi_cols(w), *peak_window) for w in miss_windows]
    change_peaks = [compute_peak_zdf_over_window(w, roi_cols(w), *peak_window) for w in change_only_windows]

    # Aggregate
    def agg(peaks: List[Dict[str, float]]) -> Dict[str, float]:
        keys = set().union(*[p.keys() for p in peaks]) if peaks else set()
        return {k: float(np.nanmean([p.get(k, np.nan) for p in peaks])) for k in keys}

    hit_mean = agg(hit_peaks)
    miss_mean = agg(miss_peaks)
    change_mean = agg(change_peaks)

    return {
        "n_trials": len(session_df),
        "hit_peak_0to1s": hit_mean,
        "miss_peak_0to1s": miss_mean,
        "change_peak_0to1s": change_mean,
    }


def run(mouse_dir: str, out_dir: Optional[str] = None, limit: Optional[int] = None) -> None:
    return run_with_options(mouse_dir, out_dir=out_dir, limit=limit, save_plots=False)


def concat_windows(windows: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    windows = [w for w in windows if w is not None and not w.empty]
    if not windows:
        return None
    try:
        return pd.concat(windows, axis=0)
    except Exception:
        return None


def plot_melted_and_save(melted: pd.DataFrame, behave_event: str, out_png: str, title: Optional[str] = None) -> None:
    sns.set_context('talk')
    colors_categories = ["#808080", "#ffa500", "#e60000"]
    # Use relplot similar to helper but save to file
    if behave_event in ("hit", "miss", "change"):
        g = sns.relplot(
            x='seconds from event', y='zscored signal', data=melted,
            row='hemisphere', kind='line', hue='change category',
            palette=colors_categories, hue_order=['no_change', 'small', 'big']
        )
    else:
        g = sns.relplot(
            x='seconds from event', y='zscored signal', data=melted,
            row='hemisphere', kind='line'
        )
    for ax in g.axes.flatten():
        ax.axvline(x=0.0, color='black', linestyle='--')
        sns.despine(ax=ax)
        if title:
            ax.set_title(ax.get_title(), pad=14)
    if title:
        plt.suptitle(title)
    plt.tight_layout(pad=0.5)
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)


def generate_session_plots(session_df: pd.DataFrame, base_out_dir: str) -> List[str]:
    """Generate and save plots for Hit, Miss, FA, abort, Ref, Change, Baseline."""
    saved: List[str] = []
    os.makedirs(base_out_dir, exist_ok=True)

    # Helper to process a windows list and save
    def _save_from_windows(windows: List[pd.DataFrame], behave_event: str):
        df = concat_windows(windows)
        if df is None or df.empty:
            return
        melted = melt_signals(df, behave_event)
        mouse = session_df.get('mouse_id', pd.Series(["unknown"]))[:1].iloc[0]
        sdate = session_df.get('session_date', pd.Series(["unknown"]))[:1].iloc[0]
        out_png = os.path.join(base_out_dir, f"{mouse}_{sdate}_{behave_event}.png")
        title = f"{mouse} {sdate} - {behave_event}"
        plot_melted_and_save(melted, behave_event, out_png, title=title)
        saved.append(out_png)

    # Hit
    try:
        hit_windows, change_windows_for_hits, base_for_hits = extract_photom_windows_from_session_s(session_df, "hit")
        _save_from_windows(hit_windows, "hit")
        # Change for relevant trials (hits/misses handled below separately too)
        _save_from_windows(change_windows_for_hits, "change")
    except Exception:
        pass

    # Miss
    try:
        miss_windows, _, base_for_miss = extract_photom_windows_from_session_s(session_df, "miss")
        _save_from_windows(miss_windows, "miss")
    except Exception:
        pass

    # FA
    try:
        early_FA_signals, late_FA_signals, early_FA_baseline_signals, late_FA_baseline_signals = extract_photom_windows_from_session_s(session_df, "FA")
        _save_from_windows(early_FA_signals + late_FA_signals, "FA")
    except Exception:
        pass

    # abort
    try:
        early_aborts_signals, late_aborts_signals, early_aborts_baseline_signals, late_aborts_baseline_signals = extract_photom_windows_from_session_s(session_df, "abort")
        _save_from_windows(early_aborts_signals + late_aborts_signals, "abort")
    except Exception:
        pass

    # change-only (excluding FA/abort where change not encountered), from helper
    try:
        change_only_windows, _ = extract_photom_windows_from_session_s(session_df, "change")
        _save_from_windows(change_only_windows, "change_all")
    except Exception:
        pass

    # baseline (all outcomes, excluding early FA/abort <2s per helper)
    try:
        baseline_windows, _ = extract_photom_windows_from_session_s(session_df, "baseline")
        _save_from_windows(baseline_windows, "baseline")
    except Exception:
        pass

    # Ref outcome (not handled in helper) — align to reaction and baseline
    try:
        ref_df = session_df[session_df['outcomes'] == 'Ref']
        if not ref_df.empty:
            ref_react = ref_df.apply(lambda row: extract_signal_window_from_trial_df(row, row['reaction_time_Timestamps']), axis=1).tolist()
            _save_from_windows(ref_react, "ref")
    except Exception:
        pass

    return saved


def export_per_trial_metrics(session_df: pd.DataFrame, out_dir: str, mouse_id: str, session_date: str) -> str:
    """
    Export per-trial peak z-dF/F metrics for key events (hit, miss, change) for this session.
    Uses manual extraction to retain trial indices from session_df and aligns to reaction/change.
    Output CSV path is returned.
    """
    peak_window = (0.0, 1.0)

    def roi_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("zscored_") and c.endswith("_clean_signal_dff")]

    rows: List[Dict[str, object]] = []

    # Helper to process a selection of trials into per-trial peak rows
    def process_sel(sel: pd.DataFrame, event: str, align_col: str):
        for trial_idx, row in sel.iterrows():
            try:
                w = extract_signal_window_from_trial_df(row, row[align_col])
                peaks = compute_peak_zdf_over_window(w, roi_cols(w), *peak_window)
            except Exception:
                peaks = {}
            rec: Dict[str, object] = {
                "mouse_id": mouse_id,
                "session_date": session_date,
                "trial_index": trial_idx,
                "event": event,
                "outcome": row.get("outcomes"),
            }
            # Add ROI peaks and region means
            for k, v in peaks.items():
                rec[k] = v
            # Region averages (DMS from G0/G2, VLS from G4/G5 when present)
            dms_vals = [peaks.get(k) for k in peaks.keys() if _roi_to_region(k) == "DMS"]
            vls_vals = [peaks.get(k) for k in peaks.keys() if _roi_to_region(k) == "VLS"]
            if any(pd.notna(dms_vals)):
                rec["DMS_mean"] = float(np.nanmean(dms_vals))
            if any(pd.notna(vls_vals)):
                rec["VLS_mean"] = float(np.nanmean(vls_vals))
            rows.append(rec)

    # Build selections and process
    try:
        sel_hit = session_df[session_df["outcomes"] == "Hit"]
        process_sel(sel_hit, "hit", "reaction_time_Timestamps")
    except Exception:
        pass
    try:
        sel_miss = session_df[session_df["outcomes"] == "Miss"]
        process_sel(sel_miss, "miss", "reaction_time_Timestamps")
    except Exception:
        pass
    try:
        # change-only trials: exclude FA/abort where change wasn't encountered
        sel_change = session_df[~session_df["outcomes"].isin(["FA", "abort"])].copy()
        process_sel(sel_change, "change", "change_time_Timestamps")
    except Exception:
        pass

    df_out = pd.DataFrame(rows)
    per_trial_dir = os.path.join(out_dir, "per_trial", f"BG_{mouse_id}", str(session_date))
    os.makedirs(per_trial_dir, exist_ok=True)
    out_csv = os.path.join(per_trial_dir, f"per_trial_metrics_{mouse_id}_{session_date}.csv")
    df_out.to_csv(out_csv, index=False)
    print("Saved per-trial metrics:", out_csv)
    return out_csv


def parse_genotype_file(path: str) -> Dict[str, Dict[str, object]]:
    """Parse mouse_genotypes_and_procedeures.txt into a mapping.
    Returns mapping from 'BG_016' -> {'genotype': 'A2a'|'Drd1', 'regions': ['DMS', 'VLS', ...]}
    """
    mapping: Dict[str, Dict[str, object]] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('BG_'):
                continue
            # Example: BG_016 A2a male  'LL '  , DMS;
            parts = line.replace(';', '').split(',')
            left = parts[0]
            regions_part = (parts[1] if len(parts) > 1 else '').strip()
            tokens = left.split()
            if len(tokens) >= 2:
                mouse = tokens[0]
                genotype = tokens[1]
            else:
                continue
            regions = [r.strip() for r in regions_part.split('&')] if regions_part else []
            mapping[mouse] = {"genotype": genotype, "regions": regions}
    return mapping


def run_with_options(mouse_dir: str, out_dir: Optional[str] = None, limit: Optional[int] = None, save_plots: bool = False, recursive: bool = False, learning_plots: bool = False, force: bool = False, per_trial: bool = False) -> None:
    sessions = find_all_sessions(mouse_dir, recursive=recursive)
    if not sessions:
        print(f"No complete sessions found under: {mouse_dir}")
        return

    if out_dir is None:
        out_dir = os.path.join(REPO_ROOT, "pdf_output")
    os.makedirs(out_dir, exist_ok=True)
    plots_root = os.path.join(out_dir, "photom_plots")
    if save_plots:
        os.makedirs(plots_root, exist_ok=True)

    # Genotype mapping
    geno_map = parse_genotype_file(os.path.join(REPO_ROOT, 'photom_data', 'mouse_genotypes_and_procedeures.txt'))

    summaries: List[Dict[str, object]] = []

    processed = 0
    skipped = 0
    failures = 0
    failure_records: List[Dict[str, object]] = []

    for i, s in enumerate(sessions):
        if limit is not None and i >= limit:
            break
        print("\n================ SESSION", i + 1, "of", len(sessions), "================")
        print("trials:          ", os.path.basename(s["trials"]))
        print("session_settings:", os.path.basename(s["session_settings"]))
        print("photom:          ", os.path.basename(s["photom"]))
        print("photom_io:       ", os.path.basename(s["photom_io"]))

        # Resume/skip logic based on expected output CSV existing
        mouse_guess, date_guess = infer_session_keys_from_paths(s["trials"]) 
        expected_csv = None
        if mouse_guess and date_guess:
            expected_csv = os.path.join(out_dir, f"photom_summary_{mouse_guess}_{date_guess}.csv")
            if os.path.exists(expected_csv) and not force:
                print("Skipping session (already processed):", os.path.basename(expected_csv))
                skipped += 1
                continue

        try:
            session_df = process_session_data(
                [s["photom"]], [s["photom_io"]], [s["session_settings"]], [s["trials"]]
            )
        except Exception as e:
            print("Failed to process session:", e)
            failures += 1
            failure_records.append({
                "stage": "process",
                "session_index": i + 1,
                "trials": os.path.basename(s.get("trials", "")),
                "mouse_id": mouse_guess,
                "session_date": date_guess,
                "error": str(e),
            })
            continue

        if session_df is None or session_df.empty:
            print("Session returned no data. Skipping.")
            failures += 1
            continue

        # Summarize (robust to failures)
        try:
            summary = summarize_session(session_df)
        except Exception as e:
            print("Failed to summarize session:", e)
            summary = {"n_trials": len(session_df)}
            failure_records.append({
                "stage": "summarize",
                "session_index": i + 1,
                "trials": os.path.basename(s.get("trials", "")),
                "mouse_id": mouse_guess,
                "session_date": date_guess,
                "error": str(e),
            })
        # Add keys from df
        mouse_id = session_df["mouse_id"].iloc[0] if "mouse_id" in session_df.columns else None
        session_date = session_df["session_date"].iloc[0] if "session_date" in session_df.columns else None
        summary.update({
            "mouse_id": mouse_id,
            "session_date": session_date,
        })

        # Attach genotype/region if available
        mouse_key = f"BG_{mouse_id}" if mouse_id else None
        if mouse_key and mouse_key in geno_map:
            summary["genotype"] = geno_map[mouse_key].get("genotype")
            summary["regions"] = "&".join(geno_map[mouse_key].get("regions", []))
        summaries.append(summary)

        # Optionally save a thin CSV of trial-level metrics
        out_csv = os.path.join(out_dir, f"photom_summary_{summary.get('mouse_id','unknown')}_{summary.get('session_date','unknown')}.csv")
        # Flatten dicts for CSV
        rows = []
        def flat(prefix: str, d: Dict[str, float]):
            return {f"{prefix}_{k}": v for k, v in (d or {}).items()}
        rows.append({
            "mouse_id": summary.get("mouse_id"),
            "session_date": summary.get("session_date"),
            "n_trials": summary.get("n_trials"),
            **flat("hit", summary.get("hit_peak_0to1s")),
            **flat("miss", summary.get("miss_peak_0to1s")),
            **flat("change", summary.get("change_peak_0to1s")),
        })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("Saved:", out_csv)
        processed += 1

        # Plots per session
        if save_plots:
            sess_plot_dir = os.path.join(plots_root, f"BG_{mouse_id}", str(session_date))
            try:
                saved = generate_session_plots(session_df, sess_plot_dir)
            except Exception as e:
                print("Plotting failed:", e)
                saved = []
                failure_records.append({
                    "stage": "plot",
                    "session_index": i + 1,
                    "trials": os.path.basename(s.get("trials", "")),
                    "mouse_id": mouse_id,
                    "session_date": session_date,
                    "error": str(e),
                })
            for p in saved:
                print("Saved plot:", p)

        # Optional per-trial metrics export (hit/miss/change aligned windows)
        if per_trial and mouse_id is not None and session_date is not None:
            try:
                export_per_trial_metrics(session_df, out_dir, str(mouse_id), str(session_date))
            except Exception as e:
                print("Per-trial export failed:", e)

    if summaries:
        # Print a compact overall summary message
        print("\n=== Overall photometry summary ===")
        print(f"Processed {len(summaries)} sessions under {mouse_dir}.")
        print(f"Counts — processed: {processed}, skipped: {skipped}, failures: {failures}")
        # Example: show average hit peak (DMS left/right) across sessions if present
        keys = set()
        for s in summaries:
            for cat in ["hit_peak_0to1s", "miss_peak_0to1s", "change_peak_0to1s"]:
                keys.update((s.get(cat) or {}).keys())
        for cat in ["hit_peak_0to1s", "miss_peak_0to1s", "change_peak_0to1s"]:
            vals_by_key = {k: [] for k in keys}
            for s in summaries:
                for k in keys:
                    v = (s.get(cat) or {}).get(k)
                    if v is not None and not np.isnan(v):
                        vals_by_key[k].append(v)
            means = {k: (float(np.mean(v)) if v else np.nan) for k, v in vals_by_key.items()}
            print(f"{cat}:", {k: round(v, 3) if v == v else None for k, v in means.items()})

        # Save aggregated CSV
        def _flat(prefix: str, d: Dict[str, float]):
            return {f"{prefix}_{k}": v for k, v in (d or {}).items()}

        def _rebuild_aggregate_from_per_session(out_dir: str, geno_map_local: Dict[str, Dict[str, object]]) -> pd.DataFrame:
            """Rebuild full aggregate by concatenating all per-session CSVs in out_dir.
            Ensures we don't lose previous sessions when doing incremental runs.
            """
            # Collect all per-session files: photom_summary_XXX_YYYYMMDD.csv
            pattern = os.path.join(out_dir, "photom_summary_*_*.csv")
            files = [p for p in glob(pattern) if not os.path.basename(p).startswith("photom_summary_all")]  # exclude aggregates
            rows: List[pd.DataFrame] = []
            for p in sorted(files):
                try:
                    dfp = pd.read_csv(p)
                    # Ensure mouse_id and session_date exist; best-effort
                    if "mouse_id" not in dfp.columns or "session_date" not in dfp.columns:
                        continue
                    # Attach genotype/regions via mapping
                    def _apply_map_mouse(mouse_val):
                        if pd.isna(mouse_val):
                            return None
                        try:
                            # Cast to int then zero-pad to 3 digits to match keys like BG_016
                            return f"BG_{int(mouse_val):03d}"
                        except Exception:
                            # Fall back to string handling
                            s = str(mouse_val)
                            s_digits = ''.join(ch for ch in s if ch.isdigit())
                            if s_digits:
                                return f"BG_{int(s_digits):03d}"
                            return None
                    dfp["_mouse_key"] = dfp["mouse_id"].apply(_apply_map_mouse)
                    dfp["genotype"] = dfp["_mouse_key"].apply(lambda k: geno_map_local.get(k, {}).get("genotype") if k else None)
                    dfp["regions"] = dfp["_mouse_key"].apply(lambda k: "&".join(geno_map_local.get(k, {}).get("regions", [])) if k else None)
                    dfp = dfp.drop(columns=["_mouse_key"], errors="ignore")
                    rows.append(dfp)
                except Exception:
                    continue
            if not rows:
                return pd.DataFrame()
            full = pd.concat(rows, ignore_index=True, sort=False)
            # Drop obvious duplicates (same mouse_id + session_date)
            if {"mouse_id", "session_date"}.issubset(full.columns):
                full = full.drop_duplicates(subset=["mouse_id", "session_date"], keep="last").reset_index(drop=True)
            return full

        if limit is not None:
            # Limited run: write a limited aggregate from just this run's summaries, do NOT clobber the full aggregate
            limited_rows = []
            for s in summaries:
                limited_rows.append({
                    "mouse_id": s.get("mouse_id"),
                    "session_date": s.get("session_date"),
                    "genotype": s.get("genotype"),
                    "regions": s.get("regions"),
                    **_flat("hit", s.get("hit_peak_0to1s")),
                    **_flat("miss", s.get("miss_peak_0to1s")),
                    **_flat("change", s.get("change_peak_0to1s")),
                })
            agg_df = pd.DataFrame(limited_rows)
            agg_csv = os.path.join(out_dir, f"photom_summary_all.limit_{limit}.csv")
            print(f"[warn] Writing limited aggregate due to --limit={limit}:", agg_csv)
            agg_df.to_csv(agg_csv, index=False)
            print("Saved:", agg_csv)
        else:
            # Full run or resume: rebuild the full aggregate from all per-session CSVs
            agg_df = _rebuild_aggregate_from_per_session(out_dir, geno_map)
            agg_csv = os.path.join(out_dir, "photom_summary_all.csv")
            agg_df.to_csv(agg_csv, index=False)
            print("Saved:", agg_csv)

        if learning_plots:
            learning_dir = os.path.join(out_dir, "learning_plots")
            os.makedirs(learning_dir, exist_ok=True)
            generate_learning_plots(agg_df, learning_dir)

    else:
        # No sessions processed in this run. If not a limited run, still rebuild the full aggregate
        if limit is None:
            def _rebuild_aggregate_from_per_session(out_dir: str, geno_map_local: Dict[str, Dict[str, object]]) -> pd.DataFrame:
                pattern = os.path.join(out_dir, "photom_summary_*_*.csv")
                files = [p for p in glob(pattern) if not os.path.basename(p).startswith("photom_summary_all")]  # exclude aggregates
                rows: List[pd.DataFrame] = []
                for p in sorted(files):
                    try:
                        dfp = pd.read_csv(p)
                        if "mouse_id" not in dfp.columns or "session_date" not in dfp.columns:
                            continue
                        def _apply_map_mouse(mouse_val):
                            if pd.isna(mouse_val):
                                return None
                            try:
                                return f"BG_{int(mouse_val):03d}"
                            except Exception:
                                s = str(mouse_val)
                                s_digits = ''.join(ch for ch in s if ch.isdigit())
                                if s_digits:
                                    return f"BG_{int(s_digits):03d}"
                                return None
                        dfp["_mouse_key"] = dfp["mouse_id"].apply(_apply_map_mouse)
                        dfp["genotype"] = dfp["_mouse_key"].apply(lambda k: geno_map_local.get(k, {}).get("genotype") if k else None)
                        dfp["regions"] = dfp["_mouse_key"].apply(lambda k: "&".join(geno_map_local.get(k, {}).get("regions", [])) if k else None)
                        dfp = dfp.drop(columns=["_mouse_key"], errors="ignore")
                        rows.append(dfp)
                    except Exception:
                        continue
                if not rows:
                    return pd.DataFrame()
                full = pd.concat(rows, ignore_index=True, sort=False)
                if {"mouse_id", "session_date"}.issubset(full.columns):
                    full = full.drop_duplicates(subset=["mouse_id", "session_date"], keep="last").reset_index(drop=True)
                return full

            agg_df = _rebuild_aggregate_from_per_session(out_dir, geno_map)
            agg_csv = os.path.join(out_dir, "photom_summary_all.csv")
            agg_df.to_csv(agg_csv, index=False)
            print("Saved:", agg_csv)
            if learning_plots:
                learning_dir = os.path.join(out_dir, "learning_plots")
                os.makedirs(learning_dir, exist_ok=True)
                generate_learning_plots(agg_df, learning_dir)

        # Write failures.json for diagnostics
        if failure_records:
            failures_path = os.path.join(out_dir, "failures.json")
            try:
                with open(failures_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "root": mouse_dir,
                        "counts": {"processed": processed, "skipped": skipped, "failures": failures},
                        "records": failure_records,
                    }, f, indent=2)
                print("Saved failure report:", failures_path)
            except Exception as e:
                print("Failed to write failure report:", e)


def _roi_to_region(roi_name: str) -> Optional[str]:
    """Map ROI key to region string."""
    if any(token in roi_name for token in ["G0", "G2"]):
        return "DMS"
    if any(token in roi_name for token in ["G4", "G5"]):
        return "VLS"
    return None


def _melt_agg_for_learning(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Convert aggregated wide table to long form with columns:
    mouse_id, session_date, genotype, session_index, event, region, value
    """
    df = agg_df.copy()
    # Parse session_date to sortable date and make per-mouse session index
    def _parse_date(s):
        try:
            return datetime.strptime(str(s), "%Y%m%d").date()
        except Exception:
            return None
    df["session_date_parsed"] = df["session_date"].apply(_parse_date)
    df = df.dropna(subset=["session_date_parsed"])  # keep valid
    df = df.sort_values(["mouse_id", "session_date_parsed"]).reset_index(drop=True)
    # Assign session index within mouse (1..N)
    df["session_index"] = df.groupby("mouse_id").cumcount() + 1

    rows = []
    event_prefixes = ["hit_", "miss_", "change_"]
    for _, row in df.iterrows():
        for ev in event_prefixes:
            ev_name = ev[:-1]
            # Collect ROI columns for this event
            roi_vals: Dict[str, float] = {k: row[k] for k in row.index if isinstance(k, str) and k.startswith(ev)}
            # Group by region
            region_to_vals: Dict[str, List[float]] = {"DMS": [], "VLS": []}
            for k, v in roi_vals.items():
                region = _roi_to_region(k)
                if region in region_to_vals and pd.notna(v):
                    region_to_vals[region].append(float(v))
            for region, vals in region_to_vals.items():
                if not vals:
                    continue
                value = float(np.nanmean(vals))
                rows.append({
                    "mouse_id": row.get("mouse_id"),
                    "genotype": row.get("genotype"),
                    "session_date": row.get("session_date"),
                    "session_index": int(row.get("session_index")),
                    "event": ev_name,
                    "region": region,
                    "value": value,
                })
    return pd.DataFrame(rows)


def _lineplot(df: pd.DataFrame, x: str, y: str, hue: str, col: Optional[str], title: str, out_png: str):
    sns.set_context('talk')
    g = sns.relplot(data=df, x=x, y=y, hue=hue, kind='line', col=col, facet_kws={'sharey': False}) if col else sns.relplot(data=df, x=x, y=y, hue=hue, kind='line')
    if hasattr(g, 'axes'):
        axes = g.axes.flatten() if isinstance(g.axes, np.ndarray) else [g.ax]
    else:
        axes = [g.ax]
    for ax in axes:
        ax.set_title(ax.get_title(), pad=14)
        sns.despine(ax=ax)
        ax.set_xlabel('session index')
        ax.set_ylabel('peak z-dF/F (0-1s)')
    plt.suptitle(title)
    plt.tight_layout(pad=0.8)
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)


def generate_learning_plots(agg_df: pd.DataFrame, out_dir: str) -> None:
    long_df = _melt_agg_for_learning(agg_df)
    if long_df.empty:
        print("No data for learning plots.")
        return
    # Per-mouse plots: columns by event, lines by region
    for mouse_id, df_m in long_df.groupby('mouse_id'):
        out_png = os.path.join(out_dir, f"mouse_{mouse_id}_learning.png")
        _lineplot(df_m, x='session_index', y='value', hue='region', col='event', title=f"Mouse {mouse_id} learning", out_png=out_png)
        print("Saved learning plot:", out_png)

    # Genotype-level aggregation: mean across mice at each session_index
    # To align learning, keep session_index ordinal per mouse
    agg = long_df.groupby(['genotype', 'region', 'event', 'session_index']).agg(value_mean=('value','mean'), value_sem=('value','sem'), n=('value','count')).reset_index()
    # Plot per genotype
    for genotype, df_g in agg.groupby('genotype'):
        # Expand back to per-row for shading if needed
        # Simpler: plot mean lines with hue=region and facet by event
        df_plot = df_g.rename(columns={'value_mean':'value'})
        out_png = os.path.join(out_dir, f"genotype_{genotype}_learning.png")
        # Use relplot over the mean values
        sns.set_context('talk')
        g = sns.relplot(data=df_plot, x='session_index', y='value', hue='region', kind='line', col='event', facet_kws={'sharey': False})
        for ax in g.axes.flatten():
            ax.set_title(ax.get_title(), pad=14)
            sns.despine(ax=ax)
            ax.set_xlabel('session index')
            ax.set_ylabel('mean peak z-dF/F (0-1s)')
        plt.suptitle(f"Genotype {genotype} learning")
        plt.tight_layout(pad=0.8)
        g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(g.figure)
        print("Saved learning plot:", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run photometry analysis and print/save summary messages.")
    parser.add_argument("mouse_dir", help="Path to the mouse directory under photom_data (e.g., photom_data/BG_021)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory for summary CSVs (default: pdf_output)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions processed (debug)")
    parser.add_argument("--plots", action="store_true", help="Save per-session plots for outcomes and events")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for sessions under the mouse_dir root")
    parser.add_argument("--learning-plots", action="store_true", help="Generate cross-session learning plots and group summaries")
    parser.add_argument("--force", action="store_true", help="Reprocess sessions even if per-session CSV exists")
    parser.add_argument("--per-trial", action="store_true", help="Also export per-trial metrics (hit/miss/change) per session")
    args = parser.parse_args()

    run_with_options(
        args.mouse_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        save_plots=args.plots,
        recursive=args.recursive,
        learning_plots=args.learning_plots,
        force=args.force,
        per_trial=args.per_trial,
    )
