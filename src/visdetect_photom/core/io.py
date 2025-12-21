import pandas as pd
import numpy as np
import json
import os
import re
from glob import glob
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Constants
PHOTOM_GLOB = "*__photom_*.csv"
PHOTOM_IO_GLOB = "*__photom_IO_*.csv"
TRIALS_GLOB = "*__trials.json"
SESSION_SETTINGS_GLOB = "*__session_settings.json"

DATE_IN_TRIALS_RE = re.compile(r"_(\d{8})_(\d{6})__trials\.json$")
DATE_IN_JSON_RE = re.compile(r"_(\d{8})_(\d{6})__[^/\\]+\.json$")
DATE_IN_PHOTOM_RE = re.compile(r"__(?:photom|photom_IO)_(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})\.csv$")
MOUSE_ID_FROM_BASENAME_RE = re.compile(r"^BG_(\d+)")

def load_csv_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception:
        # Fallback to python engine if C engine fails (common with some CSVs)
        return pd.read_csv(filepath, engine='python')

def load_json_data(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return json.load(file)

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
