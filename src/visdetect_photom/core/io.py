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

def _extract_subject_id_from_filename(path: str) -> Optional[str]:
    """Extract subject ID (e.g. 'BG_008') from a filename."""
    m = re.match(r"(BG_\d+)", os.path.basename(path))
    return m.group(1) if m else None


def pair_session_files(mouse_dir: str, min_photom_bytes: int = 0) -> List[Dict[str, str]]:
    """
    Build best-effort pairs of photom, photom_io, trials, session_settings per session.
    The photometry CSV timestamps rarely match trials to-the-second, so we match by date
    and choose the closest photom timestamps on the same day.

    Parameters
    ----------
    mouse_dir : str
        Directory containing session files for one mouse.
    min_photom_bytes : int
        Minimum file size for photometry CSVs. Smaller files (test/startup
        recordings) are excluded from pairing. Default 0 (no filtering).
    """
    # Infer expected subject ID from the directory name
    dir_subject_id = _extract_subject_id_from_filename(
        os.path.basename(os.path.normpath(mouse_dir)) + "_dummy"
    )
    # Simpler: try to match from the directory name itself
    dir_name = os.path.basename(os.path.normpath(mouse_dir))
    dir_m = re.match(r"(BG_\d+)", dir_name)
    dir_subject_id = dir_m.group(1) if dir_m else None

    def _belongs_to_subject(path: str) -> bool:
        """Check that a file belongs to this subject (prevent cross-contamination)."""
        if dir_subject_id is None:
            return True  # Can't filter if we don't know the subject
        file_subject = _extract_subject_id_from_filename(path)
        return file_subject is None or file_subject == dir_subject_id

    trials = sorted([t for t in glob(os.path.join(mouse_dir, TRIALS_GLOB)) if _belongs_to_subject(t)])
    sess = sorted([s for s in glob(os.path.join(mouse_dir, SESSION_SETTINGS_GLOB)) if _belongs_to_subject(s)])

    # Filter photom CSVs: exclude IO files, delete-flagged, wrong subject, and too-small files
    all_phot = glob(os.path.join(mouse_dir, PHOTOM_GLOB))
    phot = sorted([
        p for p in all_phot
        if "__photom_IO_" not in p
        and "delete" not in os.path.basename(p).lower()
        and _belongs_to_subject(p)
        and (min_photom_bytes == 0 or os.path.getsize(p) >= min_photom_bytes)
    ])
    phot_io = sorted([p for p in glob(os.path.join(mouse_dir, PHOTOM_IO_GLOB)) if _belongs_to_subject(p)])

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

        # Require photom + session_settings; photom_io is optional (old-format
        # subjects have IO data embedded in the photometry CSV itself).
        if p_best and s_best:
            sessions.append({
                "trials": t_path,
                "session_settings": s_best,
                "photom": p_best,
                "photom_io": pio_best,  # May be None for old-format subjects
            })

    # ── Compute IO offsets for sessions sharing the same photom CSV ──
    # When multiple behavioral sessions share one continuous photometry recording,
    # the embedded IO events (Input0 rising edges) are sequential across sessions.
    # We compute cumulative trial-count offsets so each session knows which
    # slice of IO events belongs to it.
    _compute_io_offsets(sessions)

    return sessions


def _compute_io_offsets(sessions: List[Dict[str, str]]) -> None:
    """Add 'io_event_offset' to sessions sharing the same photometry CSV.

    Modifies sessions in-place. Sessions sharing a photom CSV are sorted
    chronologically, and each gets an offset equal to the cumulative
    trial count of preceding sessions.
    """
    from collections import defaultdict

    # Group session indices by photom path
    by_photom: Dict[str, List[int]] = defaultdict(list)
    for idx, s in enumerate(sessions):
        by_photom[s['photom']].append(idx)

    for photom_path, indices in by_photom.items():
        if len(indices) == 1:
            sessions[indices[0]]['io_event_offset'] = 0
            continue

        # Sort sessions sharing this CSV chronologically by trials timestamp
        indices_sorted = sorted(
            indices,
            key=lambda i: parse_trials_timestamp(sessions[i]['trials']) or datetime.min
        )

        # Assign offsets based on cumulative trial counts
        offset = 0
        for idx in indices_sorted:
            sessions[idx]['io_event_offset'] = offset
            # Count trials in this session's JSON (quick peek)
            try:
                n_trials = _count_trials_in_json(sessions[idx]['trials'])
            except Exception:
                n_trials = 0
            offset += n_trials


def _count_trials_in_json(trials_path: str) -> int:
    """Quickly count trials in a JSON file without loading full trial data."""
    import json
    with open(trials_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return len(data.get('trials', []))
    return 0

def find_all_sessions(root_dir: str, recursive: bool = False,
                      min_photom_bytes: int = 0) -> List[Dict[str, str]]:
    """Discover sessions under root_dir. If recursive, search subfolders per mouse.

    Parameters
    ----------
    root_dir : str
        Top-level directory (or single mouse directory if not recursive).
    recursive : bool
        If True, walk subdirectories to find per-mouse folders.
    min_photom_bytes : int
        Minimum file size for photometry CSVs (passed to pair_session_files).
    """
    if not recursive:
        return pair_session_files(root_dir, min_photom_bytes=min_photom_bytes)
    all_sessions: List[Dict[str, str]] = []
    # Include root itself
    all_sessions.extend(pair_session_files(root_dir, min_photom_bytes=min_photom_bytes))
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
                all_sessions.extend(pair_session_files(dirpath, min_photom_bytes=min_photom_bytes))
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
