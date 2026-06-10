"""Session-level learning-stage helper backed by results/staging_manifest.csv.

Distinct from analysis/state_provider.py (trial-level behavioral state). Stages:
Naive | Learning | Expert | Disengaged | Excluded.
"""
import os
import pandas as pd

DEFAULT_MANIFEST_PATH = os.path.join("results", "staging_manifest.csv")


def load_staging_manifest(path: str = DEFAULT_MANIFEST_PATH):
    """Load the staging manifest, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _norm_subject(subject_id) -> str:
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def get_session_stage(session, manifest) -> str:
    """Stage for a session by matching session_name == session.session_id.

    If multiple manifest rows share the session_name (rare; >1 recording/day),
    the first match wins.
    """
    if manifest is None or "session_name" not in manifest.columns:
        return "Unknown"
    hit = manifest[manifest["session_name"] == session.session_id]
    if len(hit) == 0:
        return "Unknown"
    return str(hit.iloc[0]["stage"])


def excluded_mice(manifest) -> set:
    """Subjects (BG_0XX) whose every staged session is 'Excluded'."""
    if manifest is None or "stage" not in manifest.columns:
        return set()
    excl = set()
    for subj, grp in manifest.groupby("subject_id"):
        if (grp["stage"] == "Excluded").all():
            excl.add(_norm_subject(subj))
    return excl
