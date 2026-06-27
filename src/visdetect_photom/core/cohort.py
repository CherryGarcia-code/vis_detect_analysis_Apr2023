"""Cohort registry: which subjects belong to which recording cohort, their
indicator, and (for the intersectional cohort) the cortical input region.

The bulk GCaMP8m cohort and the intersectional GCaMP6f (MOs-recipient) cohort
must NEVER be pooled. `non_bulk_subjects()` feeds the central never-pool guard
in core/staging.excluded_mice.
"""
from pathlib import Path

COHORTS = {
    "bulk_8m": {
        "indicator": "GCaMP8m",
        "subjects": ["BG_008","BG_009","BG_010","BG_011","BG_013","BG_014",
                     "BG_015","BG_016","BG_017","BG_018","BG_019","BG_020"],
    },
    "intersectional_mos": {
        "indicator": "GCaMP6f",
        "subjects": ["BG_027","BG_028","BG_029","BG_030"],
        # VMS cells receive aMOs; DMS cells receive pMOs.
        "inputs": {"BG_027": "aMOs", "BG_028": "pMOs",
                   "BG_029": "pMOs", "BG_030": "aMOs"},
    },
}

def _norm(subject_id) -> str:
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s

def subjects_in(cohort_name):
    return list(COHORTS[cohort_name]["subjects"])

def cohort_of(subject_id):
    s = _norm(subject_id)
    for name, spec in COHORTS.items():
        if s in spec["subjects"]:
            return name
    return None

def indicator_of(subject_id):
    name = cohort_of(subject_id)
    return COHORTS[name]["indicator"] if name else None

def cortical_input(subject_id):
    s = _norm(subject_id)
    return COHORTS["intersectional_mos"].get("inputs", {}).get(s)

def non_bulk_subjects():
    out = []
    for name, spec in COHORTS.items():
        if name != "bulk_8m":
            out.extend(spec["subjects"])
    return out

def load_cohort_sessions(cohort_name, root_dir, max_sessions=None):
    """Load only the sessions belonging to `cohort_name` under root_dir."""
    from visdetect_photom.core import io
    from visdetect_photom.core.session import load_session_from_files
    from visdetect_photom.core.constants import MIN_PHOTOM_CSV_BYTES
    wanted = set(subjects_in(cohort_name))
    files = io.find_all_sessions(str(root_dir), recursive=True,
                                 min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    sessions, n = [], 0
    for sf in files:
        if max_sessions and n >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception:
            continue
        if _norm(sess.subject_id) not in wanted:
            continue
        sessions.append(sess)
        n += 1
    return sessions

def summarize_sessions_by_cell(per_session_df, value_cols=("delta", "auroc"),
                               cell_keys=("subject_id", "genotype", "region")):
    """Per cell: bootstrap CI of each value col over that cell's per-session values.

    This is the within-animal session-unit summary for the n=1-mouse/cell cohort.
    """
    import numpy as np
    import pandas as pd
    from visdetect_photom.analysis.group_statistics import bootstrap_ci
    if per_session_df is None or per_session_df.empty:
        return pd.DataFrame()
    cell_keys = list(cell_keys)
    out = []
    for key, g in per_session_df.groupby(cell_keys):
        row = dict(zip(cell_keys, key if isinstance(key, tuple) else (key,)))
        row["n_sessions"] = int(len(g))
        for col in value_cols:
            vals = g[col].to_numpy(dtype=float)
            ci = bootstrap_ci(vals)
            row[f"{col}_mean"] = ci["observed"]
            row[f"{col}_ci_lo"] = ci["ci_lo"]
            row[f"{col}_ci_hi"] = ci["ci_hi"]
        out.append(row)
    return pd.DataFrame(out)
