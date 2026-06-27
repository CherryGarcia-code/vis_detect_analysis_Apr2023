import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import assign_proficiency_bins
from visdetect_photom.core.constants import PROF_MIN_SESSIONS


def _sess(subj, sid, date, trials=None):
    return SimpleNamespace(subject_id=subj, session_id=sid, session_date=date,
                           trials=trials or [], photometry_data={})


def test_date_fallback_too_few_sessions_all_none():
    # manifest=None forces the date fallback. With only 2 sessions (< 2*PROF_MIN),
    # a split would give 1 session per bin -> underpowered -> all bins None.
    sessions = [_sess("013", "013_0", "20231201"),
                _sess("013", "013_1", "20231202")]
    bins = assign_proficiency_bins(sessions, manifest=None)
    assert bins["013_0"] is None and bins["013_1"] is None


def test_date_fallback_exactly_2x_floor_splits_evenly():
    # 2 * PROF_MIN_SESSIONS sessions on distinct dates -> proper split with
    # >= PROF_MIN_SESSIONS on each side.
    n = 2 * PROF_MIN_SESSIONS
    sessions = [_sess("013", f"013_{i}", f"202312{i + 1:02d}") for i in range(n)]
    bins = assign_proficiency_bins(sessions, manifest=None)
    less = [sid for sid, b in bins.items() if b == "less"]
    more = [sid for sid, b in bins.items() if b == "more"]
    none = [sid for sid, b in bins.items() if b is None]
    assert len(none) == 0
    assert len(less) >= PROF_MIN_SESSIONS
    assert len(more) >= PROF_MIN_SESSIONS
    # earliest session is 'less', latest is 'more'
    assert bins["013_0"] == "less"
    assert bins[f"013_{n - 1}"] == "more"


def test_date_fallback_just_below_2x_floor_all_none():
    # 2*PROF_MIN_SESSIONS - 1 sessions: still below the floor -> all None.
    n = 2 * PROF_MIN_SESSIONS - 1
    sessions = [_sess("013", f"013_{i}", f"202312{i + 1:02d}") for i in range(n)]
    bins = assign_proficiency_bins(sessions, manifest=None)
    assert all(b is None for b in bins.values())
