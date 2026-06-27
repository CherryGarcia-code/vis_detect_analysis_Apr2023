"""Regression test for bug #3: same-day session_id collisions in proficiency binning.

`session_id = "{subject}_{date}"` is DATE-granular, so two recordings on the same
calendar day share an identical session_id. `assign_proficiency_bins` and
`build_session_scalars` must key/tag by a unique-per-recording `recording_id`
instead, so same-day recordings are counted and binned independently and their
trial rows remain distinguishable.
"""
import numpy as np
from types import SimpleNamespace

from visdetect_photom.analysis.suppression import (
    assign_proficiency_bins, build_session_scalars,
)
from visdetect_photom.core.constants import PROF_MIN_SESSIONS


def _sess(subj, sid, date, rec_id, trials=None, photometry_data=None):
    return SimpleNamespace(subject_id=subj, session_id=sid, session_date=date,
                           recording_id=rec_id, trials=trials or [],
                           photometry_data=photometry_data or {})


def _trace(roi):
    ts = np.arange(0.0, 60.0, 0.01)
    return SimpleNamespace(roi_name=roi, signal=np.zeros_like(ts), timestamps=ts)


def test_same_day_recordings_kept_separate_in_proficiency_bins():
    # Two recordings on the SAME calendar day -> identical session_id, distinct
    # recording_id. Add enough distinct-day sessions to clear the 2*floor gate.
    n_extra = 2 * PROF_MIN_SESSIONS
    sessions = [
        _sess("013", "013_20231201", "20231201", "013_20231201_090000"),
        _sess("013", "013_20231201", "20231201", "013_20231201_140000"),
    ]
    # distinct-day filler sessions; recording_id mirrors session_id (one rec/day)
    for i in range(n_extra):
        sid = f"013_202312{i + 2:02d}"
        sessions.append(_sess("013", sid, f"202312{i + 2:02d}", sid))

    bins = assign_proficiency_bins(sessions, manifest=None)

    # (a) both same-day recordings must appear as SEPARATE keys (not collapsed
    #     to one entry, not overwritten by each other).
    assert "013_20231201_090000" in bins
    assert "013_20231201_140000" in bins
    # The two same-day recordings are the two earliest -> both land in 'less'
    # (independently assigned), and neither clobbers the other.
    assert bins["013_20231201_090000"] == "less"
    assert bins["013_20231201_140000"] == "less"
    # Returned dict is keyed by recording_id, so the colliding date-granular
    # session_id is NOT used as a key.
    assert "013_20231201" not in bins


def test_build_session_scalars_rows_carry_distinct_recording_id():
    def _trial(idx, outcome, change_size, abs_change, change_time, abs_rt):
        return SimpleNamespace(trial_index=idx, outcome=outcome,
                               change_size=change_size, change_time=change_time,
                               absolute_change_time=abs_change,
                               absolute_reaction_time=abs_rt)

    # One FA (lick) + one Hit (withhold), enough to emit rows for behavioral_fa.
    trials = [
        _trial(0, "FA", 2.0, abs_change=26.0, change_time=6.0, abs_rt=25.0),
        _trial(1, "Hit", 2.0, abs_change=14.0, change_time=4.0, abs_rt=14.5),
    ]
    photom = {"G0": _trace("G0"), "G2": _trace("G2")}
    sA = _sess("013", "013_20231201", "20231201", "013_20231201_090000", trials, photom)
    sB = _sess("013", "013_20231201", "20231201", "013_20231201_140000", trials, photom)

    rows_a = build_session_scalars(sA, track="behavioral_fa", scheme="scheme1",
                                   use_qc=False)
    rows_b = build_session_scalars(sB, track="behavioral_fa", scheme="scheme1",
                                   use_qc=False)
    assert rows_a and rows_b
    # every row carries recording_id (and still session_id)
    for r in rows_a + rows_b:
        assert "recording_id" in r
        assert "session_id" in r
    rec_a = {r["recording_id"] for r in rows_a}
    rec_b = {r["recording_id"] for r in rows_b}
    # two same-day recordings produce rows with DISTINCT recording_id
    assert rec_a == {"013_20231201_090000"}
    assert rec_b == {"013_20231201_140000"}
    assert rec_a.isdisjoint(rec_b)
