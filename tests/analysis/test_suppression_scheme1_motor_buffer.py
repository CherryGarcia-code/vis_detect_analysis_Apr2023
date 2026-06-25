import numpy as np
from visdetect_photom.analysis.suppression import scheme1_scalar


def _rec(group, onset_abs, change_time, lick_elapsed=np.nan):
    return {"group": group, "onset_abs": onset_abs, "change_time": change_time,
            "lick_abs": onset_abs + lick_elapsed if np.isfinite(lick_elapsed) else np.nan,
            "lick_elapsed": lick_elapsed}

TS = np.arange(0, 100, 0.01)
SIG = np.full_like(TS, 4.0)


def test_scheme1_withhold_hit_excluded_when_lick_just_past_window():
    # Withhold-Hit: change_time=3.05 clears the change-coincidence gate (>w1=3.0),
    # but the Hit lick lands at change_time + reaction_time = 3.05 + 0.10 = 3.15,
    # only 0.15 s after the window end (gap < motor_buffer=1.0) -> excluded (NaN).
    change_time = 3.05
    reaction_time = 0.10
    lick_elapsed = change_time + reaction_time  # 3.15
    r = _rec("withhold", onset_abs=10.0, change_time=change_time,
             lick_elapsed=lick_elapsed)
    assert np.isnan(scheme1_scalar(r, SIG, TS))


def test_scheme1_withhold_hit_included_when_lick_well_after_window():
    # Withhold-Hit whose lick is well after the window end: change_time=6.0,
    # reaction_time=0.5 -> lick_elapsed=6.5, gap = 6.5 - 3.0 = 3.5 >= 1.0 -> included.
    change_time = 6.0
    reaction_time = 0.5
    lick_elapsed = change_time + reaction_time  # 6.5
    r = _rec("withhold", onset_abs=10.0, change_time=change_time,
             lick_elapsed=lick_elapsed)
    assert scheme1_scalar(r, SIG, TS) == 4.0


def test_scheme1_withhold_no_lick_still_included():
    # Withhold Miss/CR (no lick): lick_elapsed is NaN -> only change-coincidence
    # exclusion applies, so a trial with change_time past the window is included.
    r = _rec("withhold", onset_abs=10.0, change_time=4.0, lick_elapsed=np.nan)
    assert scheme1_scalar(r, SIG, TS) == 4.0
