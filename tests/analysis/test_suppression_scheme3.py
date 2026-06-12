import numpy as np
from visdetect_photom.analysis.suppression import scheme3_scalars


def _rec(group, onset_abs, change_time, lick_elapsed=np.nan):
    return {"trial_index": 0, "group": group, "onset_abs": onset_abs,
            "change_time": change_time,
            "lick_abs": onset_abs + lick_elapsed if np.isfinite(lick_elapsed) else np.nan,
            "lick_elapsed": lick_elapsed}

TS = np.arange(0, 200, 0.01)
SIG = np.full_like(TS, 5.0)


def test_scheme3_lick_window_ends_before_lick():
    # lick at elapsed 4.0, buffer 0.5, L 1.0 -> window [2.5, 3.5] elapsed -> valid, 5.0
    a = _rec("lick", onset_abs=10.0, change_time=8.0, lick_elapsed=4.0); a["trial_index"] = 1
    w = _rec("withhold", onset_abs=50.0, change_time=8.0); w["trial_index"] = 2
    lick_vals, wh_vals = scheme3_scalars([a], [w], SIG, TS)
    assert dict(lick_vals)[1] == 5.0

def test_scheme3_withhold_hazard_matched_value():
    a = _rec("lick", onset_abs=10.0, change_time=8.0, lick_elapsed=4.0); a["trial_index"] = 1
    w = _rec("withhold", onset_abs=50.0, change_time=8.0); w["trial_index"] = 2
    _, wh_vals = scheme3_scalars([a], [w], SIG, TS)
    assert dict(wh_vals)[2] == 5.0  # constant signal -> any matched window = 5.0

def test_scheme3_withhold_nan_when_no_admissible_tau():
    # only lick elapsed is 7.9; withhold change_time=2.0 -> tau(7.9) > change -> no valid draw
    a = _rec("lick", onset_abs=10.0, change_time=10.0, lick_elapsed=7.9); a["trial_index"] = 1
    w = _rec("withhold", onset_abs=50.0, change_time=2.0); w["trial_index"] = 2
    _, wh_vals = scheme3_scalars([a], [w], SIG, TS)
    assert np.isnan(dict(wh_vals)[2])

def test_scheme3_is_deterministic():
    a = _rec("lick", onset_abs=10.0, change_time=8.0, lick_elapsed=4.0); a["trial_index"] = 1
    b = _rec("lick", onset_abs=30.0, change_time=8.0, lick_elapsed=5.0); b["trial_index"] = 3
    w = _rec("withhold", onset_abs=50.0, change_time=8.0); w["trial_index"] = 2
    r1 = scheme3_scalars([a, b], [w], SIG, TS)
    r2 = scheme3_scalars([a, b], [w], SIG, TS)
    assert dict(r1[1]) == dict(r2[1])
