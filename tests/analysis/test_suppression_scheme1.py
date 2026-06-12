import numpy as np
from visdetect_photom.analysis.suppression import scheme1_scalar


def _rec(group, onset_abs, change_time, lick_elapsed=np.nan):
    return {"group": group, "onset_abs": onset_abs, "change_time": change_time,
            "lick_abs": onset_abs + lick_elapsed if np.isfinite(lick_elapsed) else np.nan,
            "lick_elapsed": lick_elapsed}

TS = np.arange(0, 100, 0.01)
SIG = np.full_like(TS, 4.0)


def test_scheme1_withhold_included_before_change():
    # window (2,3) ends before change_time=4 -> valid, mean=4.0
    r = _rec("withhold", onset_abs=10.0, change_time=4.0)
    assert scheme1_scalar(r, SIG, TS) == 4.0

def test_scheme1_withhold_excluded_change_too_early():
    # change_time=2.5 < window end (3) -> NaN
    r = _rec("withhold", onset_abs=10.0, change_time=2.5)
    assert np.isnan(scheme1_scalar(r, SIG, TS))

def test_scheme1_lick_included_with_motor_buffer():
    # lick_elapsed=4.5 >= w1(3)+buffer(1)=4 -> valid
    r = _rec("lick", onset_abs=20.0, change_time=6.0, lick_elapsed=4.5)
    assert scheme1_scalar(r, SIG, TS) == 4.0

def test_scheme1_lick_excluded_when_lick_too_soon():
    # lick_elapsed=3.5 < 4 -> NaN (no clean pre-lick window; e.g. impulsive early FA)
    r = _rec("lick", onset_abs=20.0, change_time=6.0, lick_elapsed=3.5)
    assert np.isnan(scheme1_scalar(r, SIG, TS))
