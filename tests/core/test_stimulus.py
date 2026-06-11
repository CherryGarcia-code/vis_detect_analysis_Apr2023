import numpy as np
from types import SimpleNamespace
from visdetect_photom.core.stimulus import (
    baseline_onset_ts, baseline_pulse_values, n_baseline_samples,
    windowed_pulses, fast_slow_pulse_times, aligned_baseline_regressor,
    validate_change_anchor,
)


def _trial(outcome="Hit", change_time=8.0, reaction_time=0.5, onset=100.0,
           st1_pulses=None, change_size=2.0, with_realized=False):
    if st1_pulses is None:
        st1_pulses = np.ones(200)          # flat 1 Hz baseline
    st1 = np.repeat(st1_pulses, 3)         # 3 frames per 50 ms pulse
    md = {"St1TrialVector": st1.tolist(), "Stim2TF": change_size}
    if with_realized:
        fps = 60.0
        n_gray = int(round(2.0 * fps))     # 2 s gray
        n_base = int(round(change_time * fps))
        n_post = int(round(1.0 * fps))
        tf = np.concatenate([np.zeros(n_gray),
                             np.repeat(st1_pulses, 3)[:n_base],
                             np.full(n_post, change_size)])
        vbl = onset - n_gray / fps + np.arange(len(tf)) / fps  # wall-clock-ish, onset at frame n_gray
        # make vbl an arbitrary epoch but with correct deltas:
        vbl = 1.7e9 + np.arange(len(tf)) / fps
        md["TF"] = tf.tolist()
        md["vbl"] = vbl.tolist()
    abs_change = onset + change_time
    abs_rt = onset + reaction_time if outcome in ("FA", "Abort") else onset + change_time + reaction_time
    return SimpleNamespace(trial_index=0, outcome=outcome, change_time=change_time,
                           iti_duration=2.0, reaction_time=reaction_time, change_size=change_size,
                           absolute_change_time=abs_change, absolute_reaction_time=abs_rt,
                           metadata=md)


def test_baseline_onset_ts():
    assert baseline_onset_ts(_trial(change_time=8.0, onset=100.0)) == 108.0 - 8.0

def test_baseline_pulse_values_strided():
    vals = baseline_pulse_values(_trial(st1_pulses=np.arange(10.0)))
    assert np.allclose(vals, np.arange(10.0))   # [::3] of repeat-3 recovers the pulses

def test_n_baseline_samples_hit_vs_fa():
    assert n_baseline_samples(_trial(outcome="Hit", change_time=8.0)) == 160
    assert n_baseline_samples(_trial(outcome="FA", reaction_time=5.0)) == 100

def test_windowed_pulses_respects_margins():
    # Hit: window = [onset+1.0, change-1.0] = [101, 107] -> times 100+k*0.05 in [101,107] -> k in [20,140]
    vals, times = windowed_pulses(_trial(outcome="Hit", change_time=8.0, onset=100.0))
    assert times.min() >= 101.0 - 1e-9 and times.max() <= 107.0 + 1e-9

def test_fast_slow_classification():
    pulses = np.array([1.0, 1.3, 0.7, 1.0] * 60)   # 1.3 -> log2=0.38 fast; 0.7 -> -0.51 slow
    fast, slow = fast_slow_pulse_times(_trial(outcome="Hit", change_time=8.0, st1_pulses=pulses))
    assert fast.size > 0 and slow.size > 0

def test_aligned_regressor_is_log2_meancenterable():
    l2, times = aligned_baseline_regressor(_trial(st1_pulses=np.full(200, 2.0)))
    assert np.allclose(l2, 1.0)   # log2(2/1) = 1

def test_validate_change_anchor_pass():
    ok, mism = validate_change_anchor(_trial(change_size=4.0, with_realized=True))
    assert ok is True and mism < 0.05

def test_validate_change_anchor_skips_small_change():
    ok, mism = validate_change_anchor(_trial(change_size=1.25, with_realized=True))
    assert ok is True and np.isnan(mism)   # not applicable -> pass
