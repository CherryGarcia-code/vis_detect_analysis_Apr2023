import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import window_mean, trial_waiting_records


def test_window_mean_basic():
    ts = np.arange(0, 10, 0.01)
    sig = np.full_like(ts, 3.0)
    assert window_mean(sig, ts, 2.0, 3.0) == 3.0

def test_window_mean_nan_when_too_few_samples():
    ts = np.array([0.0, 5.0, 9.0])
    sig = np.array([1.0, 1.0, 1.0])
    assert np.isnan(window_mean(sig, ts, 2.0, 3.0))  # zero samples in window

def _trial(idx, outcome, change_time, change_size, abs_change, abs_rt):
    return SimpleNamespace(trial_index=idx, outcome=outcome, change_time=change_time,
                           change_size=change_size, reaction_time=None,
                           absolute_change_time=abs_change, absolute_reaction_time=abs_rt)

def _session(trials):
    return SimpleNamespace(subject_id="013", session_id="013_x", trials=trials,
                           photometry_data={})

def test_behavioral_fa_track_grouping():
    trials = [
        _trial(0, "FA",   change_time=6.0, change_size=2.0, abs_change=26.0, abs_rt=25.0),
        _trial(1, "Hit",  change_time=4.0, change_size=2.0, abs_change=14.0, abs_rt=14.5),
        _trial(2, "Miss", change_time=4.0, change_size=2.0, abs_change=44.0, abs_rt=None),
        _trial(3, "CR",   change_time=4.0, change_size=1.0, abs_change=54.0, abs_rt=None),
        _trial(4, "Abort",change_time=5.0, change_size=2.0, abs_change=64.0, abs_rt=61.0),
    ]
    recs = {r["trial_index"]: r for r in trial_waiting_records(_session(trials), "behavioral_fa")}
    assert recs[0]["group"] == "lick"
    assert recs[1]["group"] == "withhold" and recs[2]["group"] == "withhold"
    assert recs[3]["group"] == "withhold"
    assert recs[4]["group"] == "abort"
    # grating onset recovered as abs_change - change_time; lick_elapsed = abs_rt - onset
    assert recs[0]["onset_abs"] == 20.0
    assert recs[0]["lick_elapsed"] == 5.0

def test_sdt_fa_track_grouping():
    trials = [
        _trial(0, "Hit",  change_time=4.0, change_size=1.0, abs_change=14.0, abs_rt=18.0),  # catch lick -> lick
        _trial(1, "Miss", change_time=4.0, change_size=1.0, abs_change=24.0, abs_rt=None),  # SDT-CR -> withhold
        _trial(2, "Hit",  change_time=4.0, change_size=2.0, abs_change=34.0, abs_rt=34.5),  # go hit -> group None, skipped
        _trial(3, "FA",   change_time=6.0, change_size=2.0, abs_change=46.0, abs_rt=42.0),  # behavioral FA -> group None, skipped
    ]
    recs = {r["trial_index"]: r for r in trial_waiting_records(_session(trials), "sdt_fa")}
    assert recs[0]["group"] == "lick"
    assert recs[1]["group"] == "withhold"
    assert 2 not in recs and 3 not in recs   # group None on sdt_fa track -> record skipped
