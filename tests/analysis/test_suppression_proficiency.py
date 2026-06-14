import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import assign_proficiency_bins, session_d_prime


def _sess(subj, sid, date, trials=None):
    return SimpleNamespace(subject_id=subj, session_id=sid, session_date=date,
                           trials=trials or [], photometry_data={})


def test_staging_split_used_when_enough_sessions():
    import pandas as pd
    sessions = [_sess("013", f"013_{i}", f"2023120{i}") for i in range(6)]
    manifest = pd.DataFrame({
        "subject_id": ["013"] * 6,
        "session_name": [f"013_{i}" for i in range(6)],
        "stage": ["Learning", "Learning", "Learning", "Expert", "Expert", "Expert"],
    })
    bins = assign_proficiency_bins(sessions, manifest)
    assert bins["013_0"] == "less" and bins["013_5"] == "more"

def test_date_fallback_when_staging_thin():
    sessions = [_sess("013", f"013_{i}", f"2023120{i}") for i in range(4)]
    bins = assign_proficiency_bins(sessions, manifest=None)  # no staging -> date split
    assert bins["013_0"] == "less" and bins["013_3"] == "more"

def test_session_d_prime_empty_is_nan():
    assert np.isnan(session_d_prime(_sess("013", "013_x", "20231205")))

def test_session_d_prime_finite_with_trials():
    # go trials (change_size>1): 2 Hit, 1 Miss; catch trials (change_size~1): 1 Hit, 2 Miss
    trials = [SimpleNamespace(outcome=o, change_size=cs) for o, cs in
              [("Hit", 2.0), ("Hit", 2.0), ("Miss", 2.0),
               ("Hit", 1.0), ("Miss", 1.0), ("Miss", 1.0)]]
    d = session_d_prime(_sess("013", "013_y", "20231205", trials=trials))
    assert np.isfinite(d)
