import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import (
    build_session_scalars, build_suppression_dataset,
)


def _trace():
    ts = np.arange(0, 80, 0.01)
    return ts, np.full_like(ts, 2.0)


def _trial(idx, outcome, change_time, change_size, abs_change, abs_rt):
    return SimpleNamespace(trial_index=idx, outcome=outcome, change_time=change_time,
                           change_size=change_size, reaction_time=None,
                           absolute_change_time=abs_change, absolute_reaction_time=abs_rt)


def _d1_session():
    ts, sig = _trace()
    photom = {"G0": SimpleNamespace(roi_name="G0", timestamps=ts, signal=sig),
              "G2": SimpleNamespace(roi_name="G2", timestamps=ts, signal=sig.copy())}
    trials = [
        _trial(0, "FA",  change_time=8.0, change_size=2.0, abs_change=18.0, abs_rt=15.0),  # lick, elapsed 5
        _trial(1, "Hit", change_time=5.0, change_size=2.0, abs_change=35.0, abs_rt=35.5),  # withhold
        _trial(2, "Miss",change_time=5.0, change_size=2.0, abs_change=55.0, abs_rt=None),  # withhold
    ]
    return SimpleNamespace(subject_id="013", session_id="013_a", session_date="20231205",
                           trials=trials, photometry_data=photom)


def test_build_session_scalars_scheme1():
    rows = build_session_scalars(_d1_session(), track="behavioral_fa",
                                 scheme="scheme1", use_qc=False)
    df_groups = {(r["region"], r["group"]) for r in rows}
    assert ("DMS", "lick") in df_groups and ("DMS", "withhold") in df_groups
    assert all(r["genotype"] == "D1" for r in rows)
    assert all(r["track"] == "behavioral_fa" and r["scheme"] == "scheme1" for r in rows)
    assert all(np.isfinite(r["scalar"]) for r in rows)


def test_build_dataset_two_genotypes_and_scheme3():
    d1 = _d1_session()
    d2 = _d1_session(); d2.subject_id = "016"; d2.session_id = "016_a"  # BG_016 = D2
    df = build_suppression_dataset([d1, d2], track="behavioral_fa",
                                   scheme="scheme3", use_qc=False)
    assert set(df["genotype"]) == {"D1", "D2"}
    assert set(df["group"]) >= {"lick", "withhold"}
    assert "session_id" in df.columns
