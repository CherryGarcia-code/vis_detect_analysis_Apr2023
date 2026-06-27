# tests/core/test_cohort.py
from visdetect_photom.core import cohort

def test_registry_membership_and_lookups():
    assert set(cohort.subjects_in("intersectional_mos")) == {"BG_027","BG_028","BG_029","BG_030"}
    assert cohort.cohort_of("BG_027") == "intersectional_mos"
    assert cohort.cohort_of("BG_013") == "bulk_8m"
    assert cohort.cohort_of("BG_999") is None
    assert cohort.indicator_of("BG_029") == "GCaMP6f"
    assert cohort.indicator_of("BG_013") == "GCaMP8m"
    assert cohort.cortical_input("BG_027") == "aMOs"   # VMS
    assert cohort.cortical_input("BG_028") == "pMOs"   # DMS
    assert cohort.cortical_input("BG_013") is None
    assert set(cohort.non_bulk_subjects()) == {"BG_027","BG_028","BG_029","BG_030"}


import numpy as np, pandas as pd
from visdetect_photom.core.cohort import summarize_sessions_by_cell

def test_summarize_sessions_by_cell_bootstraps_over_sessions():
    rows = [{"subject_id":"BG_029","genotype":"D2","region":"DMS",
             "session_id":f"s{i}","auroc":0.6+0.01*i,"delta":0.1*i} for i in range(20)]
    out = summarize_sessions_by_cell(pd.DataFrame(rows))
    r = out.iloc[0]
    assert r["n_sessions"] == 20
    assert r["auroc_mean"] == pytest_approx(np.mean([0.6+0.01*i for i in range(20)]))
    assert r["auroc_ci_lo"] < r["auroc_mean"] < r["auroc_ci_hi"]

def pytest_approx(x):  # tiny local helper to avoid extra import noise
    import pytest
    return pytest.approx(x, rel=1e-6)
