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


import pytest
from visdetect_photom.core.cohort import match_cohort_cells, assert_rank_based

def test_match_cohort_cells_pairs_by_genotype_region():
    m = match_cohort_cells("D2", "DMS")
    assert m["intersectional"] == ["BG_029"]
    assert set(m["bulk"]) == {"BG_016", "BG_018", "BG_019"}
    m2 = match_cohort_cells("D1", "VMS")
    assert m2["intersectional"] == ["BG_027"]
    assert set(m2["bulk"]) == {"BG_008", "BG_009"}

def test_assert_rank_based_refuses_magnitude():
    assert_rank_based("auroc")            # ok
    with pytest.raises(ValueError):
        assert_rank_based("signed_auc")   # a magnitude metric -> refused across indicators


from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.core.constants import get_roi_region
from visdetect_photom.core.qc import get_region_pairs_for_subject

def test_cohort_genotype_and_region_maps():
    assert get_genotype("BG_027") == "D1" and get_genotype("BG_028") == "D1"
    assert get_genotype("BG_029") == "D2" and get_genotype("BG_030") == "D2"
    assert get_roi_region("G0", "BG_027") == "VMS_L"   # 027/030 = VMS
    assert get_roi_region("G2", "BG_030") == "VMS_R"
    assert get_roi_region("G0", "BG_028") == "DMS_L"   # 028/029 = DMS
    # G0/G2 map to the cell's region; (G4/G5 default to a VLS pair for every
    # G0/G2 mouse — harmless, the pipeline only extracts regions with real data).
    assert get_region_pairs_for_subject("BG_027")["VMS"] == ("G0", "G2")
    assert get_region_pairs_for_subject("BG_029")["DMS"] == ("G0", "G2")
