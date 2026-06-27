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
