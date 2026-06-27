# tests/core/test_excluded_cohort.py
from visdetect_photom.core.staging import excluded_mice

def test_excluded_mice_unions_non_bulk_cohort_even_without_manifest():
    excl = excluded_mice(None)
    for s in ("BG_027", "BG_028", "BG_029", "BG_030"):
        assert s in excl, "intersectional cohort must be excluded from the bulk default"
    assert "BG_013" not in excl  # a bulk mouse is not excluded by default
