import numpy as np
from types import SimpleNamespace
from visdetect_photom.core.qc import region_sources


def _session():
    ts = np.arange(0, 30, 0.01)
    sig = np.full_like(ts, 2.0)
    photom = {
        "G0": SimpleNamespace(roi_name="G0", timestamps=ts, signal=sig),
        "G2": SimpleNamespace(roi_name="G2", timestamps=ts, signal=sig.copy()),
    }
    return SimpleNamespace(subject_id="013", session_id="013_x",
                           session_date="20231205", trials=[], photometry_data=photom)


def test_region_sources_no_qc_averages_hemispheres():
    src = region_sources(_session(), use_qc=False)
    assert "DMS" in src
    sig, ts = src["DMS"]
    assert sig.shape == ts.shape
    assert np.allclose(sig, 2.0)
