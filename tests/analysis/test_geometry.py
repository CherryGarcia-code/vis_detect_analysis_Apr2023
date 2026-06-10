import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.geometry import compute_geometry_metrics_for_session


def _trace(amp_at_change, change_t, fs=100.0, dur=60.0):
    ts = np.arange(0, dur, 1.0 / fs)
    sig = np.zeros_like(ts)
    sig += amp_at_change * np.exp(-((ts - (change_t + 0.5)) ** 2) / (2 * 0.3 ** 2))
    return ts, sig


def _d1_session():
    """One synthetic D1 (BG_013 -> DMS via G0/G2) session: Hit at t=30, +activation."""
    ts, sig = _trace(amp_at_change=2.0, change_t=30.0)
    photom = {
        "G0": SimpleNamespace(roi_name="G0", timestamps=ts, signal=sig),
        "G2": SimpleNamespace(roi_name="G2", timestamps=ts, signal=sig.copy()),
    }
    trials = [SimpleNamespace(trial_index=0, outcome="Hit", change_size=2.0,
                              reaction_time=0.5, absolute_change_time=30.0,
                              absolute_reaction_time=30.5)]
    return SimpleNamespace(subject_id="013", session_id="013_20231205",
                           session_date="20231205", trials=trials,
                           photometry_data=photom)


def test_change_hit_activation_positive():
    rows, traces, t = compute_geometry_metrics_for_session(_d1_session(), use_qc=False)
    change = [r for r in rows if r["region"] == "DMS" and r["epoch"] == "change_hit"]
    assert len(change) == 1
    r = change[0]
    assert r["genotype"] == "D1"
    assert r["signed_peak"] > 0
    assert r["activation"] > 0
    # A pure positive bump should register no meaningful suppression. The tiny
    # negative (~-0.03) is a baseline-subtraction tail artifact: the Gaussian's
    # rising edge leaks into the (-2,0)s baseline, nudging the baseline mean up,
    # which pushes the bump's far tail slightly below zero after subtraction.
    assert np.isnan(r["suppression"]) or r["suppression"] > -0.1
    assert ("DMS", "change_hit") in traces
    assert t is not None


from visdetect_photom.analysis.geometry import (
    build_geometry_dataset, run_pushpull_tests,
)


def _d2_session():
    """Synthetic D2 (BG_016 -> DMS) Hit session with a SUPPRESSION at change."""
    ts, sig = _trace(amp_at_change=-2.0, change_t=30.0)
    photom = {
        "G0": SimpleNamespace(roi_name="G0", timestamps=ts, signal=sig),
        "G2": SimpleNamespace(roi_name="G2", timestamps=ts, signal=sig.copy()),
    }
    trials = [SimpleNamespace(trial_index=0, outcome="Hit", change_size=2.0,
                              reaction_time=0.5, absolute_change_time=30.0,
                              absolute_reaction_time=30.5)]
    return SimpleNamespace(subject_id="016", session_id="016_20231214",
                           session_date="20231214", trials=trials,
                           photometry_data=photom)


def test_build_dataset_two_genotypes():
    df, traces_by_group, t = build_geometry_dataset(
        [_d1_session(), _d2_session()], use_qc=False)
    assert set(df["genotype"]) == {"D1", "D2"}
    assert ("D1", "DMS", "change_hit") in traces_by_group
    assert t is not None


def test_pushpull_opposite_sign_d1_vs_d2():
    # two D1 mice (positive) vs two D2 mice (negative) at change_hit
    d1a, d1b = _d1_session(), _d1_session(); d1b.subject_id = "020"; d1b.session_id = "020_x"
    d2a, d2b = _d2_session(), _d2_session(); d2b.subject_id = "018"; d2b.session_id = "018_x"
    df, _, _ = build_geometry_dataset([d1a, d1b, d2a, d2b], use_qc=False)
    stats = run_pushpull_tests(df, metric="signed_auc")
    row = stats[(stats["region"] == "DMS") & (stats["epoch"] == "change_hit")].iloc[0]
    assert row["d1_sign"] == 1 and row["d2_sign"] == -1
