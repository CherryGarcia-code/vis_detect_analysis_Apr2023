"""Regression test: extract_onset_latency uses a TWO-SIDED absolute-deviation
crossing, so it detects SUPPRESSION (negative steps), not only activation.

Locks in the documented behavior: the onset is the first time |trace - baseline_mean|
exceeds threshold_n_std * baseline_std for n_consecutive bins.
"""
import numpy as np

from visdetect_photom.analysis.group_statistics import extract_onset_latency


def test_onset_detected_for_negative_step():
    # 100 Hz time axis over [-2, 4].
    time_axis = np.arange(-2.0, 4.0, 0.01)
    rng = np.random.default_rng(0)

    trace = np.zeros_like(time_axis)
    # Tiny baseline noise so bl_std > 1e-6 (otherwise the function returns NaN).
    bl_mask = time_axis < 0.0
    trace[bl_mask] = rng.normal(0.0, 0.01, size=bl_mask.sum())

    # Clean, large NEGATIVE step (suppression) starting at t = 0.5 s.
    step_time = 0.5
    trace[time_axis >= step_time] = -5.0

    onset = extract_onset_latency(
        trace, time_axis,
        threshold_n_std=2.0,
        baseline_window=(-2.0, 0.0),
        search_window=(0.0, 2.0),
        n_consecutive=3,
    )

    assert np.isfinite(onset)
    # Onset should land at (or essentially at) the step, well before the search end.
    assert abs(onset - step_time) < 0.05


def test_flat_trace_returns_nan():
    # No deviation anywhere -> no onset.
    time_axis = np.arange(-2.0, 4.0, 0.01)
    rng = np.random.default_rng(1)
    trace = rng.normal(0.0, 0.01, size=time_axis.size)  # noise only, no step
    onset = extract_onset_latency(
        trace, time_axis,
        threshold_n_std=10.0,  # high bar: pure baseline noise never crosses
        baseline_window=(-2.0, 0.0),
        search_window=(0.0, 2.0),
        n_consecutive=3,
    )
    assert np.isnan(onset)
