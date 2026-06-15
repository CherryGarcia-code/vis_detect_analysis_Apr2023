import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.tf_kernel import (
    lag_grid, fit_trf, build_region_design, kernel_timescale, shuffle_null,
)


def test_lag_grid_spans_range():
    lags = lag_grid()
    assert lags[0] == -0.5 and lags[-1] == 2.0
    assert np.allclose(np.diff(lags), 0.05)


def test_fit_trf_recovers_known_kernel():
    rng = np.random.default_rng(0)
    lags = lag_grid()
    ls = np.round(lags / 0.05).astype(int)
    true_k = np.exp(-((lags - 0.2) ** 2) / (2 * 0.1 ** 2))  # bump at +0.2 s
    x = rng.standard_normal(4000)
    y = np.zeros_like(x)
    for w, s in zip(true_k, ls):
        if s >= 0:
            y[s:] += w * x[:len(x) - s]
        else:
            y[:s] += w * x[-s:]
    y += 0.01 * rng.standard_normal(len(y))
    out_lags, kernel = fit_trf([(x, y)], lags=lags)
    assert out_lags[np.nanargmax(kernel)] == np.float64(0.2) or abs(out_lags[np.nanargmax(kernel)] - 0.2) <= 0.05


def test_build_region_design_returns_segments():
    onset = 100.0
    st1 = np.repeat(np.ones(200), 3).tolist()
    tr = SimpleNamespace(trial_index=0, outcome="Hit", change_time=8.0, iti_duration=2.0,
                         reaction_time=0.5, change_size=2.0, absolute_change_time=108.0,
                         absolute_reaction_time=108.5, metadata={"St1TrialVector": st1, "Stim2TF": 2.0})
    sess = SimpleNamespace(trials=[tr])
    ts = np.arange(95.0, 115.0, 0.01)
    sig = np.sin(ts)
    segs = build_region_design(sess, sig, ts, validate=False)
    assert len(segs) == 1
    assert segs[0][0].size == segs[0][1].size and segs[0][0].size > 50


def test_build_region_design_counts():
    onset = 100.0
    st1 = np.repeat(np.ones(200), 3).tolist()
    good_tr = SimpleNamespace(trial_index=0, outcome="Hit", change_time=8.0, iti_duration=2.0,
                              reaction_time=0.5, change_size=2.0, absolute_change_time=108.0,
                              absolute_reaction_time=108.5, metadata={"St1TrialVector": st1, "Stim2TF": 2.0})
    # change_time=1.5 -> usable window [onset+1.0, onset+0.5] is empty -> n_empty_window
    empty_tr = SimpleNamespace(trial_index=1, outcome="Hit", change_time=1.5, iti_duration=2.0,
                               reaction_time=0.5, change_size=2.0, absolute_change_time=101.5,
                               absolute_reaction_time=102.0, metadata={"St1TrialVector": st1, "Stim2TF": 2.0})
    sess = SimpleNamespace(trials=[good_tr, empty_tr])
    ts = np.arange(95.0, 115.0, 0.01)
    sig = np.sin(ts)
    segs, counts = build_region_design(sess, sig, ts, validate=False, return_counts=True)
    assert len(segs) == 1
    assert counts["n_seen"] == 2
    assert counts["n_kept"] == 1
    assert counts["n_empty_window"] == 1
    assert counts["n_pulses"] == segs[0][0].size
    # backward compat: default returns just the list
    segs_only = build_region_design(sess, sig, ts, validate=False)
    assert isinstance(segs_only, list) and len(segs_only) == 1
    # state_keep filters trials out of n_seen
    _, counts2 = build_region_design(sess, sig, ts, validate=False, return_counts=True, state_keep={1})
    assert counts2["n_seen"] == 1 and counts2["n_kept"] == 0 and counts2["n_empty_window"] == 1


def test_kernel_timescale_peak():
    lags = lag_grid()
    k = np.zeros_like(lags); k[np.argmin(np.abs(lags - 0.3))] = 2.0
    out = kernel_timescale(lags, k)
    assert out["peak_lag"] == 0.3 and out["signed_peak"] == 2.0


def test_shuffle_null_shape():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000); y = rng.standard_normal(2000)
    lags, lo, hi = shuffle_null([(x, y)], n_shuffles=20)
    assert lo.shape == lags.shape and hi.shape == lags.shape and np.all(hi >= lo)


def test_fit_trf_segments_isolated():
    """Primary invariant: lag embedding never crosses trial boundaries.

    Two L=10 segments are each too short to yield any lag-embedded rows (smax=11),
    so the kernel is all-NaN. The same 20 samples concatenated WOULD yield rows --
    confirming segments are kept isolated rather than merged.
    """
    lags = np.arange(0.0, 0.6, 0.05)  # causal-only grid, smax=11
    short_segs = [(np.ones(10), np.zeros(10)), (np.ones(10), np.zeros(10))]
    _, k_isolated = fit_trf(short_segs, lags=lags)
    assert not np.any(np.isfinite(k_isolated)), "cross-trial lag bleed detected"
    _, k_merged = fit_trf([(np.ones(20), np.zeros(20))], lags=lags)
    assert np.any(np.isfinite(k_merged))


from visdetect_photom.analysis.tf_kernel import pulse_triggered_average, detrend_pulse_trace


def test_pulse_triggered_recovers_bump():
    fs = 100.0
    ts = np.arange(0, 60, 1 / fs)
    sig = np.zeros_like(ts)
    pulses = np.array([10.0, 20.0, 30.0, 40.0])
    for p in pulses:
        sig += 1.5 * np.exp(-((ts - (p + 0.2)) ** 2) / (2 * 0.05 ** 2))
    t_vec, mean, sem = pulse_triggered_average(sig, ts, pulses, fs=fs)
    post = (t_vec >= 0.1) & (t_vec <= 0.3)
    assert np.nanmax(mean[post]) > 2.0   # z-scored bump


def test_detrend_removes_linear_trend():
    t = np.linspace(-0.4, 0.5, 90)
    trace = 5.0 * t + 0.0       # pure linear, no post-pulse feature
    for i, tt in enumerate(t):
        if 0.1 <= tt <= 0.2:
            trace[i] += 3.0      # planted post-pulse peak
    detr, zmax, zmin = detrend_pulse_trace(t, trace)
    assert zmax > 2.0 and abs(np.mean(detr[t < 0.0])) < 0.5


def test_pulse_triggered_returns_none_when_no_pulses():
    ts = np.arange(0, 10, 0.01)
    sig = np.zeros_like(ts)
    assert pulse_triggered_average(sig, ts, []) is None
    assert pulse_triggered_average(sig, ts, [np.nan, np.nan]) is None


def test_detrend_fallback_few_baseline_points():
    # No samples in the baseline window (-0.4, -0.01) -> raw trace returned,
    # peak/trough measured on the post window (0.0, 0.3).
    t = np.array([0.0, 0.1, 0.2])
    trace = np.array([1.0, 5.0, 2.0])
    detr, zmax, zmin = detrend_pulse_trace(t, trace)
    assert np.allclose(detr, trace) and zmax == 5.0 and zmin == 1.0
