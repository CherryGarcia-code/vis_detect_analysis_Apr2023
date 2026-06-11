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
