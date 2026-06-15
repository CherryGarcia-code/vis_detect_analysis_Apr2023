"""TRF kernel estimation for baseline TF -> dF/F (G1). numpy-only ridge."""
import numpy as np
from visdetect_photom.core.constants import (
    TRF_LAG_MIN, TRF_LAG_MAX, TRF_LAG_STEP, TF_SAMPLE_PERIOD, SAMPLING_FREQ,
    TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW,
    TF_PULSE_DETREND_BASELINE, TF_PULSE_DETREND_POST,
)
from visdetect_photom.core.stimulus import aligned_baseline_regressor, validate_change_anchor


def lag_grid():
    """TRF lag grid (TRF_LAG_MIN..TRF_LAG_MAX in TRF_LAG_STEP increments, seconds)."""
    n = int(round((TRF_LAG_MAX - TRF_LAG_MIN) / TRF_LAG_STEP)) + 1
    return np.round(np.linspace(TRF_LAG_MIN, TRF_LAG_MAX, n), 6)


def build_region_design(session, signal, timestamps, *, state_keep=None, validate=True,
                        return_counts=False):
    """Return list of (x_seg, y_seg) per valid baseline window (50 ms grid).

    x_seg = log2(TF), y_seg = dF/F interpolated onto the pulse times. Segments are
    kept separate so the lag embedding never crosses trial boundaries.

    If return_counts is True, also return a per-trial disposition dict
    (n_seen / n_validate_drop / n_empty_window / n_too_short / n_kept / n_pulses)
    for effective-N / alignment-QC reporting.
    """
    timestamps = np.asarray(timestamps, float)
    signal = np.asarray(signal, float)
    segments = []
    counts = {"n_seen": 0, "n_validate_drop": 0, "n_empty_window": 0,
              "n_too_short": 0, "n_kept": 0, "n_pulses": 0}
    for t in session.trials:
        if state_keep is not None and t.trial_index not in state_keep:
            continue
        counts["n_seen"] += 1
        if validate:
            ok, mism = validate_change_anchor(t)
            if (ok is False) and np.isfinite(mism):
                counts["n_validate_drop"] += 1
                continue
        l2, times = aligned_baseline_regressor(t)
        if l2.size == 0:
            counts["n_empty_window"] += 1
            continue
        dff = np.interp(times, timestamps, signal, left=np.nan, right=np.nan)
        good = np.isfinite(dff) & np.isfinite(l2)
        if good.sum() <= 1:
            counts["n_too_short"] += 1
            continue
        segments.append((l2[good], dff[good]))
        counts["n_kept"] += 1
        counts["n_pulses"] += int(good.sum())
    if return_counts:
        return segments, counts
    return segments


def _ridge_gcv(X, y, alphas):
    """Closed-form ridge with GCV-selected alpha (numpy only). X centered, y centered."""
    n = X.shape[0]
    XtX = X.T @ X
    Xty = X.T @ y
    evals, evecs = np.linalg.eigh(XtX)
    evals = np.clip(evals, 0, None)
    z = evecs.T @ Xty
    best_w, best_gcv = None, np.inf
    for a in alphas:
        denom = evals + a
        w = evecs @ (z / denom)
        resid = y - X @ w
        rss = float(resid @ resid)
        df = float(np.sum(evals / denom))
        gcv = (rss / n) / (1.0 - df / n) ** 2 if df < n else np.inf
        if gcv < best_gcv:
            best_gcv, best_w = gcv, w
    if best_w is None:
        raise ValueError("_ridge_gcv: no valid alpha found (alphas empty or all df >= n)")
    return best_w


def fit_trf(segments, lags=None, alpha=None):
    """Ridge time-receptive-field. Returns (lags, kernel)."""
    if lags is None:
        lags = lag_grid()
    lags = np.asarray(lags, float)
    lag_s = np.round(lags / TF_SAMPLE_PERIOD).astype(int)
    smin, smax = int(lag_s.min()), int(lag_s.max())

    # Build the lag-embedded design per segment (vectorized; never crosses trial
    # boundaries). Equivalent to the row-by-row form but fast enough for the pooled
    # shuffle-null (hundreds of refits on large pooled designs).
    X_blocks, y_blocks = [], []
    for x_seg, y_seg in segments:
        x_seg = np.asarray(x_seg, float)
        y_seg = np.asarray(y_seg, float)
        L = len(x_seg)
        i_lo = max(0, smax)
        i_hi = min(L, L + smin)  # i <= L-1+smin
        if i_hi <= i_lo:
            continue
        idxs = np.arange(i_lo, i_hi)
        rows = x_seg[idxs[:, None] - lag_s[None, :]]   # (n_rows, n_lags)
        yv = y_seg[idxs]
        finite = np.all(np.isfinite(rows), axis=1) & np.isfinite(yv)
        if finite.any():
            X_blocks.append(rows[finite])
            y_blocks.append(yv[finite])
    if not X_blocks:
        return lags, np.full(len(lags), np.nan)

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    X = X - X.mean(axis=0, keepdims=True)
    y = y - y.mean()
    if alpha is None:
        w = _ridge_gcv(X, y, np.logspace(-3, 3, 13))
    else:
        p = X.shape[1]
        w = np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)
    return lags, w


def kernel_timescale(lags, kernel):
    """signed_peak / peak_lag / center-of-mass over the causal (lag>=0) part."""
    lags = np.asarray(lags, float)
    k = np.asarray(kernel, float)
    causal = lags >= 0
    lk, kk = lags[causal], k[causal]
    if not np.any(np.isfinite(kk)):
        return {"signed_peak": np.nan, "peak_lag": np.nan, "com": np.nan}
    ip = int(np.nanargmax(np.abs(kk)))
    w = np.where(np.isfinite(kk), np.abs(kk), 0.0)
    com = float(np.sum(lk * w) / np.sum(w)) if np.sum(w) > 0 else np.nan
    return {"signed_peak": float(kk[ip]), "peak_lag": float(lk[ip]), "com": com}


def shuffle_null(segments, lags=None, n_shuffles=200, seed=42):
    """Circular-shift null band (2.5/97.5 pct) for the kernel."""
    if lags is None:
        lags = lag_grid()
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_shuffles):
        shuf = []
        for x_seg, y_seg in segments:
            if len(x_seg) < 2:
                shuf.append((x_seg, y_seg)); continue
            sh = int(rng.integers(1, len(x_seg)))
            shuf.append((np.roll(x_seg, sh), y_seg))
        _, k = fit_trf(shuf, lags=lags)
        null.append(k)
    null = np.asarray(null)
    return np.asarray(lags), np.nanpercentile(null, 2.5, axis=0), np.nanpercentile(null, 97.5, axis=0)


def pulse_triggered_average(signal, timestamps, pulse_times,
                            pre=TF_PULSE_PRE_WINDOW, post=TF_PULSE_POST_WINDOW, fs=SAMPLING_FREQ):
    """Mean +/- SEM dF/F around pulses, z-scored to the pre-pulse window.

    Returns (t_vec, mean, sem) or None if no pulses.
    """
    pulse_times = np.asarray(pulse_times, float)
    pulse_times = pulse_times[np.isfinite(pulse_times)]
    if pulse_times.size == 0:
        return None
    ts = np.asarray(timestamps, float)
    sig = np.asarray(signal, float)
    t_vec = np.arange(pre[0], post[1] + 1e-9, 1.0 / fs)
    pre_mask = (t_vec >= pre[0]) & (t_vec < pre[1])
    rows = []
    for pt in pulse_times:
        target = pt + t_vec
        idx = np.clip(np.searchsorted(ts, target), 0, len(sig) - 1)
        vals = sig[idx].astype(float)
        vals[np.abs(ts[idx] - target) >= (1.5 / fs)] = np.nan
        b = vals[pre_mask]
        m, s = np.nanmean(b), np.nanstd(b)
        vals = (vals - m) / s if (np.isfinite(s) and s > 1e-9) else vals - m
        rows.append(vals)
    rows = np.asarray(rows)
    mean = np.nanmean(rows, axis=0)
    n = np.sum(~np.isnan(rows), axis=0)
    # SEM = sample std (ddof=1) / sqrt(n); columns with n<2 yield NaN (honest: a
    # single observation has no defined SEM). maximum(n,1) only guards the divide.
    with np.errstate(invalid="ignore"):
        sem = np.nanstd(rows, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return t_vec, mean, sem


def detrend_pulse_trace(t_vec, trace,
                        baseline=TF_PULSE_DETREND_BASELINE, post=TF_PULSE_DETREND_POST):
    """Linear-detrend on the baseline window; measure post-pulse peak/trough.

    Ports the ephys detrend_tf_traces. Returns (detrended, max_post, min_post),
    where max_post/min_post are the peak/trough of the detrended trace over the
    post window in input units (e.g. z-dF/F), NOT a re-z-scored value.
    """
    t = np.asarray(t_vec, float)
    tr = np.asarray(trace, float)
    pre = (t >= baseline[0]) & (t < baseline[1])
    pm = (t >= post[0]) & (t < post[1])
    if pre.sum() < 2:
        zmax = float(np.nanmax(tr[pm])) if pm.any() else np.nan
        zmin = float(np.nanmin(tr[pm])) if pm.any() else np.nan
        return tr.copy(), zmax, zmin
    coef = np.polyfit(t[pre], tr[pre], 1)
    d = tr - np.polyval(coef, t)
    zmax = float(np.nanmax(d[pm])) if pm.any() else np.nan
    zmin = float(np.nanmin(d[pm])) if pm.any() else np.nan
    return d, zmax, zmin
