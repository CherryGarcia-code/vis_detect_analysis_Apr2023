"""TRF kernel estimation for baseline TF -> dF/F (G1). numpy-only ridge."""
import numpy as np
from visdetect_photom.core.constants import (
    TRF_LAG_MIN, TRF_LAG_MAX, TRF_LAG_STEP, TF_SAMPLE_PERIOD,
)
from visdetect_photom.core.stimulus import aligned_baseline_regressor, validate_change_anchor


def lag_grid():
    """TRF lag grid (TRF_LAG_MIN..TRF_LAG_MAX in TRF_LAG_STEP increments, seconds)."""
    n = int(round((TRF_LAG_MAX - TRF_LAG_MIN) / TRF_LAG_STEP)) + 1
    return np.round(np.linspace(TRF_LAG_MIN, TRF_LAG_MAX, n), 6)


def build_region_design(session, signal, timestamps, *, state_keep=None, validate=True):
    """Return list of (x_seg, y_seg) per valid baseline window (50 ms grid).

    x_seg = log2(TF), y_seg = dF/F interpolated onto the pulse times. Segments are
    kept separate so the lag embedding never crosses trial boundaries.
    """
    timestamps = np.asarray(timestamps, float)
    signal = np.asarray(signal, float)
    segments = []
    for t in session.trials:
        if state_keep is not None and t.trial_index not in state_keep:
            continue
        if validate:
            ok, mism = validate_change_anchor(t)
            if (ok is False) and np.isfinite(mism):
                continue
        l2, times = aligned_baseline_regressor(t)
        if l2.size == 0:
            continue
        dff = np.interp(times, timestamps, signal, left=np.nan, right=np.nan)
        good = np.isfinite(dff) & np.isfinite(l2)
        if good.sum() <= 1:
            continue
        segments.append((l2[good], dff[good]))
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

    X_rows, y_rows = [], []
    for x_seg, y_seg in segments:
        L = len(x_seg)
        i_lo = max(0, smax)
        i_hi = min(L, L + smin)  # i <= L-1+smin
        for i in range(i_lo, i_hi):
            row = x_seg[i - lag_s]
            if np.all(np.isfinite(row)) and np.isfinite(y_seg[i]):
                X_rows.append(row)
                y_rows.append(y_seg[i])
    if not X_rows:
        return lags, np.full(len(lags), np.nan)

    X = np.asarray(X_rows, float)
    y = np.asarray(y_rows, float)
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
