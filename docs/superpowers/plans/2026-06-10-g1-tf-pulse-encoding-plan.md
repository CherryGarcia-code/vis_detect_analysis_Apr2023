# G1 — TF-Pulse Evidence Encoding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate the temporal kernel relating moment-to-moment baseline TF fluctuations to bulk D1/D2 dF/F (TRF) + a fast/slow pulse-triggered companion, and compare D1 vs D2 (sign, timing, integration window).

**Architecture:** Approach B — reusable stimulus reconstruction + cross-clock alignment in `core/stimulus.py`; kernel math in `analysis/tf_kernel.py`; thin CLI `scripts/analysis/photometry/09_tf_pulse_encoding.py`. Conventions mirror the ephys repo's `tf_pulse.py` (fast/slow `log2(TF)≥±0.25`, 50 ms pulses, pre `(-0.4,0)` / post `(0,0.5)` windows).

**Tech Stack:** Python 3.10, numpy<2, pandas<2.3, scipy<2, matplotlib<3.9, seaborn<0.14, pytest 9. **No scikit-learn** (ridge is implemented in numpy). `src/` layout via `tests/conftest.py`. Invoke with `py`.

**Spec:** `docs/superpowers/specs/2026-06-09-g1-tf-pulse-encoding-design.md` (arrives in `main` via the C2 merge). Read it first.

---

## ⚠️ Prerequisite: C2 must be merged into `main` first

G1 reuses code introduced by C2: `analysis/group_statistics.pushpull_sign_contrast`, `analysis/state_provider.py`, `core/staging.py`, plus the region-source pattern. **Do not start until `main` contains the merged C2 work** (verify: `py -c "from visdetect_photom.analysis.state_provider import PooledStateProvider; from visdetect_photom.core.staging import excluded_mice; from visdetect_photom.analysis.group_statistics import pushpull_sign_contrast; print('C2 deps present')"`). Branch off the post-merge `main`.

## Background the engineer needs

- **`Trial`** (`core/session.py`) carries the raw trial dict on `Trial.metadata` — including `St1TrialVector`, `TF`, `vbl`, `Stim2TF`. Also `.outcome`, `.change_time` (stimT), `.iti_duration` (stimD), `.reaction_time`, `.absolute_change_time`, `.absolute_reaction_time`, `.change_size`, `.trial_index`.
- **Baseline onset (grating onset)** in photometry `SystemTimestamp` = `Input0` edge = `absolute_change_time − change_time`.
- **Baseline TF pulse sequence** = `St1TrialVector[::3]` (each pulse repeated 3× at 60 fps → 50 ms). Sample `k` occurs at `baseline_onset + k·0.05`.
- **No `n_seen`**: baseline length = `round(change_time/0.05)` (Hit/Miss/CR) or `round(reaction_time/0.05)` (FA/Abort; FA/abort `reaction_time` is relative to baseline onset).
- **dF/F** lives in `Trial`-independent `PhotometryTrace.signal/timestamps`; region merging is via `core/qc.merge_hemispheres`.
- Reuse from C2: `analysis/group_utils.get_genotype`, `core/qc.compute_session_roi_qc/merge_hemispheres`, `core/constants.get_roi_region`, `analysis/group_statistics.{permutation_test,pushpull_sign_contrast}`, `analysis/state_provider`, `core/staging`.

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/visdetect_photom/core/constants.py` | Modify | Append TF-pulse + TRF constants. |
| `src/visdetect_photom/core/stimulus.py` | Create | Reconstruction + Input0-anchored uniform-50 ms alignment + windowing + validation. |
| `tests/core/test_stimulus.py` | Create | Unit tests for stimulus. |
| `src/visdetect_photom/analysis/tf_kernel.py` | Create | numpy ridge TRF, design build, timescale, shuffle null, pulse-triggered + detrend. |
| `tests/analysis/test_tf_kernel.py` | Create | Unit tests for kernel math. |
| `scripts/analysis/photometry/09_tf_pulse_encoding.py` | Create | Thin CLI: discover → load → design → kernels → figures. |
| `tests/scripts/test_09_smoke.py` | Create | Smoke test (skips if `photom_data/` absent). |

---

## Task 1: Constants + `core/stimulus.py`

**Files:**
- Modify: `src/visdetect_photom/core/constants.py`
- Create: `src/visdetect_photom/core/stimulus.py`
- Create: `tests/core/test_stimulus.py`

- [ ] **Step 1: Append constants**

Append to `src/visdetect_photom/core/constants.py`:

```python
# ── TF-pulse / evidence encoding (G1) — mirrors ephys tf_pulse.py conventions ──
TF_BASE_HZ = 1.0                       # nominal base temporal frequency
TF_BASELINE_STRIDE = 3                 # St1TrialVector repeats each pulse 3x (60fps)
TF_SAMPLE_PERIOD = 0.05                # seconds per baseline pulse sample (50 ms)
TF_FAST_THRESH_LOG2 = 0.25             # fast pulse: log2(TF) >= +0.25
TF_SLOW_THRESH_LOG2 = -0.25            # slow pulse: log2(TF) <= -0.25
TF_MIN_AFTER_BASELINE = 1.0            # exclude pulses < 1.0 s after baseline onset
TF_MIN_BEFORE_CHANGE = 1.0             # exclude pulses < 1.0 s before change
TF_MIN_BEFORE_OUTCOME_FA_ABORT = 2.0   # exclude pulses < 2.0 s before FA/abort lick
TF_PULSE_PRE_WINDOW = (-0.4, 0.0)      # pre-pulse z-score baseline
TF_PULSE_POST_WINDOW = (0.0, 0.5)      # post-pulse response window
TF_PULSE_DETREND_BASELINE = (-0.4, -0.01)
TF_PULSE_DETREND_POST = (0.0, 0.3)
TF_CHANGE_VALIDATE_MIN_CS = 2.0        # only run change-anchor validation when change_size >= this
TF_CHANGE_VALIDATE_TOL = 0.05          # 50 ms mismatch tolerance
# TRF lag grid (negatives = causality control)
TRF_LAG_MIN = -0.5
TRF_LAG_MAX = 2.0
TRF_LAG_STEP = 0.05
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/test_stimulus.py`:

```python
import numpy as np
from types import SimpleNamespace
from visdetect_photom.core.stimulus import (
    baseline_onset_ts, baseline_pulse_values, n_baseline_samples,
    windowed_pulses, fast_slow_pulse_times, aligned_baseline_regressor,
    validate_change_anchor,
)


def _trial(outcome="Hit", change_time=8.0, reaction_time=0.5, onset=100.0,
           st1_pulses=None, change_size=2.0, with_realized=False):
    if st1_pulses is None:
        st1_pulses = np.ones(200)          # flat 1 Hz baseline
    st1 = np.repeat(st1_pulses, 3)         # 3 frames per 50 ms pulse
    md = {"St1TrialVector": st1.tolist(), "Stim2TF": change_size}
    if with_realized:
        fps = 60.0
        n_gray = int(round(2.0 * fps))     # 2 s gray
        n_base = int(round(change_time * fps))
        n_post = int(round(1.0 * fps))
        tf = np.concatenate([np.zeros(n_gray),
                             np.repeat(st1_pulses, 3)[:n_base],
                             np.full(n_post, change_size)])
        vbl = onset - n_gray / fps + np.arange(len(tf)) / fps  # wall-clock-ish, onset at frame n_gray
        # make vbl an arbitrary epoch but with correct deltas:
        vbl = 1.7e9 + np.arange(len(tf)) / fps
        md["TF"] = tf.tolist()
        md["vbl"] = vbl.tolist()
    abs_change = onset + change_time
    abs_rt = onset + reaction_time if outcome in ("FA", "Abort") else onset + change_time + reaction_time
    return SimpleNamespace(trial_index=0, outcome=outcome, change_time=change_time,
                           iti_duration=2.0, reaction_time=reaction_time, change_size=change_size,
                           absolute_change_time=abs_change, absolute_reaction_time=abs_rt,
                           metadata=md)


def test_baseline_onset_ts():
    assert baseline_onset_ts(_trial(change_time=8.0, onset=100.0)) == 108.0 - 8.0

def test_baseline_pulse_values_strided():
    vals = baseline_pulse_values(_trial(st1_pulses=np.arange(10.0)))
    assert np.allclose(vals, np.arange(10.0))   # [::3] of repeat-3 recovers the pulses

def test_n_baseline_samples_hit_vs_fa():
    assert n_baseline_samples(_trial(outcome="Hit", change_time=8.0)) == 160
    assert n_baseline_samples(_trial(outcome="FA", reaction_time=5.0)) == 100

def test_windowed_pulses_respects_margins():
    # Hit: window = [onset+1.0, change-1.0] = [101, 107] -> times 100+k*0.05 in [101,107] -> k in [20,140]
    vals, times = windowed_pulses(_trial(outcome="Hit", change_time=8.0, onset=100.0))
    assert times.min() >= 101.0 - 1e-9 and times.max() <= 107.0 + 1e-9

def test_fast_slow_classification():
    pulses = np.array([1.0, 1.3, 0.7, 1.0] * 60)   # 1.3 -> log2=0.38 fast; 0.7 -> -0.51 slow
    fast, slow = fast_slow_pulse_times(_trial(outcome="Hit", change_time=8.0, st1_pulses=pulses))
    assert fast.size > 0 and slow.size > 0

def test_aligned_regressor_is_log2_meancenterable():
    l2, times = aligned_baseline_regressor(_trial(st1_pulses=np.full(200, 2.0)))
    assert np.allclose(l2, 1.0)   # log2(2/1) = 1

def test_validate_change_anchor_pass():
    ok, mism = validate_change_anchor(_trial(change_size=4.0, with_realized=True))
    assert ok is True and mism < 0.05

def test_validate_change_anchor_skips_small_change():
    ok, mism = validate_change_anchor(_trial(change_size=1.25, with_realized=True))
    assert ok is True and np.isnan(mism)   # not applicable -> pass
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `py -m pytest tests/core/test_stimulus.py -v`
Expected: FAIL — module `stimulus` does not exist.

- [ ] **Step 4: Implement `core/stimulus.py`**

Create `src/visdetect_photom/core/stimulus.py`:

```python
"""Baseline-stimulus (TF-pulse) reconstruction + Input0-anchored alignment (G1).

Places baseline TF pulses on a uniform 50 ms grid off the baseline-onset
(Input0) timestamp — identical convention to the ephys tf_pulse.py — and
validates per-trial timing against the change anchor.
"""
import numpy as np
from visdetect_photom.core.constants import (
    TF_BASELINE_STRIDE, TF_SAMPLE_PERIOD, TF_MIN_AFTER_BASELINE,
    TF_MIN_BEFORE_CHANGE, TF_MIN_BEFORE_OUTCOME_FA_ABORT,
    TF_FAST_THRESH_LOG2, TF_SLOW_THRESH_LOG2, TF_BASE_HZ,
    TF_CHANGE_VALIDATE_MIN_CS, TF_CHANGE_VALIDATE_TOL,
)

_FA_LIKE = ("FA", "Abort")


def baseline_onset_ts(trial):
    """Baseline-grating onset in photometry SystemTimestamp (Input0). None if N/A."""
    if trial.absolute_change_time is None or trial.change_time is None:
        return None
    return float(trial.absolute_change_time - trial.change_time)


def baseline_pulse_values(trial, stride=TF_BASELINE_STRIDE):
    """The 50 ms baseline pulse sequence = St1TrialVector[::stride]. None if missing."""
    md = getattr(trial, "metadata", None) or {}
    st1 = md.get("St1TrialVector")
    if st1 is None:
        return None
    arr = np.asarray(st1, dtype=float).ravel()
    if arr.size == 0:
        return None
    return arr[::stride]


def n_baseline_samples(trial, sample_period=TF_SAMPLE_PERIOD):
    """Number of baseline pulses actually shown before change (go/CR) or lick (FA/abort)."""
    o = trial.outcome
    if o in _FA_LIKE and trial.reaction_time is not None:
        return int(round(trial.reaction_time / sample_period))
    if trial.change_time is not None:
        return int(round(trial.change_time / sample_period))
    return 0


def windowed_pulses(trial, sample_period=TF_SAMPLE_PERIOD):
    """(values, abs_times) for baseline pulses inside the usable, margin-trimmed window."""
    onset = baseline_onset_ts(trial)
    vals = baseline_pulse_values(trial)
    if onset is None or vals is None:
        return np.array([]), np.array([])
    n = min(n_baseline_samples(trial, sample_period), len(vals))
    if n <= 0:
        return np.array([]), np.array([])
    vals = vals[:n]
    times = onset + np.arange(n) * sample_period
    start = onset + TF_MIN_AFTER_BASELINE
    if trial.outcome in _FA_LIKE:
        end = onset + (trial.reaction_time or 0.0) - TF_MIN_BEFORE_OUTCOME_FA_ABORT
    else:
        end = onset + (trial.change_time or 0.0) - TF_MIN_BEFORE_CHANGE
    mask = (times >= start) & (times <= end)
    return vals[mask], times[mask]


def _log2_tf(vals):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(np.asarray(vals, float) / TF_BASE_HZ)


def fast_slow_pulse_times(trial):
    """(fast_times, slow_times) by log2(TF) vs +/-0.25 within the usable window."""
    vals, times = windowed_pulses(trial)
    if vals.size == 0:
        return np.array([]), np.array([])
    l2 = _log2_tf(vals)
    return times[l2 >= TF_FAST_THRESH_LOG2], times[l2 <= TF_SLOW_THRESH_LOG2]


def aligned_baseline_regressor(trial):
    """(log2_tf, abs_times) for the windowed baseline (continuous TRF input)."""
    vals, times = windowed_pulses(trial)
    if vals.size == 0:
        return np.array([]), np.array([])
    l2 = _log2_tf(vals)
    good = np.isfinite(l2)
    return l2[good], times[good]


def validate_change_anchor(trial, tol=TF_CHANGE_VALIDATE_TOL):
    """Best-effort timing QC using realized TF + vbl.

    Returns (ok, mismatch_s). Only applicable when change_size is large enough to
    detect the post-change TF level (>= TF_CHANGE_VALIDATE_MIN_CS); otherwise
    returns (True, nan) = 'not applicable, do not drop'.
    """
    md = getattr(trial, "metadata", None) or {}
    tf, vbl = md.get("TF"), md.get("vbl")
    onset = baseline_onset_ts(trial)
    cs = trial.change_size
    if cs is None or cs < TF_CHANGE_VALIDATE_MIN_CS:
        return True, np.nan
    if tf is None or vbl is None or onset is None or trial.absolute_change_time is None:
        return True, np.nan
    tf = np.asarray(tf, float); vbl = np.asarray(vbl, float)
    if tf.size != vbl.size or tf.size == 0:
        return True, np.nan
    nz = np.where(tf > 0)[0]
    if nz.size == 0:
        return True, np.nan
    onset_frame = nz[0]
    stim2 = md.get("Stim2TF")
    if stim2 is None:
        return True, np.nan
    after = np.arange(tf.size) > onset_frame
    near = np.abs(tf - stim2) <= 0.1 * abs(stim2)
    cand = np.where(after & near)[0]
    if cand.size == 0:
        return True, np.nan
    mapped = onset + (vbl[cand[0]] - vbl[onset_frame])
    mism = abs(mapped - trial.absolute_change_time)
    return (mism <= tol), float(mism)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `py -m pytest tests/core/test_stimulus.py -v`
Expected: PASS (8 passed).

- [ ] **Step 6: Commit**

```bash
git add src/visdetect_photom/core/constants.py src/visdetect_photom/core/stimulus.py tests/core/test_stimulus.py
git commit -m "Add TF-pulse constants and stimulus reconstruction/alignment (G1)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `analysis/tf_kernel.py` — ridge TRF, design, timescale, null

**Files:**
- Create: `src/visdetect_photom/analysis/tf_kernel.py`
- Create: `tests/analysis/test_tf_kernel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_tf_kernel.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_tf_kernel.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement (design + ridge + timescale + null)**

Create `src/visdetect_photom/analysis/tf_kernel.py`:

```python
"""TRF kernel estimation for baseline TF -> dF/F (G1). numpy-only ridge."""
import numpy as np
from visdetect_photom.core.constants import (
    TRF_LAG_MIN, TRF_LAG_MAX, TRF_LAG_STEP, TF_SAMPLE_PERIOD,
)
from visdetect_photom.core.stimulus import aligned_baseline_regressor, validate_change_anchor


def lag_grid():
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_tf_kernel.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/tf_kernel.py tests/analysis/test_tf_kernel.py
git commit -m "Add numpy ridge TRF, design builder, timescale, shuffle null (G1)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Pulse-triggered average + detrend (append to `tf_kernel.py`)

**Files:**
- Modify: `src/visdetect_photom/analysis/tf_kernel.py`
- Modify: `tests/analysis/test_tf_kernel.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_tf_kernel.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_tf_kernel.py -k "pulse or detrend" -v`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement (append to `analysis/tf_kernel.py`)**

Add the import at the top of `tf_kernel.py` (extend the existing constants import):

```python
from visdetect_photom.core.constants import (
    TF_PULSE_PRE_WINDOW, TF_PULSE_POST_WINDOW,
    TF_PULSE_DETREND_BASELINE, TF_PULSE_DETREND_POST,
)
```

Append the functions:

```python
def pulse_triggered_average(signal, timestamps, pulse_times,
                            pre=TF_PULSE_PRE_WINDOW, post=TF_PULSE_POST_WINDOW, fs=100.0):
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
    sem = np.nanstd(rows, axis=0) / np.sqrt(np.maximum(n, 1))
    return t_vec, mean, sem


def detrend_pulse_trace(t_vec, trace,
                        baseline=TF_PULSE_DETREND_BASELINE, post=TF_PULSE_DETREND_POST):
    """Linear-detrend on the baseline window; measure post-pulse peak/trough.

    Ports the ephys detrend_tf_traces. Returns (detrended, z_max_post, z_min_post).
    """
    t = np.asarray(t_vec, float)
    tr = np.asarray(trace, float)
    pre = (t >= baseline[0]) & (t < baseline[1])
    pm = (t >= post[0]) & (t < post[1])
    if pre.sum() < 2:
        zmax = float(np.nanmax(tr[pm])) if pm.any() else np.nan
        zmin = float(np.nanmin(tr[pm])) if pm.any() else np.nan
        return tr, zmax, zmin
    coef = np.polyfit(t[pre], tr[pre], 1)
    d = tr - np.polyval(coef, t)
    zmax = float(np.nanmax(d[pm])) if pm.any() else np.nan
    zmin = float(np.nanmin(d[pm])) if pm.any() else np.nan
    return d, zmax, zmin
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_tf_kernel.py -v`
Expected: PASS (7 passed total).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/tf_kernel.py tests/analysis/test_tf_kernel.py
git commit -m "Add pulse-triggered average and detrend (G1)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Script `09_tf_pulse_encoding.py` + smoke test

**Files:**
- Create: `scripts/analysis/photometry/09_tf_pulse_encoding.py`
- Create: `tests/scripts/test_09_smoke.py`

- [ ] **Step 1: Implement the script**

Create `scripts/analysis/photometry/09_tf_pulse_encoding.py`:

```python
"""G1 — TF-Pulse Evidence Encoding (D1 vs D2 baseline-TF kernel + pulse-triggered).

D1 and D2 are DIFFERENT animals: all comparisons are GROUP-LEVEL.
The kernel reflects neural response convolved with GCaMP kinetics (timescale = upper bound).

Usage:
    py scripts/analysis/photometry/09_tf_pulse_encoding.py
    py scripts/analysis/photometry/09_tf_pulse_encoding.py --no-qc
    py scripts/analysis/photometry/09_tf_pulse_encoding.py --state-filter Engaged --state-results-dir results/hmm/BG_013
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import GENOTYPE_COLORS, MIN_PHOTOM_CSV_BYTES, get_roi_region
from visdetect_photom.core.qc import compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres
from visdetect_photom.core.staging import load_staging_manifest, excluded_mice
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.analysis.group_statistics import pushpull_sign_contrast, format_stats_table
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider, filter_trials_by_state
from visdetect_photom.analysis.tf_kernel import (
    lag_grid, build_region_design, fit_trf, kernel_timescale, shuffle_null,
    pulse_triggered_average,
)
from visdetect_photom.core.stimulus import fast_slow_pulse_times

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _subject_full(sid):
    s = str(sid)
    return f"BG_{s.zfill(3)}" if (not s.startswith("BG_") and s.isdigit()) else s


def _region_sources(session, subj, use_qc):
    if use_qc:
        merged = merge_hemispheres(session, qc_results=compute_session_roi_qc(session))
        return {r: (m["signal"], m["timestamps"]) for r, m in merged.items()}
    by = defaultdict(list)
    for roi, tr in session.photometry_data.items():
        region = get_roi_region(roi, subj)
        if region:
            by[region.rsplit("_", 1)[0]].append((tr.signal, tr.timestamps))
    out = {}
    for r, trs in by.items():
        if len(trs) == 1:
            out[r] = trs[0]
        elif len(trs) >= 2:
            n = min(len(s) for s, _ in trs)
            out[r] = (np.mean([s[:n] for s, _ in trs], axis=0), trs[0][1][:n])
    return out


def collect(session_files, *, use_qc, state_provider, keep_states, max_sessions):
    lags = lag_grid()
    # per (genotype, region): {subject: list of kernels}; and pulse-triggered traces
    kern = defaultdict(lambda: defaultdict(list))
    pta = defaultdict(lambda: defaultdict(lambda: {"fast": [], "slow": []}))
    ptv = {"t": None}
    n = 0
    for sf in session_files:
        if max_sessions and n >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception:
            continue
        geno = get_genotype(sess.subject_id)
        if geno == "Unknown":
            continue
        subj = _subject_full(sess.subject_id)
        if use_qc and not check_behavioral_engagement(sess)["pass"]:
            continue
        keep = None
        if state_provider is not None and keep_states is not None:
            keep = filter_trials_by_state(sess, state_provider, keep_states)
        sources = _region_sources(sess, subj, use_qc)
        for region, (sig, ts) in sources.items():
            segs = build_region_design(sess, sig, ts, state_keep=keep, validate=True)
            if len(segs) >= 1:
                _, k = fit_trf(segs, lags=lags)
                if np.any(np.isfinite(k)):
                    kern[(geno, region)][subj].append(k)
            # pulse-triggered companion
            fast_t, slow_t = [], []
            for tr in sess.trials:
                if keep is not None and tr.trial_index not in keep:
                    continue
                f, s = fast_slow_pulse_times(tr)
                fast_t.append(f); slow_t.append(s)
            fast_t = np.concatenate(fast_t) if fast_t else np.array([])
            slow_t = np.concatenate(slow_t) if slow_t else np.array([])
            for label, times in (("fast", fast_t), ("slow", slow_t)):
                res = pulse_triggered_average(sig, ts, times)
                if res is not None:
                    ptv["t"], mean, _ = res
                    pta[(geno, region)][subj][label].append(mean)
        n += 1
        if n % 20 == 0:
            logging.info(f"  processed {n}")
    return lags, kern, pta, ptv["t"]


def _per_mouse_mean(subj_map):
    """{subject: [arrays]} -> list of (subject, mean_array)."""
    return [(s, np.nanmean(np.array(a), axis=0)) for s, a in subj_map.items() if a]


def main():
    ap = argparse.ArgumentParser(description="G1: TF-pulse evidence encoding")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "G1_tf_pulse_encoding"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None)
    ap.add_argument("--state-results-dir", default=None)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    use_qc = not args.no_qc
    out = Path(args.output_dir)
    files = io.find_all_sessions(args.root_dir, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(files)} session files.")

    excl = excluded_mice(load_staging_manifest())
    files = [f for f in files]  # mouse-level exclusion handled after load (subject from path is reliable)

    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir"); sys.exit(1)
        provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
    else:
        provider, keep_states = PooledStateProvider(), ["All"]

    # exclude mice: filter sessions whose subject is excluded
    def _excluded(sf):
        sid = sf.get("trials", "")
        for m in excl:
            if m.replace("BG_", "") in str(sid) or m in str(sid):
                return True
        return False
    files = [f for f in files if not _excluded(f)]
    if excl:
        logging.info(f"Excluding mice: {sorted(excl)}")

    lags, kern, pta, ptv_t = collect(files, use_qc=use_qc, state_provider=provider,
                                     keep_states=keep_states, max_sessions=args.max_sessions)
    if not kern:
        logging.error("No kernels computed."); sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)

    # ── per-mouse kernel summary + stats ──
    rows, stat_rows = [], []
    regions = sorted({r for (_, r) in kern})
    for region in regions:
        for geno in ("D1", "D2"):
            for subj, mean_k in _per_mouse_mean(kern.get((geno, region), {})):
                ts_ = kernel_timescale(lags, mean_k)
                rows.append({"subject_id": subj, "genotype": geno, "region": region, **ts_})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "G1_kernels.csv", index=False)

    for region in regions:
        sub = metrics[metrics["region"] == region]
        d1 = sub[sub["genotype"] == "D1"]["signed_peak"].values
        d2 = sub[sub["genotype"] == "D2"]["signed_peak"].values
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "metric": "kernel_signed_peak"})
        stat_rows.append(res)
    if stat_rows:
        format_stats_table(stat_rows, save_path=str(out / "G1_stats.csv"))

    # ── figures: per region (kernel D1 vs D2 + pulse-triggered) ──
    for region in regions:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"G1 — TF-pulse encoding — {region}\n(D1/D2 different animals; kernel = neural ⊛ GCaMP)", fontsize=11)
        # kernel
        ax = axes[0]
        for geno in ("D1", "D2"):
            km = _per_mouse_mean(kern.get((geno, region), {}))
            if not km:
                continue
            K = np.array([k for _, k in km])
            mean = np.nanmean(K, axis=0)
            sem = np.nanstd(K, axis=0) / np.sqrt(max(K.shape[0], 1))
            c = GENOTYPE_COLORS[geno]
            ax.plot(lags, mean, color=c, lw=1.5, label=f"{geno} ({K.shape[0]} mice)")
            ax.fill_between(lags, mean - sem, mean + sem, color=c, alpha=0.2)
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6); ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("Lag (s): TF → dF/F"); ax.set_ylabel("kernel weight"); ax.set_title("TRF kernel")
        ax.legend(fontsize=8); sns.despine(ax=ax)
        # pulse-triggered (fast solid / slow dashed)
        ax = axes[1]
        if ptv_t is not None:
            for geno in ("D1", "D2"):
                c = GENOTYPE_COLORS[geno]
                for label, style in (("fast", "-"), ("slow", "--")):
                    traces = [np.nanmean(np.array(v[label]), axis=0)
                              for v in pta.get((geno, region), {}).values() if v[label]]
                    if traces:
                        m = np.nanmean(np.array(traces), axis=0)
                        ax.plot(ptv_t, m, color=c, ls=style, lw=1.3,
                                label=f"{geno} {label}")
            ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.set_xlabel("Time from pulse (s)"); ax.set_ylabel("z-dF/F (pre-pulse)")
            ax.set_title("Fast vs slow pulse-triggered"); ax.legend(fontsize=7); sns.despine(ax=ax)
        p = out / f"G1_{region}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        logging.info(f"Saved {p}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke test**

Create `tests/scripts/test_09_smoke.py`:

```python
import os
import subprocess
import sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "analysis", "photometry", "09_tf_pulse_encoding.py")
DATA = os.path.join(REPO, "photom_data")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="photom_data/ not present")
def test_runs_on_small_subset(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--max_sessions", "4", "--output_dir", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "G1_kernels.csv").exists()
```

- [ ] **Step 3: Run the smoke test**

Run: `py -m pytest tests/scripts/test_09_smoke.py -v`
Expected: PASS on local `photom_data/` (or SKIP if absent). If FAIL, read `proc.stderr` and fix.

- [ ] **Step 4: Run the full suite**

Run: `py -m pytest tests/ -v`
Expected: all PASS/SKIP.

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/photometry/09_tf_pulse_encoding.py tests/scripts/test_09_smoke.py
git commit -m "Add G1 script 09 (TF-pulse encoding CLI + figures) and smoke test" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during authoring)

**Spec coverage:** TRF kernel → Task 2 (`fit_trf`, `build_region_design`, segments avoid cross-trial lag bleed). Pulse-triggered + detrend → Task 3. Uniform-50 ms Input0 alignment + windowing + fast/slow ±0.25 + change-anchor validation → Task 1. Integration timescale → Task 2 (`kernel_timescale`). Shuffle null → Task 2. Scope (regions, exclusion, state seam) → Task 4. D1-vs-D2 push–pull → Task 4 (`pushpull_sign_contrast` on `signed_peak`). All covered.

**Placeholder scan:** none — all code complete; ridge is numpy-only (no sklearn).

**Type consistency:** `build_region_design` returns `list[(x_seg, y_seg)]`; `fit_trf`/`shuffle_null` consume that list and return `(lags, kernel|band)`; `kernel_timescale` returns `{signed_peak, peak_lag, com}` used verbatim in the script's metrics rows and `pushpull_sign_contrast` (keys `d1_mean/d2_mean/d1_sign/d2_sign/opposite_sign/p`). `pulse_triggered_average` returns `(t_vec, mean, sem)`; the script consumes `mean`. Consistent.

**Robustness notes (not blockers):** (1) `validate_change_anchor` only fires for `change_size ≥ 2.0` (small changes can't be distinguished from baseline) — returns `(True, nan)` = pass otherwise. (2) Short FA/abort windows may yield no segments → trial skipped. (3) Learning-vs-Expert robustness split is a follow-up rerun using `core.staging.get_session_stage` to subset sessions (same code path); add as a `--stage` filter if desired. (4) GCaMP-kinetics caveat is interpretive (caption), not code.
