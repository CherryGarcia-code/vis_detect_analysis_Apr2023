# C1 — FA Suppression-Failure (MOs–D2 brake) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether waiting/decision-period bulk D1/D2 striatal activity predicts withhold-vs-lick, via per-trial waiting-period scalars, a group push–pull contrast, and a single-trial AUROC decoder, across two outcome tracks and two window schemes.

**Architecture:** Approach B — a new `analysis/suppression.py` holds C1 primitives (track grouping, two window extractors, per-mouse Δ + AUROC, proficiency split); a thin script `11_fa_suppression.py` consumes them. Reuses all merged C2 infrastructure (`group_statistics`, `state_provider`, `core/staging`, `qc`, `statistics`, `group_utils`). Adds one shared `qc.region_sources()` and one `group_statistics.auroc_score()`. The merged/tested `geometry.py` (C2) is **not modified**.

**Tech Stack:** Python (`py` on Windows), numpy<2, pandas<2.3, scipy<2, matplotlib<3.9, seaborn; pytest. src-layout (tests find the package via `tests/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-06-11-c1-fa-suppression-design.md`

**Implementation hygiene (parallel chats):** This repo is worked on by multiple chats sharing one tree. Implement on a dedicated branch/worktree off `main` (e.g. `git worktree add ../vis_detect_analysis_Apr2023-c1impl -b analysis/c1-fa-suppression main`) and **verify the branch (`git branch --show-current`) before every commit**. CRLF warnings from git on Windows are harmless.

---

## Reference: existing APIs this plan reuses (verified)

- `visdetect_photom.core.io.find_all_sessions(root, recursive=True, min_photom_bytes=...)` → list of file-dicts; `load_session_from_files(file_dict)` → `Session`.
- `Session`: `.subject_id` (e.g. `"013"`), `.session_id` (e.g. `"013_20231205"`), `.session_date`, `.trials` (list of `Trial`), `.photometry_data` (`{roi_name: PhotometryTrace}`). `Trial`: `.trial_index`, `.outcome` ∈ {`Hit`,`Miss`,`FA`,`CR`,`Abort`}, `.change_time` (stimT), `.change_size` (Stim2TF), `.reaction_time`, `.absolute_change_time`, `.absolute_reaction_time`. `PhotometryTrace`: `.signal` (session-z-scored dF/F by default), `.timestamps`, `.signal_type`.
- `visdetect_photom.core.constants`: `CATCH_THRESHOLD` (1.01), `FA_RT_SPLIT` (3.0), `CHANGE_SIZES`, `GENOTYPE_COLORS`, `MIN_PHOTOM_CSV_BYTES`, `get_roi_region(roi, subject_full)`.
- `visdetect_photom.core.qc`: `compute_session_roi_qc(session)` → `{roi: qc}`; `merge_hemispheres(session, qc_results=qc)` → `{region: {'signal','timestamps','source','rois_used'}}` (region names are base, e.g. `'DMS'`, `'VLS'`, `'VMS'`).
- `visdetect_photom.core.staging`: `load_staging_manifest()`, `get_session_stage(session, manifest)` → stage string, `excluded_mice(manifest)` → set of `BG_0XX`.
- `visdetect_photom.analysis.state_provider`: `PooledStateProvider` (all trials labeled `"All"`), `HMMStateProvider`, `filter_trials_by_state(session, provider, keep_states)` → set of trial indices.
- `visdetect_photom.analysis.group_utils.get_genotype(subject_full)` → `"D1"|"D2"|"Unknown"`.
- `visdetect_photom.analysis.group_statistics`: `bootstrap_ci(data, func=np.nanmean, n_boot=1000, seed=42, ci=95.0)` → `{observed,ci_lo,ci_hi,n}`; `permutation_test(x, y, n_perm=10000, seed=42)` → `{observed,p,n1,n2}`; `pushpull_sign_contrast(d1_vals, d2_vals)` → `{d1_mean,d2_mean,d1_sign,d2_sign,opposite_sign,p,rank_biserial_r,...}`; `mannwhitney_with_effect_size`.
- `visdetect_photom.analysis.statistics`: `extract_peth(signal, timestamps, event_times, window=, baseline_window=)` → `(time_axis, peth_matrix)`; `calculate_sdt_metrics(outcomes, change_sizes)` → `{d_prime, sdt_hit_rate, sdt_fa_rate, ...}`.
- **Test convention:** synthetic sessions are `types.SimpleNamespace` with the needed attributes; call core functions with `use_qc=False` to bypass the QC pipeline (see `tests/analysis/test_geometry.py`).

---

## File structure

| File | Responsibility | New/changed |
|---|---|---|
| `src/visdetect_photom/analysis/group_statistics.py` | add `auroc_score(scores, labels)` | MODIFY (append) |
| `src/visdetect_photom/core/qc.py` | add public `region_sources(session, use_qc=True)` | MODIFY (append) |
| `src/visdetect_photom/core/constants.py` | add C1 window constants block | MODIFY (append) |
| `src/visdetect_photom/analysis/suppression.py` | C1 core: records, window extractors, dataset, Δ/AUROC, proficiency | **CREATE** |
| `scripts/analysis/photometry/11_fa_suppression.py` | thin consumer + figures | **CREATE** |
| `tests/analysis/test_auroc.py` | `auroc_score` | **CREATE** |
| `tests/core/test_region_sources.py` | `region_sources` | **CREATE** |
| `tests/analysis/test_suppression_records.py` | records + `window_mean` | **CREATE** |
| `tests/analysis/test_suppression_scheme1.py` | Scheme-1 extractor | **CREATE** |
| `tests/analysis/test_suppression_scheme3.py` | Scheme-3 hazard extractor | **CREATE** |
| `tests/analysis/test_suppression_dataset.py` | dataset builder | **CREATE** |
| `tests/analysis/test_suppression_stats.py` | Δ/AUROC aggregation + push–pull | **CREATE** |
| `tests/analysis/test_suppression_proficiency.py` | proficiency binning | **CREATE** |
| `tests/scripts/test_11_smoke.py` | script smoke test | **CREATE** |

---

## Task 1: `auroc_score` in group_statistics

**Files:**
- Modify: `src/visdetect_photom/analysis/group_statistics.py` (append)
- Test: `tests/analysis/test_auroc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_auroc.py
import numpy as np
from visdetect_photom.analysis.group_statistics import auroc_score


def test_auroc_perfect_separation_positive():
    # positive class (1) clearly higher than negative (0) -> AUROC ~ 1.0
    scores = np.array([5.0, 6.0, 7.0, 1.0, 2.0, 3.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert auroc_score(scores, labels) == 1.0

def test_auroc_perfect_separation_negative():
    # positive class lower than negative -> AUROC ~ 0.0
    scores = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert auroc_score(scores, labels) == 0.0

def test_auroc_chance_when_interleaved():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([1, 0, 1, 0])
    assert auroc_score(scores, labels) == 0.5

def test_auroc_nan_when_one_class_empty():
    assert np.isnan(auroc_score(np.array([1.0, 2.0]), np.array([1, 1])))

def test_auroc_ignores_nonfinite():
    scores = np.array([5.0, np.nan, 7.0, 1.0, 2.0])
    labels = np.array([1, 1, 1, 0, 0])
    assert auroc_score(scores, labels) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_auroc.py -v`
Expected: FAIL — `ImportError: cannot import name 'auroc_score'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/analysis/group_statistics.py`:

```python
# ── AUROC (single-trial discriminability) ────────────────────

def auroc_score(scores, labels) -> float:
    """Area under ROC for score discriminating positive class (label==1) from
    negative (label==0). AUROC = P(score_pos > score_neg) via the Mann-Whitney U
    statistic: U / (n_pos * n_neg). Non-finite scores are dropped. Returns NaN if
    either class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    U, _ = sp_stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(U / (pos.size * neg.size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_auroc.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/group_statistics.py tests/analysis/test_auroc.py
git commit -m "feat(c1): add auroc_score single-trial discriminability metric"
```

---

## Task 2: `region_sources` in qc

**Files:**
- Modify: `src/visdetect_photom/core/qc.py` (append)
- Test: `tests/core/test_region_sources.py`

Promotes the (currently private) `geometry._region_sources` logic to a public, shared `qc.region_sources`. `geometry.py` is left untouched (it keeps its own copy); this duplication is intentional and temporary — geometry may adopt this later, but we do not modify merged C2 code now.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_region_sources.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/core/test_region_sources.py -v`
Expected: FAIL — `ImportError: cannot import name 'region_sources'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/core/qc.py`:

```python
# ── Region source resolution (shared by analyses) ────────────
from collections import defaultdict as _defaultdict


def region_sources(session, use_qc: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {region_base: (signal, timestamps)} for a session.

    use_qc=True : QC each ROI, then hemisphere-merge (both pass -> average,
                  one passes -> use it, neither -> skip).
    use_qc=False: average all ROIs that map to the same base region (no QC gate).
    """
    if use_qc:
        qc = compute_session_roi_qc(session)
        merged = merge_hemispheres(session, qc_results=qc)
        return {r: (m["signal"], m["timestamps"]) for r, m in merged.items()}

    subject_id = getattr(session, "subject_id", None)
    if subject_id and not subject_id.startswith("BG_"):
        subject_full = f"BG_{subject_id.zfill(3)}" if subject_id.isdigit() else subject_id
    else:
        subject_full = subject_id

    by_region = _defaultdict(list)
    for roi_name, trace in session.photometry_data.items():
        region = get_roi_region(roi_name, subject_full)
        if region is None:
            continue
        by_region[region.rsplit("_", 1)[0]].append((trace.signal, trace.timestamps))

    sources = {}
    for region, traces in by_region.items():
        if len(traces) == 1:
            sources[region] = traces[0]
        elif len(traces) >= 2:
            n = min(len(s) for s, _ in traces)
            avg = np.mean([s[:n] for s, _ in traces], axis=0)
            sources[region] = (avg, traces[0][1][:n])
    return sources
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/core/test_region_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/core/qc.py tests/core/test_region_sources.py
git commit -m "feat(c1): add shared qc.region_sources helper"
```

---

## Task 3: C1 constants, `window_mean`, and `trial_waiting_records`

**Files:**
- Modify: `src/visdetect_photom/core/constants.py` (append)
- Create: `src/visdetect_photom/analysis/suppression.py`
- Test: `tests/analysis/test_suppression_records.py`

- [ ] **Step 1: Add C1 constants**

Append to `src/visdetect_photom/core/constants.py`:

```python
# ── C1 waiting-period (FA suppression) windows ────────────────
SCHEME1_WINDOW        = (2.0, 3.0)  # (w0, w0+L) s after grating onset (clean late-waiting)
SCHEME1_MOTOR_BUFFER  = 1.0         # s; Scheme-1 window must end this long before an action
SCHEME3_L             = 1.0         # s; Scheme-3 window length
SCHEME3_BUFFER        = 0.5         # s; Scheme-3 motor-execution guard before the action
HAZARD_RESAMPLES      = 20          # withhold pseudo-action-time draws (Scheme 3)
HAZARD_SEED           = 42
MIN_TRIALS_PER_GROUP  = 8           # min finite scalars per (mouse, region, group) cell
PROF_MIN_SESSIONS     = 3           # min sessions per staging bin to use the staging split
WINDOW_MIN_SAMPLES    = 3           # min finite samples for a window-mean to be valid
```

- [ ] **Step 2: Write the failing test**

```python
# tests/analysis/test_suppression_records.py
import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import window_mean, trial_waiting_records


def test_window_mean_basic():
    ts = np.arange(0, 10, 0.01)
    sig = np.full_like(ts, 3.0)
    assert window_mean(sig, ts, 2.0, 3.0) == 3.0

def test_window_mean_nan_when_too_few_samples():
    ts = np.array([0.0, 5.0, 9.0])
    sig = np.array([1.0, 1.0, 1.0])
    assert np.isnan(window_mean(sig, ts, 2.0, 3.0))  # zero samples in window

def _trial(idx, outcome, change_time, change_size, abs_change, abs_rt):
    return SimpleNamespace(trial_index=idx, outcome=outcome, change_time=change_time,
                           change_size=change_size, reaction_time=None,
                           absolute_change_time=abs_change, absolute_reaction_time=abs_rt)

def _session(trials):
    return SimpleNamespace(subject_id="013", session_id="013_x", trials=trials,
                           photometry_data={})

def test_behavioral_fa_track_grouping():
    trials = [
        _trial(0, "FA",   change_time=6.0, change_size=2.0, abs_change=26.0, abs_rt=25.0),
        _trial(1, "Hit",  change_time=4.0, change_size=2.0, abs_change=14.0, abs_rt=14.5),
        _trial(2, "Miss", change_time=4.0, change_size=2.0, abs_change=44.0, abs_rt=None),
        _trial(3, "CR",   change_time=4.0, change_size=1.0, abs_change=54.0, abs_rt=None),
        _trial(4, "Abort",change_time=5.0, change_size=2.0, abs_change=64.0, abs_rt=61.0),
    ]
    recs = {r["trial_index"]: r for r in trial_waiting_records(_session(trials), "behavioral_fa")}
    assert recs[0]["group"] == "lick"
    assert recs[1]["group"] == "withhold" and recs[2]["group"] == "withhold"
    assert recs[3]["group"] == "withhold"
    assert recs[4]["group"] == "abort"
    # grating onset recovered as abs_change - change_time; lick_elapsed = abs_rt - onset
    assert recs[0]["onset_abs"] == 20.0
    assert recs[0]["lick_elapsed"] == 5.0

def test_sdt_fa_track_grouping():
    trials = [
        _trial(0, "Hit",  change_time=4.0, change_size=1.0, abs_change=14.0, abs_rt=18.0),  # catch lick
        _trial(1, "Miss", change_time=4.0, change_size=1.0, abs_change=24.0, abs_rt=None),  # SDT-CR
        _trial(2, "Hit",  change_time=4.0, change_size=2.0, abs_change=34.0, abs_rt=34.5),  # go hit -> None
        _trial(3, "FA",   change_time=6.0, change_size=2.0, abs_change=46.0, abs_rt=42.0),  # behavioral FA -> None
    ]
    recs = {r["trial_index"]: r for r in trial_waiting_records(_session(trials), "sdt_fa")}
    assert recs[0]["group"] == "lick"
    assert recs[1]["group"] == "withhold"
    assert recs[2]["group"] is None and recs[3]["group"] is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_suppression_records.py -v`
Expected: FAIL — `ModuleNotFoundError: ... suppression`.

- [ ] **Step 4: Write minimal implementation**

Create `src/visdetect_photom/analysis/suppression.py`:

```python
"""C1 — waiting/decision-period suppression-failure analysis.

Per-trial waiting-period scalars (two window schemes), two outcome tracks
(behavioral FA + SDT-FA control), per-mouse Δ(withhold-lick) and single-trial
AUROC, and a coarse proficiency split. Thin script 11 consumes this. See
docs/superpowers/specs/2026-06-11-c1-fa-suppression-design.md.

D1 and D2 are DIFFERENT animals: every D1-vs-D2 result is a GROUP-LEVEL sign
contrast, never within-animal anticorrelation.
"""
import numpy as np

from visdetect_photom.core.constants import (
    CATCH_THRESHOLD, WINDOW_MIN_SAMPLES,
)

# Groups that represent a premature action (have an action time = lick_elapsed)
_ACTION_GROUPS = ("lick", "abort")


def _subject_full(subject_id):
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def window_mean(signal, timestamps, t_start, t_end, min_samples=WINDOW_MIN_SAMPLES):
    """Mean of finite signal samples with t in [t_start, t_end]; NaN if < min_samples."""
    signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    seg = signal[mask]
    seg = seg[np.isfinite(seg)]
    if seg.size < min_samples:
        return np.nan
    return float(np.mean(seg))


def _group_for(outcome, change_size, track):
    """Map (outcome, change_size) to a C1 group for the given track, or None."""
    if track == "behavioral_fa":
        if outcome == "FA":
            return "lick"
        if outcome in ("Hit", "Miss", "CR"):
            return "withhold"
        if outcome == "Abort":
            return "abort"
        return None
    if track == "sdt_fa":
        is_catch = change_size is not None and change_size <= CATCH_THRESHOLD
        if not is_catch:
            return None
        if outcome == "Hit":
            return "lick"
        if outcome == "Miss":
            return "withhold"
        return None
    raise ValueError(f"unknown track: {track!r}")


def trial_waiting_records(session, track, keep=None):
    """List of per-trial dicts for a track.

    Each record: trial_index, group ('lick'|'withhold'|'abort'|None), onset_abs
    (grating onset = absolute_change_time - change_time), change_time, lick_abs
    (absolute_reaction_time or NaN), lick_elapsed (lick_abs - onset_abs or NaN).
    Records whose grating onset cannot be recovered are skipped.
    `keep`: optional set of trial indices to retain (state filtering).
    """
    out = []
    for t in session.trials:
        if keep is not None and t.trial_index not in keep:
            continue
        if t.absolute_change_time is None or t.change_time is None:
            continue
        group = _group_for(t.outcome, t.change_size, track)
        if group is None:
            continue
        onset_abs = float(t.absolute_change_time) - float(t.change_time)
        lick_abs = (float(t.absolute_reaction_time)
                    if t.absolute_reaction_time is not None else np.nan)
        lick_elapsed = lick_abs - onset_abs if np.isfinite(lick_abs) else np.nan
        out.append({
            "trial_index": t.trial_index, "group": group, "onset_abs": onset_abs,
            "change_time": float(t.change_time), "lick_abs": lick_abs,
            "lick_elapsed": lick_elapsed,
        })
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_suppression_records.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/visdetect_photom/core/constants.py src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_records.py
git commit -m "feat(c1): add window_mean and track grouping (behavioral FA + SDT-FA)"
```

---

## Task 4: Scheme-1 fixed-window scalar

**Files:**
- Modify: `src/visdetect_photom/analysis/suppression.py` (append)
- Test: `tests/analysis/test_suppression_scheme1.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_suppression_scheme1.py
import numpy as np
from visdetect_photom.analysis.suppression import scheme1_scalar


def _rec(group, onset_abs, change_time, lick_elapsed=np.nan):
    return {"group": group, "onset_abs": onset_abs, "change_time": change_time,
            "lick_abs": onset_abs + lick_elapsed if np.isfinite(lick_elapsed) else np.nan,
            "lick_elapsed": lick_elapsed}

TS = np.arange(0, 100, 0.01)
SIG = np.full_like(TS, 4.0)


def test_scheme1_withhold_included_before_change():
    # window (2,3) ends before change_time=4 -> valid, mean=4.0
    r = _rec("withhold", onset_abs=10.0, change_time=4.0)
    assert scheme1_scalar(r, SIG, TS) == 4.0

def test_scheme1_withhold_excluded_change_too_early():
    # change_time=2.5 < window end (3) -> NaN
    r = _rec("withhold", onset_abs=10.0, change_time=2.5)
    assert np.isnan(scheme1_scalar(r, SIG, TS))

def test_scheme1_lick_included_with_motor_buffer():
    # lick_elapsed=4.5 >= w1(3)+buffer(1)=4 -> valid
    r = _rec("lick", onset_abs=20.0, change_time=6.0, lick_elapsed=4.5)
    assert scheme1_scalar(r, SIG, TS) == 4.0

def test_scheme1_lick_excluded_when_lick_too_soon():
    # lick_elapsed=3.5 < 4 -> NaN (no clean pre-lick window; e.g. impulsive early FA)
    r = _rec("lick", onset_abs=20.0, change_time=6.0, lick_elapsed=3.5)
    assert np.isnan(scheme1_scalar(r, SIG, TS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_suppression_scheme1.py -v`
Expected: FAIL — `ImportError: cannot import name 'scheme1_scalar'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/analysis/suppression.py` (add the constants import at the top of the existing import from constants):

```python
from visdetect_photom.core.constants import (
    SCHEME1_WINDOW, SCHEME1_MOTOR_BUFFER,
)


def scheme1_scalar(record, signal, timestamps,
                   window=SCHEME1_WINDOW, motor_buffer=SCHEME1_MOTOR_BUFFER):
    """Baseline-onset-anchored fixed-window mean for one trial, or NaN if excluded.

    Window [onset+w0, onset+w1]. Excluded unless it ends before the change
    (change_time >= w1); for action groups (lick/abort) it must also end
    motor_buffer before the action (lick_elapsed >= w1 + motor_buffer).
    """
    w0, w1 = window
    if record["change_time"] is None or record["change_time"] < w1:
        return np.nan
    if record["group"] in _ACTION_GROUPS:
        le = record["lick_elapsed"]
        if not np.isfinite(le) or le < w1 + motor_buffer:
            return np.nan
    t0 = record["onset_abs"] + w0
    t1 = record["onset_abs"] + w1
    return window_mean(signal, timestamps, t0, t1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_suppression_scheme1.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_scheme1.py
git commit -m "feat(c1): add Scheme-1 baseline-onset fixed-window scalar"
```

---

## Task 5: Scheme-3 hazard-time-matched scalars

**Files:**
- Modify: `src/visdetect_photom/analysis/suppression.py` (append)
- Test: `tests/analysis/test_suppression_scheme3.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_suppression_scheme3.py
import numpy as np
from visdetect_photom.analysis.suppression import scheme3_scalars


def _rec(group, onset_abs, change_time, lick_elapsed=np.nan):
    return {"trial_index": 0, "group": group, "onset_abs": onset_abs,
            "change_time": change_time,
            "lick_abs": onset_abs + lick_elapsed if np.isfinite(lick_elapsed) else np.nan,
            "lick_elapsed": lick_elapsed}

TS = np.arange(0, 200, 0.01)
SIG = np.full_like(TS, 5.0)


def test_scheme3_lick_window_ends_before_lick():
    # lick at elapsed 4.0, buffer 0.5, L 1.0 -> window [2.5, 3.5] elapsed -> valid, 5.0
    a = _rec("lick", onset_abs=10.0, change_time=8.0, lick_elapsed=4.0); a["trial_index"] = 1
    w = _rec("withhold", onset_abs=50.0, change_time=8.0); w["trial_index"] = 2
    lick_vals, wh_vals = scheme3_scalars([a], [w], SIG, TS)
    assert dict(lick_vals)[1] == 5.0

def test_scheme3_withhold_hazard_matched_value():
    a = _rec("lick", onset_abs=10.0, change_time=8.0, lick_elapsed=4.0); a["trial_index"] = 1
    w = _rec("withhold", onset_abs=50.0, change_time=8.0); w["trial_index"] = 2
    _, wh_vals = scheme3_scalars([a], [w], SIG, TS)
    assert dict(wh_vals)[2] == 5.0  # constant signal -> any matched window = 5.0

def test_scheme3_withhold_nan_when_no_admissible_tau():
    # only lick elapsed is 7.9; withhold change_time=2.0 -> tau(7.9) > change -> no valid draw
    a = _rec("lick", onset_abs=10.0, change_time=10.0, lick_elapsed=7.9); a["trial_index"] = 1
    w = _rec("withhold", onset_abs=50.0, change_time=2.0); w["trial_index"] = 2
    _, wh_vals = scheme3_scalars([a], [w], SIG, TS)
    assert np.isnan(dict(wh_vals)[2])

def test_scheme3_is_deterministic():
    a = _rec("lick", onset_abs=10.0, change_time=8.0, lick_elapsed=4.0); a["trial_index"] = 1
    b = _rec("lick", onset_abs=30.0, change_time=8.0, lick_elapsed=5.0); b["trial_index"] = 3
    w = _rec("withhold", onset_abs=50.0, change_time=8.0); w["trial_index"] = 2
    r1 = scheme3_scalars([a, b], [w], SIG, TS)
    r2 = scheme3_scalars([a, b], [w], SIG, TS)
    assert dict(r1[1]) == dict(r2[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_suppression_scheme3.py -v`
Expected: FAIL — `ImportError: cannot import name 'scheme3_scalars'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/analysis/suppression.py` (extend the constants import with `SCHEME3_L, SCHEME3_BUFFER, HAZARD_RESAMPLES, HAZARD_SEED`):

```python
from visdetect_photom.core.constants import (
    SCHEME3_L, SCHEME3_BUFFER, HAZARD_RESAMPLES, HAZARD_SEED,
)


def scheme3_scalars(action_records, withhold_records, signal, timestamps,
                    L=SCHEME3_L, buffer=SCHEME3_BUFFER,
                    n_resample=HAZARD_RESAMPLES, seed=HAZARD_SEED):
    """Hazard-time-matched waiting-period scalars (Scheme 3).

    action_records: premature-action trials (FA licks or aborts). Window
        [act-buffer-L, act-buffer], ending `buffer` before the action.
    withhold_records: trials that reached the change. For each, draw
        `n_resample` pseudo-action elapsed-times from the action group's
        elapsed-time distribution (truncated to <= change_time and window
        start >= 0), average the per-draw window means. Deterministic (seed).

    Returns (action_vals, withhold_vals), each a list of (trial_index, scalar).
    """
    action_vals = []
    elapsed_pool = []
    for r in action_records:
        le = r["lick_elapsed"]
        if np.isfinite(le):
            elapsed_pool.append(le)
        if not np.isfinite(le) or (le - buffer - L) < 0:
            action_vals.append((r["trial_index"], np.nan))
            continue
        ws = r["onset_abs"] + le - buffer - L
        we = r["onset_abs"] + le - buffer
        action_vals.append((r["trial_index"], window_mean(signal, timestamps, ws, we)))

    pool = np.asarray(elapsed_pool, dtype=float)
    rng = np.random.default_rng(seed)
    withhold_vals = []
    for r in withhold_records:
        if pool.size == 0 or not np.isfinite(r["change_time"]):
            withhold_vals.append((r["trial_index"], np.nan))
            continue
        draws = rng.choice(pool, size=n_resample, replace=True)
        means = []
        for tau in draws:
            if tau > r["change_time"] or (tau - buffer - L) < 0:
                continue
            ws = r["onset_abs"] + tau - buffer - L
            we = r["onset_abs"] + tau - buffer
            m = window_mean(signal, timestamps, ws, we)
            if np.isfinite(m):
                means.append(m)
        withhold_vals.append((r["trial_index"],
                              float(np.mean(means)) if means else np.nan))
    return action_vals, withhold_vals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_suppression_scheme3.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_scheme3.py
git commit -m "feat(c1): add Scheme-3 hazard-time-matched scalars"
```

---

## Task 6: Session + multi-session dataset builders

**Files:**
- Modify: `src/visdetect_photom/analysis/suppression.py` (append)
- Test: `tests/analysis/test_suppression_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_suppression_dataset.py
import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import (
    build_session_scalars, build_suppression_dataset,
)


def _trace():
    ts = np.arange(0, 80, 0.01)
    return ts, np.full_like(ts, 2.0)


def _trial(idx, outcome, change_time, change_size, abs_change, abs_rt):
    return SimpleNamespace(trial_index=idx, outcome=outcome, change_time=change_time,
                           change_size=change_size, reaction_time=None,
                           absolute_change_time=abs_change, absolute_reaction_time=abs_rt)


def _d1_session():
    ts, sig = _trace()
    photom = {"G0": SimpleNamespace(roi_name="G0", timestamps=ts, signal=sig),
              "G2": SimpleNamespace(roi_name="G2", timestamps=ts, signal=sig.copy())}
    trials = [
        _trial(0, "FA",  change_time=8.0, change_size=2.0, abs_change=18.0, abs_rt=15.0),  # lick, elapsed 5
        _trial(1, "Hit", change_time=5.0, change_size=2.0, abs_change=35.0, abs_rt=35.5),  # withhold
        _trial(2, "Miss",change_time=5.0, change_size=2.0, abs_change=55.0, abs_rt=None),  # withhold
    ]
    return SimpleNamespace(subject_id="013", session_id="013_a", session_date="20231205",
                           trials=trials, photometry_data=photom)


def test_build_session_scalars_scheme1():
    rows = build_session_scalars(_d1_session(), track="behavioral_fa",
                                 scheme="scheme1", use_qc=False)
    df_groups = {(r["region"], r["group"]) for r in rows}
    assert ("DMS", "lick") in df_groups and ("DMS", "withhold") in df_groups
    assert all(r["genotype"] == "D1" for r in rows)
    assert all(r["track"] == "behavioral_fa" and r["scheme"] == "scheme1" for r in rows)
    assert all(np.isfinite(r["scalar"]) for r in rows)

def test_build_dataset_two_genotypes_and_scheme3():
    d1 = _d1_session()
    d2 = _d1_session(); d2.subject_id = "016"; d2.session_id = "016_a"  # BG_016 = D2
    df = build_suppression_dataset([d1, d2], track="behavioral_fa",
                                   scheme="scheme3", use_qc=False)
    assert set(df["genotype"]) == {"D1", "D2"}
    assert set(df["group"]) >= {"lick", "withhold"}
    assert "session_id" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_suppression_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_session_scalars'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/analysis/suppression.py` (add `import pandas as pd` and the new imports at top):

```python
import pandas as pd
from visdetect_photom.core.qc import region_sources
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.analysis.state_provider import filter_trials_by_state


def build_session_scalars(session, *, track, scheme, use_qc=True,
                          state_provider=None, keep_states=None, stage="Unknown"):
    """Per-trial waiting-period scalar rows for one session (one track, one scheme).

    Row keys: subject_id, genotype, region, track, scheme, group, trial_index,
              scalar, session_id, stage.
    """
    subject_full = _subject_full(session.subject_id)
    genotype = get_genotype(subject_full)
    if genotype == "Unknown":
        return []

    keep = None
    if state_provider is not None and keep_states is not None:
        keep = filter_trials_by_state(session, state_provider, keep_states)

    records = trial_waiting_records(session, track, keep)
    by_idx = {r["trial_index"]: r for r in records}
    sources = region_sources(session, use_qc)
    rows = []

    def _emit(region, group, trial_index, scalar):
        rows.append({"subject_id": subject_full, "genotype": genotype,
                     "region": region, "track": track, "scheme": scheme,
                     "group": group, "trial_index": trial_index,
                     "scalar": scalar, "session_id": session.session_id,
                     "stage": stage})

    for region, (sig, ts) in sources.items():
        if scheme == "scheme1":
            for r in records:
                _emit(region, r["group"], r["trial_index"],
                      scheme1_scalar(r, sig, ts))
        elif scheme == "scheme3":
            # Primary lick-vs-withhold. Withhold scalars are hazard-matched to the
            # lick elapsed-time distribution. Abort is scheme1-only (exploratory):
            # it would need its own abort-matched withhold control, out of scope here.
            lick = [r for r in records if r["group"] == "lick"]
            withhold = [r for r in records if r["group"] == "withhold"]
            a_vals, w_vals = scheme3_scalars(lick, withhold, sig, ts)
            for ti, v in a_vals:
                _emit(region, "lick", ti, v)
            for ti, v in w_vals:
                _emit(region, "withhold", ti, v)
        else:
            raise ValueError(f"unknown scheme: {scheme!r}")
    return rows


def build_suppression_dataset(sessions, *, track, scheme, use_qc=True,
                              state_provider=None, keep_states=None, manifest=None):
    """Concatenate per-trial scalar rows across sessions into a DataFrame.

    If `manifest` is given, each session's learning stage is attached.
    """
    from visdetect_photom.core.staging import get_session_stage
    all_rows = []
    for sess in sessions:
        stage = get_session_stage(sess, manifest) if manifest is not None else "Unknown"
        all_rows.extend(build_session_scalars(
            sess, track=track, scheme=scheme, use_qc=use_qc,
            state_provider=state_provider, keep_states=keep_states, stage=stage))
    return pd.DataFrame(all_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_suppression_dataset.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_dataset.py
git commit -m "feat(c1): add per-session and multi-session scalar dataset builders"
```

---

## Task 7: Per-mouse Δ + AUROC and push–pull stats

**Files:**
- Modify: `src/visdetect_photom/analysis/suppression.py` (append)
- Test: `tests/analysis/test_suppression_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_suppression_stats.py
import numpy as np
import pandas as pd
from visdetect_photom.analysis.suppression import (
    compute_delta_and_auroc, run_suppression_stats,
)


def _per_trial(subject, genotype, region, lick_vals, withhold_vals):
    rows = []
    for v in lick_vals:
        rows.append({"subject_id": subject, "genotype": genotype, "region": region,
                     "track": "behavioral_fa", "scheme": "scheme1", "group": "lick",
                     "scalar": v})
    for v in withhold_vals:
        rows.append({"subject_id": subject, "genotype": genotype, "region": region,
                     "track": "behavioral_fa", "scheme": "scheme1", "group": "withhold",
                     "scalar": v})
    return rows


def test_compute_delta_and_auroc_brake_direction():
    # withhold higher than lick -> delta > 0 and AUROC > 0.5 (activity predicts withholding)
    rows = _per_trial("BG_013", "D1", "DMS",
                      lick_vals=list(np.arange(0, 10) * 0.1),
                      withhold_vals=list(1.0 + np.arange(0, 10) * 0.1))
    pm = compute_delta_and_auroc(pd.DataFrame(rows))
    r = pm.iloc[0]
    assert r["delta"] > 0
    assert r["auroc"] > 0.5

def test_compute_delta_skips_below_min_n():
    rows = _per_trial("BG_013", "D1", "DMS", lick_vals=[0.1, 0.2], withhold_vals=[1.0])
    pm = compute_delta_and_auroc(pd.DataFrame(rows))
    assert pm.empty  # < MIN_TRIALS_PER_GROUP

def test_run_suppression_stats_pushpull_opposite_sign():
    # 2 D1 mice: withhold>lick (delta>0); 2 D2 mice: withhold<lick (delta<0)
    rows = []
    for s in ("BG_013", "BG_020"):
        rows += _per_trial(s, "D1", "DMS",
                           lick_vals=list(np.zeros(10)),
                           withhold_vals=list(np.ones(10)))
    for s in ("BG_016", "BG_018"):
        rows += _per_trial(s, "D2", "DMS",
                           lick_vals=list(np.ones(10)),
                           withhold_vals=list(np.zeros(10)))
    pm = compute_delta_and_auroc(pd.DataFrame(rows))
    pp, au = run_suppression_stats(pm)
    row = pp[pp["region"] == "DMS"].iloc[0]
    assert row["d1_sign"] == 1 and row["d2_sign"] == -1
    assert set(au["genotype"]) == {"D1", "D2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_suppression_stats.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_delta_and_auroc'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/analysis/suppression.py` (add `MIN_TRIALS_PER_GROUP` to the constants import and import the stats helpers):

```python
from visdetect_photom.core.constants import MIN_TRIALS_PER_GROUP
from visdetect_photom.analysis.group_statistics import (
    auroc_score, bootstrap_ci, permutation_test, pushpull_sign_contrast,
)


def compute_delta_and_auroc(per_trial_df, min_n=MIN_TRIALS_PER_GROUP):
    """Per (subject_id, genotype, region) waiting-period summary.

    delta = mean(withhold) - mean(lick); auroc = AUROC of scalar discriminating
    withhold (positive) from lick. Cells with < min_n finite scalars in either
    group are dropped. Returns a per-mouse DataFrame.
    """
    if per_trial_df.empty:
        return pd.DataFrame()
    df = per_trial_df[per_trial_df["group"].isin(["lick", "withhold"])].copy()
    df = df[np.isfinite(df["scalar"].astype(float))]
    out = []
    for (subj, geno, region), g in df.groupby(["subject_id", "genotype", "region"]):
        lick = g[g["group"] == "lick"]["scalar"].to_numpy(dtype=float)
        wh = g[g["group"] == "withhold"]["scalar"].to_numpy(dtype=float)
        if lick.size < min_n or wh.size < min_n:
            continue
        scores = np.concatenate([wh, lick])
        labels = np.concatenate([np.ones(wh.size), np.zeros(lick.size)])
        out.append({"subject_id": subj, "genotype": geno, "region": region,
                    "n_lick": int(lick.size), "n_withhold": int(wh.size),
                    "delta": float(np.mean(wh) - np.mean(lick)),
                    "auroc": auroc_score(scores, labels)})
    return pd.DataFrame(out)


def run_suppression_stats(per_mouse_df):
    """(pushpull_df, auroc_df) over per-mouse values, per region.

    pushpull_df: D1-vs-D2 group-level sign contrast on `delta`.
    auroc_df: per genotype x region, bootstrap CI of AUROC (vs chance 0.5) plus
              D1-vs-D2 permutation p.
    """
    if per_mouse_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pp_rows = []
    for region, grp in per_mouse_df.groupby("region"):
        d1 = grp[grp["genotype"] == "D1"]["delta"].to_numpy(dtype=float)
        d2 = grp[grp["genotype"] == "D2"]["delta"].to_numpy(dtype=float)
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "metric": "delta"})
        pp_rows.append(res)

    au_rows = []
    for region, grp in per_mouse_df.groupby("region"):
        d1a = grp[grp["genotype"] == "D1"]["auroc"].to_numpy(dtype=float)
        d1a = d1a[np.isfinite(d1a)]
        d2a = grp[grp["genotype"] == "D2"]["auroc"].to_numpy(dtype=float)
        d2a = d2a[np.isfinite(d2a)]
        perm_p = (permutation_test(d1a, d2a)["p"]
                  if d1a.size >= 2 and d2a.size >= 2 else np.nan)
        for geno, vals in (("D1", d1a), ("D2", d2a)):
            ci = bootstrap_ci(vals) if vals.size >= 2 else {"observed": np.nan,
                                                            "ci_lo": np.nan, "ci_hi": np.nan}
            au_rows.append({"region": region, "genotype": geno, "n_mice": int(vals.size),
                            "auroc_mean": ci["observed"], "ci_lo": ci["ci_lo"],
                            "ci_hi": ci["ci_hi"],
                            "excludes_chance": bool(np.isfinite(ci["ci_lo"]) and
                                                    (ci["ci_lo"] > 0.5 or ci["ci_hi"] < 0.5)),
                            "perm_p_d1_vs_d2": perm_p})
    return pd.DataFrame(pp_rows), pd.DataFrame(au_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_suppression_stats.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_stats.py
git commit -m "feat(c1): add per-mouse delta/AUROC and push-pull stats"
```

---

## Task 8: Proficiency binning (staging + early/late fallback + per-bin d′)

**Files:**
- Modify: `src/visdetect_photom/analysis/suppression.py` (append)
- Test: `tests/analysis/test_suppression_proficiency.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/analysis/test_suppression_proficiency.py
import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.suppression import assign_proficiency_bins


def _sess(subj, sid, date):
    return SimpleNamespace(subject_id=subj, session_id=sid, session_date=date,
                           trials=[], photometry_data={})


def test_staging_split_used_when_enough_sessions():
    import pandas as pd
    sessions = [_sess("013", f"013_{i}", f"2023120{i}") for i in range(6)]
    manifest = pd.DataFrame({
        "subject_id": ["013"] * 6,
        "session_name": [f"013_{i}" for i in range(6)],
        "stage": ["Learning", "Learning", "Learning", "Expert", "Expert", "Expert"],
    })
    bins = assign_proficiency_bins(sessions, manifest)
    assert bins["013_0"] == "less" and bins["013_5"] == "more"

def test_date_fallback_when_staging_thin():
    sessions = [_sess("013", f"013_{i}", f"2023120{i}") for i in range(4)]
    bins = assign_proficiency_bins(sessions, manifest=None)  # no staging -> date split
    assert bins["013_0"] == "less" and bins["013_3"] == "more"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/analysis/test_suppression_proficiency.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_proficiency_bins'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/visdetect_photom/analysis/suppression.py` (add `PROF_MIN_SESSIONS` to the constants import):

```python
from collections import defaultdict
from visdetect_photom.core.constants import PROF_MIN_SESSIONS
from visdetect_photom.core.staging import get_session_stage


def assign_proficiency_bins(sessions, manifest=None,
                            min_sessions=PROF_MIN_SESSIONS):
    """Map session_id -> 'less' | 'more' | None.

    Per subject: if the staging manifest gives >= min_sessions Learning AND
    >= min_sessions Expert sessions, use Learning='less' / Expert='more' (other
    stages -> None). Otherwise fall back to a within-subject early-vs-late split
    by session_date (earlier half 'less', later half 'more'; a lone session ->
    None).
    """
    by_subject = defaultdict(list)
    for s in sessions:
        by_subject[_subject_full(s.subject_id)].append(s)

    bins = {}
    for subj, subj_sessions in by_subject.items():
        stages = {s.session_id: get_session_stage(s, manifest) for s in subj_sessions}
        n_learn = sum(v == "Learning" for v in stages.values())
        n_expert = sum(v == "Expert" for v in stages.values())
        if n_learn >= min_sessions and n_expert >= min_sessions:
            for sid, st in stages.items():
                bins[sid] = "less" if st == "Learning" else ("more" if st == "Expert" else None)
            continue
        ordered = sorted(subj_sessions, key=lambda s: str(s.session_date))
        n = len(ordered)
        if n < 2:
            for s in ordered:
                bins[s.session_id] = None
            continue
        half = n // 2
        for i, s in enumerate(ordered):
            bins[s.session_id] = "less" if i < half else "more"
    return bins


def session_d_prime(session):
    """Per-session SDT d' (go/catch by change_size), or NaN. Reporting only."""
    from visdetect_photom.analysis.statistics import calculate_sdt_metrics
    outcomes = np.array([t.outcome for t in session.trials])
    change_sizes = np.array([t.change_size if t.change_size is not None else np.nan
                             for t in session.trials], dtype=float)
    if outcomes.size == 0:
        return np.nan
    return calculate_sdt_metrics(outcomes, change_sizes).get("d_prime", np.nan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/analysis/test_suppression_proficiency.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/suppression.py tests/analysis/test_suppression_proficiency.py
git commit -m "feat(c1): add proficiency binning (staging + date fallback) and per-session d-prime"
```

---

## Task 9: Thin script `11_fa_suppression.py` + smoke test

**Files:**
- Create: `scripts/analysis/photometry/11_fa_suppression.py`
- Test: `tests/scripts/test_11_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/scripts/test_11_smoke.py
import os
import subprocess
import sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "analysis", "photometry", "11_fa_suppression.py")
DATA = os.path.join(REPO, "photom_data")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="photom_data/ not present")
def test_script_runs_on_small_subset(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--max_sessions", "5", "--output_dir", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "c1_per_trial_scalars.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/scripts/test_11_smoke.py -v`
Expected: FAIL (script does not exist → non-zero return / file missing). If `photom_data/` is absent the test is skipped — in that case verify failure by running `py scripts/analysis/photometry/11_fa_suppression.py --help` and confirming it errors (file not found) before Step 3.

- [ ] **Step 3: Write the script**

Create `scripts/analysis/photometry/11_fa_suppression.py`:

```python
"""C1 — FA suppression-failure (MOs-D2 brake): waiting-period prediction of
withhold-vs-lick from bulk D1/D2 signal.

Two tracks (behavioral_fa primary, sdt_fa control) x two window schemes
(scheme1 baseline-onset fixed, scheme3 hazard-time-matched). Per-mouse delta
(withhold-lick) + single-trial AUROC, group push-pull sign contrast, and a
coarse proficiency split. D1 and D2 are DIFFERENT animals: push-pull is a
GROUP-LEVEL sign contrast.

Usage:
    py scripts/analysis/photometry/11_fa_suppression.py
    py scripts/analysis/photometry/11_fa_suppression.py --no-qc
    py scripts/analysis/photometry/11_fa_suppression.py --max_sessions 10
    py scripts/analysis/photometry/11_fa_suppression.py --state-filter Engaged --state-results-dir results/hmm/BG_013
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import GENOTYPE_COLORS, MIN_PHOTOM_CSV_BYTES
from visdetect_photom.core.staging import load_staging_manifest, excluded_mice
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider
from visdetect_photom.analysis.suppression import (
    build_suppression_dataset, compute_delta_and_auroc, run_suppression_stats,
    assign_proficiency_bins,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

TRACKS = ["behavioral_fa", "sdt_fa"]
SCHEMES = ["scheme1", "scheme3"]


def _load_sessions(args, excl):
    files = io.find_all_sessions(args.root_dir, recursive=True,
                                 min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(files)} session files.")
    sessions, n = [], 0
    for sf in files:
        if args.max_sessions and n >= args.max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.warning(f"skip {sf.get('trials', '?')}: {e}")
            continue
        if f"BG_{str(sess.subject_id).zfill(3)}" in excl or sess.subject_id in excl:
            continue
        sessions.append(sess)
        n += 1
    logging.info(f"Loaded {len(sessions)} sessions.")
    return sessions


def _qualifying_n(per_trial_df):
    if per_trial_df.empty:
        return pd.DataFrame()
    g = per_trial_df.copy()
    g["finite"] = np.isfinite(g["scalar"].astype(float))
    return (g.groupby(["track", "scheme", "region", "genotype", "group"])["finite"]
             .agg(n_total="size", n_finite="sum").reset_index())


def _plot_delta_summary(pushpull_df, out_dir):
    if pushpull_df.empty:
        return
    regions = sorted(pushpull_df["region"].unique())
    keys = pushpull_df[["track", "scheme"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * max(len(regions), 1), 5),
                             squeeze=False)
    fig.suptitle("C1 — waiting-period Δ(withhold−lick), D1 vs D2 (group-level)", fontsize=12)
    x = np.arange(len(keys))
    for ai, region in enumerate(regions):
        ax = axes[0][ai]
        sub = pushpull_df[pushpull_df["region"] == region].set_index(["track", "scheme"])
        d1 = [sub.loc[tuple(k), "d1_mean"] if tuple(k) in sub.index else np.nan for k in keys]
        d2 = [sub.loc[tuple(k), "d2_mean"] if tuple(k) in sub.index else np.nan for k in keys]
        ax.bar(x - 0.2, d1, 0.4, color=GENOTYPE_COLORS["D1"], label="D1")
        ax.bar(x + 0.2, d2, 0.4, color=GENOTYPE_COLORS["D2"], label="D2")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["/".join(k) for k in keys], rotation=45, ha="right", fontsize=7)
        ax.set_title(region, fontsize=10)
        ax.set_ylabel("Δ z-dF/F (withhold−lick)", fontsize=8)
        ax.legend(fontsize=7)
        sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "C1_delta_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser(description="C1: FA suppression-failure (MOs-D2 brake)")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "C1_fa_suppression"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None,
                    help="comma-separated behavioral states to keep (default: pooled)")
    ap.add_argument("--state-results-dir", default=None)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    use_qc = not args.no_qc
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_staging_manifest()
    excl = excluded_mice(manifest)
    if excl:
        logging.info(f"Excluding mice (staging all-Excluded): {sorted(excl)}")

    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir")
            sys.exit(1)
        state_provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
    else:
        state_provider = PooledStateProvider()
        keep_states = ["All"]

    sessions = _load_sessions(args, excl)
    if not sessions:
        logging.error("No sessions loaded.")
        sys.exit(1)

    prof_bins = assign_proficiency_bins(sessions, manifest)

    all_trials, all_pushpull, all_auroc = [], [], []
    for track in TRACKS:
        for scheme in SCHEMES:
            df = build_suppression_dataset(
                sessions, track=track, scheme=scheme, use_qc=use_qc,
                state_provider=state_provider, keep_states=keep_states, manifest=manifest)
            if df.empty:
                continue
            df["prof_bin"] = df["session_id"].map(prof_bins)
            all_trials.append(df)

            # pooled (primary)
            pm = compute_delta_and_auroc(df)
            pp, au = run_suppression_stats(pm)
            for frame, store in ((pp, all_pushpull), (au, all_auroc)):
                if not frame.empty:
                    frame = frame.copy()
                    frame["track"], frame["scheme"], frame["prof_bin"] = track, scheme, "pooled"
                    store.append(frame)
            # proficiency split (robustness)
            for b in ("less", "more"):
                pmb = compute_delta_and_auroc(df[df["prof_bin"] == b])
                ppb, aub = run_suppression_stats(pmb)
                for frame, store in ((ppb, all_pushpull), (aub, all_auroc)):
                    if not frame.empty:
                        frame = frame.copy()
                        frame["track"], frame["scheme"], frame["prof_bin"] = track, scheme, b
                        store.append(frame)

    if not all_trials:
        logging.error("No waiting-period scalars extracted.")
        sys.exit(1)

    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.to_csv(out_dir / "c1_per_trial_scalars.csv", index=False)
    _qualifying_n(trials_df).to_csv(out_dir / "c1_qualifying_n.csv", index=False)
    if all_pushpull:
        pd.concat(all_pushpull, ignore_index=True).to_csv(out_dir / "c1_pushpull_stats.csv", index=False)
    if all_auroc:
        pd.concat(all_auroc, ignore_index=True).to_csv(out_dir / "c1_auroc_stats.csv", index=False)

    if all_pushpull:
        pooled_pp = pd.concat(all_pushpull, ignore_index=True)
        _plot_delta_summary(pooled_pp[pooled_pp["prof_bin"] == "pooled"], out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test (and the full suite)**

Run: `py -m pytest tests/scripts/test_11_smoke.py -v`
Expected: PASS if `photom_data/` is present (writes `c1_per_trial_scalars.csv`); otherwise SKIPPED.

Run the whole C1 suite to confirm nothing regressed:
Run: `py -m pytest tests/ -q`
Expected: all tests pass (C1 tests + the pre-existing C2/staging/state-provider tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/photometry/11_fa_suppression.py tests/scripts/test_11_smoke.py
git commit -m "feat(c1): add 11_fa_suppression script + smoke test"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| §3 Track A behavioral FA (lick=FA, withhold=Hit/Miss/CR, abort separate) | Task 3 (`_group_for`) |
| §3 Track B SDT-FA (catch Hit vs catch Miss, `calculate_sdt_metrics` convention) | Task 3 (`_group_for` sdt_fa) |
| §3 abort as optional exploratory group | Task 3 (`abort` group) + Task 6 (emitted, scheme3 action group) |
| §3 push–pull = group-level sign contrast | Task 7 (`run_suppression_stats` via `pushpull_sign_contrast`) |
| §4 Scheme 1 fixed window + inclusion + early-FA limitation | Task 4 |
| §4 Scheme 3 hazard-matched + buffer + determinism + transient subtraction | Task 5 |
| §4 window constants | Task 3 Step 1 |
| §5 session-z-scored, no per-trial re-baselining | Tasks 3/6 (uses `region_sources` trace signal directly; `window_mean` raw mean) |
| §6 group push–pull contrast (Δ) | Task 7 |
| §6 single-trial AUROC decoder | Task 1 + Task 7 |
| §6 proficiency split (staging + early/late + per-bin d′) | Task 8 + Task 9 wiring |
| §6 companion PETHs | Task 9 (`_plot_delta_summary`; see note below) |
| §7 module layout (suppression.py, qc.region_sources, group_statistics.auroc_score, script 11) | Tasks 1,2,9 + all |
| §7 outputs (per_trial, pushpull, auroc, qualifying_n CSVs) | Task 9 |
| §8 mouse exclusion via staging; pluggable state default pooled | Task 9 |
| §8 min-N guard + per-cell N reporting | Task 7 (`min_n`) + Task 9 (`c1_qualifying_n.csv`) |

**Note on companion PETHs (§6):** Task 9 ships the quantitative core (per-trial scalars, Δ/AUROC stats, qualifying-N) plus a Δ-summary bar figure. Full alignment-separated PETH overlays (grating-onset-aligned for Scheme 1; lick-aligned for Scheme 3, never mixed) are illustrative and **deferred to a follow-up** after the user reviews the numeric results — they don't change any statistic. This keeps the script focused; flag for the executor that adding them later is a small, isolated extension (reuse `extract_peth` with the group event times already derivable from the records).

**2. Placeholder scan:** No TBD/TODO; every code step contains complete code; every test step contains real assertions. ✓

**3. Type consistency:** `region_sources` returns `{region: (signal, timestamps)}` (Task 2) consumed as such in Task 6. `trial_waiting_records` record keys (`group`, `onset_abs`, `change_time`, `lick_abs`, `lick_elapsed`, `trial_index`) are produced in Task 3 and consumed identically in Tasks 4/5/6. `scheme3_scalars` returns `(action_vals, withhold_vals)` lists of `(trial_index, scalar)` (Task 5) consumed via `dict(...)`/iteration in Task 6. `compute_delta_and_auroc` emits columns `delta`/`auroc`/`genotype`/`region` (Task 7) consumed by `run_suppression_stats` and the script (Tasks 7/9). `auroc_score(scores, labels)` positive-class=1=withhold is consistent across Tasks 1/7. ✓

---

## Execution notes
- Run tests with `py -m pytest` (Windows; `py` not `python`). The src-layout is wired by `tests/conftest.py`.
- The smoke test (Task 9) requires `photom_data/`; it auto-skips if absent. On a data-bearing checkout, also do a manual end-to-end run: `py scripts/analysis/photometry/11_fa_suppression.py --max_sessions 20` and eyeball `FIGURES/C1_fa_suppression/c1_pushpull_stats.csv` + `c1_auroc_stats.csv` + `c1_qualifying_n.csv`.
- After all tasks: confirm no regression with `py -m pytest tests/ -q`.
