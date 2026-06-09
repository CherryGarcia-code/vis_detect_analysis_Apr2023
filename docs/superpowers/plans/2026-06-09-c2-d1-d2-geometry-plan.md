# C2 — D1/D2 Response Geometry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mode-aware D1-vs-D2 striatal-photometry "response geometry" analysis (push–pull sign test + evidence grading + commitment timing) across change, lick, and anticipation epochs, in all three regions.

**Architecture:** Approach B — reusable primitives go in the package (`analysis/group_statistics.py` mode-aware extractors, new `analysis/state_provider.py`, new `core/staging.py`, new `analysis/geometry.py` computation core), and a thin CLI script `scripts/analysis/photometry/08_d1_d2_geometry.py` consumes them. All new logic is unit-tested (TDD); the script is a thin orchestration layer + plotting.

**Tech Stack:** Python 3.10, numpy<2, pandas<2.3, scipy<2, matplotlib<3.9, seaborn<0.14, pytest 9. Package uses a `src/` layout imported via `sys.path` (NOT pip-installed) — tests add `src/` to the path via `tests/conftest.py`. Invoke Python with `py` (Windows).

**Spec:** `docs/superpowers/specs/2026-06-08-c2-d1-d2-geometry-design.md`. Read it before starting.

---

## Background the engineer needs

- **Session model** (`src/visdetect_photom/core/session.py`): `Session` has `.subject_id` (digits-only string, e.g. `"013"`), `.session_id` (e.g. `"013_20231205"`), `.session_date`, `.trials` (list of `Trial`), `.photometry_data` (dict `roi_name -> PhotometryTrace`). `Trial` has `.outcome` (`'Hit'|'Miss'|'FA'|'Abort'|'CR'`), `.change_size` (float; `>1.01` = go, `<=1.01` = catch), `.reaction_time`, `.absolute_change_time`, `.absolute_reaction_time`. `PhotometryTrace` has `.roi_name`, `.timestamps` (1-D, sorted), `.signal` (1-D session-z-scored dF/F).
- **ROI→region** (`core/constants.py`): `get_roi_region(roi, subject_id)` → e.g. `'DMS_L'`. Strip the `_L/_R` suffix for the base region. `get_genotype(subject_id)` (`analysis/group_utils.py`) → `'D1'|'D2'|'Unknown'` and accepts `'013'`, `'BG_013'`, `'13'`.
- **PETH** (`analysis/statistics.py`): `extract_peth(signal, timestamps, event_times, window=(-2,4), baseline_window=(-2,0), normalize='subtract')` → `(time_axis, peth_matrix)` where `peth_matrix` is `(n_events, n_timepoints)`, baseline-mean-subtracted. **Note the return order is `(time_axis, matrix)`.**
- **Existing stats** (`analysis/group_statistics.py`): `permutation_test(x, y, n_perm, seed)`, `bootstrap_ci(data, n_boot, seed)`, `mannwhitney_with_effect_size(x, y)`, `spearman_with_ci(x, y)`, `extract_peak_latency(trace, t, peak_window)`, `extract_onset_latency(trace, t, threshold_n_std, baseline_window, search_window, n_consecutive)`.
- **QC + merge** (`core/qc.py`): `compute_session_roi_qc(session)`, `merge_hemispheres(session, qc_results)` → dict `region -> {'signal','timestamps','source','rois_used'}`.
- **Hard constraint:** D1 and D2 are *different animals*. All push–pull claims are **group-level sign contrasts**, never within-animal anticorrelation.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `tests/conftest.py` | Create | Put `src/` on `sys.path` so `import visdetect_photom` works under pytest. |
| `src/visdetect_photom/analysis/group_statistics.py` | Modify | Add 5 mode-aware extractors + `pushpull_sign_contrast`. |
| `tests/analysis/test_group_statistics_modeaware.py` | Create | Unit tests for the 6 new functions. |
| `scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py` | Modify | Replace local `extract_peak` with import of `extract_signed_peak` (no behavior change). |
| `src/visdetect_photom/analysis/state_provider.py` | Create | `StateProvider` protocol + `PooledStateProvider` (default), `HMMStateProvider` (lazy), `filter_trials_by_state`. |
| `tests/analysis/test_state_provider.py` | Create | Tests for pooled provider + filter. |
| `src/visdetect_photom/core/staging.py` | Create | `load_staging_manifest`, `get_session_stage`, `excluded_mice`. |
| `tests/core/test_staging.py` | Create | Tests with a fixture manifest. |
| `src/visdetect_photom/analysis/geometry.py` | Create | C2 computation core: per-session metrics + dataset build + push–pull/grading stats. |
| `tests/analysis/test_geometry.py` | Create | Synthetic-Session tests for the core. |
| `scripts/analysis/photometry/08_d1_d2_geometry.py` | Create | Thin CLI: discover → load → build dataset → stats → figures. |
| `tests/scripts/test_08_smoke.py` | Create | Smoke test (skips if `photom_data/` absent). |

---

## Task 1: Mode-aware extractors + test scaffolding

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/analysis/test_group_statistics_modeaware.py`
- Modify: `src/visdetect_photom/analysis/group_statistics.py` (append functions)

- [ ] **Step 1: Create the test path shim**

Create `tests/conftest.py`:

```python
import os
import sys

# Package uses a src/ layout and is not pip-installed; put it on the path.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/analysis/test_group_statistics_modeaware.py`:

```python
import numpy as np
import pytest
from visdetect_photom.analysis.group_statistics import (
    extract_activation, extract_suppression, extract_signed_peak,
    extract_signed_auc, extract_ramp_slope, pushpull_sign_contrast,
)

T = np.linspace(-2.0, 4.0, 600)          # 100 Hz over [-2, 4]
WIN = (0.0, 1.5)


def _bump(amp, center=0.5, width=0.3):
    return amp * np.exp(-((T - center) ** 2) / (2 * width ** 2))


def test_activation_positive_bump():
    assert extract_activation(_bump(2.0), T, WIN) == pytest.approx(2.0, abs=0.05)

def test_activation_pure_dip_is_nan():
    assert np.isnan(extract_activation(_bump(-2.0), T, WIN))

def test_suppression_negative_dip():
    assert extract_suppression(_bump(-3.0), T, WIN) == pytest.approx(-3.0, abs=0.05)

def test_suppression_pure_bump_is_nan():
    assert np.isnan(extract_suppression(_bump(2.0), T, WIN))

def test_signed_peak_preserves_sign():
    assert extract_signed_peak(_bump(-3.0), T, WIN) == pytest.approx(-3.0, abs=0.05)
    assert extract_signed_peak(_bump(2.0), T, WIN) == pytest.approx(2.0, abs=0.05)

def test_signed_auc_sign():
    assert extract_signed_auc(_bump(2.0), T, WIN) > 0
    assert extract_signed_auc(_bump(-2.0), T, WIN) < 0

def test_ramp_slope_known_slope():
    trace = 3.0 * T  # slope 3 per second
    assert extract_ramp_slope(trace, T, (-1.5, 0.0)) == pytest.approx(3.0, abs=0.01)

def test_empty_window_returns_nan():
    assert np.isnan(extract_activation(_bump(2.0), T, (10.0, 11.0)))

def test_pushpull_opposite_sign_flagged():
    d1 = np.array([1.8, 2.1, 1.9, 2.3])
    d2 = np.array([-1.7, -2.0, -1.6, -2.2])
    res = pushpull_sign_contrast(d1, d2, n_perm=2000, seed=42)
    assert res["opposite_sign"] is True
    assert res["d1_sign"] == 1 and res["d2_sign"] == -1
    assert res["p"] < 0.05

def test_pushpull_same_sign_not_flagged():
    d1 = np.array([1.8, 2.1, 1.9, 2.3])
    d2 = np.array([1.5, 1.7, 1.6, 1.9])
    res = pushpull_sign_contrast(d1, d2, n_perm=2000, seed=42)
    assert res["opposite_sign"] is False
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_group_statistics_modeaware.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_activation'`.

- [ ] **Step 4: Implement the functions**

Append to `src/visdetect_photom/analysis/group_statistics.py`:

```python
# ── Mode-aware response extraction (C2) ──────────────────────

def _window_segment(trace, time_axis, window):
    """Return (finite values, their times) inside the window."""
    trace = np.asarray(trace, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    mask = (time_axis >= window[0]) & (time_axis <= window[1])
    seg = trace[mask]
    t_seg = time_axis[mask]
    finite = np.isfinite(seg)
    return seg[finite], t_seg[finite]


def extract_activation(trace, time_axis, window):
    """Peak positive deflection in window (>0), else nan (pure suppression/flat)."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    m = float(np.max(seg))
    return m if m > 0 else np.nan


def extract_suppression(trace, time_axis, window):
    """Peak negative deflection in window (<0), else nan (pure activation/flat)."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    m = float(np.min(seg))
    return m if m < 0 else np.nan


def extract_signed_peak(trace, time_axis, window):
    """Abs-max deflection in window, preserving sign (captures activation OR suppression)."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    return float(seg[np.argmax(np.abs(seg))])


def extract_signed_auc(trace, time_axis, window):
    """Mean (net signed response) over window."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    return float(np.mean(seg))


def extract_ramp_slope(trace, time_axis, window):
    """Slope (signal-units/s) of a degree-1 fit over window; offset-invariant."""
    seg, t_seg = _window_segment(trace, time_axis, window)
    if seg.size < 2:
        return np.nan
    return float(np.polyfit(t_seg, seg, 1)[0])


def pushpull_sign_contrast(d1_vals, d2_vals, n_perm=10000, seed=42):
    """Group-level D1-vs-D2 sign contrast (NOT within-animal anticorrelation).

    Returns per-genotype mean + bootstrap 95% CI, each sign, an `opposite_sign`
    flag (signs differ AND both CIs exclude 0), permutation p on (meanD1-meanD2),
    and rank-biserial effect size.
    """
    d1 = np.asarray(d1_vals, float); d1 = d1[np.isfinite(d1)]
    d2 = np.asarray(d2_vals, float); d2 = d2[np.isfinite(d2)]
    out = {"d1_n": int(d1.size), "d2_n": int(d2.size),
           "d1_mean": float(np.mean(d1)) if d1.size else np.nan,
           "d2_mean": float(np.mean(d2)) if d2.size else np.nan}

    d1_ci = bootstrap_ci(d1) if d1.size >= 2 else {"ci_lo": np.nan, "ci_hi": np.nan}
    d2_ci = bootstrap_ci(d2) if d2.size >= 2 else {"ci_lo": np.nan, "ci_hi": np.nan}
    out.update({"d1_ci_lo": d1_ci["ci_lo"], "d1_ci_hi": d1_ci["ci_hi"],
                "d2_ci_lo": d2_ci["ci_lo"], "d2_ci_hi": d2_ci["ci_hi"]})

    def _excl_zero(ci):
        return np.isfinite(ci["ci_lo"]) and (ci["ci_lo"] > 0 or ci["ci_hi"] < 0)

    out["d1_sign"] = int(np.sign(out["d1_mean"])) if np.isfinite(out["d1_mean"]) else 0
    out["d2_sign"] = int(np.sign(out["d2_mean"])) if np.isfinite(out["d2_mean"]) else 0
    out["opposite_sign"] = bool(_excl_zero(d1_ci) and _excl_zero(d2_ci)
                                and out["d1_sign"] != out["d2_sign"])

    if d1.size >= 2 and d2.size >= 2:
        out["p"] = permutation_test(d1, d2, n_perm=n_perm, seed=seed)["p"]
        out["rank_biserial_r"] = mannwhitney_with_effect_size(d1, d2)["rank_biserial_r"]
    else:
        out["p"] = np.nan
        out["rank_biserial_r"] = np.nan
    return out
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_group_statistics_modeaware.py -v`
Expected: PASS (11 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/analysis/test_group_statistics_modeaware.py src/visdetect_photom/analysis/group_statistics.py
git commit -m "Add mode-aware response extractors and push-pull sign contrast (C2)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Promote `extract_peak` in script 01 to the shared `extract_signed_peak`

**Files:**
- Modify: `scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py`

The spec says the abs-max-preserving-sign extractor must be canonical. Script 01 currently defines a local `extract_peak`. Replace it with the shared one (identical behavior) so there is a single source of truth.

- [ ] **Step 1: Update the import block**

In `scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py`, find the import:

```python
from visdetect_photom.analysis.group_statistics import (
    mannwhitney_with_effect_size, permutation_test, bootstrap_ci, format_stats_table,
)
```

Replace with:

```python
from visdetect_photom.analysis.group_statistics import (
    mannwhitney_with_effect_size, permutation_test, bootstrap_ci, format_stats_table,
    extract_signed_peak,
)
```

- [ ] **Step 2: Replace the local function with an alias**

Find the local definition (the `def extract_peak(trace, time_axis, peak_window=PEAK_WINDOW):` block, ~lines 229-240). Replace the whole `def extract_peak(...): ...` block with:

```python
def extract_peak(trace, time_axis, peak_window=PEAK_WINDOW):
    """Peak (abs-max, sign-preserving) value within peak_window. Delegates to the
    canonical implementation in group_statistics."""
    return extract_signed_peak(trace, time_axis, peak_window)
```

- [ ] **Step 3: Verify the script still imports**

Run: `py -c "import sys; sys.path.insert(0,'src'); import importlib.util; spec=importlib.util.spec_from_file_location('s01','scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('import OK', m.extract_peak.__doc__ is not None)"`
Expected: prints `import OK True` (no exceptions).

- [ ] **Step 4: Commit**

```bash
git add scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py
git commit -m "Use canonical extract_signed_peak in script 01 (no behavior change)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `state_provider.py` (swappable behavioral-state seam)

**Files:**
- Create: `src/visdetect_photom/analysis/state_provider.py`
- Create: `tests/analysis/test_state_provider.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_state_provider.py`:

```python
import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.state_provider import (
    PooledStateProvider, HMMStateProvider, filter_trials_by_state,
)


def _session(n):
    return SimpleNamespace(trials=[SimpleNamespace(trial_index=i) for i in range(n)],
                           subject_id="013")


def test_pooled_returns_all_label():
    s = _session(5)
    states = PooledStateProvider().get_trial_states(s)
    assert list(states) == ["All"] * 5


def test_filter_keeps_matching_indices():
    s = _session(4)

    class Fake:
        def get_trial_states(self, session):
            return np.array(["Engaged", "Disengaged", "Engaged", "NA"], dtype=object)

    keep = filter_trials_by_state(s, Fake(), {"Engaged"})
    assert keep == {0, 2}


def test_pooled_filter_keeps_everything():
    s = _session(3)
    keep = filter_trials_by_state(s, PooledStateProvider(), {"All"})
    assert keep == {0, 1, 2}


def test_hmm_provider_is_lazy_constructible():
    # Constructing must NOT import/load HMM artifacts.
    p = HMMStateProvider(results_dir="does/not/exist")
    assert p.results_dir == "does/not/exist"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_state_provider.py -v`
Expected: FAIL — module `state_provider` does not exist.

- [ ] **Step 3: Implement the module**

Create `src/visdetect_photom/analysis/state_provider.py`:

```python
"""Swappable trial-level behavioral-state labeling.

`StateProvider` is the seam: any object with `get_trial_states(session)` returning
one label per trial works. The HMM is just one backend (`HMMStateProvider`); the
default (`PooledStateProvider`) does no filtering. Keep this SEPARATE from the
session-level learning-stage logic in `core/staging.py`.
"""
from typing import Protocol, Iterable, Set
import numpy as np


class StateProvider(Protocol):
    def get_trial_states(self, session) -> np.ndarray:
        """Return an array of per-trial state labels, len == len(session.trials)."""
        ...


class PooledStateProvider:
    """Default: no state distinction; every trial is 'All'."""

    def get_trial_states(self, session) -> np.ndarray:
        return np.array(["All"] * len(session.trials), dtype=object)


class HMMStateProvider:
    """Trial states from a fitted GLM-HMM. Lazy: artifacts load on first use."""

    def __init__(self, results_dir, K=None):
        self.results_dir = results_dir
        self.K = K
        self._model = None
        self._labels = None

    def _ensure_loaded(self):
        if self._model is None:
            from visdetect_photom.analysis.hmm_downstream import load_hmm_results
            self._model, _, self._labels = load_hmm_results(self.results_dir, self.K)

    def get_trial_states(self, session) -> np.ndarray:
        self._ensure_loaded()
        from visdetect_photom.analysis.hmm import decode_session
        df = decode_session(self._model, session, self._labels)
        labels = np.array(["NA"] * len(session.trials), dtype=object)
        if "hmm_state_label" in df.columns and "trial_index" in df.columns:
            for _, row in df.iterrows():
                ti = int(row["trial_index"])
                if 0 <= ti < len(labels):
                    labels[ti] = row["hmm_state_label"]
        return labels


def filter_trials_by_state(session, provider: StateProvider,
                           keep_states: Iterable[str]) -> Set[int]:
    """Return the set of trial indices whose state is in keep_states."""
    states = provider.get_trial_states(session)
    keep = set(keep_states)
    return {i for i, s in enumerate(states) if s in keep}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_state_provider.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/state_provider.py tests/analysis/test_state_provider.py
git commit -m "Add swappable StateProvider seam (pooled default, lazy HMM backend)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `core/staging.py` (learning-stage manifest helper)

**Files:**
- Create: `src/visdetect_photom/core/staging.py`
- Create: `tests/core/test_staging.py`

The manifest (`results/staging_manifest.csv`) columns: `subject_id` (`BG_0XX`), `session_name` (`013_20231205`, matches `Session.session_id`), `stage` (`Naive|Learning|Expert|Disengaged|Excluded`). A mouse is "excluded" iff it has **no** non-Excluded session.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_staging.py`:

```python
import pandas as pd
from types import SimpleNamespace
from visdetect_photom.core.staging import (
    load_staging_manifest, get_session_stage, excluded_mice,
)


def _manifest(tmp_path):
    df = pd.DataFrame([
        {"subject_id": "BG_013", "session_name": "013_20231205", "stage": "Learning"},
        {"subject_id": "BG_013", "session_name": "013_20231206", "stage": "Expert"},
        {"subject_id": "BG_014", "session_name": "014_20231219", "stage": "Excluded"},
        {"subject_id": "BG_014", "session_name": "014_20231221", "stage": "Excluded"},
    ])
    p = tmp_path / "staging_manifest.csv"
    df.to_csv(p, index=False)
    return p


def test_load_missing_returns_none():
    assert load_staging_manifest("nope/missing.csv") is None


def test_get_session_stage_matches_session_id(tmp_path):
    m = load_staging_manifest(str(_manifest(tmp_path)))
    sess = SimpleNamespace(subject_id="013", session_id="013_20231206")
    assert get_session_stage(sess, m) == "Expert"


def test_get_session_stage_unknown_when_absent(tmp_path):
    m = load_staging_manifest(str(_manifest(tmp_path)))
    sess = SimpleNamespace(subject_id="999", session_id="999_20990101")
    assert get_session_stage(sess, m) == "Unknown"


def test_excluded_mice_includes_all_excluded(tmp_path):
    m = load_staging_manifest(str(_manifest(tmp_path)))
    excl = excluded_mice(m)
    assert "BG_014" in excl
    assert "BG_013" not in excl
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/core/test_staging.py -v`
Expected: FAIL — module `staging` does not exist.

- [ ] **Step 3: Implement the module**

Create `src/visdetect_photom/core/staging.py`:

```python
"""Session-level learning-stage helper backed by results/staging_manifest.csv.

Distinct from analysis/state_provider.py (trial-level behavioral state). Stages:
Naive | Learning | Expert | Disengaged | Excluded.
"""
import os
import pandas as pd

DEFAULT_MANIFEST_PATH = os.path.join("results", "staging_manifest.csv")


def load_staging_manifest(path: str = DEFAULT_MANIFEST_PATH):
    """Load the staging manifest, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _norm_subject(subject_id) -> str:
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def get_session_stage(session, manifest) -> str:
    """Stage for a session by matching session_name == session.session_id.

    If multiple manifest rows share the session_name (rare; >1 recording/day),
    the first match wins.
    """
    if manifest is None or "session_name" not in manifest.columns:
        return "Unknown"
    hit = manifest[manifest["session_name"] == session.session_id]
    if len(hit) == 0:
        return "Unknown"
    return str(hit.iloc[0]["stage"])


def excluded_mice(manifest) -> set:
    """Subjects (BG_0XX) whose every staged session is 'Excluded'."""
    if manifest is None or "stage" not in manifest.columns:
        return set()
    excl = set()
    for subj, grp in manifest.groupby("subject_id"):
        if (grp["stage"] == "Excluded").all():
            excl.add(_norm_subject(subj))
    return excl
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/core/test_staging.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/core/staging.py tests/core/test_staging.py
git commit -m "Add staging-manifest helper (session stage + excluded mice)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `geometry.py` — per-session metric computation

**Files:**
- Create: `src/visdetect_photom/analysis/geometry.py`
- Create: `tests/analysis/test_geometry.py`

This is the C2 core. `compute_geometry_metrics_for_session` returns `(rows, traces, time_axis)` where `rows` are per-(region, epoch[, change_size]) metric dicts and `traces` maps `(region, epoch) -> per-mouse mean trace` for plotting.

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_geometry.py`:

```python
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
    assert np.isnan(r["suppression"])  # pure positive bump
    assert ("DMS", "change_hit") in traces
    assert t is not None
```

- [ ] **Step 2: Run test, verify it fails**

Run: `py -m pytest tests/analysis/test_geometry.py -v`
Expected: FAIL — module `geometry` does not exist.

- [ ] **Step 3: Implement the core**

Create `src/visdetect_photom/analysis/geometry.py`:

```python
"""C2 — D1/D2 response-geometry computation core.

Per-session, mode-aware metrics across change / lick / anticipation epochs, plus
change-size grading. Thin script 08 consumes this. See spec
docs/superpowers/specs/2026-06-08-c2-d1-d2-geometry-design.md.
"""
from collections import defaultdict
import numpy as np

from visdetect_photom.core.constants import CHANGE_SIZES, CATCH_THRESHOLD, get_roi_region
from visdetect_photom.core.qc import compute_session_roi_qc, merge_hemispheres
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.analysis.group_statistics import (
    extract_activation, extract_suppression, extract_signed_peak,
    extract_signed_auc, extract_ramp_slope,
    extract_peak_latency, extract_onset_latency,
)
from visdetect_photom.analysis.state_provider import filter_trials_by_state

PETH_WINDOW = (-2.0, 4.0)
POST_WINDOW = (0.0, 1.5)
ONSET_WINDOW = (0.0, 2.0)
ANTICIP_WINDOW = (-1.5, 0.0)
ANTICIP_BASELINE = (-2.0, -1.5)
CHANGE_BASELINE = (-2.0, 0.0)

# epoch -> (event_type, kind, baseline_window)
EPOCHS = {
    "change_hit":        ("change_hit", "change",       CHANGE_BASELINE),
    "change_miss":       ("change_miss", "change",      CHANGE_BASELINE),
    "hit_lick":          ("hit_lick", "lick",           CHANGE_BASELINE),
    "fa_lick":           ("fa_lick", "lick",            CHANGE_BASELINE),
    "anticipation_hit":  ("change_hit", "anticipation", ANTICIP_BASELINE),
    "anticipation_miss": ("change_miss", "anticipation", ANTICIP_BASELINE),
    "anticipation_cr":   ("change_cr", "anticipation",  ANTICIP_BASELINE),
}

_METRIC_KEYS = ["activation", "suppression", "signed_peak", "signed_auc",
                "ramp_slope", "peak_latency", "onset_latency"]


def _subject_full(subject_id):
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def _event_times(session, event_type, keep):
    """Absolute event times for an epoch's event_type, restricted to keep (set|None)."""
    times = []
    for t in session.trials:
        if keep is not None and t.trial_index not in keep:
            continue
        o = t.outcome
        if event_type == "change_hit" and o == "Hit" and t.absolute_change_time is not None:
            times.append(t.absolute_change_time)
        elif event_type == "change_miss" and o == "Miss" and t.absolute_change_time is not None:
            times.append(t.absolute_change_time)
        elif event_type == "change_cr" and o == "CR" and t.absolute_change_time is not None:
            times.append(t.absolute_change_time)
        elif event_type == "hit_lick" and o == "Hit" and t.absolute_reaction_time is not None:
            times.append(t.absolute_reaction_time)
        elif event_type == "fa_lick" and o == "FA" and t.absolute_reaction_time is not None:
            times.append(t.absolute_reaction_time)
    return np.array(times, dtype=float)


def _change_hit_times_for_cs(session, cs, keep):
    """Hit-trial change times whose change_size rounds to canonical cs."""
    times = []
    for t in session.trials:
        if keep is not None and t.trial_index not in keep:
            continue
        if t.outcome != "Hit" or t.absolute_change_time is None:
            continue
        if t.change_size is None or t.change_size <= CATCH_THRESHOLD:
            continue
        if min(CHANGE_SIZES, key=lambda x: abs(x - t.change_size)) == cs:
            times.append(t.absolute_change_time)
    return np.array(times, dtype=float)


def _mean_trace(peth_matrix):
    """Per-mouse mean over trials with >50% finite bins; None if none qualify."""
    if peth_matrix.shape[0] == 0:
        return None
    valid = [row for row in peth_matrix
             if np.sum(np.isfinite(row)) > 0.5 * row.shape[0]]
    if not valid:
        return None
    return np.nanmean(np.array(valid), axis=0)


def _metrics_for_trace(trace, t, kind):
    out = {k: np.nan for k in _METRIC_KEYS}
    if kind in ("change", "lick"):
        out["activation"] = extract_activation(trace, t, POST_WINDOW)
        out["suppression"] = extract_suppression(trace, t, POST_WINDOW)
        out["signed_peak"] = extract_signed_peak(trace, t, POST_WINDOW)
        out["signed_auc"] = extract_signed_auc(trace, t, POST_WINDOW)
        out["peak_latency"] = extract_peak_latency(trace, t, peak_window=POST_WINDOW)
        out["onset_latency"] = extract_onset_latency(
            trace, t, threshold_n_std=2.0,
            baseline_window=CHANGE_BASELINE, search_window=ONSET_WINDOW, n_consecutive=3)
    elif kind == "anticipation":
        out["ramp_slope"] = extract_ramp_slope(trace, t, ANTICIP_WINDOW)
        out["signed_auc"] = extract_signed_auc(trace, t, ANTICIP_WINDOW)
    return out


def _region_sources(session, subject_full, use_qc):
    """Return {region_base: (signal, timestamps)} via QC-merge or no-QC averaging."""
    if use_qc:
        qc = compute_session_roi_qc(session)
        merged = merge_hemispheres(session, qc_results=qc)
        return {r: (m["signal"], m["timestamps"]) for r, m in merged.items()}
    by_region = defaultdict(list)
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


def compute_geometry_metrics_for_session(session, *, use_qc=True,
                                         state_provider=None, keep_states=None):
    """Return (rows, traces, time_axis) for one session.

    rows: list of dicts (subject_id, genotype, region, epoch, change_size, n_trials,
          + the 7 metric keys). change_size is NaN except for graded change_hit rows
          (epoch == 'change_hit_graded').
    traces: {(region, epoch): mean_trace} for pooled (non-graded) epochs only.
    time_axis: shared 1-D axis (or None if nothing extracted).
    """
    subject_full = _subject_full(session.subject_id)
    genotype = get_genotype(subject_full)
    if genotype == "Unknown":
        return [], {}, None

    keep = None
    if state_provider is not None and keep_states is not None:
        keep = filter_trials_by_state(session, state_provider, keep_states)

    sources = _region_sources(session, subject_full, use_qc)
    rows, traces, time_axis = [], {}, None

    for region, (sig, ts) in sources.items():
        for epoch, (evt, kind, bl) in EPOCHS.items():
            et = _event_times(session, evt, keep)
            if et.size == 0:
                continue
            t_ax, peth = extract_peth(sig, ts, et, window=PETH_WINDOW, baseline_window=bl)
            if time_axis is None:
                time_axis = t_ax
            mt = _mean_trace(peth)
            if mt is None:
                continue
            traces[(region, epoch)] = mt
            rows.append({"subject_id": subject_full, "genotype": genotype,
                         "region": region, "epoch": epoch, "change_size": np.nan,
                         "n_trials": int(peth.shape[0]),
                         **_metrics_for_trace(mt, t_ax, kind)})
        # change-size grading (Hit go-trials)
        for cs in CHANGE_SIZES:
            et = _change_hit_times_for_cs(session, cs, keep)
            if et.size == 0:
                continue
            t_ax, peth = extract_peth(sig, ts, et, window=PETH_WINDOW,
                                      baseline_window=CHANGE_BASELINE)
            mt = _mean_trace(peth)
            if mt is None:
                continue
            rows.append({"subject_id": subject_full, "genotype": genotype,
                         "region": region, "epoch": "change_hit_graded",
                         "change_size": float(cs), "n_trials": int(peth.shape[0]),
                         **_metrics_for_trace(mt, t_ax, "change")})

    return rows, traces, time_axis
```

- [ ] **Step 4: Run test, verify it passes**

Run: `py -m pytest tests/analysis/test_geometry.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/geometry.py tests/analysis/test_geometry.py
git commit -m "Add C2 geometry core: per-session mode-aware metrics + grading" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `geometry.py` — dataset build + push–pull + grading stats

**Files:**
- Modify: `src/visdetect_photom/analysis/geometry.py` (append)
- Modify: `tests/analysis/test_geometry.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_geometry.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_geometry.py -v`
Expected: FAIL — `build_geometry_dataset` / `run_pushpull_tests` not defined.

- [ ] **Step 3: Implement aggregation + stats**

Append to `src/visdetect_photom/analysis/geometry.py`:

```python
import pandas as pd
from visdetect_photom.analysis.group_statistics import (
    pushpull_sign_contrast, spearman_with_ci,
)


def build_geometry_dataset(sessions, *, use_qc=True, state_provider=None,
                           keep_states=None):
    """Aggregate per-session rows to a PER-MOUSE metrics DataFrame + grouped traces.

    Returns (per_mouse_df, traces_by_group, time_axis):
      per_mouse_df: one row per (subject_id, genotype, region, epoch, change_size),
                    metric columns averaged across that mouse's sessions.
      traces_by_group: {(genotype, region, epoch): [(subject_id, mean_trace), ...]}
                       (pooled epochs only; per-mouse mean traces).
      time_axis: shared 1-D axis.
    """
    all_rows = []
    trace_accum = defaultdict(lambda: defaultdict(list))  # (geno,region,epoch)->subj->[traces]
    time_axis = None

    for sess in sessions:
        rows, traces, t = compute_geometry_metrics_for_session(
            sess, use_qc=use_qc, state_provider=state_provider, keep_states=keep_states)
        if t is not None and time_axis is None:
            time_axis = t
        all_rows.extend(rows)
        if rows:
            geno = rows[0]["genotype"]
            subj = rows[0]["subject_id"]
            for (region, epoch), tr in traces.items():
                trace_accum[(geno, region, epoch)][subj].append(tr)

    if not all_rows:
        return pd.DataFrame(), {}, time_axis

    raw = pd.DataFrame(all_rows)
    group_cols = ["subject_id", "genotype", "region", "epoch", "change_size"]
    per_mouse = (raw.groupby(group_cols, dropna=False)[_METRIC_KEYS]
                    .mean().reset_index())

    traces_by_group = {}
    for key, subj_map in trace_accum.items():
        traces_by_group[key] = [
            (subj, np.nanmean(np.array(trs), axis=0)) for subj, trs in subj_map.items()
        ]
    return per_mouse, traces_by_group, time_axis


def run_pushpull_tests(per_mouse_df, metric="signed_auc"):
    """Per (region, epoch) D1-vs-D2 push–pull sign contrast over per-mouse values.

    Pooled epochs only (change_size is NaN). Returns a tidy DataFrame.
    """
    if per_mouse_df.empty:
        return pd.DataFrame()
    pooled = per_mouse_df[per_mouse_df["change_size"].isna()]
    out = []
    for (region, epoch), grp in pooled.groupby(["region", "epoch"]):
        d1 = grp[grp["genotype"] == "D1"][metric].values
        d2 = grp[grp["genotype"] == "D2"][metric].values
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "epoch": epoch, "metric": metric})
        out.append(res)
    return pd.DataFrame(out)


def run_grading(per_mouse_df, metric="signed_auc"):
    """Spearman(metric, log2(change_size)) per genotype x region over graded rows."""
    if per_mouse_df.empty:
        return pd.DataFrame()
    graded = per_mouse_df[per_mouse_df["epoch"] == "change_hit_graded"].dropna(
        subset=["change_size", metric])
    out = []
    for (geno, region), grp in graded.groupby(["genotype", "region"]):
        if grp["change_size"].nunique() < 3:
            continue
        x = np.log2(grp["change_size"].values.astype(float))
        y = grp[metric].values.astype(float)
        res = spearman_with_ci(x, y)
        res.update({"genotype": geno, "region": region, "metric": metric})
        out.append(res)
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_geometry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/geometry.py tests/analysis/test_geometry.py
git commit -m "Add C2 dataset builder, push-pull and grading stats" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Script `08_d1_d2_geometry.py` (thin CLI + figures) + smoke test

**Files:**
- Create: `scripts/analysis/photometry/08_d1_d2_geometry.py`
- Create: `tests/scripts/test_08_smoke.py`

- [ ] **Step 1: Implement the script**

Create `scripts/analysis/photometry/08_d1_d2_geometry.py`:

```python
"""C2 — D1/D2 Response Geometry (mode-aware push-pull + grading + commitment).

Per-region figures (rows = change / lick / anticipation blocks) + a cross-region
push-pull summary + stats CSVs. D1 and D2 are DIFFERENT animals: all push-pull
results are GROUP-LEVEL sign contrasts, never within-animal anticorrelation.

Usage:
    py scripts/analysis/photometry/08_d1_d2_geometry.py
    py scripts/analysis/photometry/08_d1_d2_geometry.py --no-qc
    py scripts/analysis/photometry/08_d1_d2_geometry.py --state-filter Engaged --state-results-dir results/hmm/BG_013
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import GENOTYPE_COLORS, MIN_PHOTOM_CSV_BYTES
from visdetect_photom.core.staging import load_staging_manifest, excluded_mice
from visdetect_photom.analysis.geometry import (
    build_geometry_dataset, run_pushpull_tests, run_grading,
)
from visdetect_photom.analysis.group_statistics import format_stats_table
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

TRACE_EPOCHS = ["change_hit", "change_miss", "hit_lick", "fa_lick",
                "anticipation_hit", "anticipation_miss", "anticipation_cr"]


def _aggregate(trace_list):
    """[(subj, trace)] -> grand mean/SEM over per-mouse traces (N=mice)."""
    if not trace_list:
        return None
    rows = np.array([tr for _, tr in trace_list])
    mean = np.nanmean(rows, axis=0)
    n = rows.shape[0]
    sem = np.nanstd(rows, axis=0, ddof=0) / np.sqrt(max(n, 1))
    return {"mean": mean, "sem": sem, "n_mice": n}


def _plot_traces(ax, t, agg_d1, agg_d2, title, xlabel="Time (s)"):
    for agg, geno in [(agg_d1, "D1"), (agg_d2, "D2")]:
        if agg is None:
            continue
        c = GENOTYPE_COLORS[geno]
        ax.plot(t, agg["mean"], color=c, lw=1.5, label=f"{geno} ({agg['n_mice']} mice)")
        ax.fill_between(t, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                        color=c, alpha=0.2)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Δ z-dF/F", fontsize=8)
    ax.legend(fontsize=7)
    sns.despine(ax=ax)


def _build_region_figure(region, traces_by_group, time_axis, out_dir):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"C2 — D1/D2 Response Geometry — {region}\n"
                 f"(D1 vs D2 are different animals: group-level sign contrast)", fontsize=12)
    gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.35)

    def agg(epoch):
        return (_aggregate(traces_by_group.get(("D1", region, epoch), [])),
                _aggregate(traces_by_group.get(("D2", region, epoch), [])))

    panels = [
        (0, 0, "change_hit", "Change (Hit)", "Time from change (s)"),
        (0, 1, "change_miss", "Change (Miss)", "Time from change (s)"),
        (0, 2, "anticipation_cr", "Anticipation (CR)", "Time from change (s)"),
        (1, 0, "hit_lick", "Hit lick", "Time from lick (s)"),
        (1, 1, "fa_lick", "FA lick", "Time from lick (s)"),
        (1, 2, "anticipation_hit", "Anticipation (Hit)", "Time from change (s)"),
    ]
    for r, c, epoch, title, xl in panels:
        d1, d2 = agg(epoch)
        _plot_traces(fig.add_subplot(gs[r, c]), time_axis, d1, d2, title, xl)

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"C2_geometry_{region}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {p}")


def _build_summary_figure(pushpull_df, out_dir, metric="signed_auc"):
    if pushpull_df.empty:
        return
    epochs = sorted(pushpull_df["epoch"].unique())
    regions = sorted(pushpull_df["region"].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5), squeeze=False)
    fig.suptitle(f"C2 — Push-pull sign summary ({metric}, D1 vs D2)", fontsize=12)
    x = np.arange(len(epochs))
    for ai, region in enumerate(regions):
        ax = axes[0][ai]
        sub = pushpull_df[pushpull_df["region"] == region].set_index("epoch")
        d1 = [sub.loc[e, "d1_mean"] if e in sub.index else np.nan for e in epochs]
        d2 = [sub.loc[e, "d2_mean"] if e in sub.index else np.nan for e in epochs]
        ax.bar(x - 0.2, d1, 0.4, color=GENOTYPE_COLORS["D1"], label="D1")
        ax.bar(x + 0.2, d2, 0.4, color=GENOTYPE_COLORS["D2"], label="D2")
        for xi, e in enumerate(epochs):
            if e in sub.index and bool(sub.loc[e, "opposite_sign"]):
                ax.text(xi, ax.get_ylim()[1] * 0.9, "*", ha="center", fontsize=14)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(epochs, rotation=45, ha="right", fontsize=7)
        ax.set_title(region, fontsize=10); ax.set_ylabel(metric, fontsize=8)
        ax.legend(fontsize=7); sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "C2_pushpull_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser(description="C2: D1/D2 Response Geometry")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "C2_d1_d2_geometry"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None,
                    help="comma-separated behavioral states to keep (default: pooled)")
    ap.add_argument("--state-results-dir", default=None,
                    help="HMM results dir for --state-filter")
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    use_qc = not args.no_qc
    out_dir = Path(args.output_dir)

    sessions_files = io.find_all_sessions(args.root_dir, recursive=True,
                                          min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(sessions_files)} session files.")

    manifest = load_staging_manifest()
    excl = excluded_mice(manifest)
    if excl:
        logging.info(f"Excluding mice (staging all-Excluded): {sorted(excl)}")

    state_provider, keep_states = None, None
    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir")
            sys.exit(1)
        state_provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
        logging.info(f"State filter: keep {keep_states}")
    else:
        state_provider = PooledStateProvider()
        keep_states = ["All"]

    sessions, n = [], 0
    for sf in sessions_files:
        if args.max_sessions and n >= args.max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.warning(f"skip {sf.get('trials','?')}: {e}")
            continue
        if f"BG_{str(sess.subject_id).zfill(3)}" in excl or sess.subject_id in excl:
            continue
        sessions.append(sess)
        n += 1
        if n % 20 == 0:
            logging.info(f"  loaded {n}")

    per_mouse, traces_by_group, time_axis = build_geometry_dataset(
        sessions, use_qc=use_qc, state_provider=state_provider, keep_states=keep_states)
    if time_axis is None:
        logging.error("No data extracted.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_mouse.to_csv(out_dir / "C2_geometry_metrics.csv", index=False)

    pushpull = run_pushpull_tests(per_mouse, metric="signed_auc")
    grading = run_grading(per_mouse, metric="signed_auc")
    if not pushpull.empty:
        pushpull.to_csv(out_dir / "C2_pushpull_stats.csv", index=False)
    if not grading.empty:
        grading.to_csv(out_dir / "C2_grading.csv", index=False)

    regions = sorted({k[1] for k in traces_by_group})
    for region in regions:
        _build_region_figure(region, traces_by_group, time_axis, out_dir)
    _build_summary_figure(pushpull, out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke test**

Create `tests/scripts/test_08_smoke.py`:

```python
import os
import subprocess
import sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "analysis", "photometry", "08_d1_d2_geometry.py")
DATA = os.path.join(REPO, "photom_data")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="photom_data/ not present")
def test_script_runs_on_small_subset(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--max_sessions", "3", "--output_dir", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "C2_geometry_metrics.csv").exists()
```

- [ ] **Step 3: Run the smoke test**

Run: `py -m pytest tests/scripts/test_08_smoke.py -v`
Expected: PASS (runs on local `photom_data/`), or SKIP if data absent. If it FAILS, read `proc.stderr` and fix the script.

- [ ] **Step 4: Run the full test suite**

Run: `py -m pytest tests/ -v`
Expected: all PASS/SKIP, no failures.

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/photometry/08_d1_d2_geometry.py tests/scripts/test_08_smoke.py
git commit -m "Add C2 script 08 (D1/D2 geometry CLI + figures) and smoke test" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during plan authoring)

**Spec coverage:** Block 1 sensory/grading → Tasks 5/6 (`change_hit/miss`, `change_hit_graded`, `run_grading`). Block 2 commitment → Task 5 (`hit_lick/fa_lick` + latencies). Block 3 anticipation → Task 5 (early-reference baseline `(-2,-1.5)`, ramp over `(-1.5,0)`). Mode-aware extractors → Task 1. Push–pull sign test → Tasks 1/6. StateProvider seam → Task 3 (default pooled; HMM lazy). Mouse exclusion via staging → Tasks 4/7. Figures + CSVs → Task 7. Per-mouse aggregation + permutation/effect-size → Tasks 1/6. All covered.

**Placeholder scan:** none — every code step is complete.

**Type consistency:** row schema keys (`subject_id, genotype, region, epoch, change_size, n_trials` + `_METRIC_KEYS`) are identical across `compute_geometry_metrics_for_session`, `build_geometry_dataset`, `run_pushpull_tests`, `run_grading`. `pushpull_sign_contrast` keys (`d1_mean, d2_mean, d1_sign, d2_sign, opposite_sign, p, ...`) match the summary-figure and test usage. `extract_peth` return order `(time_axis, matrix)` honored everywhere.

**Known caveats to verify at runtime (not blockers):** (1) extremely small N per region×genotype (esp. VLS) — permutation/bootstrap will return NaN p where n<2; panels still render. (2) `onset_latency` for `anticipation_*` is left NaN by design. (3) duplicate `session_name` rows in the manifest resolve to first-match (documented in `staging.py`).
