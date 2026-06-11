# B5 — D1/D2 Baseline Operating Point Across Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Test whether the D1 vs D2 striatal baseline operating point shifts across Naive→Expert, via two complementary tracks — absolute within-mouse level (raw dF/F) and z-score-robust structure — with a concordance cross-check.

**Architecture:** Approach A — opt-in raw (non-z-scored) dF/F on `Session`; analysis in `analysis/baseline.py`; thin CLI `scripts/.../10_baseline_operating_point.py`. Learning stage (from `core/staging`) is the primary axis; unit of replication = mouse.

**Tech Stack:** numpy<2, pandas<2.3, scipy<2, matplotlib<3.9, seaborn<0.14, pytest 9. No scikit-learn. `src/` layout via `tests/conftest.py`; invoke with `py`.

**Spec:** `docs/superpowers/specs/2026-06-11-b5-baseline-operating-point-design.md`.

---

## ⚠️ Prerequisite: C2 merged into `main` (G1 NOT required)

B5 depends on C2 only (`core/staging.py`, `analysis/state_provider.py`, `analysis/group_statistics.py` extractors, `analysis/group_utils._get_event_times`, `analysis/statistics.extract_peth`, `core/qc`). Verify:
`py -c "from visdetect_photom.core.staging import get_session_stage; from visdetect_photom.analysis.group_statistics import extract_ramp_slope, extract_signed_peak, pushpull_sign_contrast, spearman_with_ci; from visdetect_photom.analysis.group_utils import _get_event_times; print('C2 deps present')"`
**Do NOT import any G1 (`tf_kernel`/`stimulus`) symbols** — G1 may be unmerged. B5 defines its own window constants.

## Background

- **`Trial`**: `.absolute_start_time` (ITI/trial start), `.absolute_change_time`, `.change_time`, `.iti_duration`, `.absolute_reaction_time`, `.outcome`, `.trial_index`. `baseline_onset = absolute_change_time − change_time = absolute_start_time + iti_duration`.
- **z-scored dF/F** is on `Session.photometry_data` (default). **raw (non-z-scored) dF/F** must be opted-in (Task 1) → `Session.raw_photometry_data`.
- Reuse: `core/staging.{load_staging_manifest,get_session_stage,excluded_mice}`, `analysis/group_utils.{get_genotype,_get_event_times}`, `analysis/statistics.extract_peth`, `analysis/group_statistics.{extract_ramp_slope,extract_signed_peak,pushpull_sign_contrast,permutation_test,bootstrap_ci,spearman_with_ci}`, `core/qc.{compute_session_roi_qc,get_region_pairs_for_subject}`, `core/constants.get_roi_region`.

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/visdetect_photom/core/constants.py` | Modify | Append baseline-window constants. |
| `src/visdetect_photom/core/session.py` | Modify | `keep_raw_dff` flag + `raw_photometry_data`; factor `_build_traces_from_processed`. |
| `tests/core/test_session_raw_dff.py` | Create | Tests for the raw-dF/F trace builder + default-off guard. |
| `src/visdetect_photom/analysis/baseline.py` | Create | Windows, dual region sources, metrics, dataset, trends, contrast. |
| `tests/analysis/test_baseline.py` | Create | Unit tests for baseline analysis. |
| `scripts/analysis/photometry/10_baseline_operating_point.py` | Create | Thin CLI + figures. |
| `tests/scripts/test_10_smoke.py` | Create | Smoke test (skips if `photom_data/` absent). |

---

## Task 1: Raw dF/F on `Session` (opt-in) + constants

**Files:** Modify `core/constants.py`, `core/session.py`; Create `tests/core/test_session_raw_dff.py`.

- [ ] **Step 1: Append constants**

Append to `src/visdetect_photom/core/constants.py`:

```python
# ── Baseline operating point (B5) ──
ITI_TRIM = 0.5             # s trimmed from ITI start (trial-transition transient)
ITI_END_PAD = 0.1         # s before baseline onset to end the ITI window
GRATING_ONSET_TRIM = 1.0  # s after baseline onset to start the grating window
GRATING_MARGIN_CHANGE = 1.0   # s before change to end grating window (Hit/Miss/CR)
GRATING_MARGIN_LICK = 2.0     # s before FA/abort lick to end grating window
```

- [ ] **Step 2: Write the failing test**

Create `tests/core/test_session_raw_dff.py`:

```python
import numpy as np
import pandas as pd
from visdetect_photom.core.session import _build_traces_from_processed


def _df():
    ts = np.arange(0, 5, 0.01)
    raw = np.sin(ts) * 0.1
    z = (raw - raw.mean()) / raw.std()
    return pd.DataFrame({
        "SystemTimestamp": ts,
        "G0_clean_signal": raw * 2,
        "G0_clean_signal_dff": raw,
        "zscored_G0_clean_signal_dff": z,
    }), ts


def test_keep_raw_dff_builds_both():
    df, ts = _df()
    zt, rawt = _build_traces_from_processed(df, ts, keep_raw_dff=True)
    assert "G0" in zt and "G0" in rawt
    assert zt["G0"].signal_type == "zscored_dff"
    assert rawt["G0"].signal_type == "dff"
    assert not np.allclose(zt["G0"].signal, rawt["G0"].signal)


def test_default_off_no_raw():
    df, ts = _df()
    zt, rawt = _build_traces_from_processed(df, ts, keep_raw_dff=False)
    assert "G0" in zt and rawt == {}
```

- [ ] **Step 3: Run test, verify it fails**

Run: `py -m pytest tests/core/test_session_raw_dff.py -v`
Expected: FAIL — `_build_traces_from_processed` not defined.

- [ ] **Step 4: Implement**

In `src/visdetect_photom/core/session.py`:

(a) Add `raw_photometry_data` to the `Session` dataclass (next to `photometry_data`):

```python
    raw_photometry_data: Dict[str, PhotometryTrace] = field(default_factory=dict)
```

(b) Add the module-level helper (near the other module functions):

```python
def _build_traces_from_processed(processed_df, timestamps, keep_raw_dff=False):
    """Build z-scored (always) and raw (optional) PhotometryTrace dicts from processed columns."""
    z_traces, raw_traces = {}, {}
    suffix = "_clean_signal_dff"
    for col in processed_df.columns:
        if col.startswith("zscored_") and col.endswith(suffix):
            roi = col[len("zscored_"):-len(suffix)]
            z_traces[roi] = PhotometryTrace(roi_name=roi, timestamps=timestamps,
                                            signal=processed_df[col].values, signal_type="zscored_dff")
        elif keep_raw_dff and col.endswith(suffix) and not col.startswith("zscored_"):
            roi = col[:-len(suffix)]
            raw_traces[roi] = PhotometryTrace(roi_name=roi, timestamps=timestamps,
                                              signal=processed_df[col].values, signal_type="dff")
    return z_traces, raw_traces
```

(c) Add `keep_raw_dff: bool = False` to the `load_session_from_files` signature, and replace the existing per-column trace-building loop (the `for col in processed_df.columns: if 'zscored' in col ...` block that fills `photom_traces`) with:

```python
                photom_traces, raw_traces = _build_traces_from_processed(
                    processed_df, timestamps, keep_raw_dff=keep_raw_dff)
```

(d) Pass it into the `Session(...)` constructor at the end:

```python
        photometry_data=photom_traces,
        raw_photometry_data=raw_traces,
```

(Initialise `raw_traces = {}` before the `if photom_path:` block so it's always defined.)

- [ ] **Step 5: Run test, verify it passes**

Run: `py -m pytest tests/core/test_session_raw_dff.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Regression — existing loads unaffected**

Run: `py -m pytest tests/ -v`
Expected: all prior tests still PASS (default `keep_raw_dff=False` changes nothing).

- [ ] **Step 7: Commit**

```bash
git add src/visdetect_photom/core/constants.py src/visdetect_photom/core/session.py tests/core/test_session_raw_dff.py
git commit -m "Add opt-in raw dF/F on Session + baseline-window constants (B5)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `analysis/baseline.py` — windows, dual sources, metrics

**Files:** Create `src/visdetect_photom/analysis/baseline.py`, `tests/analysis/test_baseline.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_baseline.py`:

```python
import numpy as np
from types import SimpleNamespace
from visdetect_photom.analysis.baseline import (
    baseline_windows, region_sources_dual, extract_baseline_metrics,
)


def _trial(outcome="Hit", onset=100.0, iti=4.0, change_time=8.0, rt=0.5, idx=0):
    abs_start = onset - iti
    abs_change = onset + change_time
    abs_rt = onset + rt if outcome in ("FA", "Abort") else onset + change_time + rt
    return SimpleNamespace(trial_index=idx, outcome=outcome, change_time=change_time,
                           iti_duration=iti, reaction_time=rt, change_size=2.0,
                           absolute_start_time=abs_start, absolute_change_time=abs_change,
                           absolute_reaction_time=abs_rt)


def test_baseline_windows_iti_and_grating():
    w = baseline_windows(_trial(onset=100.0, iti=4.0, change_time=8.0))
    assert w["iti"][0] == 100.0 - 4.0 + 0.5 and abs(w["iti"][1] - (100.0 - 0.1)) < 1e-9
    assert w["grating"][0] == 101.0 and w["grating"][1] == 107.0   # onset+1, change-1


def test_baseline_windows_fa_uses_lick_margin():
    w = baseline_windows(_trial(outcome="FA", onset=100.0, iti=4.0, change_time=8.0, rt=6.0))
    assert w["grating"][1] == 100.0 + 6.0 - 2.0   # lick - 2.0


def _session_with_levels(raw_level=0.3):
    """One trial; raw dF/F flat at raw_level in baseline, z-scored ~0 baseline."""
    ts = np.arange(90.0, 115.0, 0.01)
    raw = np.full_like(ts, raw_level)
    z = np.zeros_like(ts)
    g0_z = SimpleNamespace(roi_name="G0", timestamps=ts, signal=z)
    g2_z = SimpleNamespace(roi_name="G2", timestamps=ts, signal=z.copy())
    g0_r = SimpleNamespace(roi_name="G0", timestamps=ts, signal=raw)
    g2_r = SimpleNamespace(roi_name="G2", timestamps=ts, signal=raw.copy())
    return SimpleNamespace(subject_id="013",
                           photometry_data={"G0": g0_z, "G2": g2_z},
                           raw_photometry_data={"G0": g0_r, "G2": g2_r},
                           trials=[_trial(onset=100.0)])


def test_region_sources_dual_merges_z_and_raw():
    src = region_sources_dual(_session_with_levels(), use_qc=False)
    assert "DMS" in src and src["DMS"]["z"] is not None and src["DMS"]["raw"] is not None


def test_extract_metrics_recovers_raw_level():
    sess = _session_with_levels(raw_level=0.3)
    src = region_sources_dual(sess, use_qc=False)
    m = extract_baseline_metrics(sess, src["DMS"])
    assert abs(m["iti_level"] - 0.3) < 1e-6
    assert abs(m["grating_level"] - 0.3) < 1e-6
    assert abs(m["iti_grating_offset"]) < 1e-6   # z flat -> offset ~0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_baseline.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `src/visdetect_photom/analysis/baseline.py`:

```python
"""B5 — D1/D2 baseline operating point across learning.

Track A (absolute, raw dF/F, within-mouse longitudinal) + Track B (z-score-robust
structure). Learning stage = primary axis. Depends only on C2 (not G1).
"""
from collections import defaultdict
import numpy as np

from visdetect_photom.core.constants import (
    ITI_TRIM, ITI_END_PAD, GRATING_ONSET_TRIM, GRATING_MARGIN_CHANGE,
    GRATING_MARGIN_LICK, get_roi_region,
)

_FA_LIKE = ("FA", "Abort")


def _subject_full(sid):
    s = str(sid)
    return f"BG_{s.zfill(3)}" if (not s.startswith("BG_") and s.isdigit()) else s


def baseline_windows(trial):
    """{'iti': (start,end)|None, 'grating': (start,end)|None} in SystemTimestamp."""
    if trial.absolute_change_time is None or trial.change_time is None or trial.absolute_start_time is None:
        return {"iti": None, "grating": None}
    onset = trial.absolute_change_time - trial.change_time
    iti = (trial.absolute_start_time + ITI_TRIM, onset - ITI_END_PAD)
    if iti[0] >= iti[1]:
        iti = None
    if trial.outcome in _FA_LIKE:
        end = (trial.absolute_reaction_time - GRATING_MARGIN_LICK) if trial.absolute_reaction_time is not None else None
    else:
        end = trial.absolute_change_time - GRATING_MARGIN_CHANGE
    grating = (onset + GRATING_ONSET_TRIM, end) if end is not None else None
    if grating is not None and grating[0] >= grating[1]:
        grating = None
    return {"iti": iti, "grating": grating}


def _merge(store, rois):
    traces = [store[r] for r in rois if r in store]
    if not traces:
        return None
    n = min(len(t.signal) for t in traces)
    sig = np.mean([np.asarray(t.signal, float)[:n] for t in traces], axis=0)
    return sig, np.asarray(traces[0].timestamps, float)[:n]


def region_sources_dual(session, *, use_qc=True):
    """{region: {'z': (sig,ts)|None, 'raw': (sig,ts)|None}} — same ROIs merged for z & raw."""
    subj = _subject_full(session.subject_id)
    z_store = session.photometry_data
    raw_store = getattr(session, "raw_photometry_data", {}) or {}
    region_rois = defaultdict(list)
    if use_qc:
        from visdetect_photom.core.qc import compute_session_roi_qc, get_region_pairs_for_subject
        qc = compute_session_roi_qc(session)
        for region, (l, r) in get_region_pairs_for_subject(subj).items():
            passing = [roi for roi in dict.fromkeys([l, r]) if qc.get(roi, {}).get("pass", False)]
            if passing:
                region_rois[region] = passing
    else:
        for roi in z_store:
            region = get_roi_region(roi, subj)
            if region:
                region_rois[region.rsplit("_", 1)[0]].append(roi)
    out = {}
    for region, rois in region_rois.items():
        out[region] = {"z": _merge(z_store, rois), "raw": _merge(raw_store, rois)}
    return out


def _win_mean(sig, ts, window):
    if window is None:
        return np.nan
    v = sig[(ts >= window[0]) & (ts <= window[1])]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else np.nan


def _win_samples(sig, ts, window):
    if window is None:
        return np.array([])
    v = sig[(ts >= window[0]) & (ts <= window[1])]
    return v[np.isfinite(v)]


_METRIC_KEYS = ("iti_level", "grating_level", "iti_sd",
                "iti_grating_offset", "anticipatory_ramp_slope", "modulation_depth")


def extract_baseline_metrics(session, region_dual, *, state_keep=None):
    """Track A + Track B metrics for one session x region."""
    from visdetect_photom.analysis.group_utils import _get_event_times
    from visdetect_photom.analysis.statistics import extract_peth
    from visdetect_photom.analysis.group_statistics import extract_ramp_slope, extract_signed_peak

    m = {k: np.nan for k in _METRIC_KEYS}
    z = region_dual.get("z")
    raw = region_dual.get("raw")
    if z is None:
        return m
    z_sig, z_ts = z

    iti_raw, grat_raw, iti_z_means, grat_z_means, iti_z_samples = [], [], [], [], []
    for t in session.trials:
        if state_keep is not None and t.trial_index not in state_keep:
            continue
        w = baseline_windows(t)
        if raw is not None:
            r_sig, r_ts = raw
            iti_raw.append(_win_mean(r_sig, r_ts, w["iti"]))
            grat_raw.append(_win_mean(r_sig, r_ts, w["grating"]))
        s = _win_samples(z_sig, z_ts, w["iti"])
        if s.size:
            iti_z_samples.append(s)
            iti_z_means.append(float(np.mean(s)))
        gm = _win_mean(z_sig, z_ts, w["grating"])
        if np.isfinite(gm):
            grat_z_means.append(gm)

    if iti_raw and np.any(np.isfinite(iti_raw)):
        m["iti_level"] = float(np.nanmean(iti_raw))
    if grat_raw and np.any(np.isfinite(grat_raw)):
        m["grating_level"] = float(np.nanmean(grat_raw))
    if iti_z_samples:
        m["iti_sd"] = float(np.std(np.concatenate(iti_z_samples)))
    iti_mean_z = float(np.mean(iti_z_means)) if iti_z_means else np.nan
    grat_mean_z = float(np.mean(grat_z_means)) if grat_z_means else np.nan
    if np.isfinite(iti_mean_z) and np.isfinite(grat_mean_z):
        m["iti_grating_offset"] = grat_mean_z - iti_mean_z

    # ramp + modulation via change-aligned PETH (whole-session events; state filter not applied here)
    ramp_ev = [_get_event_times(session, e) for e in ("change_hit", "change_miss", "change_cr")]
    ramp_ev = np.concatenate([e for e in ramp_ev if len(e)]) if any(len(e) for e in ramp_ev) else np.array([])
    if ramp_ev.size:
        t_ax, peth = extract_peth(z_sig, z_ts, ramp_ev, window=(-2.0, 4.0), baseline_window=(-2.0, -1.5))
        if peth.shape[0]:
            m["anticipatory_ramp_slope"] = extract_ramp_slope(np.nanmean(peth, axis=0), t_ax, (-1.0, 0.0))
    hit_ev = _get_event_times(session, "change_hit")
    if len(hit_ev):
        t_ax, peth = extract_peth(z_sig, z_ts, hit_ev, window=(-2.0, 4.0), baseline_window=(-2.0, 0.0))
        if peth.shape[0]:
            peak = extract_signed_peak(np.nanmean(peth, axis=0), t_ax, (0.0, 1.5))
            m["modulation_depth"] = (peak - iti_mean_z) if np.isfinite(iti_mean_z) else peak
    return m
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_baseline.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/baseline.py tests/analysis/test_baseline.py
git commit -m "Add baseline windows, dual region sources, and metrics (B5)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `analysis/baseline.py` — dataset, trends, contrast

**Files:** Modify `src/visdetect_photom/analysis/baseline.py`, `tests/analysis/test_baseline.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/analysis/test_baseline.py`:

```python
import pandas as pd
from visdetect_photom.analysis.baseline import fit_learning_trends, contrast_trends


def _trend_df():
    # 2 D1 mice rising, 2 D2 mice falling, in DMS, metric iti_level
    rows = []
    for subj, geno, direction in [("BG_013", "D1", +1), ("BG_020", "D1", +1),
                                  ("BG_016", "D2", -1), ("BG_018", "D2", -1)]:
        for idx in range(6):
            rows.append({"subject_id": subj, "genotype": geno, "region": "DMS",
                         "session_idx": idx, "stage": "Learning",
                         "iti_level": direction * 0.05 * idx})
    return pd.DataFrame(rows)


def test_fit_learning_trends_recovers_slope_sign():
    tr = fit_learning_trends(_trend_df(), metrics=["iti_level"])
    d1 = tr[(tr.subject_id == "BG_013") & (tr.metric == "iti_level")].iloc[0]
    assert d1["slope"] > 0
    d2 = tr[(tr.subject_id == "BG_016") & (tr.metric == "iti_level")].iloc[0]
    assert d2["slope"] < 0


def test_contrast_trends_flags_opposite():
    tr = fit_learning_trends(_trend_df(), metrics=["iti_level"])
    c = contrast_trends(tr, metric="iti_level")
    row = c[c.region == "DMS"].iloc[0]
    assert row["d1_sign"] == 1 and row["d2_sign"] == -1
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `py -m pytest tests/analysis/test_baseline.py -k "trend or contrast" -v`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement (append to `analysis/baseline.py`)**

```python
import pandas as pd


def build_baseline_dataset(sessions, *, use_qc=True, state_provider=None,
                           keep_states=None, manifest=None):
    """Per-session x region metrics + stage + chronological session_idx (per mouse)."""
    from visdetect_photom.analysis.group_utils import get_genotype
    from visdetect_photom.core.staging import get_session_stage
    from visdetect_photom.analysis.state_provider import filter_trials_by_state

    rows = []
    for sess in sessions:
        subj = _subject_full(sess.subject_id)
        geno = get_genotype(subj)
        if geno == "Unknown":
            continue
        keep = None
        if state_provider is not None and keep_states is not None:
            keep = filter_trials_by_state(sess, state_provider, keep_states)
        stage = get_session_stage(sess, manifest) if manifest is not None else "Unknown"
        if stage == "Excluded":
            continue
        sources = region_sources_dual(sess, use_qc=use_qc)
        for region, dual in sources.items():
            m = extract_baseline_metrics(sess, dual, state_keep=keep)
            rows.append({"subject_id": subj, "genotype": geno, "region": region,
                         "session_date": getattr(sess, "session_date", ""), "stage": stage, **m})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # chronological index per mouse (across all its sessions, region-agnostic ordering by date)
    df["session_date"] = df["session_date"].astype(str)
    order = (df[["subject_id", "session_date"]].drop_duplicates()
             .sort_values(["subject_id", "session_date"]))
    order["session_idx"] = order.groupby("subject_id").cumcount()
    return df.merge(order, on=["subject_id", "session_date"], how="left")


def fit_learning_trends(dataset, metrics=None):
    """Per (subject, region, metric): Spearman rho + linear slope vs session_idx + per-stage means."""
    from visdetect_photom.analysis.group_statistics import spearman_with_ci
    if metrics is None:
        metrics = list(_METRIC_KEYS)
    out = []
    for (subj, region), g in dataset.groupby(["subject_id", "region"]):
        geno = g["genotype"].iloc[0]
        for metric in metrics:
            sub = g[["session_idx", metric, "stage"]].dropna(subset=[metric])
            if sub["session_idx"].nunique() < 3:
                continue
            x = sub["session_idx"].values.astype(float)
            y = sub[metric].values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0])
            rho = spearman_with_ci(x, y)["rho"]
            row = {"subject_id": subj, "genotype": geno, "region": region,
                   "metric": metric, "slope": slope, "spearman_rho": rho, "n_sessions": len(sub)}
            for st in ("Naive", "Learning", "Expert"):
                v = sub[sub["stage"] == st][metric]
                row[f"mean_{st}"] = float(v.mean()) if len(v) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def contrast_trends(trend_df, metric="iti_level"):
    """D1-vs-D2 contrast of per-mouse slopes per region (push-pull sign + permutation)."""
    from visdetect_photom.analysis.group_statistics import pushpull_sign_contrast
    if trend_df.empty:
        return pd.DataFrame()
    sub = trend_df[trend_df["metric"] == metric]
    out = []
    for region, g in sub.groupby("region"):
        d1 = g[g["genotype"] == "D1"]["slope"].values
        d2 = g[g["genotype"] == "D2"]["slope"].values
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "metric": metric})
        out.append(res)
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `py -m pytest tests/analysis/test_baseline.py -v`
Expected: PASS (7 passed total).

- [ ] **Step 5: Commit**

```bash
git add src/visdetect_photom/analysis/baseline.py tests/analysis/test_baseline.py
git commit -m "Add baseline dataset, learning trends, and D1/D2 contrast (B5)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Script `10_baseline_operating_point.py` + smoke

**Files:** Create `scripts/analysis/photometry/10_baseline_operating_point.py`, `tests/scripts/test_10_smoke.py`.

- [ ] **Step 1: Implement the script**

Create `scripts/analysis/photometry/10_baseline_operating_point.py`:

```python
"""B5 — D1/D2 baseline operating point across learning.

Track A (absolute within-mouse raw-dF/F level) + Track B (z-score-robust structure),
learning stage as the primary axis. D1/D2 are different animals: trend contrasts are
group-level; absolute levels are never compared across mice.

Usage:
    py scripts/analysis/photometry/10_baseline_operating_point.py
    py scripts/analysis/photometry/10_baseline_operating_point.py --no-qc
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
from visdetect_photom.analysis.baseline import (
    build_baseline_dataset, fit_learning_trends, contrast_trends, _METRIC_KEYS,
)
from visdetect_photom.analysis.group_statistics import format_stats_table
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

TRACK_A = ["iti_level", "grating_level"]
TRACK_B = ["iti_sd", "iti_grating_offset", "anticipatory_ramp_slope", "modulation_depth"]


def main():
    ap = argparse.ArgumentParser(description="B5: baseline operating point across learning")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "B5_baseline_operating_point"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None)
    ap.add_argument("--state-results-dir", default=None)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    use_qc = not args.no_qc
    out = Path(args.output_dir)
    manifest = load_staging_manifest()
    excl = excluded_mice(manifest)

    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir"); sys.exit(1)
        provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
    else:
        provider, keep_states = PooledStateProvider(), ["All"]

    files = io.find_all_sessions(args.root_dir, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(files)} session files.")

    def _excluded(sf):
        s = str(sf.get("trials", ""))
        return any((m.replace("BG_", "") in s or m in s) for m in excl)
    files = [f for f in files if not _excluded(f)]

    sessions, n = [], 0
    for sf in files:
        if args.max_sessions and n >= args.max_sessions:
            break
        try:
            sessions.append(load_session_from_files(sf, keep_raw_dff=True))
            n += 1
        except Exception as e:
            logging.warning(f"skip {sf.get('trials','?')}: {e}")
        if n % 20 == 0 and n:
            logging.info(f"  loaded {n}")

    ds = build_baseline_dataset(sessions, use_qc=use_qc, state_provider=provider,
                                keep_states=keep_states, manifest=manifest)
    if ds.empty:
        logging.error("No baseline data extracted."); sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out / "B5_metrics.csv", index=False)
    trends = fit_learning_trends(ds)
    trends.to_csv(out / "B5_trends.csv", index=False)

    contrasts = []
    for metric in _METRIC_KEYS:
        c = contrast_trends(trends, metric=metric)
        if not c.empty:
            contrasts.append(c)
    if contrasts:
        allc = pd.concat(contrasts, ignore_index=True)
        allc.to_csv(out / "B5_contrasts.csv", index=False)

    # concordance: per genotype x region, sign agreement of Track A vs Track B slopes
    conc = []
    for (geno, region), g in trends.groupby(["genotype", "region"]):
        a = g[g.metric.isin(TRACK_A)]["slope"].mean()
        b = g[g.metric.isin(TRACK_B)]["slope"].mean()
        conc.append({"genotype": geno, "region": region,
                     "trackA_mean_slope": a, "trackB_mean_slope": b,
                     "concordant_sign": bool(np.isfinite(a) and np.isfinite(b) and (np.sign(a) == np.sign(b)))})
    pd.DataFrame(conc).to_csv(out / "B5_concordance.csv", index=False)

    # ── figures: per region, metric trajectories D1 vs D2 ──
    for region in sorted(ds["region"].unique()):
        metrics = TRACK_A + TRACK_B
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"B5 — baseline operating point — {region}\n"
                     f"(D1/D2 different animals; Track A absolute=within-mouse only)", fontsize=12)
        for ax, metric in zip(axes.ravel(), metrics):
            sub = ds[ds.region == region]
            for geno in ("D1", "D2"):
                gg = sub[sub.genotype == geno]
                for subj, s in gg.groupby("subject_id"):
                    s = s.dropna(subset=[metric]).sort_values("session_idx")
                    if len(s) >= 2:
                        ax.plot(s["session_idx"], s[metric], color=GENOTYPE_COLORS[geno],
                                alpha=0.35, lw=0.8)
            ax.set_title(metric, fontsize=9)
            ax.set_xlabel("session index"); sns.despine(ax=ax)
        # legend proxy
        axes.ravel()[0].plot([], [], color=GENOTYPE_COLORS["D1"], label="D1")
        axes.ravel()[0].plot([], [], color=GENOTYPE_COLORS["D2"], label="D2")
        axes.ravel()[0].legend(fontsize=8)
        if len(metrics) < 6:
            for ax in axes.ravel()[len(metrics):]:
                ax.axis("off")
        p = out / f"B5_{region}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        logging.info(f"Saved {p}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the smoke test**

Create `tests/scripts/test_10_smoke.py`:

```python
import os
import subprocess
import sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "analysis", "photometry", "10_baseline_operating_point.py")
DATA = os.path.join(REPO, "photom_data")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="photom_data/ not present")
def test_runs_on_small_subset(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--max_sessions", "6", "--output_dir", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "B5_trends.csv").exists()
```

- [ ] **Step 3: Run smoke + full suite**

Run: `py -m pytest tests/scripts/test_10_smoke.py -v` then `py -m pytest tests/ -v`
Expected: PASS/SKIP, no failures. If the script errors, read `proc.stderr` and fix.

- [ ] **Step 4: Commit**

```bash
git add scripts/analysis/photometry/10_baseline_operating_point.py tests/scripts/test_10_smoke.py
git commit -m "Add B5 script 10 (baseline operating point CLI + figures) and smoke test" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed during authoring)

**Spec coverage:** Track A (raw levels) → Task 1 (opt-in raw dF/F) + Task 2 (`iti_level`/`grating_level`). Track B → Task 2 (`iti_sd`, `iti_grating_offset`, `anticipatory_ramp_slope`, `modulation_depth`). Both ITI + grating windows → Task 2 (`baseline_windows`). Stage as primary axis + chronological index → Task 3 (`build_baseline_dataset`). Per-mouse trends + D1/D2 contrast + concordance → Task 3 + Task 4. Scope (regions/state/exclusion) → Task 4. All covered.

**Placeholder scan:** none — all code complete; numpy-only; no G1 imports.

**Type consistency:** `region_sources_dual` returns `{region: {"z","raw"}}` consumed by `extract_baseline_metrics`; metric keys `_METRIC_KEYS` are identical across `extract_baseline_metrics`, `build_baseline_dataset`, `fit_learning_trends`, and the script; `contrast_trends` consumes `trend_df` columns (`genotype`,`region`,`metric`,`slope`) produced by `fit_learning_trends`; `pushpull_sign_contrast` keys (`d1_sign`,`d2_sign`,`opposite_sign`,`p`) used in the contrast CSV. `_build_traces_from_processed` returns `(z, raw)` consumed by `load_session_from_files`.

**Known simplifications (not blockers):** (1) `anticipatory_ramp_slope`/`modulation_depth` use whole-session change events (state filter not applied to those two PETH-based metrics) — moot under the default pooled provider; documented. (2) Few mice reach Expert → trend-vs-session-index is primary; per-stage means may be sparse (NaN where a stage is absent). (3) Absolute Track-A levels are within-mouse only — the figure plots per-mouse trajectories vs session index, never pooled absolute levels.
