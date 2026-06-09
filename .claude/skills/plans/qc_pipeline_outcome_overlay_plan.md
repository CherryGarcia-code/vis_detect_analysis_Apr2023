# Plan: Photometry QC Pipeline + Outcome-Comparison Overlay Figures

## Context

The C1 analysis (D1 vs D2 response profiles) runs on all sessions/ROIs without signal quality filtering. Some ROIs have flat or noisy signals that dilute group averages. Left/right hemisphere ROIs (G0/G2 for DMS, G4/G5 for VLS) are always plotted separately — never merged — even when they measure the same structure.

The user also wants overlay figures comparing different outcomes (Hit, Miss, FA) on the same subplot with proper shared-baseline normalization per the CLAUDE.md golden rule.

**What already exists (user-written):** `src/visdetect_photom/core/qc.py` — a complete QC module with:
- `compute_trace_qc()` — variance, NaN fraction, SNR (slow/fast decomposition), baseline drift, pass/fail
- `compute_session_roi_qc()` — runs QC on all ROIs in a Session
- `get_passing_rois()` — returns list of passing ROI names
- `merge_hemispheres()` — L/R merging per region (both pass → average, one passes → use it, neither → skip)
- `check_behavioral_engagement()` — min go trials, hit rate, abort fraction checks
- `DEFAULT_QC_THRESHOLDS` and `REGION_PAIRS` constants (in qc.py, not constants.py)

---

## Feature 1: Integrate QC into Analysis Pipeline

### Step 1: Add `change_cr` event type

**File:** `src/visdetect_photom/analysis/group_utils.py` — `_get_event_times()`

Add support for CR trials aligned to the nominal change time (needed for Feature 2):
```python
elif event_type == 'change_cr' and t.outcome == 'CR':
    if t.absolute_change_time is not None:
        times.append(t.absolute_change_time)
```

### Step 2: Add QC metrics to `compute_session_summary()`

**File:** `src/visdetect_photom/analysis/group_utils.py`

After computing behavioral metrics, call `compute_session_roi_qc(session)` and add per-ROI columns:
- `qc_{roi}_passed`, `qc_{roi}_variance`, `qc_{roi}_snr`, `qc_{roi}_nan_frac`
- `n_rois_passing` (count of ROIs that pass)

This makes QC data available in session manifest CSVs for the YAML-based filter pipeline.

### Step 3: Add hemisphere-merged PETH extraction helper

**File:** `src/visdetect_photom/core/qc.py` — new function `extract_merged_region_peths()`

```python
def extract_merged_region_peths(session, event_type, qc_results=None,
                                 window=(-2,4), baseline_window=(-2,0)):
    """
    For each region (DMS, VLS), merge hemispheres via QC, then extract PETHs.

    Returns: {region: (peth_matrix, time_axis, merge_strategy)}
    """
```

Strategy: use `merge_hemispheres()` to get a merged signal per region, then call `extract_peth()` on the merged signal. This produces one PETH matrix per region instead of per ROI.

### Step 4: Standardize trial-level QC threshold

The C1 script uses `>50% finite` (line 135); `aggregate_peth_by_group` uses `not all NaN` (line 188). Standardize to **>50% finite** everywhere. Add a constant:
```python
# in qc.py
MIN_TRIAL_VALID_FRACTION = 0.5
```

---

## Feature 2: Outcome-Comparison Overlay Figures

### Normalization Design Decision

**Key principle:** Only overlay conditions that share the same alignment event and baseline context.

| Subplot | Conditions | Alignment | Baseline |
|---------|-----------|-----------|----------|
| Change-aligned | Hit, Miss, CR | Change onset | [-2, 0]s pre-change |
| Lick-aligned | Hit-lick, FA-lick | Lick time | [-2, 0]s pre-lick |
| FA subtypes | Early FA, Late FA | FA lick time | [-2, 0]s pre-FA-lick |

Within each subplot, all conditions use the **same event alignment and same baseline window** → per-trial z-scoring is scientifically valid because the baseline context is identical. No pooled-baseline machinery needed.

### New script: `scripts/analysis/photometry/02_outcome_comparison.py`

**Figure layout:** One figure per region (DMS, VLS), 3 rows × 2 columns:
- Rows: Change-aligned, Lick-aligned, FA subtypes
- Columns: D1 genotype, D2 genotype
- Each cell: overlaid outcome traces with `OUTCOME_COLORS`, mean ± SEM shading

**Condition colors** (from `constants.py` + extensions):
```python
CONDITION_COLORS = {
    'Hit': '#2ca02c',      # green
    'Miss': '#9467bd',     # purple
    'FA': '#d62728',       # red
    'CR': '#17becf',       # cyan
    'Early FA': '#ff7f0e', # orange
    'Late FA': '#8c564b',  # brown
}
```

**Data collection:** `collect_region_peths_across_sessions()`
1. Load session → determine genotype (skip Unknown)
2. Run `compute_session_roi_qc()` + `check_behavioral_engagement()` → skip failing sessions
3. For each event type, call `extract_merged_region_peths()` → one merged PETH per region
4. Apply >50% valid trial filter
5. Store `(subject_id, trial_trace)` keyed by `[genotype][region][event_type]`

**Aggregation:** Reuse C1's `aggregate_traces()` pattern (grand mean, SEM, per-mouse means).

**Plotting:** `plot_outcome_overlay(ax, time_axis, condition_aggs, title)` — loops over conditions, plots each as `ax.plot()` + `ax.fill_between()` with condition-specific color.

**Statistics per subplot:** Pairwise Mann-Whitney U on per-mouse peak z-dF/F in [0, 1.5]s:
- Change-aligned: Hit vs Miss, Hit vs CR
- Lick-aligned: Hit-lick vs FA-lick
- FA subtypes: Early FA vs Late FA

**Outputs:**
- `FIGURES/C2_outcome_comparison/C2_outcome_DMS.png`
- `FIGURES/C2_outcome_comparison/C2_outcome_VLS.png` (if VLS data exists)
- `FIGURES/C2_outcome_comparison/C2_stats_summary.csv`
- `FIGURES/C2_outcome_comparison/C2_qc_summary.csv` (per-session per-ROI QC results)

---

## Implementation Sequence

```
Step 1: Add change_cr to _get_event_times()                   [group_utils.py]
Step 2: Add QC metrics to compute_session_summary()           [group_utils.py]
Step 3: Add extract_merged_region_peths() + MIN_TRIAL_VALID   [qc.py]
Step 4: Create 02_outcome_comparison.py                       [new script]
Step 5: Run on full dataset, inspect figures + QC summary
```

## Verification

1. **QC sanity check**: Run `compute_session_roi_qc()` on all 148 sessions, save CSV with all metrics. Inspect distribution of variance, SNR, NaN fraction. Confirm dead/flat fibers are flagged.
2. **Hemisphere merging validation**: Log L/R Pearson r for every merged region. Confirm most are highly correlated (r > 0.5).
3. **Outcome overlay correctness**: Verify that Hit > Miss > CR in the change-aligned subplot (expected from task structure). Verify FA lick-aligned traces show pre-lick ramp.
4. **Stats**: All pairwise comparisons saved with U, p, effect size, n1, n2.

## Key Files

| File | Action |
|------|--------|
| `src/visdetect_photom/core/qc.py` | Add `extract_merged_region_peths()`, `MIN_TRIAL_VALID_FRACTION` |
| `src/visdetect_photom/analysis/group_utils.py` | Add `change_cr` event, QC in session summary |
| `scripts/analysis/photometry/02_outcome_comparison.py` | **NEW** — outcome overlay figures |
| `src/visdetect_photom/core/constants.py` | No changes needed (QC thresholds live in qc.py) |
