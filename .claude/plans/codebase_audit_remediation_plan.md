# Plan: Codebase Audit Remediation + Scientific Analysis Roadmap

## Context

A comprehensive audit of the Apr2023 photometry project revealed **3 CRITICAL**, **5 HIGH**, and **11 MEDIUM** issues across scientific correctness, code quality, and project organization. Additionally, comparison with the Sep2025 ephys project identified a clear analysis roadmap to achieve the same scientific aims (D1 vs D2 corticostriatal circuit function during perceptual learning) using the photometry dataset's unique strengths: multi-subject group statistics, genetically pure D1/D2 populations, and dual-site DMS+VLS recordings.

This plan is organized into three sequential phases: (A) Fix critical bugs, (B) Reorganize the codebase, (C) Build new publication-grade analyses.

---

## Phase A: Fix Critical Bugs (do first — everything else depends on correct data)

### A1. Fix SDT d' Computation (CRITICAL) ✅ DONE

**Problem**: Every d' calculation in the codebase was wrong. All three implementations used behavioral FA (premature lick) as the SDT false alarm rate instead of catch-trial hits. No code used `change_size` to separate go and catch trials.

**Files fixed**:
- `src/visdetect_photom/analysis/statistics.py` — new `calculate_sdt_metrics()` using change_size
- `scripts/analysis/behavior/plot_session_behavior.py` — now imports from `statistics.py`
- `scripts/behavior_metrics.py` — rewritten with correct Stim2TF-based SDT

**Fix applied**: Correct SDT classification:
```
SDT Hit Rate = count(outcome='Hit' AND change_size > 1.01) / count(change_size > 1.01 AND outcome in ['Hit','Miss'])
SDT FA Rate = count(outcome='Hit' AND change_size <= 1.01) / count(change_size <= 1.01 AND outcome in ['Hit','Miss'])
d' = z(hit_rate) - z(fa_rate), rates clipped to [0.01, 0.99]
criterion c = -0.5 * (z(hit_rate) + z(fa_rate))
```

### A2. Fix Legacy SavGol Even Window Length (HIGH)

**Problem**: `scripts/vis_detect_helpers_v9.py:577-578` uses `window_length=90` and `window_length=40` (even numbers). SavGol requires odd windows.

**Fix**: Change to 91 and 41 (matching the documented values and the new package).

### A3. Verify Outcome Label Casing (MEDIUM) ✅ DONE

**Problem**: JSON data uses `'abort'` (lowercase) and `'Ref'` instead of `'Abort'` and `'CR'`.

**Fix applied**: Normalization map in `constants.py` (`OUTCOME_NORMALIZATION`), applied at load time in `session.py:load_session_from_files()`. Also updated `behavior_metrics.py` with its own local normalization map.

---

## Phase B: Reorganize the Codebase

### B1. Create Central Constants Module ✅ DONE

**Created**: `src/visdetect_photom/core/constants.py`

Contains: SAMPLING_FREQ, TRIM_SECONDS, SAVGOL params, PETH_WINDOW, CHANGE_SIZES, CATCH_THRESHOLD, FA_RT_SPLIT, OUTCOME_LABELS, EVENT_VALID_OUTCOMES, OUTCOME_NORMALIZATION, ROI_TO_REGION, SUBJECT_GENOTYPE, color palettes.

### B2. Delete Duplicate Code

1. **Delete `scripts/photom_helpers.py`** — every function is duplicated in either `vis_detect_helpers_v9.py` or `src/visdetect_photom/`. Update any importers to use the new package.

2. **Remove verbatim copy functions from `scripts/photometry_analysis.py` lines 66-228** (`parse_trials_timestamp`, `parse_session_json_timestamp`, `parse_photom_timestamp`, `pair_session_files`, `find_all_sessions`, `infer_session_keys_from_paths`, `compute_peak_zdf_over_window`) — replace with imports from `src/visdetect_photom/`.

3. **Consolidate `_roi_to_region`** — exists in 3 files. Move to `constants.py`, import everywhere.

### B3. Reorganize File Structure

**Move scripts to proper subdirectories**:
```
scripts/
  legacy/                           ← NEW
    vis_detect_helpers_v9.py        ← MOVE from scripts/
    photometry_analysis.py          ← MOVE from scripts/
  analysis/
    behavior/
      plot_session_behavior.py      (keep)
      batch_run_behavior.py         (keep)
    photometry/                     ← NEW
      photom_group_analysis.py      ← MOVE from scripts/
      photom_report.py              ← MOVE from scripts/
  batch_processing/
    01_batch_session_analysis.py    (keep)
    02_aggregate_learning.py        (implement or delete)
    03_population_analysis.py       (implement or delete)
  data_management/
    filter_sessions.py              (keep)
    create_session_manifest.py      (keep)
    export_photom_matlab.py         (keep)
    copy_exported_figures.py        (keep)
    plot_subject_sessions_grid.py   (keep)
```

**Delete debug/scratch files**: `debug_read.py`, `debug_trials.py`, `check_matlab_export.py`, `plot_matlab_export_check.py`, `single_session_test.py`.

**Clean root directory**: Add to `.gitignore`: `output_*.log`, `*.7z`, `photometry_export_matlab.csv`, `photometry_export_check_plot.png`, `exported_figures/`, `__pycache__/`.

### B4. Replace Hardcoded Absolute Paths

In these files, replace `E:/python_analysis/git_repos/vis_detect_analysis_Apr2023/...` with `Path(__file__).resolve().parents[N]`-relative defaults:
- `01_batch_session_analysis.py:624-625`
- `02_aggregate_learning.py:44-45`
- `03_population_analysis.py:22-23`
- `batch_run_behavior.py:48-49`

### B5. Smaller Fixes

- Fix division-by-zero in legacy z-score (`vis_detect_helpers_v9.py:606`) — add `if std < 1e-6: std = 1.0` guard ✅ DONE (already present)
- Change `nan_to_num` to use NaN instead of 0.0 (`preprocessing.py:99`)
- Fix legacy `late_FAs` overlap bug (`vis_detect_helpers_v9.py:723`) — use `>` instead of `>=` at boundary
- Remove duplicate `flatten_nested_df` definition (`vis_detect_helpers_v9.py:501-514`)
- Remove unused imports from `vis_detect_helpers_v9.py` (`statistics`, `scipy.io`)
- Update legacy FA RT split from 2.0s to 3.0s

---

## Phase C: New Analyses — Photometry-Tailored Scientific Roadmap

These analyses leverage the photometry dataset's unique strengths (multi-subject, D1 vs D2 genotypes, DMS + VLS dual-site) to pursue the same scientific questions as the ephys project.

### Analysis Infrastructure (build first, used by all new analyses)

**C0a. Proper Statistics Module** — `src/visdetect_photom/analysis/group_statistics.py` (NEW)

Reusable functions:
- `mannwhitney_with_effect_size(x, y)` → U, p, rank-biserial r
- `kruskal_with_effect_size(*groups)` → H, p, η²_H
- `spearman_with_ci(x, y, n_boot=1000)` → ρ, p, CI95
- `bootstrap_ci(data, func, n_boot=1000, seed=42)` → CI95
- `wilcoxon_with_effect_size(x, y)` → W, p, matched-pairs r
- `permutation_test(x, y, n_perm=1000)` → observed, p, null_dist
- `format_stats_table(results)` → formatted markdown + CSV export

**C0b. Group Aggregation Utilities** — `src/visdetect_photom/analysis/group_utils.py` (NEW)

- `load_genotype_map()` → dict mapping subject_id to 'D1'/'D2'
- `aggregate_peth_by_genotype(sessions, event, genotype_map)` → D1_mean, D2_mean, per-mouse means
- `compute_session_summary(session)` → dict with peak dF/F per event, per ROI, d', RT, etc.
- Color palettes: `GENOTYPE_COLORS`, `REGION_COLORS`, `OUTCOME_COLORS`

### C1. D1 vs D2 Population Response Profiles (highest impact)

**Script**: `scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py`

**Panels**:
- A: Grand-mean D1 vs D2 PETH traces ± SEM, aligned to Change_ON (Hit trials), per region
- B: Same for Miss trials
- C: Same for FA lick time (early + late combined)
- D: Peak z-dF/F bar plot: D1 vs D2, per event type, per region, with Mann-Whitney U stats
- E: Hit-Miss difference traces (D1 vs D2) — the "outcome selectivity" signal
- F: Stats summary table

### C2. D1 vs D2 Impulsivity Signatures

**Script**: `scripts/analysis/photometry/02_d1_vs_d2_impulsivity.py`

**Panels**:
- A: D1 vs D2 PETH traces for Early FA (RT ≤ 3s) aligned to FA lick
- B: D1 vs D2 PETH traces for Late FA (RT > 3s) aligned to FA lick
- C: Pre-lick ramp magnitude (-1s to 0s) comparison: D1 vs D2, Early vs Late
- D: FA Early/Late ratio across sessions, D1 vs D2 (does impulsivity differ by genotype?)
- E: Stats table

### C3. Behavioral State Analysis (HMM)

**Script**: `scripts/analysis/behavior/03_hmm_behavioral_states.py`

Port the HMM fitting code from `visdetect/analysis/hmm.py` (Sep2025 repo) adapted for photometry trial structure:
- Fit K=2,3,4 GLM-HMM per mouse, select best K by BIC
- Compare state fractions (Engaged/Disengaged/Impulsive) between D1 and D2 groups
- Condition photometry PETHs on HMM state: does engagement state modulate D1/D2 differently?
- Track state prevalence across learning sessions

### C4. DMS vs VLS Regional Comparison

**Script**: `scripts/analysis/photometry/04_dms_vs_vls_comparison.py`

For dual-site mice:
- DMS vs VLS grand-mean PETHs per event, per genotype
- Region × genotype interaction tests (2×2 design)
- Trial-by-trial DMS-VLS correlation: Pearson r per trial type, compare across outcomes

### C5. Neural Psychometric Functions (Change-Size Dose Response)

**Script**: `scripts/analysis/photometry/05_neural_psychometric.py`

- Peak z-dF/F as a function of change size [1.25, 1.35, 1.5, 2.0, 4.0] for Hit trials
- Separate curves for D1 vs D2, DMS vs VLS
- Fit sigmoidal dose-response and compare slope/threshold across genotypes
- Overlay with behavioral psychometric curve (hit rate vs change_size)

### C6. Session-Level Learning Trajectories

**Script**: `scripts/analysis/photometry/06_learning_trajectories.py`

- Per-mouse time series of peak dF/F (per event), d', hit rate, FA rate across sessions
- Spearman trend tests with bootstrap CI per mouse
- Group-level: D1 vs D2 learning rate comparison (slope of d' vs session_index)
- Neural-behavioral coupling: Spearman rho of peak dF/F vs d' across sessions, per genotype

### C7. Baseline Prediction of Trial Outcome

**Script**: `scripts/analysis/photometry/07_baseline_outcome_prediction.py`

The photometry analog of the ephys "pre-trial state" analysis:
- Pre-change baseline dF/F (-1.0 to 0.0s) on Hit vs Miss trials
- ROC analysis: AUC for baseline dF/F predicting Hit vs Miss, per ROI, per genotype
- Compare D1 vs D2: does pre-change state matter more for one pathway?

### C8. Variance Partitioning (Photometry Analog of 2D Decomposition)

**Script**: `scripts/analysis/photometry/08_variance_partitioning.py`

The photometry-appropriate analog of the Lohse et al. task-state × sensory AND-gate:

Per session per ROI, fit logistic regression:
```
P(Hit) ~ β_task × baseline_dF/F + β_sensory × change_response_dF/F + β_interaction
```
- Compute unique variance explained by each predictor
- Track across sessions: does sensory component grow with learning?
- Compare D1 vs D2: does D1 carry more sensory variance? D2 more task-state?
- For dual-site mice: substitute DMS and VLS signals as the two dimensions

### C9. Proper Psychometric Curve Fitting

**Script**: `scripts/analysis/behavior/09_psychometric_fitting.py`

Replace the current linear slope with proper logistic function:
```
hit_rate(x) = lapse + (1 - lapse - guess) / (1 + exp(-slope * (log(x) - threshold)))
```
Extract threshold, slope, lapse rate per session per mouse. Track across learning. Compare D1 vs D2.

---

## Execution Order

```
Phase A (Critical fixes):          ~1 day
  A1. Fix SDT d' computation       ✅ DONE
  A2. Fix SavGol window
  A3. Verify outcome casing         ✅ DONE

Phase B (Reorganization):          ~1 day
  B1. Create constants module       ✅ DONE
  B2. Delete duplicates
  B3. Reorganize files
  B4. Fix hardcoded paths
  B5. Smaller fixes

Phase C (New analyses):            ~2-3 weeks
  C0. Statistics + group utilities  (infrastructure, do first)
  C1. D1 vs D2 response profiles   (highest impact)
  C2. D1 vs D2 impulsivity         (high impact)
  C9. Psychometric fitting          (quick, fixes existing gap)
  C5. Neural psychometric           (leverages C9)
  C6. Learning trajectories         (leverages C0)
  C7. Baseline outcome prediction   (quick, high impact)
  C4. DMS vs VLS comparison         (dual-site mice only)
  C8. Variance partitioning         (most novel, Lohse analog)
  C3. HMM behavioral states         (requires porting code)
```

## Verification

After each phase:
- **Phase A**: Regenerate session manifest, compare old vs new d' values. Verify d' makes scientific sense (positive for engaged sessions, near 0 for naive).
- **Phase B**: Run `01_batch_session_analysis.py` on one session — verify it still works. Check all imports resolve. Run `py -c "from visdetect_photom.core.constants import SAMPLING_FREQ; print(SAMPLING_FREQ)"`.
- **Phase C**: Each new script should produce: (1) a figure saved to `FIGURES/`, (2) a stats CSV with test results, (3) printed summary with effect sizes and sample sizes.
