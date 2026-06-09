# Plan: Codebase Audit Remediation + Scientific Analysis Roadmap

## Context

A comprehensive audit of the Apr2023 photometry project revealed **3 CRITICAL**, **5 HIGH**, and **11 MEDIUM** issues across scientific correctness, code quality, and project organization. Additionally, comparison with the Sep2025 ephys project identified a clear analysis roadmap to achieve the same scientific aims (D1 vs D2 corticostriatal circuit function during perceptual learning) using the photometry dataset's unique strengths: multi-subject group statistics, genetically pure D1/D2 populations, and dual-site DMS+VLS recordings.

This plan is organized into three sequential phases: (A) Fix critical bugs, (B) Reorganize the codebase, (C) Build new publication-grade analyses.

---

## Phase A: Fix Critical Bugs (do first — everything else depends on correct data)

### A1. Fix SDT d' Computation (CRITICAL) — DONE

**Problem**: Every d' calculation in the codebase is wrong. All three implementations use behavioral FA (premature lick) as the SDT false alarm rate instead of catch-trial hits. No code uses `change_size` to separate go and catch trials.

**Files fixed**:
- `src/visdetect_photom/analysis/statistics.py` — new `calculate_sdt_metrics()` using change_size
- `scripts/analysis/behavior/plot_session_behavior.py` — updated to use canonical SDT
- `scripts/behavior_metrics.py` — updated to use canonical SDT

**Fix implemented**: Correct SDT classification using change_size:
```
SDT Hit Rate = count(outcome='Hit' AND change_size > 1.01) / count(change_size > 1.01 AND outcome in ['Hit','Miss'])
SDT FA Rate = count(outcome='Hit' AND change_size <= 1.01) / count(change_size <= 1.01 AND outcome in ['Hit','Miss'])
d' = z(hit_rate) - z(fa_rate), rates clipped to [0.01, 0.99]
criterion c = -0.5 * (z(hit_rate) + z(fa_rate))
```

### A2. Fix Legacy SavGol Even Window Length (HIGH) — DONE

**Problem**: `scripts/vis_detect_helpers_v9.py:577-578` uses `window_length=90` and `window_length=40` (even numbers). SavGol requires odd windows.

**Fix**: Changed to 91 and 41 (matching the documented values and the new package).

### A3. Normalize Outcome Label Casing (MEDIUM) — DONE

**Problem**: JSON has `'abort'` (lowercase) and `'Ref'`, but code checks for `'Abort'` and `'CR'`.

**Fix**: Added `OUTCOME_NORMALIZATION` dict to `constants.py` and normalization at load time in `session.py:load_session_from_files()`.

---

## Phase B: Reorganize the Codebase

### B1. Create Central Constants Module — DONE

Created `src/visdetect_photom/core/constants.py` with all shared parameters:
- Photometry acquisition (sampling freq, trim)
- Signal processing (SavGol params)
- PETH extraction windows
- Task parameters (change sizes, catch threshold, FA RT split)
- Outcome labels and normalization
- ROI mapping, subject genotypes
- Visualization defaults (genotype/region/outcome colors)

### B2. Delete Duplicate Code — DONE

Deleted `scripts/photom_helpers.py` (every function was duplicated).

### B3. Reorganize File Structure (DEFERRED)

File moves require careful `git mv` and import updating. Deferred to avoid disrupting active analysis work.

### B4. Replace Hardcoded Absolute Paths — DONE

Replaced `E:/python_analysis/git_repos/vis_detect_analysis_Apr2023/...` with `Path(__file__).resolve().parents[N]`-relative defaults in 4 files, 9 occurrences.

### B5. Smaller Fixes — DONE

- Fixed nan_to_num to use NaN instead of 0.0 in `preprocessing.py`
- Fixed z-score guard to produce NaN for dead fibers in `preprocessing.py`
- Removed duplicate `flatten_nested_df` definition in legacy helpers
- Removed unused imports from legacy helpers

---

## Phase C: New Analyses — Photometry-Tailored Scientific Roadmap

### C0. Statistics + Group Utilities — DONE

Created:
- `src/visdetect_photom/analysis/group_statistics.py` — Mann-Whitney, Wilcoxon, Kruskal, Spearman, bootstrap, permutation tests
- `src/visdetect_photom/analysis/group_utils.py` — genotype map, session summary, PETH aggregation

### C1. D1 vs D2 Population Response Profiles — DONE

**Script**: `scripts/analysis/photometry/01_d1_vs_d2_response_profiles.py`

Panels: Change-aligned (Hit/Miss), FA-lick-aligned, peak z-dF/F bars with Mann-Whitney U, Hit-Miss difference traces, Early FA comparison. 4 figures (G0/G2/G4/G5), stats CSV, session summaries.

### C2. D1 vs D2 Impulsivity Signatures (PENDING)

**Script**: `scripts/analysis/photometry/02_d1_vs_d2_impulsivity.py`

### C3. Behavioral State Analysis — HMM (PENDING)

### C4. DMS vs VLS Regional Comparison (PENDING)

### C5. Neural Psychometric Functions (PENDING)

### C6. Session-Level Learning Trajectories (PENDING)

### C7. Baseline Prediction of Trial Outcome (PENDING)

### C8. Variance Partitioning (PENDING)

### C9. Proper Psychometric Curve Fitting (PENDING)

---

## Execution Order

```
Phase A (Critical fixes):          DONE
Phase B (Reorganization):          MOSTLY DONE (B3 deferred)
Phase C (New analyses):            IN PROGRESS
  C0. Statistics + group utilities  DONE
  C1. D1 vs D2 response profiles   DONE
  C2. D1 vs D2 impulsivity         NEXT
  C9. Psychometric fitting
  C5. Neural psychometric
  C6. Learning trajectories
  C7. Baseline outcome prediction
  C4. DMS vs VLS comparison
  C8. Variance partitioning
  C3. HMM behavioral states
```

## Verification

After each phase:
- **Phase A**: Regenerate session manifest, compare old vs new d' values. Verify d' makes scientific sense (positive for engaged sessions, near 0 for naive).
- **Phase B**: Run `01_batch_session_analysis.py` on one session — verify it still works. Check all imports resolve. Run `py -c "from visdetect_photom.core.constants import SAMPLING_FREQ; print(SAMPLING_FREQ)"`.
- **Phase C**: Each new script should produce: (1) a figure saved to `FIGURES/`, (2) a stats CSV with test results, (3) printed summary with effect sizes and sample sizes.
