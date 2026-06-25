# Phase-0 correctness audit — reused C1/C2 machinery

**Date:** 2026-06-23
**Method:** Workflow `wf_6f00c7d3-124` — 22 review dimensions across 6 modules, each on Opus/high-effort, every raw defect adversarially verified (default-refuted) by an independent Opus reviewer.
**Result:** 22 dimensions reviewed, 18 raw defects, **7 confirmed** after verification (6 Important, 1 Minor).

Confirmed defects below. "Bulk impact" = whether fixing it changes the already-reviewed bulk GCaMP8m C1/C2/G1 outputs. "Cohort impact" = whether it affects the BG_027–030 analysis being built.

---

## 1. Scheme-1 motor-buffer asymmetry (suppression.py ~L115) — Important
**Mechanism:** The 1.0 s motor buffer is applied only to `_ACTION_GROUPS=('lick','abort')`. But the `behavioral_fa` **withhold** group includes **Hit** trials, which contain a post-change lick. The only protection for withhold-Hit is the change-coincidence exclusion (`change_time > 3.0`), which clears the *change* from the window but not the *lick*. A fast Hit with `change_time` just above 3.0 (e.g. 3.05) and small RT lands its lick ~0.15 s after the (inclusive) window end — while a comparable FA lick trial is rejected unless its lick is ≥1.0 s after the window. → asymmetric imminent-motor contamination of `mean(withhold)`, biasing the central `delta` and `AUROC` C1 contrast.
**Recommended fix:** apply the same `SCHEME1_MOTOR_BUFFER` to withhold trials that have a lick (Hit): require the Hit-lick elapsed (`change_time + reaction_time`) to be ≥ `w1 + MOTOR_BUFFER`. Miss/CR withholds (no lick) unaffected.
**Bulk impact:** YES (bulk C1 scheme1 behavioral_fa). **Cohort impact:** YES.

## 2. Proficiency date-fallback bypasses the per-bin session floor (suppression.py ~L332) — Important
**Mechanism:** `assign_proficiency_bins` enforces `PROF_MIN_SESSIONS` (3) only on the staging branch. The date fallback requires only `n≥2`, so n=2 yields **one session per bin** — exactly the underpowered split the floor was meant to prevent. The downstream `MIN_TRIALS_PER_GROUP=8` scalar guard doesn't catch it (one dense session clears 8 scalars).
**Recommended fix:** require `≥ PROF_MIN_SESSIONS` sessions on *each* side of the date split too; otherwise assign `None` (no proficiency contrast for that mouse).
**Bulk impact:** YES (bulk C1 proficiency rows). **Cohort impact:** YES (cohort is unstaged → uses the fallback; though the cohort scripts currently compute but don't split on `prof_bin`).

## 3. Same-day session_id collision in proficiency binning (suppression.py ~L325) — Important
**Mechanism:** `session_id = subject_date` (date-granular; the codebase's own `_compute_io_offsets` + `get_session_stage` docstring confirm >1 recording/day happens). The staging branch builds a dict keyed by `session_id`, so two same-day recordings **collapse** (undercount → can spuriously fail the staging gate). The fallback's `bins[session_id]=...` **overwrites**, forcing both same-day recordings into one bin (and can empty the other bin).
**Recommended fix:** key stages/bins by a per-recording identity (trials path or an index), not the date-granular `session_id`. **Note:** this also affects the cohort's session-unit `compute_session_delta_and_auroc` (Task 6), which groups by `session_id` — same-day recordings would pool into one "session". Consider a unique per-recording id upstream.
**Bulk impact:** YES. **Cohort impact:** YES (cohort has same-day fragment sessions).

## 4. `run_grading` mouse-level pseudo-replication (geometry.py ~L266) — Important
**Mechanism:** Pools per-mouse graded rows (up to 5 change_sizes/mouse) into one Spearman per (genotype, region) without collapsing to a per-mouse statistic. N becomes (mice × change_size); `spearman_with_ci` reports `n=len(x)` and bootstraps those within-mouse points as independent → inflated significance, tight CI. The `nunique()≥3` gate checks change-size levels, not mice.
**Recommended fix:** compute one grading statistic per mouse (per-mouse Spearman over its change sizes), then aggregate across mice; or at minimum report N as mice and gate on mouse count. (Decision needed — see question.)
**Bulk impact:** YES (bulk C2 grading). **Cohort impact:** NO (cohort C2 script doesn't call `run_grading`).

## 5. `rank_biserial_r` sign inverted vs the mean-difference convention (group_statistics.py ~L406) — Important
**Mechanism:** `pushpull_sign_contrast` reports `observed = meanD1 − meanD2` (positive ⇒ D1>D2) but `rank_biserial_r = 1 − 2U/(n1·n2)` comes out **negative** when D1>D2. The two effect-direction signals in the same output row point opposite ways; a consumer pairing `d1_mean>d2_mean` with a negative `r` reads the contrast as D2>D1.
**Recommended fix:** negate so `rank_biserial_r` agrees with `meanD1 − meanD2` (i.e. positive ⇒ D1>D2). Existing tests never check the sign, so it's uncaught.
**Bulk impact:** YES (sign of reported effect size in bulk C1/C2/G1 CSVs). **Cohort impact:** YES (cross-compare uses push–pull sign).

## 6. `spearman_with_ci` CI poisoned by NaN (group_statistics.py ~L124) — Important
**Mechanism:** Uses `np.percentile` on the bootstrap rho array; a single constant resample (common at n≈5, the per-mouse regime) makes `spearmanr` return NaN, poisoning both CI bounds to NaN while the point estimate looks fine.
**Recommended fix:** `np.nanpercentile` (consistent with the non-parametric small-N intent).
**Bulk impact:** YES (bulk C2 grading CI). **Cohort impact:** NO (cohort doesn't call grading).

## 7. `extract_onset_latency` dead variable + docstring mismatch (group_statistics.py ~L302) — Minor
**Mechanism:** Computes `threshold = bl_mean + n_std*bl_std` but never uses it; the actual test is two-sided abs-deviation (correct — catches suppression). Docstring says one-sided "exceeds". **No wrong output** — documentation/dead-code only.
**Recommended fix:** remove the dead variable; fix the docstring to describe the two-sided test.
**Bulk impact:** NO (output unchanged). **Cohort impact:** NO.

---

## Disposition
- **Clear-cut, low-risk fixes (5 partially, 6, 7):** sign correction, `nanpercentile`, dead-code/docstring — apply with regression tests.
- **Method-level fixes needing a decision (1, 2, 3, 4):** they change the analysis method and alter bulk results; the fix shape for #3 (per-recording id) and #4 (per-mouse grading) involves a real choice. Surface to the user before patching shared code.
- All fixes ship TDD (failing regression test → fix → green), so the bulk suites gain coverage they currently lack.
