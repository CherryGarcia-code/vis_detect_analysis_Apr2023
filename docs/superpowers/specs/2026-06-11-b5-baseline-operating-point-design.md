# B5 — D1/D2 Baseline Operating Point Across Learning — Design Spec

- **Date:** 2026-06-11
- **Status:** Design approved (brainstorm); pending plan
- **Question ID:** B5 (cross-repo question landscape)
- **Repo:** `vis_detect_analysis_Apr2023` (photometry; Aim 2 Approach 2)
- **Implementation prerequisite:** **C2 merged into `main`** (B5 reuses `core/staging.py`, `analysis/state_provider.py`, `analysis/group_statistics.py` extractors, region-source pattern). C2 is already in `main`.

---

## 1. Question

**Does learning re-set the D1 vs D2 striatal baseline operating point?** Lab hypotheses #1/#4: cortex (atMOs) adjusts the *tonic* D1/D2 baseline across Naive→Expert to clamp the task-state operating point and tune the impulsivity/sensitivity balance. Test whether the baseline level / structure shifts across learning, and whether D1 and D2 shift differently (an "operating-point reset", potentially opposite-signed).

Part of the spine: *how mice learn to suppress impulsivity and boost sensitivity.* B5 is the **longitudinal baseline** complement to C2 (static evoked geometry).

## 2. The core methodological tension (why two tracks)

The photometry is **session-z-scored** (`process_photometry_signals(session_zscored=True)` forces each session's dF/F to mean 0 / SD 1). That deliberately **removes the absolute baseline level** — the very quantity the hypothesis is about. So B5 runs **two complementary tracks**:

- **Track A — absolute level (within-mouse longitudinal).** Bypass z-scoring; use the raw isosbestic-corrected dF/F to measure the absolute baseline level, and ask whether it *changes* across learning **within each mouse**.
- **Track B — structural (z-score-robust).** Metrics that survive within-session normalization (variance, offset, ramp, modulation depth), as an independent cross-check.

**Concordance:** if A and B agree on direction → robust; if they diverge → flags Track A as a possible expression/bleaching artifact. This is the central interpretive safeguard.

## 3. Hard constraints (state in figures + paper)

1. **Absolute level is only comparable *within mouse*.** Bulk GCaMP level varies with expression, fiber coupling, and day-to-day re-coupling. Track A therefore reports only the **per-mouse trend** (slope/sign of baseline level vs session index), never absolute level across mice; aggregation is over per-mouse *trends*.
2. **D1/D2 are different animals** → any D1-vs-D2 "reset" is a **group-level** contrast of trends, never within-animal.
3. **Limited Expert coverage.** Most mice stay in Learning (only a couple reach Expert). → the continuous **session-index trend** is the robust primary; stage-categorical (Naive/Learning/Expert) means are secondary.
4. **Day-to-day bleaching/re-coupling** can mimic a slow baseline drift even within mouse → Track B concordance + reporting required; consider isosbestic-channel drift as a control.
5. **No movement regressors** (no video); session z-scoring rationale is exactly why Track A needs raw dF/F.

## 4. Scope (mirrors C2/G1)

| Dimension | Decision |
|---|---|
| Regions | DMS, VMS, VLS (low N flagged). |
| Genotype | D1 vs D2 (group-level trend contrast). |
| **Learning stage** | **PRIMARY axis.** Per-session stage from `core/staging` (Naive/Learning/Expert); chronological session index per mouse. |
| Behavioral state | Pooled default; optional `--state-filter` via `StateProvider`. |
| Mouse exclusion | BG_014 + any all-Excluded mouse (`core/staging.excluded_mice`). |
| Baseline periods | **Both** ITI (gray, tonic rest) **and** baseline-grating (pre-change, task-state). |
| Unit of replication | **Mouse** (per-mouse trend; group stats across mice). |

## 5. Metrics

### Baseline windows (per trial, in photometry `SystemTimestamp`)
- `baseline_onset = absolute_change_time − change_time` (= `absolute_start_time + iti_duration`).
- **ITI window:** `[absolute_start_time + ITI_TRIM, baseline_onset − ITI_END_PAD]` (ITI_TRIM=0.5 s drops the trial-transition transient; ITI_END_PAD=0.1 s).
- **Grating window:** `[baseline_onset + GRATING_ONSET_TRIM, t_end − margin]`, `t_end =` change (`absolute_change_time`, margin `GRATING_MARGIN_CHANGE`; Hit/Miss/CR) or first lick (`absolute_reaction_time`, margin `GRATING_MARGIN_LICK`; FA/Abort) — avoids onset-transient, pre-change ramp, and peri-lick motor.

> **Dependency note:** B5 must depend only on **C2** (merged), NOT on G1 (mid-flight, unmerged). The grating-window margins mirror the ephys/G1 convention in *value* but B5 defines its **own** constants (`GRATING_ONSET_TRIM=1.0`, `GRATING_MARGIN_CHANGE=1.0`, `GRATING_MARGIN_LICK=2.0`) so there is no import dependency on unmerged G1 code. (If both land, the small constant overlap is a trivial later cleanup.)

### Track A — absolute (raw, non-z-scored dF/F)
- `iti_level` = mean raw dF/F over ITI window.
- `grating_level` = mean raw dF/F over grating window.
Per session × region.

### Track B — structural (session-z-scored dF/F)
- `iti_sd` = SD of z-scored dF/F over ITI window (z-units).
- `iti_grating_offset` = mean(grating, z) − mean(ITI, z) (task-engagement offset).
- `anticipatory_ramp_slope` = `extract_ramp_slope` over `[change − 1.0, change]` on change-reaching trials (z/s).
- `modulation_depth` = `extract_signed_peak` over change-Hit post window `(0, 1.5)` (PETH, baseline-subtracted) − mean(ITI, z). *(The one structural metric that touches the evoked response; bridges to baseline→gain. Overlaps C2 conceptually — kept per the brainstorm's option-1 list.)*
Per session × region.

### Per-mouse learning trends (both tracks)
For each mouse × region × metric: order sessions chronologically; fit **slope vs session index** (Spearman ρ + linear slope) and compute **per-stage means**. Drop `Excluded` sessions.

## 6. Architecture (Approach A)

**`core/session.py`** (modify): add `keep_raw_dff: bool = False` to `load_session_from_files`. When True, also store the non-z-scored `{roi}_clean_signal_dff` traces (already computed in `process_photometry_signals`) as `session.raw_photometry_data` (dict `roi -> PhotometryTrace(signal_type="dff")`). Default False → no change to C2/G1/existing loads.

**`analysis/baseline.py`** (new):
- `baseline_windows(trial)` → `{"iti": (start, end), "grating": (start, end)}` (None where unavailable).
- `region_sources_dual(session, *, use_qc)` → `{region: {"z": (sig,ts), "raw": (sig,ts)}}` — merges hemispheres using the **same** QC verdict (decided on z-scored) applied to both z and raw traces.
- `extract_baseline_metrics(session, region_dual, *, state_keep=None)` → dict of Track A + Track B metrics for one session×region (reuses `extract_ramp_slope`, `extract_signed_peak`, `extract_peth`).
- `build_baseline_dataset(sessions, *, use_qc, state_provider, keep_states, manifest)` → per-session×region metrics DataFrame + `stage` + chronological `session_idx`.
- `fit_learning_trends(dataset)` → per-mouse×region×metric slope (Spearman + linear) + per-stage means → tidy DataFrame.
- `contrast_trends(trend_df, metric)` → D1-vs-D2 trend contrast via `pushpull_sign_contrast` (on per-mouse slopes) + `permutation_test`.

**`scripts/analysis/photometry/10_baseline_operating_point.py`** (thin CLI): discover → load (`keep_raw_dff=True`) → exclude → build dataset → trends → contrasts → concordance → figures/CSVs. Flags: `--no-qc`, `--state-filter`, `--state-results-dir`, `--max_sessions`, `--root_dir`, `--output_dir`.

## 7. Statistics

- Per-mouse slope (Spearman ρ of metric vs `session_idx`, + linear slope); test the **distribution of per-mouse slopes vs 0** (sign / one-sample permutation) per genotype×region×metric.
- **D1 vs D2**: `pushpull_sign_contrast` + `permutation_test` on per-mouse slopes (opposite-sign trends = "operating-point reset").
- Per-stage means with `bootstrap_ci`.
- **Concordance**: per genotype×region, agreement of Track A vs Track B slope signs (report a concordance table).
- Per-mouse N on every panel; flag mice with <3 usable sessions (no reliable trend).

## 8. Outputs

`FIGURES/B5_baseline_operating_point/`:
- Per region (`B5_<REGION>.png`): Track A — `iti_level` & `grating_level` vs session index, D1 vs D2 (per-mouse faint lines + group trend); Track B — `iti_sd`/`offset`/`ramp`/`modulation_depth` trajectories, D1 vs D2.
- Concordance summary (`B5_concordance.png`): Track A vs Track B slope signs per genotype×region.
- CSVs: `B5_metrics.csv` (per session×region), `B5_trends.csv` (per-mouse slopes + per-stage means), `B5_contrasts.csv` (D1-vs-D2), `B5_concordance.csv`.

## 9. Testing (TDD)

- `core/session`: `keep_raw_dff=True` stores `raw_photometry_data` with non-z-scored values (≠ z-scored); default False stores nothing (regression guard).
- `analysis/baseline`: `baseline_windows` bounds (ITI vs grating, margins); `extract_baseline_metrics` on a synthetic session (known raw level + known z structure → expected `iti_level`, `iti_grating_offset`, `iti_sd`); `build_baseline_dataset` returns stage + chronological index; `fit_learning_trends` recovers a planted positive slope; `contrast_trends` flags opposite-sign D1/D2 slopes.
- Script smoke (skip if `photom_data/` absent): runs on `--max_sessions 6`, writes `B5_trends.csv`.

## 10. Caveats (captions/paper)
1. Absolute level **within-mouse only**; aggregate trends, never cross-mouse levels.
2. D1/D2 different animals → group-level trend contrast.
3. Few mice reach Expert → session-index trend primary, stage means secondary.
4. Bleaching/re-coupling can masquerade as drift → Track B concordance is the guard; optionally report isosbestic drift.
5. `modulation_depth` overlaps C2's evoked territory — interpret as baseline→gain, not a new evoked claim.
6. No movement regressors.

## 11. References
- Program + cautions: `memory/photometry-question-landscape.md`, `memory/cross-repo-context.md`.
- Hypotheses: Sep2025 `memory/scientific_context.md` (lab hyp #1/#4), `memory/analysis_frontiers.md` (§4 baseline dSPN/iSPN), `synthesis-phase3-pathways.md`, `synthesis-phase3-bg-architecture.md` (RPE-grows / APE-decays divergent trajectories).
- Reused (in `main` via C2): `core/staging.py`, `analysis/state_provider.py`, `analysis/group_statistics.py` (`extract_ramp_slope`, `extract_signed_peak`, `pushpull_sign_contrast`, `permutation_test`, `bootstrap_ci`, `spearman_with_ci`), `analysis/statistics.extract_peth`, `core/qc.merge_hemispheres`, `analysis/group_utils.get_genotype`.
- Sibling specs: C2 `2026-06-08-c2-d1-d2-geometry-design.md`, G1 `2026-06-09-g1-tf-pulse-encoding-design.md`.
