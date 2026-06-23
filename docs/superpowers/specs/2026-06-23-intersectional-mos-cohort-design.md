# Intersectional MOs-recipient SPN cohort (BG_027–030) — C1+C2 in a 2×2

**Date:** 2026-06-23
**Status:** Design (pre-plan)
**Cohort:** BG_027–030 — input-defined, intersectional **GCaMP6f** in the MOs-recipient subset of D1/D2 SPNs.
**Reuses:** the audited C1 (`suppression.py`) and C2 (`geometry.py`) machinery + shared `group_statistics`/`qc`/`statistics` primitives. Defines a separate cohort; **never pools with the bulk GCaMP8m cohort (BG_008–020).**

---

## 0. Prerequisite (base-branch coordination)

The C1 *code* (`src/visdetect_photom/analysis/suppression.py`, `scripts/analysis/photometry/11_fa_suppression.py`) is currently present **only on local `main` (b2736d2)** — bundled with the parallel G1 chat's unpushed commits. `origin/main` (0843b35) carries the C1 *docs* but not the code. C2 (`geometry.py`) and the shared primitives (`group_statistics.py`, `qc.py`, `statistics.py`) **are** on origin/main.

**Consequence for implementation (not for this spec):** the cohort implementation must branch from a commit that contains the C1 code. Preferred order: (a) the C1 code is pushed to `origin/main`, then the implementation worktree bases off origin/main; or (b) base the implementation worktree on local `main` (b2736d2). This design document itself is base-independent and is authored on a worktree off `origin/main`.

---

## 1. Motivation — the proposal's most direct test

The bulk GCaMP8m cohort tests "D1 vs D2" with cell-type identity but **anatomically undifferentiated** SPNs. BG_027–030 add the missing axis: GCaMP is expressed (via Cre-on/Flp-on intersection + AAV1 transsynaptic delivery from a specific MOs subregion) **only in the SPNs that receive MOs input**. This is the literal population in the central claim *"MOs→D2 inhibits impulsive action; MOs→D1 promotes action."*

The cohort is a clean **2×2**: cell-type (D1/D2) × region+cortical-input (DMS-pMOs / VMS-aMOs).

| Mouse | Genotype | Region | Cortical input | ROI |
|---|---|---|---|---|
| BG_027 | D1 (Drd1) | VMS | aMOs | G0,G2 |
| BG_028 | D1 (Drd1) | DMS | pMOs | G0,G2 |
| BG_029 | D2 (A2a)  | DMS | pMOs | G0,G2 |
| BG_030 | D2 (A2a)  | VMS | aMOs | G0,G2 |

**BG_029 and BG_030 are the MOs→D2 mice** — the most direct test of the suppression/brake (C1) hypothesis.

Two analyses are ported, both in the 2×2 frame:
- **C1 — the brake.** Does *waiting/decision-period* activity predict withhold-vs-lick, especially in MOs-recipient D2?
- **C2 — response geometry.** D1-vs-D2 push–pull sign and timing at the change/lick, within the cortically-innervated subset.

G1 (TF-pulse evidence encoding) is **deferred**: most data-hungry, least central to the input-specificity story, and noisiest at n=1/cell.

---

## 2. Critical discipline (inherited + cohort-specific)

Inherited from C1/C2 (non-negotiable):
- **`fa` ≠ SDT-FA.** Behavioral `FA` = anticipatory lick during baseline (impulsivity). SDT-FA = catch-trial lick (`outcome=='Hit'` with `change_size ≤ CATCH_THRESHOLD`). Both tracks run separately; every result names which.
- **Alignment.** Never align FA/abort to a change that did not occur. Grating onset recovered arithmetically as `absolute_change_time − change_time` (defined for all trials).
- **D1 and D2 are different animals.** The D1-vs-D2 comparison is a group/cell-level **sign contrast**, never a within-animal anticorrelation.
- **Normalization.** Window scalars use the **session-z-scored** dF/F already on the trace, **no per-trial re-baselining** (would null the waiting-period level being tested).
- **Baseline is not neutral** (Markowitz); **no FaceMap** → Scheme-3 lick-proximal windows are motion-confounded (mitigated by matched-control subtraction + motor buffer; stated as a limitation).
- N aggregated to avoid pseudo-replication; non-parametric, two-sided, effect sizes alongside every p-value.

Cohort-specific:
- **Not poolable with bulk-8m.** Two independent disqualifiers: indicator (GCaMP6f vs 8m → different kinetics/SNR/dynamic range) and population (MOs-recipient subset vs bulk SPNs). Pooling would conflate "all D1/D2" with "MOs-recipient D1/D2" and contaminate both. The bulk C1/C2 outputs are left untouched.
- **n = 1 mouse per cell.** No between-animal inference is available. The per-mouse aggregation that protects the bulk analysis has no N here → see §5 (session as the unit; within-animal framing).

---

## 3. Phase 0 — Opus correctness audit of the reused machinery (a GATE)

Because parts of the reused code may have been written/run with cheaper models, **the cohort builds only on audited/fixed primitives.** A **Workflow** of parallel Opus reviewers audits each module/dimension, adversarially verifies findings, and synthesizes. Bugs are fixed **in the package, with regression tests** — improving the bulk analyses too.

**Audit targets**
- `analysis/suppression.py` (C1 core: track grouping, the two window extractors, hazard resampler, per-mouse Δ/AUROC, dataset builders, proficiency split).
- `analysis/geometry.py` (C2 core: region sourcing, push–pull extractors, response geometry).
- `analysis/group_statistics.py` (`auroc_score`, `pushpull_sign_contrast`, `permutation_test`, `bootstrap_ci`, `mannwhitney_with_effect_size`, `extract_peak`, latency extractors).
- `core/qc.py` (`region_sources`, `compute_session_roi_qc`/`compute_trace_qc`, `merge_hemispheres`, `get_region_pairs_for_subject`).
- `analysis/statistics.py` (`calculate_sdt_metrics`, `extract_peth`).
- `core/session.py` (abs_rt computation; grating-onset back-out).

**Audit criteria (the discipline, made checkable)**
1. Track grouping respects `fa` ≠ SDT-FA; SDT masking matches `calculate_sdt_metrics`.
2. Alignment: FA never aligned to change; onset = `absolute_change_time − change_time`; `extract_peth` time axis / unpacking correct.
3. Window inclusion: Scheme-1 motor buffer + change-coincidence exclusion; Scheme-3 hazard match (`τ` drawn from lick-group empirical elapsed-time distribution, truncated to `τ ≤ change_time`, `τ − BUFFER − L ≥ 0`).
4. Determinism: `HAZARD_SEED=42`, `HAZARD_RESAMPLES=20`; bootstrap/permutation seeds.
5. Normalization: session-z, **no per-trial re-baseline** on window means.
6. AUROC orientation: withhold = positive class (1), lick = 0; `AUROC=(rank_biserial+1)/2`; `>0.5` ⇒ brake.
7. `extract_peak` uses abs-max (sign-preserving) where suppression is possible.
8. Per-mouse aggregation in the bulk path (no trial-count N inflation); min-N guards (`MIN_TRIALS_PER_GROUP`).
9. QC: hemisphere merge logic (both pass → average, one → use it, neither → skip); region pairing subject-aware.

**Output:** an audit report at `docs/superpowers/audits/2026-06-23-intersectional-mos-machinery-audit.md` + a fix list; each fix shipped with a regression test. **This phase gates Phase 2/3.**

---

## 4. Phase 1 — cohort wiring (constants, registry, reproducible staging)

- **`core/constants.py`:** add 027–030 to `SUBJECT_GENOTYPE` (027/028 = D1; 029/030 = D2) and `SUBJECT_ROI_REGION` (027/030: G0,G2 → **VMS**; 028/029: G0,G2 → **DMS**).
- **New `core/cohort.py`** — a small registry so every script can target a cohort explicitly:
  - `COHORTS = {'bulk_8m': {...008–020, indicator 'GCaMP8m'}, 'intersectional_mos': {...027–030, indicator 'GCaMP6f', input map 027/030→'aMOs', 028/029→'pMOs'}}`.
  - Helpers: `cohort_of(subject)`, `subjects_in(cohort)`, `indicator_of(subject)`, `cortical_input(subject)`.
  - Existing code paths default to bulk → **no behavior change** to the bulk pipeline.
- **`scripts/data_management/stage_intersectional_cohort.py`** — committed, **idempotent** reproduction of the staging done ad-hoc earlier (the temp script was deleted): copy **top-level** csv+json from ceph (`X:\public\projects\BeJG_20230130_VisDetect\wIntersectGCaMP6F\BG_0XX`), **normalize the stale `BG_027__photom_*` / `BG_027__photom_IO_*` names to the correct subject** (the documented mislabel), single→double underscore, skip foreign strays, size-filtered Dec-tail top-up for 027/028. **Never writes to ceph.** Raw data stays gitignored; only the script is committed.
- **Discovery:** confirm `find_all_sessions` includes `photom_data/intrsct_GCaMP6f/`; the loader already handles this (new) format.

---

## 5. Phases 2–3 — the 2×2 analysis (C1 + C2 scoped); session-unit stats + trial-pooled PETHs

New scripts under **`scripts/analysis/intersectional/`** import the audited primitives (Approach B; the bulk C1/C2 scripts are untouched).

### Statistical unit
- **Inferential statistic = the session** (n ≈ 44–58 sessions/cell). Per cell: per-session scalar (C1 window-mean / AUROC; C2 peak/latency/push–pull), then **session-bootstrap CI** (1000 resamples, seed 42, percentile). The 2×2 contrasts are **descriptive**. Framed explicitly as **within-animal**: results generalize to this mouse's sessions, not to a population of mice (the single-subject precedent is the Sep2025 ephys repo, BG_046).
- **Illustration = trial-pooled PETHs/heatmaps** per cell (companion traces), **never mixing alignments** on a panel.
- A small reusable **session-unit aggregator** lives in `core/cohort.py` and is shared by both C1-cohort and C2-cohort scripts.

### C1 — the brake (scoped to the cohort)
- **Track A (primary):** FA-lick vs withhold waiting-period activity; **both** Scheme 1 (baseline-onset fixed window) and Scheme 3 (hazard-time-matched); reported for concordance.
- **Track B (control):** SDT-FA (catch-lick vs SDT-CR). **Included but flagged thin** — this cohort has huge FA counts and **sparse CR** (catch trials rare), so Track B is expected underpowered; treat as stimulus-matched corroboration, report N prominently.
- **Headline 2×2 read:** brake-AUROC (withhold = positive) vs 0.5 per cell; is it larger in **D2 than D1**, and present in the **MOs-recipient D2** cells (BG_029 DMS-pMOs, BG_030 VMS-aMOs)? Per-cell session-bootstrap CIs; D1-vs-D2 reported as a cell-level sign contrast (not a test across n=1).
- **Proficiency split:** the cohort is **not in the staging manifest** → fall back to **early-vs-late-by-date** or **d′-based** split (C1 already supports this), with per-bin SDT d′ reported.

### C2 — response geometry (scoped to the cohort)
- D1-vs-D2 **push–pull sign** + **onset/peak latency** at change (Hit/Miss) and at the Hit lick, within the cortically-innervated subset, by region.
- Per-session scalars → session-bootstrap CIs per cell; trial-pooled PETHs for illustration.

### QC (same pipeline as the bulk cohort)
- **Identical logic and thresholds**, reusing the audited `qc.py`: `compute_session_roi_qc`/`compute_trace_qc` per ROI, `merge_hemispheres` (G0/G2 → the cell's region; both pass → average, one → use it), behavioral engagement, merge-then-extract (`extract_merged_region_peths`). QC on by default; `--no-qc` escape hatch, mirroring the bulk scripts.
- **6f calibration reporting (cohort-specific, not a logic change):** the trace-quality thresholds were tuned on **GCaMP8m**; 6f has lower SNR / smaller dynamic range. The cohort scripts **report 6f pass/fail rates and the QC-metric distributions per cell**. An **indicator-aware threshold is introduced only if the data demand it — and then documented, never silent** (a constant keyed by indicator; identical gate logic). This is a Phase-1/2 reporting step (the `qc.py` *correctness* is covered by Phase 0).

---

## 6. Phase 4 — cross-cohort comparison (rank-based, secondary)

Every intersectional cell has a matched bulk cell (same genotype × region):

| Intersectional | Matched bulk-8m cell |
|---|---|
| BG_027 (D1·VMS) | bulk D1·VMS = BG_008, BG_009 |
| BG_028 (D1·DMS) | bulk D1·DMS = BG_013, BG_020 |
| BG_029 (D2·DMS) | bulk D2·DMS = BG_016, BG_018, BG_019 |
| BG_030 (D2·VMS) | bulk D2·VMS = BG_010, BG_011 |

- Compare **only indicator-invariant quantities**: AUROC (C1 brake), the **sign** of the D1-vs-D2 push–pull (C2), rank-biserial effect sizes, and **onset/peak latencies** (shape-based). **Never raw dF/F magnitude.**
- A **cross-cohort matcher** enforces correct genotype×region pairing and **refuses magnitude comparison across indicators** (guard in code + test).
- Caveated throughout: 6f vs 8m, n=1/cell, input-defined vs bulk; the bulk side carries its own (small) between-animal N.

---

## 7. Architecture (Approach B — dedicated cohort scripts + shared registry)

| Where | What | New / changed |
|---|---|---|
| `src/visdetect_photom/core/constants.py` | add 027–030 to `SUBJECT_GENOTYPE` + `SUBJECT_ROI_REGION` | **EDIT (additive)** |
| `src/visdetect_photom/core/cohort.py` | cohort registry + `cohort_of`/`subjects_in`/`indicator_of`/`cortical_input` + session-unit aggregator | **NEW module** |
| `scripts/data_management/stage_intersectional_cohort.py` | idempotent ceph→local staging with name-normalization | **NEW script** |
| `scripts/analysis/intersectional/c1_cohort_suppression.py` | C1 scoped to 027–030 (2×2, session-unit + PETHs); imports `suppression.py` | **NEW script** |
| `scripts/analysis/intersectional/c2_cohort_geometry.py` | C2 scoped to 027–030 (2×2, session-unit + PETHs); imports `geometry.py` | **NEW script** |
| `scripts/analysis/intersectional/cohort_cross_compare.py` | rank-based bulk-vs-intersectional comparison + 2×2 synthesis figure | **NEW script** |
| package modules (`suppression.py`, `geometry.py`, `group_statistics.py`, `qc.py`, `statistics.py`) | **Phase-0 bug fixes only** (with regression tests) | **EDIT if audit finds issues** |

- Reuses without re-implementation: `build_suppression_dataset` + window extractors + hazard resampler (`suppression.py`); push–pull + geometry extractors (`geometry.py`); `auroc_score`/`pushpull_sign_contrast`/`permutation_test`/`bootstrap_ci`/`mannwhitney_with_effect_size`/`extract_peak`/latencies (`group_statistics.py`); `region_sources`/`compute_session_roi_qc`/`merge_hemispheres` (`qc.py`); `calculate_sdt_metrics`/`extract_peth` (`statistics.py`); `get_genotype` (`group_utils`).
- **The bulk C1/C2 scripts are not modified** (parallel-chat safety + behavior preservation).

### Outputs (`FIGURES/intersectional_mos/`)
- `cohort_c1_session_scalars.csv`, `cohort_c1_auroc.csv`, `cohort_c1_pushpull.csv`, `cohort_c1_qualifying_n.csv`.
- `cohort_c2_session_scalars.csv`, `cohort_c2_geometry.csv` (latencies, push–pull sign).
- `cohort_qc_report.csv` (6f pass rates + metric distributions per cell).
- `cohort_cross_compare.csv` (rank-based bulk-vs-intersectional).
- **The 2×2 headline figure**; the cross-cohort rank-based figure; companion PETHs/heatmaps per cell.

---

## 8. Scope, edge cases, limitations

- **Scope:** cohort = BG_027–030 only; regions DMS/VMS (G0/G2; no VLS in these mice); both C1 tracks + C2 geometry; G1 deferred.
- **Edge cases:** a (cell, group) below `MIN_TRIALS_PER_GROUP` → skipped + reported; Track B may drop a cell for too-few catch licks (reported); Scheme-3 withhold trials with no admissible `τ` excluded.
- **Limitations:** (1) **n = 1 mouse/cell** → within-animal only, no population inference; session-unit CIs generalize to this mouse's sessions. (2) **6f vs 8m** → cross-cohort via rank/standardized metrics only. (3) **input-defined subset** → not poolable with bulk. (4) **sparse CR** → Track B thin. (5) **no staging manifest** → date/d′ proficiency split. (6) **no FaceMap** → Scheme-3 motion caveat. (7) leaky Cre labels (~6–7% hybrid SPNs). (8) 6f-on-8m-tuned QC → calibration reported, indicator-aware threshold only if/when documented.

---

## 9. Success criteria

- Phase 0 audit runs, produces a report + fix list; any fixes land with regression tests; the bulk C1/C2 suites still pass.
- 027–030 wired into constants + the cohort registry; the staging script reproduces the local cohort idempotently.
- Both C1 tracks (× both schemes) and C2 geometry run end-to-end on the cohort, writing all CSVs/PNGs with per-cell N and the QC calibration report.
- The 2×2 brake read is **testable per cell** (AUROC vs 0.5, D2-vs-D1 sign contrast, MOs-recipient D2 = BG_029/030) — reported regardless of whether positive, null, or reversed.
- The cross-cohort comparison reports **only** rank/standardized quantities, with a code guard that refuses magnitude comparison across indicators.
- New unit tests cover: cohort registry, session-unit aggregator, cross-cohort matcher (correct pairing + magnitude-comparison refusal), staging name-normalization + idempotency, plus Phase-0 regression tests.
