# C1 — False alarms as a suppression-failure signature (the MOs–D2 brake)

**Date:** 2026-06-11
**Status:** Design (approved, pre-plan)
**Question (Sep2025 landscape, tier T2):** Does *waiting/decision-period* striatal activity — especially D2/indirect — predict whether the animal **withholds** vs **licks**, and does that predictive suppression **strengthen with proficiency**?
**Depends on:** C2 (merged) infrastructure only. Defines its own window constants. Does **not** depend on G1 or B5.

---

## 1. Motivation and grounding

Liu 2023 (M2→DS): a fast indirect/D2-pathway signal in the **waiting/decision-forming period** suppresses (SDT) false alarms — a cortically-driven "brake" on impulsive action. Cruz 2022 (DMS dSPN promotes action; iSPN suppresses) and the Mink surround model give the same push–pull frame. This maps onto the proposal's *"MOs–D2 inhibits impulsive action."*

The photometry readout is the bulk D1 vs D2 signal with clean cell-type identity (Cre lines), so it can test the brake claim directly: if D2 waiting-period activity is a brake, it should be **higher on trials the animal withholds than on trials it licks**, and that predictive difference should **grow with learning**.

This question is the impulsivity/suppression complement to C2 (response geometry) and B5 (baseline operating point).

### Relationship to existing analyses (what makes C1 distinct)

`02_outcome_comparison.py` and C2's `08_d1_d2_geometry.py` already produce peri-event PETHs (incl. FA-lick-aligned, Early/Late-FA splits) and the D1/D2 push–pull extractors. **C1's distinct niche is the *waiting/decision* epoch framed as *prediction*** (does waiting activity predict the upcoming withhold-vs-lick choice), with a single-trial decoder and a proficiency axis. It is not a re-run of peri-lick PETHs.

---

## 2. Critical discipline (non-negotiable)

- **`fa` ≠ SDT-FA.** The behavioral `FA` label = an **anticipatory lick during baseline** (impulsivity readout). The SDT false alarm = a **catch-trial lick** (`outcome=='Hit'` with `change_size ≤ CATCH_THRESHOLD`). C1 analyzes **both**, on separate tracks, and every result names which.
- **D1 and D2 are different animals** (Drd1-Cre vs A2a/Drd2-Cre). The D1-vs-D2 "brake/release" comparison is a **group-level sign contrast**, never a within-animal anticorrelation.
- **Grating onset is the real anchor.** It is recovered per trial as `absolute_change_time − change_time` (populated for FA/catch trials too, since it is computed independent of outcome). `absolute_change_time` is used here **only arithmetically** to back out the grating-onset timestamp — the PETH is never aligned to a change that did not occur, so the alignment rule is respected.
- **Baseline is not neutral** (Markowitz) — the waiting period itself carries structure; interpret levels accordingly.
- **No FaceMap.** Photometry has no movement regressor here. Lick-proximal (Scheme 3) windows are motion-confounded; this is mitigated by design (matched control subtraction, motor buffer) and stated as a limitation.

---

## 3. Scientific structure — two tracks

Unit of analysis = the **waiting/decision period** (baseline grating, before any lick/change). For each track we contrast a **lick** group vs a **withhold** group, per region × genotype.

### Track A — behavioral FA (PRIMARY, impulsivity)
- **lick** = `outcome == 'FA'` (anticipatory lick *during baseline*).
- **withhold** = trials that held through baseline and reached the change = `{Hit, Miss, CR}`.
- **Aborts** (premature **wheel/locomotor** movement during the baseline grating — *not* the ITI) are **excluded from the primary contrast** because they are a locomotor suppression-failure that maximally contaminates photometry. They are offered as an **optional exploratory third group** (`abort` vs `withhold`), always flagged motion-confounded.

### Track B — SDT-FA (CONTROL, Liu-faithful)
- **lick** = `outcome == 'Hit' & change_size ≤ CATCH_THRESHOLD` (catch-trial lick).
- **withhold** = `outcome == 'Miss' & change_size ≤ CATCH_THRESHOLD` (SDT correct rejection).
- Stimulus-matched (both catch trials; only the choice differs) → cleaner causal read, lower N. Uses the **same masking convention as `calculate_sdt_metrics`** ([statistics.py](../../../src/visdetect_photom/analysis/statistics.py)). The standalone `'CR'` outcome label (normalized from `'Ref'`) is **distinct** from SDT-CR and is **not** used for Track B.

> Temporal note: Track A FA licks occur *during* the baseline (`lick_elapsed = reaction_time` from grating onset). Track B catch licks occur near the nominal change (`lick_elapsed = change_time + reaction_time`). The window machinery handles both via per-trial elapsed-time.

---

## 4. Waiting-period windows — two schemes (run separately, reported for concordance)

Per trial, per region, the analysis produces a **scalar** = mean of the **session-z-scored** signal within the window (normalization in §5). Two window definitions are computed independently; their agreement is itself a robustness result.

### Scheme 1 — baseline-onset-anchored fixed window (clean, late-waiting read)
- Window `[w0, w0 + L]` seconds after grating onset.
- Defaults: `w0 = 2.0` s (after the onset transient decays), `L = 1.0` s → `(2.0, 3.0)` s. **`w0` must be validated against the actual grating-onset average** (diagnostic step in the plan); it is a configurable constant.
- Inclusion: the window must end **≥ `MOTOR_BUFFER` (1.0 s) before** the lick (lick trials) and **before the change** (all trials). I.e. require `lick_elapsed ≥ w0 + L + MOTOR_BUFFER` and `change_time ≥ w0 + L (+ε)`.
- **Known limitation:** impulsive *early* FAs (RT ≤ 3.0 s) lick before this window can fit, so Scheme 1 predominantly captures **sustained/late waiting** (late FAs + withholds) and loses the most impulsive trials. Qualifying-N per cell is reported prominently. This is precisely why Scheme 3 exists.

### Scheme 3 — hazard-time-matched sampling (run-up-to-action read; primary for impulsivity)
- "Hazard time" = elapsed time into the baseline (how long the mouse has waited). Licks happen *late* in the wait; comparing late pre-lick activity to an early fixed window would confound **lick-vs-withhold** with **time-in-trial drift**. Hazard-matching removes this.
- For **lick** trials: window `[τ − BUFFER − L, τ − BUFFER]`, where `τ = lick_elapsed`.
- For **withhold** trials: draw `τ` from the **lick group's empirical elapsed-time distribution** (truncated to `τ ≤ change_time` and `τ − BUFFER − L ≥ 0`), `R = HAZARD_RESAMPLES (20)` deterministic resamples (`seed = 42`), and **average** the per-resample window means. Both groups are thereby sampled at the same elapsed-time distribution.
- Defaults: `L = 1.0` s, `BUFFER = 0.5` s (motor-execution guard; configurable up to 1.0 s for a stricter motor-free read).
- **Why this is robust where Scheme 1 is not:** the hazard-matched withhold control carries the **same onset transient and the same time-in-trial drift** at the same elapsed times, so `lick − withhold` cancels both. Scheme 3 can therefore **include early impulsive FAs** (shared transient subtracts out) while excluding motor *execution* via `BUFFER`. Scheme 1 (place window late) and Scheme 3 (matched subtraction) are complementary.

### Window constants (new, defined in C1; do not import from G1)
```
SCHEME1_WINDOW   = (2.0, 3.0)   # (w0, w0+L) s after grating onset
SCHEME1_MOTOR_BUFFER = 1.0      # s; window must end this long before lick/change
SCHEME3_L        = 1.0          # s window length
SCHEME3_BUFFER   = 0.5          # s motor-execution guard before lick
HAZARD_RESAMPLES = 20           # withhold pseudo-lick-time draws
HAZARD_SEED      = 42
MIN_TRIALS_PER_GROUP = 8        # min-N guard per (mouse, region, group) cell
PROF_MIN_SESSIONS    = 3        # min sessions per staging bin to use staging split
```

---

## 5. Normalization (the key C1-specific choice)

Use the **session-z-scored dF/F already stored** on the trace (`signal_type == 'zscored'`); take the window mean **with no per-trial re-baselining.**

- Rationale: lick and withhold trials come from the **same sessions**, so session-z-scoring is a fair shared reference (controls session-level bleaching/scale). Per-trial baseline subtraction would null out exactly the waiting-period level being tested.
- Consequence: C1 needs **no raw-dF/F machinery** (unlike B5, which compares *across* sessions/stages and therefore needs absolute level). C1 stays entirely on the existing z-scored pipeline.
- For Scheme 3's `lick − withhold` difference, the shared onset transient / drift cancels (see §4).

---

## 6. Metrics & statistical tests

Per trial → scalar window-mean (per region, per scheme). Then, per **region × genotype**:

1. **Group push–pull contrast (descriptive primary).**
   - Per-mouse `Δ = mean(scalar | withhold) − mean(scalar | lick)`.
   - Bootstrap CI of `Δ` per genotype × region (1000 resamples, seed 42, percentile).
   - `pushpull_sign_contrast(D1_Δ, D2_Δ)` for the group-level sign contrast.

2. **Single-trial decoder (the "predicts" claim).**
   - Per (mouse, region): **AUROC** of window-activity discriminating **withhold (= positive class, 1)** vs **lick (0)**. `AUROC > 0.5` ⇒ higher waiting activity predicts withholding (the D2-brake prediction).
   - `AUROC = (rank_biserial + 1) / 2`, derived from the existing `mannwhitney_with_effect_size` → thin new `auroc_score(scores, labels)` in `group_statistics.py`.
   - Aggregate per-mouse AUROCs: bootstrap CI vs chance (0.5) per genotype × region; **permutation test** D1-vs-D2 (10 000, rank-biserial reported).

3. **Proficiency axis (coarse robustness split).**
   - Recompute `Δ` and AUROC within a **less-proficient** vs **more-proficient** bin; **pooled is primary**.
   - Bin assignment priority: **(1)** staging Learning vs Expert where a mouse has ≥ `PROF_MIN_SESSIONS` sessions in both; **(2)** else within-mouse **early-vs-late split by session date**. Report **per-bin SDT d′** to confirm the more-proficient bin is genuinely better (d′ can define the split directly if recency does not track performance). Honest that Expert staging N is thin (≈ BG_009/010 only).

4. **Companion PETHs (illustration, not the primary statistic).**
   - Scheme-1 **grating-onset-aligned** traces and Scheme-3 **lick-aligned** traces, lick vs withhold, per region × genotype.
   - **Never mix the two alignments on one panel** (the standing rule).

All neural-data tests are **non-parametric**, **two-sided**, with **effect sizes** reported alongside every p-value, and N aggregated **per mouse** (no pseudo-replication).

---

## 7. Architecture (Approach B — package primitives + thin script)

| Where | What | New / changed |
|---|---|---|
| `src/visdetect_photom/analysis/suppression.py` | C1 core: track grouping (A/B/abort), the two window extractors, per-trial scalars, per-mouse `Δ` + AUROC, hazard resampler, proficiency split, `build_suppression_dataset` | **NEW module** |
| `src/visdetect_photom/core/qc.py` | promote a public `region_sources(session, use_qc=True)` (the logic currently private as `geometry._region_sources`), used by C1 | **ADD function** (geometry.py left untouched) |
| `src/visdetect_photom/analysis/group_statistics.py` | `auroc_score(scores, labels)` (thin, from rank-biserial) | **ADD function** |
| `scripts/analysis/photometry/11_fa_suppression.py` | thin consumer; pluggable `StateProvider` (default `PooledStateProvider`); CLI mirrors `08` (`--no-qc`, `--max_sessions`, `--state-filter`) | **NEW script** |

- Reuses without modification: `extract_*` extractors, `pushpull_sign_contrast`, `permutation_test`, `bootstrap_ci`, `mannwhitney_with_effect_size` (`group_statistics`); `StateProvider`/`PooledStateProvider`/`HMMStateProvider` (`state_provider`); `load_staging_manifest`/`get_session_stage`/`excluded_mice` (`core/staging`); `compute_session_roi_qc`/`merge_hemispheres` (`qc`); `calculate_sdt_metrics` (`statistics`); `get_genotype` (`group_utils`); `extract_peth` (`statistics`).
- **Does not touch** the merged/tested `geometry.py` (C2). `region_sources` is *added* to qc.py; geometry may adopt it later but is not modified now.
- Script number: C2=08, G1=09, B5=10 → **C1 = 11**.

### Outputs (`FIGURES/C1_fa_suppression/`)
- `c1_per_trial_scalars.csv` — per (subject, genotype, region, track, scheme, group, trial) window-mean + qualifying flags.
- `c1_pushpull_stats.csv` — per (track, scheme, region [, prof-bin]) Δ bootstrap CI + push–pull sign contrast.
- `c1_auroc_stats.csv` — per (track, scheme, region [, prof-bin]) per-genotype AUROC vs 0.5 + D1-vs-D2 permutation.
- `c1_qualifying_n.csv` — per-cell N (transparency on trial attrition from window inclusion).
- PNGs: companion PETHs (per scheme), Δ and AUROC summary plots, proficiency-split panels.

---

## 8. Scope, edge cases, limitations

- **Scope** mirrors C2/G1/B5: regions **DMS / VMS / VLS**; pool all mice with **staging exclusions** (BG_014 + no-data BG_015/017, via `excluded_mice`); pluggable states default **pooled**.
- **Edge cases:** a (mouse, region, group) cell below `MIN_TRIALS_PER_GROUP` → skipped and reported in `c1_qualifying_n.csv`; a mouse missing one track (e.g., too few catch licks for Track B) drops from that track's aggregation, reported; Scheme-3 withhold trials with no admissible `τ` window → excluded.
- **Limitations:** (1) no FaceMap → Scheme-3 lick-proximal windows are motion-confounded (mitigated by matched-control subtraction + `BUFFER`; Scheme 1 is the cleaner read → concordance matters). (2) Track B catch-lick sparsity → low power; treat as a stimulus-matched corroboration, not a stand-alone result. (3) Scheme-1 loses early impulsive FAs (covered by Scheme 3). (4) Cell-type labels are leaky (~6–7% hybrid SPNs). (5) Expert staging N is thin → the proficiency split is robustness, not a primary claim. (6) Baseline-not-neutral.

---

## 9. Success criteria

- Both tracks × both schemes run end-to-end on `photom_data/` and write all CSVs/PNGs, with per-cell N reported.
- The descriptive Δ contrast and the AUROC decoder agree in direction within a track/scheme (internal consistency), and the two schemes are concordant (cross-scheme robustness).
- The D2-brake prediction is **testable**: D2 waiting-period AUROC vs 0.5 and the D1-vs-D2 push–pull sign contrast are reported per region with effect sizes and CIs — regardless of whether the effect is positive, null, or reversed.
- New unit tests (TDD) cover: track grouping (incl. the `fa` vs SDT-FA distinction), both window extractors (inclusion logic, hazard resampling determinism), `auroc_score`, and `region_sources`.
