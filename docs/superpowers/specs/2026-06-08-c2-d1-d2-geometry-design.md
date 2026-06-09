# C2 — D1/D2 Response Geometry — Design Spec

- **Date:** 2026-06-08
- **Status:** Design approved (brainstorm); pending plan
- **Question ID:** C2 (from the cross-repo question landscape)
- **Repo:** `vis_detect_analysis_Apr2023` (photometry; Aim 2 Approach 2)

---

## 1. Question

**Do D1 (direct/dSPN) and D2 (indirect/iSPN) striatal populations carry the *same-sign* sensory/decision signal, or an *opposite-sign* push–pull, around the key task epochs?** Characterize the full D1-vs-D2 **response geometry** — sign, magnitude, timing, evidence-grading — at change onset, at the lick (commitment), and during pre-change anticipation.

This is question **C2** in the program (`memory/photometry-question-landscape.md`); it is the **foundational** analysis the other questions (B5, C1, G1) build on, because it establishes the D1/D2 response geometry they reference.

### Spine
Part of the overarching question: *how do mice learn to suppress impulsivity and increase sensitivity to informative stimuli.* C2 establishes the static geometry; B5 adds the learning trajectory.

## 2. Background & hypotheses (literature grounding)

- **Mode-asymmetry (the novel core).** dSPN code is *activation*-biased; iSPN code is *silencing*-biased — both decode behavior, but the iSPN signal often lives in its **suppression** (Varin 2023). → We must extract **both** activation *and* suppression, especially for D2; activation-only peak metrics (as in existing script 04) miss the D2 signal.
- **Opposite-sign push–pull** of cortical content by D1/D2 (van Beest 2022); D1 fast-phasic at commitment (Balewski 2022).
- **AND-gate / orthogonal axes** in the sister study on the *same task* (Lohse 2025): task-state ⊥ sensory-evidence. C2's anticipation block touches the task-state/temporal-expectation axis.
- **Regional function**: DMS dSPN *promotes* action, iSPN *suppresses* (Cruz 2022); medial striatum ≈ associative/goal-directed node. **Subregion discipline**: frameworks only, not numbers, from DLS/visual-tail/auditory-pDS work.

### Predictions (to be tested, not assumed)
- H1 (push–pull): at change/lick, D1 group shows net **activation**, D2 group shows net **suppression** (opposite sign).
- H1′ (shared axis, null): both genotypes show same-sign activation (no push–pull at the bulk level).
- H2 (grading): D1 signed response grades **up** with `change_size` (drift/sensitivity); D2 grades toward **suppression**.
- H3 (commitment timing): D2 fast / D1 slow at the lick (replicating script-05 hint), with possibly opposite sign.

## 3. Hard constraint (must be stated in the figure + paper)

**D1 and D2 are recorded in *different animals*** (D1-Cre vs A2a/Adora2a-Cre). There is **no simultaneous D1+D2 recording** in one mouse. Therefore "push–pull" here is a **group-level sign contrast** (D1-group vs D2-group), **not** a within-animal anticorrelation. Every push–pull claim is phrased accordingly.

## 4. Scope

| Dimension | Decision |
|---|---|
| Regions | **All three** — DMS, VMS, VLS. VLS genotype contrast is likely n<2/group; render only where ≥1 mouse/genotype, flag low N on every panel. |
| Epochs | **Change** (Hit/Miss), **Lick** (hit_lick/fa_lick), **Anticipation** (pre-change on Hit/Miss/CR). |
| Learning stage | **Pool all** behaviorally-engaged QC-passed sessions (primary, max power) **+ one Learning-vs-Expert robustness split** (Block-1 key contrasts). |
| Behavioral state | **Pooled by default**; optional `--state-filter` via swappable `StateProvider` (default HMM, lazy). |
| Mouse exclusion | Exclude **BG_014** + any mouse with no valid (non-Excluded) staged sessions. BG_015/017 absent from data. Driven off the staging manifest. |
| Unit of replication | **Mouse** (per-mouse means; group stats across mice). |

**Out of scope (own specs):** TF-pulse / fast-slow-pulse evidence encoding → **G1 sibling spec** (needs stimulus reconstruction + cross-clock alignment). HMM fitting (paused). Movement/FaceMap regressors (no video here).

## 5. Architecture (Approach B — shared primitives + thin script)

### 5.1 New shared functions — `src/visdetect_photom/analysis/group_statistics.py`
All operate on a 1-D trace + matching `time_axis`, over a `window=(start,end)` in seconds. Return `np.nan` when no finite samples in window.

- `extract_activation(trace, t, window) -> float` — peak **positive** deflection (max, clipped ≥0; nan if none > 0).
- `extract_suppression(trace, t, window) -> float` — peak **negative** deflection (min, clipped ≤0; nan if none < 0).
- `extract_signed_peak(trace, t, window) -> float` — abs-max preserving sign. **Promote** the existing `extract_peak` from script 01 to here; update 01 to import it (no behavior change).
- `extract_signed_auc(trace, t, window) -> float` — mean of trace over window (net signed response).
- `extract_ramp_slope(trace, t, window) -> float` — slope of a degree-1 polyfit over window (signal-units/s); robust to constant offset.
- `pushpull_sign_contrast(d1_vals, d2_vals, n_perm=10000, seed=42) -> dict` — per-genotype mean + bootstrap 95% CI; sign of each; `opposite_sign` flag (signs differ AND both CIs exclude 0); permutation p on (mean D1 − mean D2); rank-biserial effect size (from MWU). Reuses existing `permutation_test`, `bootstrap_ci`, `mannwhitney_with_effect_size`.

### 5.2 New module — `src/visdetect_photom/analysis/state_provider.py`
- `StateProvider` (typing.Protocol): `get_trial_states(session) -> np.ndarray[str]` (len = n trials; labels or `'NA'`).
- `PooledStateProvider` (**default**): returns `'All'` for every trial — no filtering, no HMM dependency.
- `HMMStateProvider(results_dir, K=None)`: **lazy** — imports `hmm_downstream.load_hmm_results` + `hmm.decode_session` only when instantiated; returns per-trial labels. Only used if `--state-filter` is passed.
- `filter_trials_by_state(session, provider, keep_states) -> set[int]` — trial indices to keep; downstream `_get_event_times`-style selection intersects with this set.

### 5.3 Learning-stage helper
Light reader for `results/staging_manifest.csv` (produced by `scripts/data_management/stage_sessions.py`): `load_staging_manifest()` + `get_session_stage(session) -> str` (`Naive`/`Learning`/`Expert`/`Disengaged`/`Excluded`/`Unknown`) and `excluded_mice()`. Place in `core/io.py` or a small `core/staging.py` (decide in plan; prefer `core/staging.py` for isolation).

### 5.4 Data flow (reuses existing pipeline)
```
find_all_sessions(recursive, min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
  → drop excluded mice (staging)
  → load_session_from_files
  → check_behavioral_engagement (skip fail)  [unless --no-qc]
  → compute_session_roi_qc + merge_hemispheres → one signal per region   [QC path]
  → (optional) filter_trials_by_state(provider, keep_states)
  → extract_peth per epoch (window, baseline_window, normalize='subtract')
  → per-mouse mean trace → mode-aware extraction (5.1)
  → genotype × region aggregation (per-mouse values)
  → stats (Section 7)
```
Follow the established `collect_*_by_region` pattern from scripts 01/04/05 (subject-aware region grouping in both QC and `--no-qc` paths).

## 6. Analysis blocks

Constants (reuse repo conventions): PETH `WINDOW=(-2.0,4.0)`, post-event metric window `(0.0,1.5)`, onset search `(0.0,2.0)`.

### Block 1 — Sensory / evidence geometry (change-aligned)
- Events: `change_hit`, `change_miss` (valid outcomes only — never align FA/Abort to change).
- Per region×genotype, per-mouse mean PETH (`baseline_window=(-2,0)`, `normalize='subtract'`).
- Metrics in `(0,1.5)s`: `extract_activation`, `extract_suppression`, `extract_signed_peak`, `extract_signed_auc`. Report activation **and** suppression for both genotypes (D2 suppression is the key signal).
- **Push–pull test** (`pushpull_sign_contrast`) on `signed_auc` (primary) and `signed_peak` (secondary) at change_hit.
- **Evidence grading**: per `change_size ∈ CHANGE_SIZES`, signed response (Hit go-trials), per genotype×region; Spearman(signed_response, log2(change_size)) + bootstrap CI. Extends script 04 to **both signs and both genotypes** (04 is Hit/activation-only). Also report D2 suppression-vs-change_size.

### Block 2 — Commitment geometry (lick-aligned)
- Events: `hit_lick`, `fa_lick`.
- Metrics: `extract_signed_peak` (0,1.5); `extract_peak_latency` (0,1.5); `extract_onset_latency` (search 0,2.0; n_consecutive=3). Quantify D1-slow vs D2-fast asymmetry + sign.
- **Push–pull test** at the lick.
- `hit_lick` vs `fa_lick` contrast per genotype (shared commitment signal? bridges to C1).

### Block 3 — Anticipation geometry (pre-change)
- Events: `change_hit`, `change_miss`, `change_cr` (CR = catch, nominal change time; temporal-expectation ramp).
- **Normalization care (anti-circularity):** the anticipation measurement window overlaps the default PETH baseline `(-2,0)`. Normalize these traces to an **early** reference `(-2.0,-1.5)s` and measure the ramp over `(-1.5,0)s`, so baseline-subtraction does not force the ramp to start at zero. (Ramp *slope* is offset-invariant; `signed_auc` is not — hence the early reference.)
- Metrics: `extract_ramp_slope` (-1.5,0); `extract_signed_auc` (-1.5,0). D1 vs D2 anticipatory ramp (sign + magnitude); push–pull test on the ramp.

## 7. Statistics

- **Per-mouse aggregation**; all group contrasts use `permutation_test` (10k, seed=42) + rank-biserial effect size (MWU has ~zero power at n<5; repo convention).
- **Sign reliability**: `bootstrap_ci` (1000, seed=42) per genotype; sign is "reliable" iff CI excludes 0.
- **Grading**: Spearman ρ + bootstrap CI per genotype×region.
- **Robustness split**: re-run Block-1 push–pull + grading within Learning vs Expert (staging manifest).
- **Reporting**: `format_stats_table` → CSV with significance stars; **per-mouse N on every panel/row**; low-N (n<3/group) flagged, not hidden.

## 8. Outputs

- **Script:** `scripts/analysis/photometry/08_d1_d2_geometry.py` (06/07 taken by HMM scripts). CLI: `--root_dir`, `--output_dir`, `--no-qc`, `--state-filter <state[,state]>`, `--state-results-dir`, `--max_sessions`.
- **Figures** → `FIGURES/C2_d1_d2_geometry/`:
  - One figure per region (`C2_geometry_<REGION>.png`), rows = the 3 blocks' key panels (D1 vs D2 mean±SEM traces + signed metric bars + grading curve).
  - One cross-region **push–pull sign summary** (`C2_pushpull_summary.png`): signed response (D1 vs D2) per region×epoch with opposite-sign flags.
- **CSVs:** `C2_geometry_metrics.csv` (per mouse×region×genotype×epoch×metric), `C2_stats_summary.csv` (contrasts), `C2_grading.csv`, `C2_session_summaries.csv`.

## 9. Testing (TDD)

Write tests first (`tests/` mirroring package path):
- `extract_activation/suppression/signed_peak/signed_auc/ramp_slope` on synthetic traces with known shapes (pure-positive bump, pure-negative dip, mixed, flat→nan, linear ramp of known slope).
- `pushpull_sign_contrast`: synthetic opposite-sign groups → `opposite_sign=True`, small p; same-sign groups → `opposite_sign=False`.
- `PooledStateProvider.get_trial_states` returns all `'All'`; `filter_trials_by_state` keeps correct indices.
- Staging helper parses a fixture manifest; `excluded_mice()` includes BG_014.
- Smoke test: script runs end-to-end on `--max_sessions 3` and writes expected files.

## 10. Caveats (carry into figure captions / paper)
1. **D1/D2 = different animals** → group-level sign contrast, not within-animal anticorrelation.
2. **Small N** per region×genotype (esp. VLS) → per-mouse N on every panel.
3. **No movement regressors** (no video) → lick-aligned signals may include motor; flagged.
4. **Sign + timing > absolute magnitude**: expression/gain vary across mice/fibers; emphasize sign and latency (script-05 rationale). Inputs are session-z-scored dF/F, baseline-mean-subtracted (Δ z-dF/F).
5. **Cell-type leak**: ~6–7% hybrid D1/D2-coexpressing SPNs (Bonnavion 2024).

## 11. References
- Program + cautions: `memory/photometry-question-landscape.md`, `memory/cross-repo-context.md`.
- Literature (Sep2025 repo): `…/memory/literature/synthesis-phase3-pathways.md` (Lohse 2025, Varin 2023, Cruz 2022, van Beest), `synthesis-phase3-celltypes.md`, `synthesis-batch01-foundations.md`.
- Existing code reused: `core/qc.py` (merge_hemispheres, QC), `analysis/statistics.py` (extract_peth, calculate_sdt_metrics), `analysis/group_statistics.py` (permutation_test, bootstrap_ci, extract_*_latency), `analysis/group_utils.py` (_get_event_times, get_genotype), `core/constants.py`.
- Sibling: G1 TF-pulse spec (to be written).
