# G1 — TF-Pulse Evidence Encoding — Design Spec

- **Date:** 2026-06-09
- **Status:** Design approved (brainstorm); pending plan
- **Question ID:** G1 (sibling of C2 in the cross-repo question landscape)
- **Repo:** `vis_detect_analysis_Apr2023` (photometry; Aim 2 Approach 2)
- **Implementation prerequisite:** **C2 must be merged first** — G1 reuses C2's `analysis/state_provider.py`, `core/staging.py`, and region-source helpers.

---

## 1. Question

**Does bulk D1 vs D2 striatal signal track the moment-to-moment baseline TF fluctuations ("fast/slow pulses")?** Estimate, per region × genotype, the temporal **kernel** mapping ongoing baseline temporal-frequency to dF/F — the *evidence axis at finest grain* — and compare D1 vs D2 (sign, timing, integration window), plus an intuitive fast-vs-slow pulse-triggered view.

Scientific motivation (see `memory/photometry-question-landscape.md`, `synthesis-batch06`, `synthesis-phase3-pathways`): pulse-triggered / reverse-correlation kernels are the canonical evidence-integration probe (Huk-Shadlen motion pulses, Pagan Poisson pulses; the Lohse sensory-CD is built on these 50 ms TF pulses). Directly informs the integration-timescale debate (~0.25 s vs ~1 s) and whether D1/D2 carry sensory evidence with the same or opposite sign.

## 2. Consistency with the ephys repo (REQUIRED)

G1 mirrors the conventions in `vis_detect_analysis_Sep2025/src/visdetect/analysis/tf_pulse.py` so the two modalities are directly comparable. Adopted constants (add to `core/constants.py`):

| Constant | Value | Meaning |
|---|---|---|
| `TF_FAST_THRESH_LOG2` | `+0.25` | fast pulse: `log2(TF) ≥ +0.25` (≥1.19× base) |
| `TF_SLOW_THRESH_LOG2` | `-0.25` | slow pulse: `log2(TF) ≤ -0.25` (≤0.84× base) |
| `TF_BASELINE_STRIDE` | `3` | baseline vector repeats each pulse 3× (60 fps → 50 ms) |
| `TF_SAMPLE_PERIOD` | `0.05` | seconds per baseline pulse sample |
| `TF_MIN_AFTER_BASELINE` | `1.0` | exclude pulses < 1.0 s after baseline onset (= onset trim) |
| `TF_MIN_BEFORE_CHANGE` | `1.0` | exclude pulses < 1.0 s before change |
| `TF_MIN_BEFORE_OUTCOME_FA_ABORT` | `2.0` | exclude pulses < 2.0 s before FA/abort lick |
| `TF_PULSE_PRE_WINDOW` | `(-0.4, 0.0)` | pre-pulse z-score baseline |
| `TF_PULSE_POST_WINDOW` | `(0.0, 0.5)` | post-pulse response window |
| `TF_PULSE_DETREND_BASELINE` | `(-0.4, -0.01)` | linear-detrend fit window |
| `TF_PULSE_DETREND_POST` | `(0.0, 0.3)` | post-pulse peak/trough measurement |
| `TRF_LAGS` | `-0.5 … +2.0 s @ 0.05 s` | kernel lag grid (negatives = causality control) |

## 3. Data facts established (verified on BG_013)

- Trial JSON stores `St1TrialVector` (baseline grating TF sequence, each pulse repeated 3×), `St2TrialVector`, realized `TF` (== St1 during baseline), `vbl` (per-frame wall-clock), `stimT` (change time from baseline onset), `stimD` (ITI). **No `n_seen`.**
- `St1[::3]` is the 50 ms pulse sequence (values identical to the realized baseline TF).
- **Anchor:** baseline (grating) onset in photometry `SystemTimestamp` = `Input0` rising edge = `absolute_change_time − change_time` (equivalently `absolute_start_time + iti_duration`). `session.py` stores the raw trial dict on `Trial.metadata`, so `St1TrialVector`/`vbl`/`TF` are available without re-reading JSON.
- **Gray ≠ stimD** (leading-zero frames ≠ ITI) → never assume gray duration; anchor on Input0.
- Baseline sample count (no `n_seen`): `n_baseline = round(change_time / 0.05)` for Hit/Miss/CR; `round(reaction_time / 0.05)` for FA/abort (FA/abort `reaction_time` is relative to baseline onset, per `session.py`).

## 4. Alignment (uniform-50 ms off Input0, with vbl validation)

For each trial:
1. `baseline_ts = absolute_change_time − change_time` (Input0 time, `SystemTimestamp`).
2. Pulse sequence = `St1TrialVector[::3]`, trimmed to `n_baseline`.
3. Pulse time for sample `k`: `baseline_ts + k · 0.05`.
4. **Validation (vbl):** independently locate the change frame in the realized `TF` array (first frame reaching the post-change level ≈ `Stim2TF`), map it via `Input0 + (vbl[change_frame] − vbl[onset_frame])`, and confirm it lands within **50 ms** of `absolute_change_time`. Drop trials that fail (dropped-frame / mis-anchor guard). Report drop counts.

Rationale: baseline TF updates uniformly every 50 ms (3 frames @ 60 fps); the uniform grid is identical to the ephys convention and avoids `vbl` cross-clock resampling, while the change anchor still validates timing.

## 5. Scope (mirrors C2)

| Dimension | Decision |
|---|---|
| Regions | DMS, VMS, VLS (low N flagged; VLS likely n<2/genotype). |
| Stage | Pool engaged QC-passed sessions + Learning-vs-Expert robustness split. |
| Behavioral state | Pooled default; optional `--state-filter` via C2's `StateProvider` (HMM lazy). |
| Mouse exclusion | BG_014 + any all-Excluded mouse (via `core/staging.excluded_mice`). |
| Trials | Hit/Miss/CR (baseline to change − 1.0 s) **and** FA/abort (baseline to lick − 2.0 s), all from baseline onset + 1.0 s. |
| Unit of replication | **Mouse**. |

## 6. Architecture (Approach B)

**`src/visdetect_photom/core/stimulus.py`** (new — reconstruction + alignment + validation):
- `baseline_onset_ts(trial)` → `Input0` time (= `absolute_change_time − change_time`); `None` if unavailable.
- `baseline_pulse_values(trial, stride=TF_BASELINE_STRIDE)` → `St1[::stride]` array; `None` if missing.
- `n_baseline_samples(trial, sample_period=TF_SAMPLE_PERIOD)` → int (from `change_time` or FA/abort `reaction_time`).
- `windowed_pulses(trial)` → `(values, abs_times)` for samples in `[onset + TF_MIN_AFTER_BASELINE, t_end − margin]` (margin = `TF_MIN_BEFORE_CHANGE` for change-reaching, `TF_MIN_BEFORE_OUTCOME_FA_ABORT` for FA/abort).
- `fast_slow_pulse_times(trial)` → `(fast_times, slow_times)` via `log2(value)` vs `±0.25`, within the same window.
- `aligned_baseline_regressor(trial)` → `(log2tf, abs_times)` for the windowed samples (continuous TRF input; mean-centering applied later at design-build).
- `validate_change_anchor(trial, tol=0.05)` → `(ok: bool, mismatch_s: float)` using realized `TF` + `vbl`.

**`src/visdetect_photom/analysis/tf_kernel.py`** (new — kernel math):
- `build_region_design(session, signal, timestamps, *, state_keep=None)` → `(X_tf, y_dff)` on the 50 ms grid: for each valid trial, `aligned_baseline_regressor` + interpolate `signal` onto `abs_times`; concatenate; mean-center `X_tf`. Skips trials failing `validate_change_anchor`.
- `fit_trf(x_tf, y_dff, lags=TRF_LAGS, alpha=None)` → ridge kernel (time-embedded design, one weight per lag). Returns `(lags, kernel)`. `alpha` selected by `RidgeCV` over a log-spaced grid (1e-3 … 1e3) with within-session generalized CV; falls back to `1.0` if CV is unavailable.
- `pulse_triggered_average(signal, timestamps, pulse_times, pre=TF_PULSE_PRE_WINDOW, post=TF_PULSE_POST_WINDOW)` → `(t_vec, mean_trace, sem)`, z-scored to pre-window.
- `detrend_pulse_trace(t_vec, trace)` → linear-detrend on `TF_PULSE_DETREND_BASELINE`, return `(detrended, z_max, z_min)` measured on `TF_PULSE_DETREND_POST` (ports the ephys `detrend_tf_traces`).
- `kernel_timescale(lags, kernel)` → `signed_peak`, `peak_lag`, `decay_halfwidth` / center-of-mass.
- `shuffle_null(x_tf, y_dff, lags, n_shuffles=200, seed=42)` → null kernel band via circular shift of `x_tf`.

**`scripts/analysis/photometry/09_tf_pulse_encoding.py`** (thin CLI): discover → load → exclude → per region×genotype: build design, fit kernel per session → per-mouse mean → group; pulse-triggered fast/slow; figures + CSVs. Flags: `--no-qc`, `--state-filter`, `--state-results-dir`, `--max_sessions`, `--root_dir`, `--output_dir`.

## 7. Statistics

- Per-session kernel (and per-session fast/slow pulse-triggered) → per-mouse mean → group mean ± SEM (N = mice).
- **D1 vs D2**: `permutation_test` + `pushpull_sign_contrast` (from C2's `group_statistics`) on kernel `signed_peak` and on `z_max_fast/z_min_slow` responsiveness metrics (mode-aware: does D1 track TF with positive kernel, D2 with suppression?).
- **Significance**: `shuffle_null` band per group; kernel is "real" where it exits the null band.
- **Integration timescale**: report `peak_lag` + decay per region×genotype, D1 vs D2 (permutation). Frame against 0.25 s vs 1 s.
- **Robustness**: re-run within Learning vs Expert.
- Per-mouse N on every panel; low-N (n<3/group) flagged.

## 8. Outputs

`FIGURES/G1_tf_pulse_encoding/`:
- Per region (`G1_<REGION>.png`): D1 vs D2 TRF kernel ± SEM (+ null band); fast & slow pulse-triggered dF/F (D1 vs D2); timescale/responsiveness bars.
- Cross-region summary (`G1_summary.png`): signed kernel peak + integration timescale, D1 vs D2 × region, with opposite-sign flags.
- CSVs: `G1_kernels.csv` (per-mouse kernel + timescale), `G1_pulse_triggered.csv` (per-mouse z_max/z_min fast/slow), `G1_stats.csv` (contrasts), `G1_alignment_qc.csv` (per-session validation drop counts).

## 9. Testing (TDD)

- `core/stimulus`: synthetic trial (known `St1`, `change_time`, `absolute_change_time`) → `baseline_onset_ts`, `n_baseline_samples`, `windowed_pulses` bounds + margins, `fast_slow_pulse_times` classification at ±0.25, `validate_change_anchor` pass/fail at the 50 ms tolerance.
- `analysis/tf_kernel`: synthetic `y = known_kernel ⊛ x_tf + noise` → `fit_trf` recovers the kernel (peak lag + sign); `pulse_triggered_average` on a known bump; `detrend_pulse_trace` removes a linear trend and recovers a planted post-pulse peak; `kernel_timescale` on a known exponential.
- Script smoke (skip if `photom_data/` absent): runs on `--max_sessions 3`, writes `G1_kernels.csv`.

## 10. Caveats (into spec + captions)

1. **D1/D2 = different animals** → group-level sign contrast, never within-animal anticorrelation.
2. **GCaMP kinetics smear the kernel** — measured kernel = neural response ⊛ indicator dynamics, so the integration timescale is an **upper bound** (calcium decay, not spiking). Report the raw calcium kernel; note optional indicator deconvolution as a future extension. *(Most important interpretive caveat — also a real difference vs the ephys spike-based screen.)*
3. **No movement regressors** (no video) → baseline movement may co-vary with TF-driven licking and dF/F; flagged. (FA/abort 2.0 s lick margin mitigates peri-lick motor.)
4. **Uniform-50 ms assumption** validated per trial by the change anchor (>50 ms mismatch dropped).
5. **TF autocorrelation** limits effective kernel resolution; ridge regularization required.
6. **Small N** per region×genotype, esp. VLS.

## 11. References

- Program + cautions: `memory/photometry-question-landscape.md`, `memory/cross-repo-context.md`.
- Ephys conventions ported: `vis_detect_analysis_Sep2025/src/visdetect/analysis/tf_pulse.py`, `.../analysis/constants.py` (`TF_PULSE_*`, `TF_FAST/SLOW_THRESH_LOG2`, `LOHSE_*`).
- Reused (post-C2): `core/qc.py`, `analysis/statistics.py` (extract_peth), `analysis/group_statistics.py` (`permutation_test`, `bootstrap_ci`, `pushpull_sign_contrast`), `analysis/state_provider.py`, `core/staging.py`, `analysis/group_utils.py`, `core/constants.py`.
- Sibling spec: `docs/superpowers/specs/2026-06-08-c2-d1-d2-geometry-design.md`.
