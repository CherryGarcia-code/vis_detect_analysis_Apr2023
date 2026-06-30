# Good-signal exploration (2026-06)

Exploratory analyses of the GOOD-quality striatal signal, run as a two-track
fan-out (Workflow). **Two cohorts are NEVER pooled** (GCaMP8m bulk vs GCaMP6f
intersectional; different indicator + population).

## Pipeline (two stages)

**Stage 1 — pre-extraction** (loads raw sessions ONCE → compact pkl cache in
`results/explore_cache/`, gitignored):
- `preextract_bulk.py` → `bulk_extract.pkl` — bulk-8m, **per-mouse** per-region
  per-condition mean traces (change- and lick-aligned), early/late splits, RT
  terciles, per-mouse behavioral meta (d′, FA rate, …). Good bulk mice only
  (D1 008/009/013/020; D2 010/011/016/018/019; 014/015/017 excluded).
- `preextract_good_d1.py` → `good_d1_extract.pkl` — intersectional good-D1
  **per-trial** event-aligned snippets + metadata (BG_027 G0/G2, BG_028 G0;
  `fiber_quality` flags BG_028 G2 as weak).

**Stage 2 — angle scripts** (read the pkls, write figures to
`FIGURES/intersectional_mos/explore/`):
- bulk track (per-mouse, N=mice): `bulk_outcome_geometry` (detection
  selectivity Hit−Miss + CR temporal-expectation), `bulk_psychometric`
  (evidence scaling), `bulk_reward_rpe`, `bulk_rt_coding`, `bulk_hit_vs_fa`,
  `bulk_learning` (early/late).
- intersectional good-D1 track (n=1/cell, **descriptive/within-animal only**):
  `int_hit_vs_fa`, `int_psychometric`, `int_rt_coding`, `int_reward`.

## Headline outcome (adversarially verified)
- **ROBUST:** D1 detection selectivity (Hit−Miss) ≈4× D2, perfect per-mouse rank
  separation (MWU p=0.016), holds in DMS and VMS (`bulk_outcome_geometry`).
- **REAL but reword:** neural psychometric — responses scale with evidence in
  both cell types; the robust difference is D1≫D2 **amplitude** (not scaling).
- **HINTS only (underpowered/outlier-driven):** D1 late reward Hit>FA (refuted as
  worded — 1 D2 outlier), D1-grows/D2-shrinks learning, D2-DMS inverse-RT
  impulsivity ramp. See the cohort memory for the full ranked synthesis.

## Caveats / re-running
- Paths assume the analysis-branch checkout (src on path) + the
  `vis_detect_analysis_Apr2023` data root. Run Stage 1 before Stage 2.
- Bulk unit = MOUSE (N=mice); intersectional is n=1/cell descriptive.
- These are exploratory provenance scripts, not the audited package pipeline.
