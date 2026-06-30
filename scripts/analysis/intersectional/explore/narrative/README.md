# Impulsivity-vs-sensitivity narrative (2026-06)

Stage 1: `preextract_per_session.py` -> `results/explore_cache/per_session_metrics.csv`
(per session x region neural+behavioral metrics; bulk + intersectional good-D1).
Stage 2: angle scripts read that CSV (+ `bulk_extract.pkl`) -> figs in
`FIGURES/intersectional_mos/narrative/`.

- `sens_mixedeffects` (D1>D2 detection sel, regions separate, MixedLM+permutation)
- `motor_vs_value` (peri-lick ramp is motor-locked)
- `impulsivity_axis`, `two_axis_dissociation`, `learning_trajectories`,
  `neural_behavioral_coupling`, `int_corroboration`
- `motor_residualized_sensitivity` (DECISIVE: early pre-lick / motor-free Hit-Miss,
  region-matched). KEY: D1>D2 selectivity is largely peri-action; motor-free
  early-sensory D1>D2 survives only in VMS (n=2v2, underpowered). template-resid
  variant is unreliable (oversubtracts) - use the early-window control.

Unit = MOUSE (bulk N=4 D1 / 5 D2); intersectional n=2 D1 descriptive; cohorts NEVER pooled.
