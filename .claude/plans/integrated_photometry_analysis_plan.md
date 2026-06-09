# Integrated Photometry Analysis Plan — DMS × VMS × VLS, D1 × D2
# Created: 2026-04-09
# Status: Active

## Dataset Summary
- **12 mice**: D1 (BG_008, 009, 013, 014, 015, 020); D2 (BG_010, 011, 016-019)
- **3 regions**: DMS (BG_013-020), VMS (BG_008-011), VLS (BG_019, 020)
- **~452 sessions** total (148 DMS/VLS + 304 VMS)
- Complementary to ephys project (vis_detect_analysis_Sep2025, BG_046)

## Phase 1: Update Foundation for 12-Mouse Dataset
- [ ] 1a. Re-run C1 (D1 vs D2 profiles) organized by **region** (DMS, VMS) instead of raw ROI
- [ ] 1b. Re-run C2 (outcome overlays) by region
- [ ] 1c. Update QC + hemisphere merging to handle VMS subjects

## Phase 2: Core Story — Regional Specialization
- [ ] 2a. DMS vs VMS Response Profiles (headline finding)
  - Same-genotype cross-region: D1-DMS vs D1-VMS, D2-DMS vs D2-VMS
  - Change-aligned (Hit, Miss) and lick-aligned (Hit-lick, FA-lick)
  - Hypothesis: DMS = perceptual/decision, VMS = motivation/reward
- [ ] 2b. Genotype × Region Interaction (2×2 design)
  - Peak z-dF/F: genotype (D1/D2) × region (DMS/VMS)
  - Permutation-based interaction test
- [ ] 2c. Neural Psychometric Functions
  - Peak z-dF/F vs change_size, sigmoidal fits per region × genotype
  - Compare neural sensitivity to behavioral d'

## Phase 3: HMM Behavioral States (port from ephys)
- [ ] 3a. Port Bernoulli GLM-HMM from ephys repo
  - Same 5 covariates: bias, log2(change_size), prev_choice, prev_reward, prev_early_lick
  - Fit per-subject, K=2-5, select by BIC
- [ ] 3b. State-Conditioned Photometry
  - PETHs split by HMM state (Engaged vs Disengaged vs Impulsive)
  - Does D1 vs D2 signal differ more in one state than another?
- [ ] 3c. HMM State Fractions × Genotype
  - Do D1 mice spend more/less time in Impulsive state?
  - Cross-subject validation of learning trajectory (N=12 vs ephys N=1)

## Phase 4: Impulsivity Deep-Dive
- [ ] 4a. Pre-FA Signals by Region and Genotype
  - Lick-aligned FA PETHs, pre-lick ramping: D1 vs D2, DMS vs VMS
- [ ] 4b. Single-Trial Outcome Prediction (ROC)
  - Pre-stimulus baseline predicts Hit vs Miss? Hit vs FA?
  - AUC per region × genotype
- [ ] 4c. Impulsivity Regression
  - P(Impulsive) from HMM as continuous predictor of pre-trial dF/F

## Phase 5: Learning Trajectories
- [ ] 5a. Session-by-Session Neural Evolution
  - Peak z-dF/F per outcome across sessions, per region × genotype
  - Correlate with d' trajectory
- [ ] 5b. Does DMS or VMS "Learn" First?
  - Session of neural Hit-Miss discrimination emergence, DMS vs VMS
- [ ] 5c. HMM State × Learning
  - State fractions over sessions per genotype
  - Disengaged→Engaged transition vs neural signal changes

## Phase 6: Bridge Analyses (Advanced)
- [ ] 6a. Simplified Lick-Hazard Model with Photometry Predictor
  - Adapt discrete-time survival framework; dF/F as additional covariate
- [ ] 6b. Variance Partitioning
  - Linear mixed model: outcome, change_size, HMM state, session, subject → dF/F

## Execution Order
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

## Key Design Principles
- Normalize-then-average (not average-then-normalize)
- Shared baseline across conditions within same alignment
- Per-mouse averaging before group stats (avoid pseudo-replication)
- Non-parametric tests (Mann-Whitney, Kruskal-Wallis) given small n
- Bootstrap CI (1000 resamples, seed=42) for key estimates
- Region resolved via get_roi_region(roi, subject_id) — subject-dependent mapping
- QC + hemisphere merging for all analyses
