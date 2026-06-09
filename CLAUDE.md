# CLAUDE.md — Project Manual for Claude Code

## Project Identity

Multi-subject fiber photometry recordings from medial striatum during a **visual change-detection task** in mice. GCaMP8m is expressed in D1 (Drd1-Cre) or D2 (A2a/Drd2-Cre) striatal projection neurons. Recording sites include DMS (dorsomedial striatum) and VLS (ventrolateral striatum). The project studies how genetically-defined striatal circuits support perceptual learning, impulsivity regulation, and action selection in a sensory decision-making task.

**Subjects**: BG_013, BG_014, BG_015 (D1/Drd1); BG_016, BG_017, BG_018, BG_019 (D2/A2a); BG_020 (D1/Drd1). Some have dual-site recordings (DMS + VLS).

**Related work**: Khilkevich & Lohse, Nature 2024 (brain-wide dynamics, ~250 ms integration timescale); Lohse et al. 2025 (frontal cortex gates striatal dynamics via task-state x sensory AND-gate).

---

## Critical Analysis Rules

These rules are **non-negotiable**. Violating them produces scientifically invalid results.

### Task Structure and Trial Types

The task has a **stimulus change detection** structure:
- **Inter-trial interval (ITI)**: Gray screen
- **Baseline period**: Drifting grating at 1 Hz base TF with stochastic fluctuations
- **Change event**: TF changes to [1.25, 1.35, 1.5, 2.0, 4.0] Hz (go trials) or stays ~1.0 (catch trials)
- **Response window**: Mouse must lick within 2.15s of change for reward

Trial classification by `change_size`:
- **Go trial**: `change_size > 1.0` (stimulus actually changed)
- **Catch trial**: `change_size ≈ 1.0` (no real change)

### Outcome Definitions

| Label | Meaning |
|-------|---------|
| `Hit` | Correct detection — lick in response window after real change |
| `Miss` | Failed to detect — no lick after real change |
| `FA` | False Alarm — anticipatory/impulsive lick before change event |
| `Abort` | Premature lick during ITI (trial terminated early) |
| `CR` | Correct Rejection — correctly withheld lick on catch trial |

**FA subtypes** (important for impulsivity analysis):
- **Early FA**: Reaction time ≤ 3.0 s (impulsive)
- **Late FA**: Reaction time > 3.0 s (potentially stimulus-driven)

**SDT (Signal Detection Theory) classification** for d' and psychometrics:
| SDT Category | How Defined |
|-------------|-------------|
| **SDT Hit** | `outcome='Hit'` AND `change_size > 1.0` |
| **SDT Miss** | `outcome='Miss'` AND `change_size > 1.0` |
| **SDT False Alarm** | `outcome='Hit'` AND `change_size ≈ 1.0` (catch trial lick) |
| **SDT Correct Rejection** | `outcome='Miss'` AND `change_size ≈ 1.0` |

**The behavioral `FA` label is NOT an SDT false alarm.** FA means the mouse licked before the change event. SDT false alarms are catch-trial `Hit` outcomes.

### Event Alignment Rules — CRITICAL

| Event | Valid Outcomes | Why |
|-------|---------------|-----|
| `Change` | `Hit`, `Miss` ONLY | On `FA`/`Abort` trials, **the change stimulus was never presented**. |
| `FA lick` | `FA` only | Motor-aligned: the lick itself is the event |
| `Hit lick` | `Hit` only | Motor-aligned: lick after detected change |
| `Baseline_ON` | All outcomes | Every trial has a baseline period |

### Key Constants

| Constant | Value | Used For |
|----------|-------|----------|
| Sampling rate | 100 Hz per channel (200 Hz raw, interleaved) | Photometry acquisition |
| Trim first | 10 s (1000 samples) | Remove startup artifacts |
| Isosbestic fit | `np.polyfit(deg=1)` | Wavelength-dependent bleaching correction |
| SavGol (isosbestic) | window=91, poly=3 | Smoothing isosbestic channel |
| SavGol (signal) | window=41, poly=2 | Smoothing signal channel |
| dF/F | `(signal - iso_fitted) / iso_fitted` | Photometry signal |
| PETH window | [-2, +4] s (default) or [-2, +1.5] s | Event-aligned extraction |
| FA RT split | 3.0 s | Early vs Late FA threshold |
| CHANGE_SIZES | [1.25, 1.35, 1.5, 2.0, 4.0] | Go-trial TF change ratios |

---

## Neuroscience Best Practices

### Photometry Analysis
- **Always correct for motion artifacts** using isosbestic (405 nm) channel via linear fit
- **Trim first 10 seconds** — startup transients corrupt signal
- **De-interleave** LedState 1 (isosbestic/405nm) and LedState 2 (signal/470nm) before processing
- **Trial-level z-scoring** for PETHs: subtract pre-event baseline mean, divide by baseline std

### Normalization Best Practices

**The Golden Rule**: Normalize each ROI/trace using a **shared baseline definition** across all conditions being compared.

| Analysis Type | Method | Rationale |
|---------------|--------|-----------|
| Session trace | dF/F from isosbestic fit | Corrects bleaching and motion |
| PETH heatmaps | Per-trial z-score (baseline-subtracted) | Equalizes across sessions |
| Cross-condition comparisons | Shared baseline z-score | Preserves relative magnitude |
| Population averages | Normalize-then-average | Prevents high-signal ROIs from dominating |

**Critical Pitfalls**:
1. **Circular baseline**: Each condition normalized to its own baseline inflates low-activity conditions
2. **Average-then-normalize**: High-signal ROIs dominate the average — always normalize first
3. **Division by zero**: Guard `if std < 1e-6: std = 1.0`

### Signal Detection Theory
- d' = z(hit_rate) - z(fa_rate), with log-linear correction clipping rates to [0.01, 0.99]
- Hit rate on **go trials only**. FA rate on **catch trials only**.
- Always report criterion c alongside d' when assessing response bias

### Statistical Standards
- **Non-parametric by default** for neural data (Kruskal-Wallis, Mann-Whitney U, Spearman rho)
- **Two-sided tests** unless hypothesis is strongly directional
- **Bootstrap CI** (1000 resamples, seed=42, percentile method) for key estimates
- **Report effect sizes** alongside every p-value

---

## Architecture

### Dual-System (Legacy + New Package)

The codebase has two coexisting architectures:

1. **Legacy monolithic**: `scripts/vis_detect_helpers_v9.py` (814 lines) + `scripts/photometry_analysis.py` (937 lines)
2. **New modular package**: `src/visdetect_photom/` — proper dataclasses, separated concerns

Scripts progressively adopt the new package. **Prefer the new package for all new work.**

### New Package: `src/visdetect_photom/`

```
visdetect_photom/
  core/
    io.py          File discovery, session pairing (find_all_sessions, pair_session_files)
    session.py     Dataclasses: Trial, PhotometryTrace, Session; load_session_from_files()
    qc.py          Stub for QC pipeline
  analysis/
    preprocessing.py  Photometry signal processing (de-interleave, isosbestic fit, SavGol, dF/F)
    statistics.py     PETH extraction (extract_peth), performance metrics, behavioral metrics
    optogenetics.py   Stub
  viz/
    plotting.py       Heatmap + melted trace plots
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/batch_processing/01_batch_session_analysis.py` | Primary batch pipeline (new package) |
| `scripts/photometry_analysis.py` | Legacy main pipeline (937 lines) |
| `scripts/photom_group_analysis.py` | Group-level analysis by genotype/region |
| `scripts/behavior_metrics.py` | Batch behavioral metrics export |
| `scripts/photom_report.py` | PDF report generation |
| `scripts/analysis/behavior/plot_session_behavior.py` | Single-session behavior plots |
| `scripts/data_management/filter_sessions.py` | QC-based session filtering |
| `scripts/data_management/create_session_manifest.py` | Session manifest builder |

### Data Flow

```
Raw Files (photom_data/<subject>/)
├── Photometry CSV    (FrameCounter, SystemTimestamp, LedState, G0, R1, G2, R3 [, G4, G5])
├── Photometry IO CSV (Input0=baseline onset, Input1=licks)
├── Trials JSON       (per-trial behavioral data)
└── Session Settings  (protocol, parameters)
            ↓
    io.find_all_sessions() + io.pair_session_files()
            ↓
    session.load_session_from_files()
    ├── Trial dataclasses (outcome, change_time, change_size, RT)
    ├── PhotometryTrace per ROI (dF/F, z-scored)
    └── Absolute timestamps (from photom_IO baseline events)
            ↓
    statistics.extract_peth()  (align signal to events)
            ↓
    Visualization + Summary CSVs + FIGURES/
```

### ROI Naming

- `G0` = DMS Left, `G2` = DMS Right (green channels, primary)
- `G4` = VLS Left, `G5` = VLS Right (green channels, dual-site mice only)
- `R1`, `R3` = Red channel references (not typically analyzed)

---

## How to Write a New Analysis Script

### Template

```python
"""Description of analysis."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from visdetect_photom.core.io import find_all_sessions, pair_session_files
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.preprocessing import process_photometry_signals

# ── Configuration ─────────────────────────────────────────────
DATA_ROOT = r"photom_data"
FIGURES_ROOT = r"FIGURES"

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sessions = find_all_sessions(DATA_ROOT)
    for subject_id, file_paths in sessions.items():
        session = load_session_from_files(file_paths)
        # ... analysis ...
```

### Checklist before finalizing
- [ ] Uses new package (`src/visdetect_photom/`) not legacy helpers
- [ ] Constants not hardcoded (sampling rate, windows, thresholds)
- [ ] Event alignment filters outcomes correctly
- [ ] dF/F computed via isosbestic correction (not raw signal)
- [ ] Output saved to appropriate `FIGURES/` subfolder
- [ ] Memory cleanup if processing many sessions

---

## Gotchas and Pitfalls

| Gotcha | Detail |
|--------|--------|
| `py` not `python` | Windows + Git Bash requires `py` to invoke Python |
| Interleaved sampling | Raw photometry CSV alternates LedState 1 and 2 rows — must de-interleave |
| First 10s trimmed | Startup artifacts — `preprocessing.py` handles this |
| FA ≠ SDT false alarm | Behavioral FA = early lick. SDT FA = catch-trial hit. |
| `change_size` determines trial type | Go vs catch from stimulus, NOT from outcome label |
| FA RT split differs | New code: 3.0 s. Legacy code: 2.0 s. Use 3.0 s for new analyses. |
| Protocol inference | Protocols 1-5 inferred from session settings (hazardtype, pprobe0, Trewdavailable) |
| Dual-site mice | Only some mice have G4/G5 (VLS) channels |
| Search before writing | **Always search the codebase for existing functions before writing new ones** |

## Environment

- Windows 10, Git Bash shell
- Python invoked via `py` (not `python`)
- Dependencies: numpy<2.0, pandas<2.3, scipy<2.0, matplotlib<3.9, seaborn<0.14, pyyaml, tqdm

## Skills

Six specialized skills in `.claude/skills/`:

### Scientific Workflow
| Skill | When to Use |
|-------|-------------|
| **Research Visualizer** | Figure design, color choices, layout, multi-option proposals |
| **Research Statistician** | Test selection, effect sizes, multiple comparisons, results tables |
| **Research Notes Summarizer** | Methods documentation, results summaries, scientific writing |

### Engineering Workflow
| Skill | When to Use |
|-------|-------------|
| **Codebase Auditor** | Systematic quality audit: alignment safety, constants, naming, dependencies |
| **Systematic Debugging** | Any bug, test failure, or unexpected behavior — root cause before fixes |
| **Verification Before Completion** | Evidence before claims — run verification before declaring success |

Skills activate automatically based on context.
