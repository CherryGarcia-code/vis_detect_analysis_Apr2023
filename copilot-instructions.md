# AI Assistant Instructions for Visual Detection Photometry Analysis

**Purpose**: Comprehensive instructions for AI assistants working on this fiber photometry neuroscience analysis repository.
**Priority**: This is the **canonical** instruction file. See `CLAUDE.md` for the detailed project manual.

---

## Project Context

### What This Project Is
Multi-subject fiber photometry analysis of medial striatal neurons (D1/D2 SPNs) during a visual change-detection task. Mice express GCaMP8m in genetically-defined striatal populations (Drd1-Cre or A2a/Drd2-Cre) and are recorded from DMS and/or VLS.

### Scientific Framework
- **Reference**: Khilkevich & Lohse, Nature 2024; Lohse et al. 2025 (frontal cortex gates striatal dynamics)
- **Task**: Mouse detects changes in temporal frequency (TF) of a drifting grating
- **Trial Types**: Go trials (change_size > 1.0), Catch trials (~1.0)
- **Key Questions**: How do D1 vs D2 neurons encode decisions? How does impulsivity balance with stimulus sensitivity?

---

## Project Structure

### Core Architecture
- **`src/visdetect_photom/`** — New modular Python package (preferred for new work)
  - `core/io.py` — File discovery, session pairing
  - `core/session.py` — Trial/PhotometryTrace/Session dataclasses
  - `analysis/preprocessing.py` — Isosbestic correction, dF/F, smoothing
  - `analysis/statistics.py` — PETH extraction, performance metrics
  - `viz/plotting.py` — Heatmaps, trace plots
- **`scripts/`** — Analysis scripts organized by domain
  - `batch_processing/` — Primary batch pipelines
  - `analysis/behavior/` — Behavioral analysis
  - `data_management/` — QC, manifests, exports
- **`scripts/vis_detect_helpers_v9.py`** — Legacy monolithic helper (814 lines)
- **`scripts/photometry_analysis.py`** — Legacy main pipeline (937 lines)

### Key Data Files
| Location | Purpose |
|----------|---------|
| `photom_data/<subject>/` | Raw data (photometry CSV, IO CSV, trials JSON, settings JSON) |
| `FIGURES/` | All output figures, manifests, batch output |
| `config/session_qc.yml` | QC rule profiles for session filtering |

---

## Agent Roles

- **DataWrangler**: Parse and normalize session data into dataclasses/DataFrames. Tools: pandas, numpy.
- **NeuroAnalyst**: Event-aligned photometry analysis, PETHs, behavioral metrics. Tools: scipy, scikit-learn.
- **VizBot**: Publication-quality figures. Tools: matplotlib, seaborn.

---

## Coding Conventions

### Modern Python
```python
# Use type hints, docstrings, and the new package
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.analysis.statistics import extract_peth

def analyze_session(file_paths: dict) -> pd.DataFrame:
    """Analyze a single session's photometry data."""
    session = load_session_from_files(file_paths)
    # ...
```

### Critical Rules
- Use `src/visdetect_photom/` for new work (not legacy helpers)
- Never align FA/Abort trials to change onset (change was never presented)
- Always apply isosbestic correction before analysis
- FA ≠ SDT false alarm (behavioral FA = early lick; SDT FA = catch-trial hit)
- Constants: sampling rate=100Hz, trim=10s, FA RT split=3.0s

### Notebook Conventions
- Existing cells must preserve `metadata.id` and specify `metadata.language`
- New cells do not need `metadata.id` but must include `metadata.language`
- Include a top-level markdown cell describing purpose and environment

### Environment
```bash
py -c "import visdetect_photom; print('OK')"  # Windows + Git Bash
```

---

## Scientific Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHANGE_SIZES` | [1.25, 1.35, 1.5, 2.0, 4.0] | Go-trial TF ratios |
| `FA_RT_SPLIT` | 3.0 s | Early vs late FA split |
| `PETH_WINDOW` | [-2, +4] s | Default event-aligned window |
| Sampling rate | 100 Hz per channel | Photometry acquisition |
| Isosbestic | 405 nm (LedState 1) | Motion/bleaching control |
| Signal | 470 nm (LedState 2) | Calcium-dependent |

---

## Priority Rules

When instructions conflict:
1. **Safety & Privacy**: Never expose credentials or sensitive data
2. **Explicit user request** in the current conversation
3. **`CLAUDE.md`** (detailed project manual with critical analysis rules)
4. **This file** (`copilot-instructions.md`)
5. **`README.md`** and other project documentation

---

## Best Practices

### Code Quality
- Prefer new package imports over legacy helpers
- Search codebase for existing functions before writing new ones
- Handle missing data gracefully (not all mice have VLS channels)
- Avoid hardcoded paths or subject names

### Analysis Standards
- Use shared baseline normalization for cross-condition comparisons
- Apply isosbestic correction (never analyze raw 470nm signal alone)
- Use non-parametric statistics by default for neural data
- Include effect sizes with p-values
- Report sample sizes for every test

### Documentation
- Clear docstrings with parameter descriptions
- Figure captions that explain the analysis
- Document assumptions and limitations

---

## Allowed and Disallowed Actions
- Allowed (with caution): read repository files, create/modify code and notebooks, run local lint/tests if requested, suggest environment changes
- Disallowed: making external network requests or disclosing secrets; do not publish private data to external services without explicit user consent

## If You Need Clarification
- Ask the user for missing details. When in doubt, propose a short plan and request approval before large changes.

---

*This is the canonical AI assistant instruction file. See `CLAUDE.md` for the comprehensive project manual.*
