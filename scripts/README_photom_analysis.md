# Photometry analysis quickstart

This repo already contains utilities to process fiber photometry CSVs and align them to behavior. The script below wraps them to produce small session summaries and CSV outputs you can explore immediately.

## What it does
- Pairs `__trials.json` + `__session_settings.json` with `__photom_*.csv` and `__photom_IO_*.csv` on the same day (closest timestamps)
- Preprocesses photometry (isosbestic regression, Savitzky–Golay smoothing, motion correction, dF/F, optional z-score)
- Aligns to events (baseline, change, reaction)
- Extracts 0–1 s peak z-dF/F per ROI for Hits, Misses, and at Change time
- Saves a compact per-session CSV under `pdf_output/`

## Run it
From the repo root:

```bash
# Example: process all sessions found under this folder (BG_016 has sample files in root of photom_data)
python -m scripts.photometry_analysis photom_data

# Or point to a single-mouse folder (e.g., BG_021)
python -m scripts.photometry_analysis photom_data/BG_021

# Limit to first N sessions (debug)
python -m scripts.photometry_analysis photom_data --limit 1

# Choose output directory (default: pdf_output)
python -m scripts.photometry_analysis photom_data --out pdf_output
```

Outputs look like:
- `pdf_output/photom_summary_<mouse_id>_<session_date>.csv` containing per-session peak metrics
- Console printout with aggregate means across sessions

## Requirements
This analysis uses:
- numpy
- pandas
- scipy
- matplotlib
- seaborn

Install with:

```bash
python -m pip install -r requirements-photom.txt
```

If you encounter a pandas/numpy binary mismatch ("numpy.dtype size changed"), upgrade both to compatible wheels:

```bash
python -m pip install --upgrade --force-reinstall --no-cache-dir "numpy>=1.23,<2.0" "pandas>=1.5,<2.3" scipy matplotlib seaborn
```

## Notes
- Photometry CSVs can be large (>50MB). The script streams them with pandas; VS Code may not preview them.
- If a session has mismatched counts of trials vs baseline events in IO, it's skipped for safety.
- ROIs expected: `G0`, `G2` (DMS) and optionally `G4`, `G5` (VLS). The helper auto-detects presence of `G4/G5`.
