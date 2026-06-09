# Plan: Integrate BG_008–011 (VMS Subjects) into Pipeline

## Context

Four new subjects (BG_008, BG_009, BG_010, BG_011) recorded from **VMS (ventromedial striatum)** need to be incorporated. They use an older Neurophotometrics CSV format with different column names and embedded IO data.

- **BG_008, BG_009**: D1 (Drd1-Cre), VMS
- **BG_010, BG_011**: D2 (A2a), VMS
- **Source**: `X:\public\projects\BeJG_20230130_VisDetect\wPhotometry\matched\`
- **Trials JSON format**: Identical to BG_013+ (no changes needed)
- **Session settings format**: Identical to BG_013+ (no changes needed)

## Key Differences: Old vs New Photometry Format

| Feature | BG_008–011 (old) | BG_013+ (current) |
|---------|-------------------|--------------------|
| Timestamp column | `Timestamp` | `SystemTimestamp` |
| Channel columns | `Region0G`, `Region1R`, `Region2G`, `Region3R` | `G0`, `R1`, `G2`, `R3` |
| IO data | **Embedded** in photometry CSV (`Input0`, `Input1` columns) | Separate `*_IO_*.csv` file |
| Extra columns | `Stimulation`, `Output0`, `Output1` | `ComputerTimestamp` |
| VLS channels | None | `G4`, `G5` (some mice) |

## Recording Region Mapping

These subjects record from VMS, not DMS. The same physical channels (G0/G2) map to different brain regions:

| Channel | BG_008–011 | BG_013+ |
|---------|-----------|---------|
| G0 (Region0G) | **VMS_L** | DMS_L |
| G2 (Region2G) | **VMS_R** | DMS_R |

This requires a **subject-dependent ROI→region mapping**.

---

## Phase 1: Copy Data Locally

1. Copy from `X:\...\matched\BG_008\`, `BG_009\`, `BG_010\`, `BG_011\` to local `photom_data/`
2. **Exclude** heavy inner subject folders (e.g., `BG_008/BG_008/` with .bin, .mp4 files)
3. Only copy: `*__photom_*.csv`, `*__trials.json`, `*__session_settings.json`, `*__computer_settings.json`, Bonsai/config files
4. Handle BG_011 trial files found inside BG_010's folder → copy to BG_011 folder
5. Skip files named `*_delete_*`

## Phase 2: Update Constants & Configuration (`constants.py`)

6. Add new subjects to `SUBJECT_GENOTYPE`:
   - `BG_008: 'D1'`, `BG_009: 'D1'`, `BG_010: 'D2'`, `BG_011: 'D2'`

7. Add subject-dependent region mapping (`SUBJECT_ROI_REGION`):
   ```python
   # Default: G0/G2 = DMS (BG_013+)
   # Override for VMS subjects:
   SUBJECT_ROI_REGION = {
       'BG_008': {'G0': 'VMS_L', 'G2': 'VMS_R'},
       'BG_009': {'G0': 'VMS_L', 'G2': 'VMS_R'},
       'BG_010': {'G0': 'VMS_L', 'G2': 'VMS_R'},
       'BG_011': {'G0': 'VMS_L', 'G2': 'VMS_R'},
       # All others fall back to ROI_TO_REGION default (DMS)
   }
   ```

8. Add VMS to region colors, QC region pairs

9. Add column name mapping for old format:
   ```python
   OLD_FORMAT_COLUMN_MAP = {
       'Timestamp': 'SystemTimestamp',
       'Region0G': 'G0', 'Region1R': 'R1',
       'Region2G': 'G2', 'Region3R': 'R3',
   }
   ```

## Phase 3: Update IO Layer (`io.py`)

10. Update `pair_session_files()` to work when photom_IO files don't exist:
    - Currently requires all 4 files (photom, photom_IO, trials, session_settings)
    - Make photom_IO **optional** — return `None` for photom_io when absent
    - Sessions will extract IO events from embedded columns in the photometry CSV instead

11. Add CSV size filtering:
    - Skip photometry CSVs smaller than ~50KB (test/startup recordings)
    - Only pair the largest CSV per date (the real session recording)

12. Handle cross-subject contamination:
    - Filter files by matching subject ID from filename to the folder being processed

## Phase 4: Update Preprocessing (`preprocessing.py`)

13. Add column renaming at the start of `process_photometry_signals()`:
    - Detect old format by checking for `Region0G` column
    - Apply `OLD_FORMAT_COLUMN_MAP` renaming
    - Rest of pipeline works unchanged after renaming

## Phase 5: Update Session Loading (`session.py`)

14. Extract IO events from embedded photometry CSV when no IO file exists:
    - Detect `Input0` / `Input1` columns in the photometry DataFrame
    - Extract baseline onset times: detect rising edges (0→1 transitions) in `Input0`
    - Extract lick times: detect rising edges in `Input1`
    - Create equivalent of the IO DataFrame that `load_session_from_files()` currently expects

15. Update `load_session_from_files()` to handle `photom_io=None`:
    - If photom_io path is None, extract IO events from the photometry CSV
    - Pass extracted IO events to the same downstream timestamp computation

## Phase 6: Update QC & Analysis Infrastructure

16. Update `qc.py` REGION_PAIRS to include VMS:
    ```python
    REGION_PAIRS = {'DMS': ('G0', 'G2'), 'VLS': ('G4', 'G5')}
    # VMS subjects also use G0/G2 but map to VMS — handle via subject lookup
    ```

17. Helper function `get_roi_region(roi_name, subject_id)`:
    - Checks `SUBJECT_ROI_REGION` first, falls back to `ROI_TO_REGION`
    - Used everywhere a region label is needed

18. Update existing analysis scripts to be aware of VMS as a third region

## Phase 7: Validation & Testing

19. Test single-session load from each new subject
20. Run batch pipeline and verify:
    - Correct number of sessions loaded per subject
    - dF/F traces look reasonable (not flat, not all NaN)
    - IO events correctly extracted (baseline timestamps align with trials)
    - Trial counts match between JSON and IO events
21. Spot-check PETH extraction with a known session

## Risks & Edge Cases

- **Multiple CSVs per date**: Size-based filtering should select the real session; verify pairing is correct
- **Cross-subject files in BG_010**: Subject ID from filename must match folder being processed
- **IO event extraction accuracy**: Rising-edge detection on embedded Input0/Input1 must produce same event count as separate IO file would
- **Old format lacks ComputerTimestamp**: Not used in our pipeline, so no impact
- **`Stimulation`/`Output0`/`Output1` columns**: Ignored — just extra columns in the CSV

## Execution Order

1. Copy data (Phase 1) — independent, do first
2. Update constants (Phase 2) — independent
3. Update io.py (Phase 3) — depends on understanding format
4. Update preprocessing.py (Phase 4) — small change
5. Update session.py (Phase 5) — most complex change
6. Update QC/analysis (Phase 6) — depends on 2-5 being done
7. Validate (Phase 7) — final step
