"""
Central constants for the visdetect_photom package.

All magic numbers and shared configuration values live here.
Import from this module instead of hardcoding values in scripts.
"""

# ── Photometry Acquisition ────────────────────────────────────
SAMPLING_FREQ = 100           # Hz per channel (after de-interleaving)
RAW_SAMPLING_FREQ = 200       # Hz interleaved (LedState 1 + 2)
TRIM_SECONDS = 10             # Startup artifact removal (seconds)
TRIM_SAMPLES = TRIM_SECONDS * SAMPLING_FREQ  # 1000 samples

# ── Signal Processing ─────────────────────────────────────────
SAVGOL_ISO_WINDOW = 91        # Isosbestic smoothing window (must be odd)
SAVGOL_ISO_POLY = 3           # Isosbestic smoothing polynomial order
SAVGOL_SIG_WINDOW = 41        # Signal smoothing window (must be odd)
SAVGOL_SIG_POLY = 2           # Signal smoothing polynomial order
ISO_FIT_DEGREE = 1            # Polynomial degree for isosbestic fit

# ── PETH Extraction ───────────────────────────────────────────
PETH_WINDOW = (-2.0, 4.0)    # Default event-aligned window (seconds)
PETH_BASELINE = (-2.0, 0.0)  # Default baseline for z-scoring (seconds)

# ── Task Parameters ───────────────────────────────────────────
CHANGE_SIZES = [1.25, 1.35, 1.5, 2.0, 4.0]  # Go-trial TF change ratios
CATCH_THRESHOLD = 1.01        # change_size <= this = catch trial
FA_RT_SPLIT = 3.0             # Early vs Late FA threshold (seconds)
RESPONSE_WINDOW = 2.15        # Response window duration (seconds)

# ── Outcome Labels ────────────────────────────────────────────
OUTCOME_LABELS = ['Hit', 'Miss', 'FA', 'Abort', 'CR']

# Outcomes valid for each event alignment type
EVENT_VALID_OUTCOMES = {
    'change': ['Hit', 'Miss'],           # Only trials where change was presented
    'fa_lick': ['FA'],                   # Motor-aligned to the FA lick
    'hit_lick': ['Hit'],                 # Motor-aligned to the hit lick
    'baseline_on': OUTCOME_LABELS,       # Every trial has a baseline period
}

# Raw JSON outcome normalization (applied at load time in session.py)
OUTCOME_NORMALIZATION = {
    'abort': 'Abort',
    'Abort': 'Abort',
    'Ref': 'CR',
    'ref': 'CR',
    'CR': 'CR',
    'Hit': 'Hit',
    'hit': 'Hit',
    'Miss': 'Miss',
    'miss': 'Miss',
    'FA': 'FA',
    'fa': 'FA',
}

# ── ROI Mapping ───────────────────────────────────────────────
ROI_TO_REGION = {
    'G0': 'DMS_L',
    'G2': 'DMS_R',
    'G4': 'VLS_L',
    'G5': 'VLS_R',
}

# Subject-specific ROI→region overrides (VMS subjects use G0/G2 for VMS, not DMS)
SUBJECT_ROI_REGION = {
    'BG_008': {'G0': 'VMS_L', 'G2': 'VMS_R'},
    'BG_009': {'G0': 'VMS_L', 'G2': 'VMS_R'},
    'BG_010': {'G0': 'VMS_L', 'G2': 'VMS_R'},
    'BG_011': {'G0': 'VMS_L', 'G2': 'VMS_R'},
}


def get_roi_region(roi_name, subject_id=None):
    """Get brain region for an ROI, accounting for subject-specific mappings.

    Parameters
    ----------
    roi_name : str
        ROI channel name (e.g. 'G0', 'G2', 'G4', 'G5')
    subject_id : str, optional
        Subject identifier (e.g. 'BG_008'). If provided, checks for
        subject-specific overrides (e.g. VMS subjects).

    Returns
    -------
    str or None
        Region name (e.g. 'DMS_L', 'VMS_R') or None if unknown.
    """
    if subject_id and subject_id in SUBJECT_ROI_REGION:
        region = SUBJECT_ROI_REGION[subject_id].get(roi_name)
        if region is not None:
            return region
    return ROI_TO_REGION.get(roi_name)

# ── Subject Genotypes ─────────────────────────────────────────
SUBJECT_GENOTYPE = {
    'BG_008': 'D1',
    'BG_009': 'D1',
    'BG_010': 'D2',
    'BG_011': 'D2',
    'BG_013': 'D1',
    'BG_014': 'D1',
    'BG_015': 'D1',
    'BG_020': 'D1',
    'BG_016': 'D2',
    'BG_017': 'D2',
    'BG_018': 'D2',
    'BG_019': 'D2',
}

# ── Visualization Defaults ────────────────────────────────────
GENOTYPE_COLORS = {'D1': '#2ca02c', 'D2': '#1f77b4'}  # green, blue
REGION_COLORS = {'DMS': '#d62728', 'VLS': '#ff7f0e', 'VMS': '#9467bd'}  # red, orange, purple
OUTCOME_COLORS = {
    'Hit': '#2ca02c',
    'Miss': '#9467bd',
    'FA': '#d62728',
    'Abort': '#7f7f7f',
    'CR': '#17becf',
}

# ── Old-Format (BG_008–011) Column Mapping ───────────────────
# These subjects used an older Neurophotometrics Bonsai export format
# with different column names and IO data embedded in the photometry CSV.
OLD_FORMAT_COLUMN_MAP = {
    'Timestamp': 'SystemTimestamp',
    'Region0G': 'G0',
    'Region1R': 'R1',
    'Region2G': 'G2',
    'Region3R': 'R3',
}

# Minimum photometry CSV file size (bytes) to consider a real session.
# Smaller files are test/startup recordings and should be skipped.
MIN_PHOTOM_CSV_BYTES = 50_000

# ── TF-pulse / evidence encoding (G1) — mirrors ephys tf_pulse.py conventions ──
TF_BASE_HZ = 1.0                       # nominal base temporal frequency
TF_BASELINE_STRIDE = 3                 # St1TrialVector repeats each pulse 3x (60fps)
TF_SAMPLE_PERIOD = 0.05                # seconds per baseline pulse sample (50 ms)
TF_FAST_THRESH_LOG2 = 0.25             # fast pulse: log2(TF) >= +0.25
TF_SLOW_THRESH_LOG2 = -0.25            # slow pulse: log2(TF) <= -0.25
TF_MIN_AFTER_BASELINE = 1.0            # exclude pulses < 1.0 s after baseline onset
TF_MIN_BEFORE_CHANGE = 1.0             # exclude pulses < 1.0 s before change
TF_MIN_BEFORE_OUTCOME_FA_ABORT = 2.0   # exclude pulses < 2.0 s before FA/abort lick
TF_PULSE_PRE_WINDOW = (-0.4, 0.0)      # pre-pulse z-score baseline
TF_PULSE_POST_WINDOW = (0.0, 0.5)      # post-pulse response window
TF_PULSE_DETREND_BASELINE = (-0.4, -0.01)
TF_PULSE_DETREND_POST = (0.0, 0.3)
TF_CHANGE_VALIDATE_MIN_CS = 2.0        # only run change-anchor validation when change_size >= this
TF_CHANGE_VALIDATE_TOL = 0.05          # 50 ms mismatch tolerance
# TRF lag grid (negatives = causality control)
TRF_LAG_MIN = -0.5
TRF_LAG_MAX = 2.0
TRF_LAG_STEP = 0.05
