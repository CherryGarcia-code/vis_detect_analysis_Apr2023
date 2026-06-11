"""Baseline-stimulus (TF-pulse) reconstruction + Input0-anchored alignment (G1).

Places baseline TF pulses on a uniform 50 ms grid off the baseline-onset
(Input0) timestamp — identical convention to the ephys tf_pulse.py — and
validates per-trial timing against the change anchor.
"""
import numpy as np
from visdetect_photom.core.constants import (
    TF_BASELINE_STRIDE, TF_SAMPLE_PERIOD, TF_MIN_AFTER_BASELINE,
    TF_MIN_BEFORE_CHANGE, TF_MIN_BEFORE_OUTCOME_FA_ABORT,
    TF_FAST_THRESH_LOG2, TF_SLOW_THRESH_LOG2, TF_BASE_HZ,
    TF_CHANGE_VALIDATE_MIN_CS, TF_CHANGE_VALIDATE_TOL,
)

_FA_LIKE = ("FA", "Abort")


def baseline_onset_ts(trial):
    """Baseline-grating onset in photometry SystemTimestamp (Input0). None if N/A."""
    if trial.absolute_change_time is None or trial.change_time is None:
        return None
    return float(trial.absolute_change_time - trial.change_time)


def baseline_pulse_values(trial, stride=TF_BASELINE_STRIDE):
    """The 50 ms baseline pulse sequence = St1TrialVector[::stride]. None if missing."""
    md = getattr(trial, "metadata", None) or {}
    st1 = md.get("St1TrialVector")
    if st1 is None:
        return None
    arr = np.asarray(st1, dtype=float).ravel()
    if arr.size == 0:
        return None
    return arr[::stride]


def n_baseline_samples(trial, sample_period=TF_SAMPLE_PERIOD):
    """Number of baseline pulses actually shown before change (go/CR) or lick (FA/abort)."""
    o = trial.outcome
    if o in _FA_LIKE and trial.reaction_time is not None:
        return int(round(trial.reaction_time / sample_period))
    if trial.change_time is not None:
        return int(round(trial.change_time / sample_period))
    return 0


def windowed_pulses(trial, sample_period=TF_SAMPLE_PERIOD):
    """(values, abs_times) for baseline pulses inside the usable, margin-trimmed window."""
    onset = baseline_onset_ts(trial)
    vals = baseline_pulse_values(trial)
    if onset is None or vals is None:
        return np.array([]), np.array([])
    n = min(n_baseline_samples(trial, sample_period), len(vals))
    if n <= 0:
        return np.array([]), np.array([])
    vals = vals[:n]
    times = onset + np.arange(n) * sample_period
    start = onset + TF_MIN_AFTER_BASELINE
    if trial.outcome in _FA_LIKE:
        end = onset + (trial.reaction_time or 0.0) - TF_MIN_BEFORE_OUTCOME_FA_ABORT
    else:
        end = onset + (trial.change_time or 0.0) - TF_MIN_BEFORE_CHANGE
    mask = (times >= start) & (times <= end)
    return vals[mask], times[mask]


def _log2_tf(vals):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(np.asarray(vals, float) / TF_BASE_HZ)


def fast_slow_pulse_times(trial):
    """(fast_times, slow_times): absolute photometry timestamps (not TF values) of
    fast/slow pulses, classified by log2(TF) vs +/-0.25 within the usable window."""
    vals, times = windowed_pulses(trial)
    if vals.size == 0:
        return np.array([]), np.array([])
    l2 = _log2_tf(vals)
    return times[l2 >= TF_FAST_THRESH_LOG2], times[l2 <= TF_SLOW_THRESH_LOG2]


def aligned_baseline_regressor(trial):
    """(log2_tf, abs_times) for the windowed baseline (continuous TRF input)."""
    vals, times = windowed_pulses(trial)
    if vals.size == 0:
        return np.array([]), np.array([])
    l2 = _log2_tf(vals)
    good = np.isfinite(l2)
    return l2[good], times[good]


def validate_change_anchor(trial, tol=TF_CHANGE_VALIDATE_TOL):
    """Best-effort timing QC using realized TF + vbl.

    Returns (ok, mismatch_s). Only applicable when change_size is large enough to
    detect the post-change TF level (>= TF_CHANGE_VALIDATE_MIN_CS); otherwise
    returns (True, nan) = 'not applicable, do not drop'.
    """
    md = getattr(trial, "metadata", None) or {}
    tf, vbl = md.get("TF"), md.get("vbl")
    onset = baseline_onset_ts(trial)
    cs = trial.change_size
    if cs is None or cs < TF_CHANGE_VALIDATE_MIN_CS:
        return True, np.nan
    if tf is None or vbl is None or onset is None or trial.absolute_change_time is None:
        return True, np.nan
    tf = np.asarray(tf, float)
    vbl = np.asarray(vbl, float)
    if tf.size != vbl.size or tf.size == 0:
        return True, np.nan
    # Onset = first nonzero TF frame. Assumes the pre-baseline (ITI gray) period
    # is exactly TF == 0, which holds for this task protocol. A nonzero pre-baseline
    # (protocol variant / corrupt gray frames) would mis-locate onset and silently
    # drop otherwise-valid trials, so this contract is worth knowing if data changes.
    nz = np.where(tf > 0)[0]
    if nz.size == 0:
        return True, np.nan
    onset_frame = nz[0]
    stim2 = md.get("Stim2TF")
    if stim2 is None:
        return True, np.nan
    after = np.arange(tf.size) > onset_frame
    near = np.abs(tf - stim2) <= 0.1 * abs(stim2)
    cand = np.where(after & near)[0]
    if cand.size == 0:
        return True, np.nan
    mapped = onset + (vbl[cand[0]] - vbl[onset_frame])
    mism = abs(mapped - trial.absolute_change_time)
    return bool(mism <= tol), float(mism)
