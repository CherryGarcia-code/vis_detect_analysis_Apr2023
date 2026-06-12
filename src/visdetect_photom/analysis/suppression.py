"""C1 — waiting/decision-period suppression-failure analysis.

Per-trial waiting-period scalars (two window schemes), two outcome tracks
(behavioral FA + SDT-FA control), per-mouse Δ(withhold-lick) and single-trial
AUROC, and a coarse proficiency split. Thin script 11 consumes this. See
docs/superpowers/specs/2026-06-11-c1-fa-suppression-design.md.

D1 and D2 are DIFFERENT animals: every D1-vs-D2 result is a GROUP-LEVEL sign
contrast, never within-animal anticorrelation.
"""
import numpy as np

from visdetect_photom.core.constants import (
    CATCH_THRESHOLD, WINDOW_MIN_SAMPLES, SCHEME1_WINDOW, SCHEME1_MOTOR_BUFFER,
)

# Groups that represent a premature action (have an action time = lick_elapsed)
_ACTION_GROUPS = ("lick", "abort")


def _subject_full(subject_id):
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def window_mean(signal, timestamps, t_start, t_end, min_samples=WINDOW_MIN_SAMPLES):
    """Mean of finite signal samples with t in [t_start, t_end]; NaN if < min_samples."""
    signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    seg = signal[mask]
    seg = seg[np.isfinite(seg)]
    if seg.size < min_samples:
        return np.nan
    return float(np.mean(seg))


def _group_for(outcome, change_size, track):
    """Map (outcome, change_size) to a C1 group for the given track, or None."""
    if track == "behavioral_fa":
        if outcome == "FA":
            return "lick"
        if outcome in ("Hit", "Miss", "CR"):
            return "withhold"
        if outcome == "Abort":
            return "abort"
        return None
    if track == "sdt_fa":
        is_catch = change_size is not None and change_size <= CATCH_THRESHOLD
        if not is_catch:
            return None
        if outcome == "Hit":
            return "lick"
        if outcome == "Miss":
            return "withhold"
        return None
    raise ValueError(f"unknown track: {track!r}")


def trial_waiting_records(session, track, keep=None):
    """List of per-trial dicts for a track.

    Each record: trial_index, group ('lick'|'withhold'|'abort'), onset_abs
    (grating onset = absolute_change_time - change_time), change_time, lick_abs
    (absolute_reaction_time or NaN), lick_elapsed (lick_abs - onset_abs or NaN).
    Trials whose group is None (not part of this track) or whose grating onset
    cannot be recovered are skipped.
    `keep`: optional set of trial indices to retain (state filtering).
    """
    out = []
    for t in session.trials:
        if keep is not None and t.trial_index not in keep:
            continue
        if t.absolute_change_time is None or t.change_time is None:
            continue
        group = _group_for(t.outcome, t.change_size, track)
        if group is None:
            continue
        onset_abs = float(t.absolute_change_time) - float(t.change_time)
        lick_abs = (float(t.absolute_reaction_time)
                    if t.absolute_reaction_time is not None else np.nan)
        lick_elapsed = lick_abs - onset_abs if np.isfinite(lick_abs) else np.nan
        out.append({
            "trial_index": t.trial_index, "group": group, "onset_abs": onset_abs,
            "change_time": float(t.change_time), "lick_abs": lick_abs,
            "lick_elapsed": lick_elapsed,
        })
    return out


def scheme1_scalar(record, signal, timestamps,
                   window=SCHEME1_WINDOW, motor_buffer=SCHEME1_MOTOR_BUFFER):
    """Baseline-onset-anchored fixed-window mean for one trial, or NaN if excluded.

    Window [onset+w0, onset+w1]. Excluded unless it ends before the change
    (change_time >= w1); for action groups (lick/abort) it must also end
    motor_buffer before the action (lick_elapsed >= w1 + motor_buffer).
    """
    w0, w1 = window
    if record["change_time"] is None or record["change_time"] < w1:
        return np.nan
    if record["group"] in _ACTION_GROUPS:
        le = record["lick_elapsed"]
        if not np.isfinite(le) or le < w1 + motor_buffer:
            return np.nan
    t0 = record["onset_abs"] + w0
    t1 = record["onset_abs"] + w1
    return window_mean(signal, timestamps, t0, t1)
