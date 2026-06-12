"""C1 — waiting/decision-period suppression-failure analysis.

Per-trial waiting-period scalars (two window schemes), two outcome tracks
(behavioral FA + SDT-FA control), per-mouse Δ(withhold-lick) and single-trial
AUROC, and a coarse proficiency split. Thin script 11 consumes this. See
docs/superpowers/specs/2026-06-11-c1-fa-suppression-design.md.

D1 and D2 are DIFFERENT animals: every D1-vs-D2 result is a GROUP-LEVEL sign
contrast, never within-animal anticorrelation.
"""
import numpy as np
import pandas as pd

from visdetect_photom.core.constants import (
    CATCH_THRESHOLD, WINDOW_MIN_SAMPLES, SCHEME1_WINDOW, SCHEME1_MOTOR_BUFFER,
    SCHEME3_L, SCHEME3_BUFFER, HAZARD_RESAMPLES, HAZARD_SEED,
    MIN_TRIALS_PER_GROUP,
)
from visdetect_photom.analysis.group_statistics import (
    auroc_score, bootstrap_ci, permutation_test, pushpull_sign_contrast,
)
from visdetect_photom.core.qc import region_sources
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.analysis.state_provider import filter_trials_by_state
from visdetect_photom.core.staging import get_session_stage

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

    Window [onset+w0, onset+w1]. Excluded unless it ends strictly before the
    change (change_time > w1); for action groups (lick/abort) it must also end
    motor_buffer before the action (lick_elapsed >= w1 + motor_buffer).
    """
    w0, w1 = window
    if record["change_time"] is None or record["change_time"] <= w1:
        return np.nan
    if record["group"] in _ACTION_GROUPS:
        le = record["lick_elapsed"]
        if not np.isfinite(le) or le < w1 + motor_buffer:
            return np.nan
    t0 = record["onset_abs"] + w0
    t1 = record["onset_abs"] + w1
    return window_mean(signal, timestamps, t0, t1)


def scheme3_scalars(action_records, withhold_records, signal, timestamps,
                    L=SCHEME3_L, buffer=SCHEME3_BUFFER,
                    n_resample=HAZARD_RESAMPLES, seed=HAZARD_SEED):
    """Hazard-time-matched waiting-period scalars (Scheme 3).

    action_records: premature-action trials (FA licks or aborts). Window
        [act-buffer-L, act-buffer], ending `buffer` before the action.
    withhold_records: trials that reached the change. For each, draw
        `n_resample` pseudo-action elapsed-times from the action group's
        elapsed-time distribution (truncated to <= change_time and window
        start >= 0), average the per-draw window means. Deterministic (seed).

    Returns (action_vals, withhold_vals), each a list of (trial_index, scalar).
    """
    action_vals = []
    elapsed_pool = []
    for r in action_records:
        le = r["lick_elapsed"]
        if np.isfinite(le):
            elapsed_pool.append(le)
        if not np.isfinite(le) or (le - buffer - L) < 0:
            action_vals.append((r["trial_index"], np.nan))
            continue
        ws = r["onset_abs"] + le - buffer - L
        we = r["onset_abs"] + le - buffer
        action_vals.append((r["trial_index"], window_mean(signal, timestamps, ws, we)))

    pool = np.asarray(elapsed_pool, dtype=float)
    rng = np.random.default_rng(seed)
    withhold_vals = []
    for r in withhold_records:
        if pool.size == 0 or not np.isfinite(r["change_time"]):
            withhold_vals.append((r["trial_index"], np.nan))
            continue
        draws = rng.choice(pool, size=n_resample, replace=True)
        means = []
        for tau in draws:
            if tau > r["change_time"] or (tau - buffer - L) < 0:
                continue
            ws = r["onset_abs"] + tau - buffer - L
            we = r["onset_abs"] + tau - buffer
            m = window_mean(signal, timestamps, ws, we)
            if np.isfinite(m):
                means.append(m)
        withhold_vals.append((r["trial_index"],
                              float(np.mean(means)) if means else np.nan))
    return action_vals, withhold_vals


_DATASET_COLUMNS = ["subject_id", "genotype", "region", "track", "scheme",
                    "group", "trial_index", "scalar", "session_id", "stage"]


def build_session_scalars(session, *, track, scheme, use_qc=True,
                          state_provider=None, keep_states=None, stage="Unknown"):
    """Per-trial waiting-period scalar rows for one session (one track, one scheme).

    Row keys: subject_id, genotype, region, track, scheme, group, trial_index,
              scalar, session_id, stage.
    """
    subject_full = _subject_full(session.subject_id)
    genotype = get_genotype(subject_full)
    if genotype == "Unknown":
        return []

    keep = None
    if state_provider is not None and keep_states is not None:
        keep = filter_trials_by_state(session, state_provider, keep_states)

    records = trial_waiting_records(session, track, keep)
    sources = region_sources(session, use_qc)
    rows = []

    def _emit(region, group, trial_index, scalar):
        rows.append({"subject_id": subject_full, "genotype": genotype,
                     "region": region, "track": track, "scheme": scheme,
                     "group": group, "trial_index": trial_index,
                     "scalar": scalar, "session_id": session.session_id,
                     "stage": stage})

    for region, (sig, ts) in sources.items():
        if scheme == "scheme1":
            for r in records:
                _emit(region, r["group"], r["trial_index"],
                      scheme1_scalar(r, sig, ts))
        elif scheme == "scheme3":
            # Primary lick-vs-withhold. Withhold scalars are hazard-matched to the
            # lick elapsed-time distribution. Abort is scheme1-only (exploratory):
            # it would need its own abort-matched withhold control, out of scope here.
            lick = [r for r in records if r["group"] == "lick"]
            withhold = [r for r in records if r["group"] == "withhold"]
            a_vals, w_vals = scheme3_scalars(lick, withhold, sig, ts)
            for ti, v in a_vals:
                _emit(region, "lick", ti, v)
            for ti, v in w_vals:
                _emit(region, "withhold", ti, v)
        else:
            raise ValueError(f"unknown scheme: {scheme!r}")
    return rows


def build_suppression_dataset(sessions, *, track, scheme, use_qc=True,
                              state_provider=None, keep_states=None, manifest=None):
    """Concatenate per-trial scalar rows across sessions into a DataFrame.

    If `manifest` is given, each session's learning stage is attached.
    """
    all_rows = []
    for sess in sessions:
        stage = get_session_stage(sess, manifest) if manifest is not None else "Unknown"
        all_rows.extend(build_session_scalars(
            sess, track=track, scheme=scheme, use_qc=use_qc,
            state_provider=state_provider, keep_states=keep_states, stage=stage))
    if not all_rows:
        return pd.DataFrame(columns=_DATASET_COLUMNS)
    return pd.DataFrame(all_rows)


def compute_delta_and_auroc(per_trial_df, min_n=MIN_TRIALS_PER_GROUP):
    """Per (subject_id, genotype, region) waiting-period summary.

    delta = mean(withhold) - mean(lick); auroc = AUROC of scalar discriminating
    withhold (positive) from lick. Cells with < min_n finite scalars in either
    group are dropped. Returns a per-mouse DataFrame.

    Caller must pass a DataFrame already filtered to a single (track, scheme)
    combination; the groupby keys do not include track/scheme, so mixing them
    would silently pool scalars from different windows.
    """
    if per_trial_df.empty:
        return pd.DataFrame()
    df = per_trial_df[per_trial_df["group"].isin(["lick", "withhold"])].copy()
    df = df[np.isfinite(df["scalar"].astype(float))]
    out = []
    for (subj, geno, region), g in df.groupby(["subject_id", "genotype", "region"]):
        lick = g[g["group"] == "lick"]["scalar"].to_numpy(dtype=float)
        wh = g[g["group"] == "withhold"]["scalar"].to_numpy(dtype=float)
        if lick.size < min_n or wh.size < min_n:
            continue
        scores = np.concatenate([wh, lick])
        labels = np.concatenate([np.ones(wh.size), np.zeros(lick.size)])
        out.append({"subject_id": subj, "genotype": geno, "region": region,
                    "n_lick": int(lick.size), "n_withhold": int(wh.size),
                    "delta": float(np.mean(wh) - np.mean(lick)),
                    "auroc": auroc_score(scores, labels)})
    return pd.DataFrame(out)


def run_suppression_stats(per_mouse_df):
    """(pushpull_df, auroc_df) over per-mouse values, per region.

    pushpull_df: D1-vs-D2 group-level sign contrast on `delta`.
    auroc_df: per genotype x region, bootstrap CI of AUROC (vs chance 0.5) plus
              D1-vs-D2 permutation p.
    """
    if per_mouse_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pp_rows = []
    for region, grp in per_mouse_df.groupby("region"):
        d1 = grp[grp["genotype"] == "D1"]["delta"].to_numpy(dtype=float)
        d2 = grp[grp["genotype"] == "D2"]["delta"].to_numpy(dtype=float)
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "metric": "delta"})
        pp_rows.append(res)

    au_rows = []
    for region, grp in per_mouse_df.groupby("region"):
        d1a = grp[grp["genotype"] == "D1"]["auroc"].to_numpy(dtype=float)
        d1a = d1a[np.isfinite(d1a)]
        d2a = grp[grp["genotype"] == "D2"]["auroc"].to_numpy(dtype=float)
        d2a = d2a[np.isfinite(d2a)]
        perm_p = (permutation_test(d1a, d2a)["p"]
                  if d1a.size >= 2 and d2a.size >= 2 else np.nan)
        for geno, vals in (("D1", d1a), ("D2", d2a)):
            ci = bootstrap_ci(vals)
            au_rows.append({"region": region, "genotype": geno, "n_mice": int(vals.size),
                            "auroc_mean": ci["observed"], "ci_lo": ci["ci_lo"],
                            "ci_hi": ci["ci_hi"],
                            "excludes_chance": bool(np.isfinite(ci["ci_lo"]) and
                                                    (ci["ci_lo"] > 0.5 or ci["ci_hi"] < 0.5)),
                            "perm_p_d1_vs_d2": perm_p})
    return pd.DataFrame(pp_rows), pd.DataFrame(au_rows)
