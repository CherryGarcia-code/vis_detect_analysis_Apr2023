"""C2 — D1/D2 response-geometry computation core.

Per-session, mode-aware metrics across change / lick / anticipation epochs, plus
change-size grading. Thin script 08 consumes this. See spec
docs/superpowers/specs/2026-06-08-c2-d1-d2-geometry-design.md.
"""
from collections import defaultdict
import numpy as np

from visdetect_photom.core.constants import CHANGE_SIZES, CATCH_THRESHOLD, get_roi_region
from visdetect_photom.core.qc import compute_session_roi_qc, merge_hemispheres
from visdetect_photom.analysis.statistics import extract_peth
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.analysis.group_statistics import (
    extract_activation, extract_suppression, extract_signed_peak,
    extract_signed_auc, extract_ramp_slope,
    extract_peak_latency, extract_onset_latency,
)
from visdetect_photom.analysis.state_provider import filter_trials_by_state

PETH_WINDOW = (-2.0, 4.0)
POST_WINDOW = (0.0, 1.5)
ONSET_WINDOW = (0.0, 2.0)
ANTICIP_WINDOW = (-1.5, 0.0)
ANTICIP_BASELINE = (-2.0, -1.5)
CHANGE_BASELINE = (-2.0, 0.0)

# epoch -> (event_type, kind, baseline_window)
EPOCHS = {
    "change_hit":        ("change_hit", "change",       CHANGE_BASELINE),
    "change_miss":       ("change_miss", "change",      CHANGE_BASELINE),
    "hit_lick":          ("hit_lick", "lick",           CHANGE_BASELINE),
    "fa_lick":           ("fa_lick", "lick",            CHANGE_BASELINE),
    "anticipation_hit":  ("change_hit", "anticipation", ANTICIP_BASELINE),
    "anticipation_miss": ("change_miss", "anticipation", ANTICIP_BASELINE),
    "anticipation_cr":   ("change_cr", "anticipation",  ANTICIP_BASELINE),
}

_METRIC_KEYS = ["activation", "suppression", "signed_peak", "signed_auc",
                "ramp_slope", "peak_latency", "onset_latency"]


def _subject_full(subject_id):
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def _event_times(session, event_type, keep):
    """Absolute event times for an epoch's event_type, restricted to keep (set|None)."""
    times = []
    for t in session.trials:
        if keep is not None and t.trial_index not in keep:
            continue
        o = t.outcome
        if event_type == "change_hit" and o == "Hit" and t.absolute_change_time is not None:
            times.append(t.absolute_change_time)
        elif event_type == "change_miss" and o == "Miss" and t.absolute_change_time is not None:
            times.append(t.absolute_change_time)
        elif event_type == "change_cr" and o == "CR" and t.absolute_change_time is not None:
            times.append(t.absolute_change_time)
        elif event_type == "hit_lick" and o == "Hit" and t.absolute_reaction_time is not None:
            times.append(t.absolute_reaction_time)
        elif event_type == "fa_lick" and o == "FA" and t.absolute_reaction_time is not None:
            times.append(t.absolute_reaction_time)
    return np.array(times, dtype=float)


def _change_hit_times_for_cs(session, cs, keep):
    """Hit-trial change times whose change_size rounds to canonical cs."""
    times = []
    for t in session.trials:
        if keep is not None and t.trial_index not in keep:
            continue
        if t.outcome != "Hit" or t.absolute_change_time is None:
            continue
        if t.change_size is None or t.change_size <= CATCH_THRESHOLD:
            continue
        if min(CHANGE_SIZES, key=lambda x: abs(x - t.change_size)) == cs:
            times.append(t.absolute_change_time)
    return np.array(times, dtype=float)


def _mean_trace(peth_matrix):
    """Per-mouse mean over trials with >50% finite bins; None if none qualify."""
    if peth_matrix.shape[0] == 0:
        return None
    valid = [row for row in peth_matrix
             if np.sum(np.isfinite(row)) > 0.5 * row.shape[0]]
    if not valid:
        return None
    return np.nanmean(np.array(valid), axis=0)


def _metrics_for_trace(trace, t, kind):
    out = {k: np.nan for k in _METRIC_KEYS}
    if kind in ("change", "lick"):
        out["activation"] = extract_activation(trace, t, POST_WINDOW)
        out["suppression"] = extract_suppression(trace, t, POST_WINDOW)
        out["signed_peak"] = extract_signed_peak(trace, t, POST_WINDOW)
        out["signed_auc"] = extract_signed_auc(trace, t, POST_WINDOW)
        out["peak_latency"] = extract_peak_latency(trace, t, peak_window=POST_WINDOW)
        out["onset_latency"] = extract_onset_latency(
            trace, t, threshold_n_std=2.0,
            baseline_window=CHANGE_BASELINE, search_window=ONSET_WINDOW, n_consecutive=3)
    elif kind == "anticipation":
        out["ramp_slope"] = extract_ramp_slope(trace, t, ANTICIP_WINDOW)
        out["signed_auc"] = extract_signed_auc(trace, t, ANTICIP_WINDOW)
    return out


def _region_sources(session, subject_full, use_qc):
    """Return {region_base: (signal, timestamps)} via QC-merge or no-QC averaging."""
    if use_qc:
        qc = compute_session_roi_qc(session)
        merged = merge_hemispheres(session, qc_results=qc)
        return {r: (m["signal"], m["timestamps"]) for r, m in merged.items()}
    by_region = defaultdict(list)
    for roi_name, trace in session.photometry_data.items():
        region = get_roi_region(roi_name, subject_full)
        if region is None:
            continue
        by_region[region.rsplit("_", 1)[0]].append((trace.signal, trace.timestamps))
    sources = {}
    for region, traces in by_region.items():
        if len(traces) == 1:
            sources[region] = traces[0]
        elif len(traces) >= 2:
            n = min(len(s) for s, _ in traces)
            avg = np.mean([s[:n] for s, _ in traces], axis=0)
            sources[region] = (avg, traces[0][1][:n])
    return sources


def compute_geometry_metrics_for_session(session, *, use_qc=True,
                                         state_provider=None, keep_states=None):
    """Return (rows, traces, time_axis) for one session.

    rows: list of dicts (subject_id, genotype, region, epoch, change_size, n_trials,
          + the 7 metric keys). change_size is NaN except for graded change_hit rows
          (epoch == 'change_hit_graded').
    traces: {(region, epoch): mean_trace} for pooled (non-graded) epochs only.
    time_axis: shared 1-D axis (or None if nothing extracted).
    """
    subject_full = _subject_full(session.subject_id)
    genotype = get_genotype(subject_full)
    if genotype == "Unknown":
        return [], {}, None

    keep = None
    if state_provider is not None and keep_states is not None:
        keep = filter_trials_by_state(session, state_provider, keep_states)

    sources = _region_sources(session, subject_full, use_qc)
    rows, traces, time_axis = [], {}, None

    for region, (sig, ts) in sources.items():
        for epoch, (evt, kind, bl) in EPOCHS.items():
            et = _event_times(session, evt, keep)
            if et.size == 0:
                continue
            t_ax, peth = extract_peth(sig, ts, et, window=PETH_WINDOW, baseline_window=bl)
            if time_axis is None:
                time_axis = t_ax
            mt = _mean_trace(peth)
            if mt is None:
                continue
            traces[(region, epoch)] = mt
            rows.append({"subject_id": subject_full, "genotype": genotype,
                         "region": region, "epoch": epoch, "change_size": np.nan,
                         "n_trials": int(peth.shape[0]),
                         **_metrics_for_trace(mt, t_ax, kind)})
        # change-size grading (Hit go-trials)
        for cs in CHANGE_SIZES:
            et = _change_hit_times_for_cs(session, cs, keep)
            if et.size == 0:
                continue
            t_ax, peth = extract_peth(sig, ts, et, window=PETH_WINDOW,
                                      baseline_window=CHANGE_BASELINE)
            mt = _mean_trace(peth)
            if mt is None:
                continue
            rows.append({"subject_id": subject_full, "genotype": genotype,
                         "region": region, "epoch": "change_hit_graded",
                         "change_size": float(cs), "n_trials": int(peth.shape[0]),
                         **_metrics_for_trace(mt, t_ax, "change")})

    return rows, traces, time_axis


import pandas as pd
from visdetect_photom.analysis.group_statistics import (
    pushpull_sign_contrast, spearman_with_ci,
)


def build_geometry_dataset(sessions, *, use_qc=True, state_provider=None,
                           keep_states=None):
    """Aggregate per-session rows to a PER-MOUSE metrics DataFrame + grouped traces.

    Returns (per_mouse_df, traces_by_group, time_axis):
      per_mouse_df: one row per (subject_id, genotype, region, epoch, change_size),
                    metric columns averaged across that mouse's sessions.
      traces_by_group: {(genotype, region, epoch): [(subject_id, mean_trace), ...]}
                       (pooled epochs only; per-mouse mean traces).
      time_axis: shared 1-D axis.
    """
    all_rows = []
    trace_accum = defaultdict(lambda: defaultdict(list))  # (geno,region,epoch)->subj->[traces]
    time_axis = None

    for sess in sessions:
        rows, traces, t = compute_geometry_metrics_for_session(
            sess, use_qc=use_qc, state_provider=state_provider, keep_states=keep_states)
        if t is not None and time_axis is None:
            time_axis = t
        all_rows.extend(rows)
        if rows:
            geno = rows[0]["genotype"]
            subj = rows[0]["subject_id"]
            for (region, epoch), tr in traces.items():
                trace_accum[(geno, region, epoch)][subj].append(tr)

    if not all_rows:
        return pd.DataFrame(), {}, time_axis

    raw = pd.DataFrame(all_rows)
    group_cols = ["subject_id", "genotype", "region", "epoch", "change_size"]
    per_mouse = (raw.groupby(group_cols, dropna=False)[_METRIC_KEYS]
                    .mean().reset_index())

    traces_by_group = {}
    for key, subj_map in trace_accum.items():
        traces_by_group[key] = [
            (subj, np.nanmean(np.array(trs), axis=0)) for subj, trs in subj_map.items()
        ]
    return per_mouse, traces_by_group, time_axis


def run_pushpull_tests(per_mouse_df, metric="signed_auc"):
    """Per (region, epoch) D1-vs-D2 push–pull sign contrast over per-mouse values.

    Pooled epochs only (change_size is NaN). Returns a tidy DataFrame.
    """
    if per_mouse_df.empty:
        return pd.DataFrame()
    pooled = per_mouse_df[per_mouse_df["change_size"].isna()]
    out = []
    for (region, epoch), grp in pooled.groupby(["region", "epoch"]):
        d1 = grp[grp["genotype"] == "D1"][metric].values
        d2 = grp[grp["genotype"] == "D2"][metric].values
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "epoch": epoch, "metric": metric})
        out.append(res)
    return pd.DataFrame(out)


def run_grading(per_mouse_df, metric="signed_auc"):
    """Spearman(metric, log2(change_size)) per genotype x region over graded rows."""
    if per_mouse_df.empty:
        return pd.DataFrame()
    graded = per_mouse_df[per_mouse_df["epoch"] == "change_hit_graded"].dropna(
        subset=["change_size", metric])
    out = []
    for (geno, region), grp in graded.groupby(["genotype", "region"]):
        if grp["change_size"].nunique() < 3:
            continue
        x = np.log2(grp["change_size"].values.astype(float))
        y = grp[metric].values.astype(float)
        res = spearman_with_ci(x, y)
        res.update({"genotype": geno, "region": region, "metric": metric})
        out.append(res)
    return pd.DataFrame(out)
