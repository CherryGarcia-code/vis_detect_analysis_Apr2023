"""Cohort headline figure: D1 pre-FA "go-ramp", HEMISPHERE-RESOLVED.

Cohort = intersectional GCaMP6f in MOs-recipient D1 SPNs (AAV1 anterograde-
transsynaptic Cre in LEFT MOs). IT cortico-striatal neurons project bilaterally;
PT project ipsilaterally. So in striatum:
    G0 = left  = IPSILATERAL  = IT+PT recipients
    G2 = right = CONTRALATERAL = IT-only recipients

The cohort's strongest result: D1 waiting-period activity RAMPS UP before
impulsive FA licks (a pre-FA "go" signal), strongest in D1·DMS. With
withhold=positive labelling, AUROC < 0.5 IS this pre-FA go-ramp (the waiting
signal is higher before licks than before matched withholds).

HEADLINE RESULT: the AUROC go-ramp. D1·DMS (BG_028) waiting AUROC is well
below 0.5 (CIs exclude chance) vs bulk-8m D1·DMS ~0.545 (no ramp). AUROC is
rank-based, so this is robust to fiber strength. This is the cohort's clean,
proposal-relevant finding (MOs->D1 promotes impulsive action).

HEMISPHERE AMPLITUDE IS *NOT* INTERPRETABLE AS BIOLOGY. The col-1 PETH amplitude
looks ipsi(G0)-dominant in BG_028, but the across-event control
(d1_hemisphere_event_control.py) shows BG_028's contra (G2) fiber is uniformly
weak across change_hit/hit_lick/fa_lick (contra/ipsi ratio 0.06-0.21), while the
positive control BG_027 has healthy bilateral fibers (ratio ~0.91 all events).
So BG_028's ipsi>>contra amplitude is a WEAK CONTRA FIBER, not an IT+PT-vs-IT-only
effect. The anatomical labels (ipsi=IT+PT, contra=IT-only) are kept for
reference, but amplitude differences must NOT be read as input-class biology.

n=1 mouse/cell: BG_028 (D1·DMS), BG_027 (D1·VMS). BG_029/030 are D2 and
compromised — NOT included. Session is the within-animal replication unit.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import cohort
from visdetect_photom.core.constants import MIN_TRIALS_PER_GROUP
from visdetect_photom.analysis.suppression import (
    trial_waiting_records, scheme3_scalars,
)
from visdetect_photom.analysis.group_statistics import auroc_score, bootstrap_ci
from visdetect_photom.analysis.statistics import extract_peth

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ──────────────────────────────────────────────
# D1 cells: mouse -> region label. BG_028 records DMS, BG_027 records VMS.
D1_MICE = [("BG_028", "DMS"), ("BG_027", "VMS")]
# Hemisphere ROI -> (label, interpretation). G0=left=ipsi (IT+PT recipients);
# G2=right=contra (IT-only recipients). (Confirmed: G0=*_L, G2=*_R in constants.)
HEMISPHERES = [
    ("G0", "ipsi (IT+PT)", "#d62728"),    # red
    ("G2", "contra (IT-only)", "#1f77b4"),  # blue
]

# FA-lick-aligned PETH (the pre-FA ramp)
PETH_WINDOW = (-2.0, 1.0)
PETH_BASELINE = (-2.0, -1.5)

# Min finite scalars per group to score a session's AUROC
# (mirrors compute_session_delta_and_auroc gate).
MIN_N = MIN_TRIALS_PER_GROUP


def _norm(subject_id):
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def session_hemisphere_auroc(session, roi):
    """Waiting-period AUROC (withhold=positive vs FA-lick) for one ROI in one
    session, via Scheme-3 scalars on that ROI's trace. Returns NaN if the ROI is
    absent or < MIN_N finite scalars in either group.
    """
    trace = session.photometry_data.get(roi)
    if trace is None:
        return np.nan
    records = trial_waiting_records(session, track="behavioral_fa")
    lick = [r for r in records if r["group"] == "lick"]
    withhold = [r for r in records if r["group"] == "withhold"]
    a_vals, w_vals = scheme3_scalars(lick, withhold, trace.signal, trace.timestamps)
    lick_s = np.array([v for _, v in a_vals], dtype=float)
    wh_s = np.array([v for _, v in w_vals], dtype=float)
    lick_s = lick_s[np.isfinite(lick_s)]
    wh_s = wh_s[np.isfinite(wh_s)]
    if lick_s.size < MIN_N or wh_s.size < MIN_N:
        return np.nan
    scores = np.concatenate([wh_s, lick_s])
    labels = np.concatenate([np.ones(wh_s.size), np.zeros(lick_s.size)])
    return auroc_score(scores, labels)


def session_hemisphere_fa_peth(session, roi):
    """FA-lick-aligned mean PETH trace (one row = session mean) for one ROI.

    Returns (time_axis, mean_trace) or (None, None) if no usable FA events / ROI.
    """
    trace = session.photometry_data.get(roi)
    if trace is None:
        return None, None
    event_times = [t.absolute_reaction_time for t in session.trials
                   if t.outcome == "FA" and t.absolute_reaction_time is not None
                   and np.isfinite(t.absolute_reaction_time)]
    if not event_times:
        return None, None
    time_axis, peth = extract_peth(
        trace.signal, trace.timestamps, np.asarray(event_times, dtype=float),
        window=PETH_WINDOW, baseline_window=PETH_BASELINE, normalize="subtract")
    if peth.size == 0:
        return None, None
    mean_trace = np.nanmean(peth, axis=0)
    if np.all(np.isnan(mean_trace)):
        return None, None
    return time_axis, mean_trace


def collect_mouse_hemisphere(sessions, roi):
    """Across a mouse's sessions, collect per-session AUROCs and per-session mean
    FA-aligned PETH traces for one ROI.

    Returns dict: aurocs (list), n_sessions_auroc, time_axis, mean_trace (per-mouse
    average of per-session means), n_sessions_peth.
    """
    aurocs = []
    peth_traces = []
    time_axis = None
    for sess in sessions:
        au = session_hemisphere_auroc(sess, roi)
        if np.isfinite(au):
            aurocs.append(au)
        ta, tr = session_hemisphere_fa_peth(sess, roi)
        if tr is not None:
            peth_traces.append(tr)
            if time_axis is None:
                time_axis = ta

    mean_trace = None
    if peth_traces:
        n = min(len(t) for t in peth_traces)
        stacked = np.vstack([t[:n] for t in peth_traces])
        mean_trace = np.nanmean(stacked, axis=0)
        if time_axis is not None:
            time_axis = time_axis[:n]

    ci = bootstrap_ci(np.asarray(aurocs, dtype=float)) if aurocs else {
        "observed": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    return {
        "aurocs": aurocs,
        "n_sessions_auroc": len(aurocs),
        "auroc_mean": ci["observed"],
        "auroc_ci_lo": ci["ci_lo"],
        "auroc_ci_hi": ci["ci_hi"],
        "time_axis": time_axis,
        "mean_trace": mean_trace,
        "n_sessions_peth": len(peth_traces),
    }


def load_bulk_reference(bulk_csv):
    """{region: auroc_mean} for bulk-8m D1, track=behavioral_fa, scheme=scheme3,
    prof_bin=pooled, region in {DMS, VMS}. Rank-based reference only."""
    ref = {}
    if not bulk_csv or not Path(bulk_csv).exists():
        logging.warning("Bulk C1 CSV not found at %s — reference lines skipped.", bulk_csv)
        return ref
    df = pd.read_csv(bulk_csv)
    sub = df[(df.get("track") == "behavioral_fa")
             & (df.get("scheme") == "scheme3")
             & (df.get("prof_bin") == "pooled")
             & (df.get("genotype") == "D1")
             & (df.get("region").isin(["DMS", "VMS"]))]
    for _, r in sub.iterrows():
        ref[r["region"]] = float(r["auroc_mean"])
    return ref


def build_figure(results, bulk_ref, out_dir):
    """2 rows (D1·DMS=BG_028, D1·VMS=BG_027) x 2 cols (PETH, AUROC bars)."""
    n_rows = len(D1_MICE)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.6 * n_rows), squeeze=False)

    fig.suptitle(
        "D1 MOs-recipient SPNs — pre-FA GO-RAMP (headline = AUROC, col 2)\n"
        "D1·DMS waiting AUROC <0.5 (pre-FA go-ramp) vs bulk-8m (no ramp); AUROC rank-based → robust to fiber strength\n"
        "⚠ col-1 AMPLITUDE is fiber-confounded — BG_028 contra (G2) is a WEAK FIBER (see d1_hemisphere_event_control),\n"
        "    NOT IT+PT-vs-IT-only biology. n=1 mouse/cell; bulk = rank-based ref (6f vs 8m, magnitudes NOT compared)",
        fontsize=10)

    for ri, (mouse, region) in enumerate(D1_MICE):
        res = results[mouse]

        # ── Col 1: FA-lick-aligned mean PETH (the pre-FA ramp) ──
        ax = axes[ri][0]
        for roi, label, color in HEMISPHERES:
            r = res[roi]
            if r["mean_trace"] is not None and r["time_axis"] is not None:
                ax.plot(r["time_axis"], r["mean_trace"], color=color, lw=2,
                        label=f"{roi} {label} (n={r['n_sessions_peth']} sess)")
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.axhline(0.0, color="0.6", ls=":", lw=0.6)
        ax.set_xlabel("Time from FA lick (s)")
        ax.set_ylabel(r"$\Delta$ z-dF/F (baseline-subtracted)")
        ax.set_title(f"{mouse} D1·{region} — pre-FA ramp by hemisphere\n"
                     "(amplitude fiber-confounded; cf. event-control)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

        # ── Col 2: waiting-period AUROC bars (withhold vs FA-lick) ──
        ax = axes[ri][1]
        xs, heights, colors, labels = [], [], [], []
        yerr_lo, yerr_hi = [], []
        for i, (roi, label, color) in enumerate(HEMISPHERES):
            r = res[roi]
            xs.append(i)
            heights.append(r["auroc_mean"])
            colors.append(color)
            labels.append(f"{roi}\n{label}\n(n={r['n_sessions_auroc']})")
            m = r["auroc_mean"]
            lo = m - r["auroc_ci_lo"] if np.isfinite(r["auroc_ci_lo"]) else 0.0
            hi = r["auroc_ci_hi"] - m if np.isfinite(r["auroc_ci_hi"]) else 0.0
            yerr_lo.append(max(lo, 0.0))
            yerr_hi.append(max(hi, 0.0))
        ax.bar(xs, heights, color=colors,
               yerr=[yerr_lo, yerr_hi], capsize=5, width=0.6)
        ax.axhline(0.5, color="k", ls="--", lw=0.9, label="chance (0.5)")
        # Bulk rank-based reference line for this region
        if region in bulk_ref:
            ax.axhline(bulk_ref[region], color="0.4", ls="-.", lw=1.4,
                       label=f"bulk-8m D1·{region} (rank ref)")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("AUROC (withhold vs FA)")
        ax.set_title(f"{mouse} D1·{region} — waiting-period AUROC\n"
                     "<0.5 = pre-FA go-ramp", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d1_hemisphere_story.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Hemisphere-resolved D1 pre-FA go-ramp figure "
                    "(ipsi IT+PT vs contra IT-only).")
    rr = _repo_root
    ap.add_argument("--root_dir",
                    default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--bulk_c1",
                    default=str(rr / "FIGURES" / "C1_fa_suppression_corrected"
                                / "c1_auroc_stats.csv"))
    ap.add_argument("--output_dir",
                    default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)

    # Load all cohort sessions, then group by mouse (D1 only).
    sessions = cohort.load_cohort_sessions(
        "intersectional_mos", args.root_dir, max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded from %s", args.root_dir)
        sys.exit(1)
    logging.info("Loaded %d cohort sessions.", len(sessions))

    by_mouse = {}
    for sess in sessions:
        by_mouse.setdefault(_norm(sess.subject_id), []).append(sess)

    bulk_ref = load_bulk_reference(args.bulk_c1)
    logging.info("Bulk-8m D1 reference AUROCs: %s", bulk_ref)

    results = {}
    summary_rows = []
    for mouse, region in D1_MICE:
        mouse_sessions = by_mouse.get(mouse, [])
        # Only sessions that HAVE both G0 and G2.
        usable = [s for s in mouse_sessions
                  if "G0" in s.photometry_data and "G2" in s.photometry_data]
        logging.info("%s (D1·%s): %d sessions, %d with both G0+G2.",
                     mouse, region, len(mouse_sessions), len(usable))
        res = {}
        for roi, label, _ in HEMISPHERES:
            r = collect_mouse_hemisphere(usable, roi)
            res[roi] = r
            summary_rows.append({
                "subject_id": mouse, "region": region, "roi": roi,
                "hemisphere": label, "n_sessions_auroc": r["n_sessions_auroc"],
                "n_sessions_peth": r["n_sessions_peth"],
                "auroc_mean": r["auroc_mean"], "auroc_ci_lo": r["auroc_ci_lo"],
                "auroc_ci_hi": r["auroc_ci_hi"],
            })
            logging.info(
                "  %s %s: AUROC=%.3f [%.3f, %.3f] (n=%d sess); PETH n=%d",
                roi, label, r["auroc_mean"], r["auroc_ci_lo"],
                r["auroc_ci_hi"], r["n_sessions_auroc"], r["n_sessions_peth"])
        results[mouse] = res

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "d1_hemisphere_story_auroc.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info("Saved %s", summary_csv)

    build_figure(results, bulk_ref, out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
