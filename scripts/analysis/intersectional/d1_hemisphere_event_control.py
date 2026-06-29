"""Per-hemisphere ACROSS-EVENT control: technical weak-fiber vs IT+PT/IT-only biology.

Cohort = intersectional GCaMP6f in MOs-recipient D1 SPNs (AAV1 anterograde-
transsynaptic Cre in LEFT MOs). IT cortico-striatal neurons project bilaterally;
PT project ipsilaterally. So in striatum:
    G0 = left  = IPSILATERAL  = IT+PT recipients
    G2 = right = CONTRALATERAL = IT-only recipients   (confirmed; left-MOs AAV1)

THE PROBLEM THIS SCRIPT SOLVES
------------------------------
BG_028's D1·DMS shows a big pre-FA "go-ramp" in IPSI (G0) but a FLAT signal in
CONTRA (G2). Two competing explanations:
    (1) TECHNICAL: the contra fiber is weak/poorly coupled, so G2 is flat for
        EVERY event (low amplitude across change_hit, hit_lick AND fa_lick).
    (2) BIOLOGICAL: the contra (IT-only) recipients genuinely lack the pre-FA go
        signal, but respond NORMALLY to sensory/motor events — i.e. flatness is
        SPECIFIC to fa_lick.

DIAGNOSTIC LOGIC
----------------
Compare the contra-vs-ipsi response AMPLITUDE ACROSS MULTIPLE events:
    - If contra/ipsi amplitude is UNIFORMLY LOW across change_hit, hit_lick AND
      fa_lick  ->  TECHNICAL (weak fiber).
    - If contra responds NORMALLY to change_hit/hit_lick (ratio ~1) but is
      SPECIFICALLY LOW for fa_lick  ->  BIOLOGY (IT-only lacks the pre-FA go
      signal).
BG_027 (whose contra looked healthy) is the POSITIVE CONTROL: we expect
contra/ipsi ratios ~1 across all events.

Method mirrors d1_hemisphere_story.py: load cohort D1 mice, bypass QC/region
merge, read session.photometry_data['G0'|'G2'] (session-z-scored signal +
timestamps) directly per hemisphere, use only sessions that have BOTH ROIs.

n=1 mouse/cell. Within-mouse, within-indicator amplitude (session-z); magnitudes
NOT compared to bulk-8m. Session is the within-animal replication unit.
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
from visdetect_photom.analysis.statistics import extract_peth

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ──────────────────────────────────────────────
# D1 cells: mouse -> region label. BG_028 records DMS, BG_027 records VMS.
D1_MICE = [("BG_028", "DMS"), ("BG_027", "VMS")]

# Hemisphere ROI -> (label, color). G0=left=ipsi (IT+PT recipients);
# G2=right=contra (IT-only recipients). (Confirmed: G0=*_L, G2=*_R in constants.)
HEMISPHERES = [
    ("G0", "ipsi (IT+PT)", "#d62728"),      # red
    ("G2", "contra (IT-only)", "#1f77b4"),  # blue
]

# Events: name -> (alignment attr, required outcome, window, baseline, xlabel).
# change_hit aligns to the change presentation (Hit only -> change WAS shown).
# hit_lick / fa_lick are motor-aligned to the lick (reaction time).
EVENTS = [
    ("change_hit", "absolute_change_time", "Hit", (-2.0, 2.0), (-2.0, -1.5),
     "Time from change (s)"),
    ("hit_lick", "absolute_reaction_time", "Hit", (-2.0, 1.5), (-2.0, -1.5),
     "Time from Hit lick (s)"),
    ("fa_lick", "absolute_reaction_time", "FA", (-2.0, 1.0), (-2.0, -1.5),
     "Time from FA lick (s)"),
]

# Amplitude metric: peak of |mean_trace| within this POST window (comparable
# across events of different total length).
POST_WINDOW = (0.0, 1.0)


def _norm(subject_id):
    s = str(subject_id)
    if not s.startswith("BG_") and s.isdigit():
        return f"BG_{s.zfill(3)}"
    return s


def session_event_peth(session, roi, align_attr, outcome, window, baseline):
    """Mean PETH trace (session mean) for one ROI, one event, in one session.

    Returns (time_axis, mean_trace) or (None, None) if no usable events / ROI.
    """
    trace = session.photometry_data.get(roi)
    if trace is None:
        return None, None
    event_times = []
    for t in session.trials:
        if t.outcome != outcome:
            continue
        et = getattr(t, align_attr, None)
        if et is not None and np.isfinite(et):
            event_times.append(et)
    if not event_times:
        return None, None
    time_axis, peth = extract_peth(
        trace.signal, trace.timestamps, np.asarray(event_times, dtype=float),
        window=window, baseline_window=baseline, normalize="subtract")
    if peth.size == 0:
        return None, None
    mean_trace = np.nanmean(peth, axis=0)
    if np.all(np.isnan(mean_trace)):
        return None, None
    return time_axis, mean_trace


def collect_mouse_hemisphere_event(sessions, roi, align_attr, outcome,
                                   window, baseline):
    """Per-mouse mean trace (mean of per-session means) for one ROI + event.

    Returns dict: time_axis, mean_trace, n_sessions.
    """
    peth_traces = []
    time_axis = None
    for sess in sessions:
        ta, tr = session_event_peth(sess, roi, align_attr, outcome,
                                    window, baseline)
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

    return {
        "time_axis": time_axis,
        "mean_trace": mean_trace,
        "n_sessions": len(peth_traces),
    }


def amplitude_metrics(time_axis, mean_trace):
    """Peak of |mean_trace| within POST_WINDOW; return (abs_peak, signed_value).

    Returns (nan, nan) if no usable trace / no samples in the post window.
    """
    if time_axis is None or mean_trace is None:
        return np.nan, np.nan
    post_mask = ((time_axis >= POST_WINDOW[0]) & (time_axis <= POST_WINDOW[1])
                 & np.isfinite(mean_trace))
    if not np.any(post_mask):
        return np.nan, np.nan
    seg = mean_trace[post_mask]
    idx = int(np.argmax(np.abs(seg)))
    signed = float(seg[idx])
    return abs(signed), signed


def _ratio(contra, ipsi):
    """contra/ipsi amplitude ratio with divide-by-zero guard."""
    if not np.isfinite(contra) or not np.isfinite(ipsi):
        return np.nan
    if abs(ipsi) < 1e-9:
        return np.nan
    return contra / ipsi


def build_figure(results, ratios, out_dir):
    """Grid: 2 mice (rows) x 3 events (cols) PETH overlays, plus a summary
    ratio panel spanning the bottom."""
    n_rows = len(D1_MICE)
    n_cols = len(EVENTS)
    # Extra row at bottom for the contra/ipsi ratio summary (spans full width).
    fig = plt.figure(figsize=(4.6 * n_cols, 4.2 * n_rows + 3.0))
    gs = fig.add_gridspec(n_rows + 1, n_cols,
                          height_ratios=[1.0] * n_rows + [0.9])

    fig.suptitle(
        "D1 MOs-recipient SPNs — per-hemisphere ACROSS-EVENT control\n"
        "TEST: is contra (G2, IT-only) flat EVERYWHERE (technical/weak fiber) "
        "or only for fa_lick (biology: IT+PT vs IT-only)?\n"
        "G0=ipsi(IT+PT) vs G2=contra(IT-only). n=1 mouse/cell; within-mouse "
        "within-indicator amplitude (session-z); magnitudes NOT compared to bulk.",
        fontsize=11)

    for ri, (mouse, region) in enumerate(D1_MICE):
        for ci, (event, _attr, _oc, _win, _bl, xlabel) in enumerate(EVENTS):
            ax = fig.add_subplot(gs[ri, ci])
            for roi, label, color in HEMISPHERES:
                r = results[mouse][event][roi]
                if r["mean_trace"] is not None and r["time_axis"] is not None:
                    ax.plot(r["time_axis"], r["mean_trace"], color=color, lw=2,
                            label=f"{roi} {label} (n={r['n_sessions']} sess)")
            ax.axvline(0.0, color="k", ls="--", lw=0.8)
            ax.axhline(0.0, color="0.6", ls=":", lw=0.6)
            ax.axvspan(POST_WINDOW[0], POST_WINDOW[1], color="0.85", alpha=0.3,
                       zorder=0)
            ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel(r"$\Delta$ z-dF/F (baseline-subtracted)")
            ax.set_title(f"{mouse} D1·{region} — {event}", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")

    # ── Summary: contra/ipsi amplitude ratio per (mouse, event) ──
    ax = fig.add_subplot(gs[n_rows, :])
    event_names = [e[0] for e in EVENTS]
    x = np.arange(len(event_names))
    width = 0.36
    mouse_colors = {"BG_028": "#9467bd", "BG_027": "#2ca02c"}
    for mi, (mouse, region) in enumerate(D1_MICE):
        vals = [ratios[mouse][ev] for ev in event_names]
        offset = (mi - (n_rows - 1) / 2.0) * width
        bars = ax.bar(x + offset, [v if np.isfinite(v) else 0.0 for v in vals],
                      width=width, color=mouse_colors.get(mouse, "0.5"),
                      edgecolor="k",
                      label=f"{mouse} D1·{region}"
                            + (" [target]" if mouse == "BG_028" else ""))
        for b, v in zip(bars, vals):
            txt = f"{v:.2f}" if np.isfinite(v) else "n/a"
            ax.text(b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.02, txt, ha="center", va="bottom",
                    fontsize=8,
                    fontweight="bold" if mouse == "BG_028" else "normal")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="ratio = 1 (contra==ipsi)")
    ax.set_xticks(x)
    ax.set_xticklabels(event_names)
    ax.set_ylabel("contra/ipsi amplitude ratio\n(G2 / G0, peak |mean| in 0..1 s)")
    ax.set_title(
        "Contra/ipsi amplitude ratio per event.  "
        "BG_028: ratio~1 for change_hit/hit_lick but LOW for fa_lick => BIOLOGY; "
        "uniformly LOW => TECHNICAL.  BG_027 = positive control (expect ~1).",
        fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "d1_hemisphere_event_control.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Per-hemisphere across-event control for D1 MOs-recipient "
                    "SPNs (technical weak-fiber vs IT+PT/IT-only biology).")
    rr = _repo_root
    ap.add_argument("--root_dir",
                    default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir",
                    default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)

    sessions = cohort.load_cohort_sessions(
        "intersectional_mos", args.root_dir, max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded from %s", args.root_dir)
        sys.exit(1)
    logging.info("Loaded %d cohort sessions.", len(sessions))

    by_mouse = {}
    for sess in sessions:
        by_mouse.setdefault(_norm(sess.subject_id), []).append(sess)

    results = {}        # mouse -> event -> roi -> collect dict
    ratios = {}         # mouse -> event -> contra/ipsi ratio
    amp_rows = []       # per (mouse, hemisphere, event)
    ratio_rows = []     # per (mouse, event)

    for mouse, region in D1_MICE:
        mouse_sessions = by_mouse.get(mouse, [])
        # Only sessions that HAVE both G0 and G2.
        usable = [s for s in mouse_sessions
                  if "G0" in s.photometry_data and "G2" in s.photometry_data]
        logging.info("%s (D1·%s): %d sessions, %d with both G0+G2.",
                     mouse, region, len(mouse_sessions), len(usable))

        results[mouse] = {}
        ratios[mouse] = {}
        for event, attr, outcome, window, baseline, _xl in EVENTS:
            results[mouse][event] = {}
            amps = {}
            for roi, label, _color in HEMISPHERES:
                r = collect_mouse_hemisphere_event(
                    usable, roi, attr, outcome, window, baseline)
                results[mouse][event][roi] = r
                abs_peak, signed = amplitude_metrics(r["time_axis"],
                                                     r["mean_trace"])
                amps[roi] = abs_peak
                amp_rows.append({
                    "subject_id": mouse, "region": region, "event": event,
                    "roi": roi, "hemisphere": label,
                    "n_sessions": r["n_sessions"],
                    "abs_peak_amp": abs_peak, "signed_peak_amp": signed,
                })
                logging.info(
                    "  %s %-7s %s %-14s: |peak|=%.4f signed=%+.4f (n=%d sess)",
                    mouse, event, roi, label, abs_peak, signed, r["n_sessions"])
            ratio = _ratio(amps.get("G2", np.nan), amps.get("G0", np.nan))
            ratios[mouse][event] = ratio
            ratio_rows.append({
                "subject_id": mouse, "region": region, "event": event,
                "ipsi_G0_abs_peak": amps.get("G0", np.nan),
                "contra_G2_abs_peak": amps.get("G2", np.nan),
                "contra_over_ipsi_ratio": ratio,
            })
            logging.info("  %s %-9s contra/ipsi ratio = %s", mouse, event,
                         f"{ratio:.3f}" if np.isfinite(ratio) else "n/a")

    out_dir.mkdir(parents=True, exist_ok=True)
    amp_df = pd.DataFrame(amp_rows)
    ratio_df = pd.DataFrame(ratio_rows)
    amp_csv = out_dir / "d1_hemisphere_event_control_amplitudes.csv"
    ratio_csv = out_dir / "d1_hemisphere_event_control_ratios.csv"
    amp_df.to_csv(amp_csv, index=False)
    ratio_df.to_csv(ratio_csv, index=False)
    logging.info("Saved %s", amp_csv)
    logging.info("Saved %s", ratio_csv)

    build_figure(results, ratios, out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
