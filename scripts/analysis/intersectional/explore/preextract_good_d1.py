"""Pre-extract the GOOD-quality D1 cohort signal once into a compact per-trial
pkl, so exploration agents can analyze without re-loading ~115 sessions each.

Good fibers (per across-event control): BG_027 G0+G2 (both healthy), BG_028 G0
(ipsi, healthy); BG_028 G2 tagged 'weak' (contra weak fiber). D2 mice excluded.

Output pkl: dict with 'change' and 'lick' each = {meta: DataFrame, traces:
float32[N,T], time: array}. One row per (trial, ROI).
  change-aligned: Hit/Miss trials (change presented), aligned to absolute_change_time
  lick-aligned:   Hit/FA trials, aligned to absolute_reaction_time
meta cols: subject, roi, hemisphere, fiber_quality, session_id, recording_id,
           trial_index, outcome, change_size, reaction_time
"""
import os, sys, pickle
import numpy as np
import pandas as pd

WT = r"e:/python_analysis/git_repos/_wt_intersectional_mos"
MAIN = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023"
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect_photom.core import cohort
from visdetect_photom.analysis.statistics import extract_peth

DATA = os.path.join(MAIN, "photom_data", "intrsct_GCaMP6f")
OUT_PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/good_d1_extract.pkl"

GOOD_MICE = {"BG_028": "DMS", "BG_027": "VMS"}   # D1 mice with real signal
WEAK = {("BG_028", "G2")}                          # contra weak fiber
ROIS = ["G0", "G2"]
HEMI = {"G0": "ipsi(IT+PT)", "G2": "contra(IT-only)"}

CHANGE_WIN, CHANGE_BL = (-2.0, 4.0), (-2.0, 0.0)
LICK_WIN, LICK_BL = (-2.0, 3.0), (-2.0, -1.5)

def _norm(s):
    s = str(s)
    return s if s.startswith("BG_") else f"BG_{s.zfill(3)}"

def main():
    sessions = cohort.load_cohort_sessions("intersectional_mos", DATA, max_sessions=None)
    sessions = [s for s in sessions if _norm(s.subject_id) in GOOD_MICE]
    print(f"Loaded {len(sessions)} good-D1 sessions")

    change_meta, change_tr, change_time = [], [], None
    lick_meta, lick_tr, lick_time = [], [], None

    for s in sessions:
        subj = _norm(s.subject_id)
        for roi in ROIS:
            tr = s.photometry_data.get(roi)
            if tr is None:
                continue
            fq = "weak" if (subj, roi) in WEAK else "good"
            sig, ts = tr.signal, tr.timestamps

            # change-aligned: Hit/Miss
            ch_trials = [(i, t) for i, t in enumerate(s.trials)
                         if t.outcome in ("Hit", "Miss")
                         and t.absolute_change_time is not None
                         and np.isfinite(t.absolute_change_time)]
            if ch_trials:
                ev = np.array([t.absolute_change_time for _, t in ch_trials], float)
                tax, peth = extract_peth(sig, ts, ev, window=CHANGE_WIN,
                                         baseline_window=CHANGE_BL, normalize="subtract")
                if peth.size:
                    if change_time is None: change_time = tax
                    for k, (idx, t) in enumerate(ch_trials):
                        change_meta.append(dict(subject=subj, roi=roi, hemisphere=HEMI[roi],
                            fiber_quality=fq, session_id=s.session_id,
                            recording_id=getattr(s, "recording_id", s.session_id),
                            trial_index=idx, outcome=t.outcome,
                            change_size=t.change_size, reaction_time=t.reaction_time))
                        change_tr.append(peth[k].astype(np.float32))

            # lick-aligned: Hit/FA
            lk_trials = [(i, t) for i, t in enumerate(s.trials)
                         if t.outcome in ("Hit", "FA")
                         and t.absolute_reaction_time is not None
                         and np.isfinite(t.absolute_reaction_time)]
            if lk_trials:
                ev = np.array([t.absolute_reaction_time for _, t in lk_trials], float)
                tax, peth = extract_peth(sig, ts, ev, window=LICK_WIN,
                                         baseline_window=LICK_BL, normalize="subtract")
                if peth.size:
                    if lick_time is None: lick_time = tax
                    for k, (idx, t) in enumerate(lk_trials):
                        lick_meta.append(dict(subject=subj, roi=roi, hemisphere=HEMI[roi],
                            fiber_quality=fq, session_id=s.session_id,
                            recording_id=getattr(s, "recording_id", s.session_id),
                            trial_index=idx, outcome=t.outcome,
                            change_size=t.change_size, reaction_time=t.reaction_time))
                        lick_tr.append(peth[k].astype(np.float32))

    out = {
        "change": {"meta": pd.DataFrame(change_meta),
                   "traces": np.vstack(change_tr).astype(np.float32) if change_tr else np.zeros((0,0),np.float32),
                   "time": change_time},
        "lick": {"meta": pd.DataFrame(lick_meta),
                 "traces": np.vstack(lick_tr).astype(np.float32) if lick_tr else np.zeros((0,0),np.float32),
                 "time": lick_time},
        "good_mice": GOOD_MICE, "weak_fibers": list(WEAK),
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("change traces:", out["change"]["traces"].shape, "lick traces:", out["lick"]["traces"].shape)
    print("change meta:", len(out["change"]["meta"]), "lick meta:", len(out["lick"]["meta"]))
    print("Saved:", OUT_PKL)

if __name__ == "__main__":
    main()
