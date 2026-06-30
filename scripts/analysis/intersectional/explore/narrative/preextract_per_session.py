"""Per-session NEURAL + BEHAVIORAL metrics for the impulsivity-vs-sensitivity
-across-learning narrative. One row per (cohort, mouse, region/fiber, session).

Substrate for: mixed-effects sensitivity test (regions separate), learning
trajectories, neural<->behavioral correlation (sensitivity<->d', impulsivity<->FA),
and motor-vs-value. Two cohorts tagged, NEVER pooled.

NEURAL (per session x region): detection selectivity (peak of Hit-minus-Miss
change-aligned diff, 0-2s) [SENSITIVITY], change_hit_peak, evidence_slope
(within-session Spearman of per-trial change peak vs log2 change_size) [SENSITIVITY],
prefa_ramp (mean FA-lick trace [-0.5,0)) [IMPULSIVITY], hit_lick_peak.
BEHAVIORAL (per session): d_prime, sdt rates [SENSITIVITY], fa_rate_beh, n_FA
[IMPULSIVITY], hit_rate, median_rt_fa. + chronological session ordinal/frac (LEARNING).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WT = r"e:/python_analysis/git_repos/_wt_intersectional_mos"
MAIN = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023"
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import MIN_PHOTOM_CSV_BYTES, CHANGE_SIZES, CATCH_THRESHOLD
from visdetect_photom.core.qc import region_sources
from visdetect_photom.analysis.statistics import extract_peth, calculate_sdt_metrics

OUTCSV = MAIN + "/results/explore_cache/per_session_metrics.csv"
BULK = {"BG_008":"D1","BG_009":"D1","BG_013":"D1","BG_020":"D1",
        "BG_010":"D2","BG_011":"D2","BG_016":"D2","BG_018":"D2","BG_019":"D2"}
INT = {"BG_027":"D1","BG_028":"D1"}              # good intersectional D1
INT_FIBERS = {"BG_027":["G0","G2"], "BG_028":["G0"]}  # good fibers only
CW, CBL = (-2.0, 4.0), (-2.0, 0.0)
LW, LBL = (-2.0, 3.0), (-2.0, -1.5)

def _norm(s):
    s = str(s); return s if s.startswith("BG_") else f"BG_{s.zfill(3)}"

def _speak(tr, t, win):
    m = (t >= win[0]) & (t <= win[1]); seg = tr[m]
    if seg.size == 0 or np.all(np.isnan(seg)): return np.nan
    return float(seg[np.nanargmax(np.abs(seg))])

def _mean(sig, ts, ev, win, bl):
    if len(ev) == 0: return None, None
    t, p = extract_peth(sig, ts, np.asarray(ev, float), window=win, baseline_window=bl, normalize="subtract")
    if p.size == 0: return None, None
    return t, p

def metrics_for(sig, ts, sess):
    """neural metrics for one (signal, timestamps) = one region/fiber in one session."""
    go = lambda tr: (tr.change_size or 0) > CATCH_THRESHOLD
    def ev_ch(pred): return [tt.absolute_change_time for tt in sess.trials if pred(tt)
                             and tt.absolute_change_time is not None and np.isfinite(tt.absolute_change_time)]
    def ev_lk(pred): return [tt.absolute_reaction_time for tt in sess.trials if pred(tt)
                             and tt.absolute_reaction_time is not None and np.isfinite(tt.absolute_reaction_time)]
    out = dict(chg_hit_peak=np.nan, chg_miss_peak=np.nan, detect_sel=np.nan,
               evidence_slope=np.nan, prefa_ramp=np.nan, hit_lick_peak=np.nan,
               n_chg_hit=0, n_fa_lick=0)
    th, ph = _mean(sig, ts, ev_ch(lambda t: t.outcome=="Hit" and go(t)), CW, CBL)
    tm, pm = _mean(sig, ts, ev_ch(lambda t: t.outcome=="Miss" and go(t)), CW, CBL)
    if th is not None:
        hit_mean = np.nanmean(ph, axis=0); out["chg_hit_peak"] = _speak(hit_mean, th, (0,2)); out["n_chg_hit"]=ph.shape[0]
        if tm is not None and pm.shape[0] >= 3:
            miss_mean = np.nanmean(pm, axis=0); out["chg_miss_peak"] = _speak(miss_mean, tm, (0,2))
            n = min(len(hit_mean), len(miss_mean))
            out["detect_sel"] = _speak(hit_mean[:n]-miss_mean[:n], th[:n], (0,2))
    # evidence slope: per-trial change-hit peak vs change_size (within session)
    hit_trials = [(t.absolute_change_time, t.change_size) for t in sess.trials
                  if t.outcome=="Hit" and go(t) and t.absolute_change_time is not None
                  and np.isfinite(t.absolute_change_time) and t.change_size is not None]
    if len(hit_trials) >= 6:
        evs = np.array([a for a,_ in hit_trials], float); css = np.array([c for _,c in hit_trials], float)
        t2, p2 = extract_peth(sig, ts, evs, window=CW, baseline_window=CBL, normalize="subtract")
        if p2.size:
            peaks = np.array([_speak(p2[k], t2, (0,1.5)) for k in range(p2.shape[0])])
            ok = np.isfinite(peaks) & np.isfinite(css)
            if ok.sum() >= 6 and len(np.unique(css[ok])) >= 3:
                rho, _ = spearmanr(np.log2(css[ok]), peaks[ok])
                out["evidence_slope"] = float(rho)
    tf, pf = _mean(sig, ts, ev_lk(lambda t: t.outcome=="FA"), LW, LBL)
    if tf is not None:
        fa_mean = np.nanmean(pf, axis=0); out["n_fa_lick"]=pf.shape[0]
        mwin = (tf >= -0.5) & (tf < 0.0); out["prefa_ramp"] = float(np.nanmean(fa_mean[mwin]))
    tl, pl = _mean(sig, ts, ev_lk(lambda t: t.outcome=="Hit"), LW, LBL)
    if tl is not None:
        out["hit_lick_peak"] = _speak(np.nanmean(pl, axis=0), tl, (0,1))
    return out

def behav(sess):
    outs = np.array([t.outcome for t in sess.trials])
    css = np.array([t.change_size if t.change_size is not None else np.nan for t in sess.trials], float)
    n = len(outs)
    sdt = calculate_sdt_metrics(outs, css)
    rt_fa = [t.reaction_time for t in sess.trials if t.outcome=="FA" and t.reaction_time]
    return dict(n_trials=n, n_FA=int((outs=="FA").sum()), n_Hit=int((outs=="Hit").sum()),
                n_Miss=int((outs=="Miss").sum()),
                d_prime=sdt.get("d_prime"), sdt_hit_rate=sdt.get("sdt_hit_rate"),
                sdt_fa_rate=sdt.get("sdt_fa_rate"),
                fa_rate_beh=int((outs=="FA").sum())/max(1,n),
                hit_rate=int((outs=="Hit").sum())/max(1,(outs=="Hit").sum()+(outs=="Miss").sum()),
                median_rt_fa=float(np.median(rt_fa)) if rt_fa else np.nan)

def main():
    files = io.find_all_sessions(MAIN+"/photom_data", recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    by_mouse = {}
    for sf in files:
        try: s = load_session_from_files(sf)
        except Exception: continue
        m = _norm(s.subject_id)
        if m in BULK or m in INT:
            by_mouse.setdefault(m, []).append(s)
    rows = []
    for m, slist in by_mouse.items():
        slist = sorted(slist, key=lambda s: getattr(s,"recording_id",s.session_id))
        N = len(slist); geno = BULK.get(m) or INT.get(m)
        cohort = "bulk" if m in BULK else "intersectional"
        for i, s in enumerate(slist):
            b = behav(s)
            if cohort == "bulk":
                try: srcs = region_sources(s, use_qc=True)
                except Exception: srcs = {}
                units = [(reg, sig, ts) for reg,(sig,ts) in srcs.items()]
            else:
                units = []
                for roi in INT_FIBERS[m]:
                    tr = s.photometry_data.get(roi)
                    if tr is not None:
                        reg = "VMS" if m in ("BG_027",) else "DMS"
                        units.append((f"{reg}_{roi}", tr.signal, tr.timestamps))
            for reg, sig, ts in units:
                nm = metrics_for(sig, ts, s)
                rows.append(dict(cohort=cohort, mouse=m, genotype=geno, region=reg,
                    session_id=s.session_id, recording_id=getattr(s,"recording_id",s.session_id),
                    sess_ordinal=i, sess_frac=(i/(N-1) if N>1 else 0.0), **b, **nm))
        print(f"{m} ({geno},{cohort}): {N} sessions")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
    df.to_csv(OUTCSV, index=False)
    print("rows:", len(df), "-> ", OUTCSV)
    print(df.groupby(["cohort","genotype"]).size())

if __name__ == "__main__":
    main()
