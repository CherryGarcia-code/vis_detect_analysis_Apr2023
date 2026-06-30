"""Pre-extract BULK GCaMP8m signal (per-mouse, per-region, per-condition MEAN
traces) once, so exploration agents analyze without re-loading ~600 sessions.

N = mice is the unit (bulk discipline). Hemispheres merged via qc.region_sources
(bulk has no laterality asymmetry, unlike the intersectional cohort). Good bulk
mice only (014/015/017 excluded by explicit list -> no manifest dependency).
NEVER pooled with the 6f intersectional cohort.

Output pkl: per-mouse mean traces keyed by condition, plus learning (early/late)
and RT-tercile lick means, plus per-mouse scalar meta.
"""
import os, sys, pickle
from collections import defaultdict
import numpy as np
import pandas as pd

WT = r"e:/python_analysis/git_repos/_wt_intersectional_mos"
MAIN = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023"
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import MIN_PHOTOM_CSV_BYTES, CHANGE_SIZES, CATCH_THRESHOLD
from visdetect_photom.core.qc import region_sources
from visdetect_photom.analysis.statistics import extract_peth, calculate_sdt_metrics

OUT_PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
DATA = os.path.join(MAIN, "photom_data")

BULK = {"BG_008":"D1","BG_009":"D1","BG_013":"D1","BG_020":"D1",
        "BG_010":"D2","BG_011":"D2","BG_016":"D2","BG_018":"D2","BG_019":"D2"}
CHANGE_WIN, CHANGE_BL = (-2.0, 4.0), (-2.0, 0.0)
LICK_WIN, LICK_BL = (-2.0, 3.0), (-2.0, -1.5)

def _norm(s):
    s = str(s)
    return s if s.startswith("BG_") else f"BG_{s.zfill(3)}"

def _mean_peth(sig, ts, ev, win, bl):
    if len(ev) == 0:
        return None, None
    tax, peth = extract_peth(sig, ts, np.asarray(ev, float), window=win,
                             baseline_window=bl, normalize="subtract")
    if peth.size == 0:
        return None, None
    mt = np.nanmean(peth, axis=0)
    return tax, (None if np.all(np.isnan(mt)) else mt)

def main():
    files = io.find_all_sessions(DATA, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    # accumulate per (mouse, region, condition) -> list of per-session mean traces
    chg = defaultdict(list); lck = defaultdict(list)
    chg_early = defaultdict(list); chg_late = defaultdict(list)
    lck_early = defaultdict(list); lck_late = defaultdict(list)
    rt_pool = defaultdict(list)   # (mouse,region,event) -> list of (rt, lick_trace) per trial-mean? we store per-session not per-trial; for RT tercile we need per-trial
    # For RT terciles we need per-trial lick traces; collect per (mouse,region,event) capped.
    rt_trials = defaultdict(lambda: {"rt": [], "tr": []})
    meta_rows = []
    sess_by_mouse = defaultdict(list)
    chg_time = {"t": None}; lck_time = {"t": None}

    # first pass: group sessions by mouse, ordered by recording_id (date)
    loaded = []
    n = 0
    for sf in files:
        try:
            s = load_session_from_files(sf)
        except Exception:
            continue
        if _norm(s.subject_id) not in BULK:
            continue
        loaded.append(s); n += 1
    for s in loaded:
        sess_by_mouse[_norm(s.subject_id)].append(s)
    print(f"Loaded {len(loaded)} bulk sessions across {len(sess_by_mouse)} mice")

    for mouse, slist in sess_by_mouse.items():
        geno = BULK[mouse]
        slist = sorted(slist, key=lambda s: getattr(s, "recording_id", s.session_id))
        half = len(slist) // 2
        # per-mouse outcome/rt accumulators
        outs_all, css_all, rt_fa, rt_hit = [], [], [], []
        for si, s in enumerate(slist):
            is_late = si >= half
            try:
                sources = region_sources(s, use_qc=True)
            except Exception:
                continue
            outs_all += [t.outcome for t in s.trials]
            css_all += [t.change_size if t.change_size is not None else np.nan for t in s.trials]
            for region, (sig, ts) in sources.items():
                def ev_change(pred):
                    return [t.absolute_change_time for t in s.trials if pred(t)
                            and t.absolute_change_time is not None and np.isfinite(t.absolute_change_time)]
                def ev_lick(pred):
                    return [t.absolute_reaction_time for t in s.trials if pred(t)
                            and t.absolute_reaction_time is not None and np.isfinite(t.absolute_reaction_time)]
                go = lambda t: (t.change_size or 0) > CATCH_THRESHOLD
                conds_change = {
                    "change_hit": ev_change(lambda t: t.outcome=="Hit" and go(t)),
                    "change_miss": ev_change(lambda t: t.outcome=="Miss" and go(t)),
                    "anticip_cr": ev_change(lambda t: t.outcome=="CR"),
                }
                for cs in CHANGE_SIZES:
                    conds_change[f"change_hit_cs{cs}"] = ev_change(
                        lambda t, cs=cs: t.outcome=="Hit" and abs((t.change_size or 0)-cs) < 1e-6)
                conds_lick = {
                    "hit_lick": ev_lick(lambda t: t.outcome=="Hit"),
                    "fa_lick": ev_lick(lambda t: t.outcome=="FA"),
                }
                for cond, ev in conds_change.items():
                    tax, mt = _mean_peth(sig, ts, ev, CHANGE_WIN, CHANGE_BL)
                    if mt is not None:
                        chg_time["t"] = tax; chg[(mouse,geno,region,cond)].append(mt)
                        (chg_late if is_late else chg_early)[(mouse,geno,region,cond)].append(mt)
                for cond, ev in conds_lick.items():
                    tax, mt = _mean_peth(sig, ts, ev, LICK_WIN, LICK_BL)
                    if mt is not None:
                        lck_time["t"] = tax; lck[(mouse,geno,region,cond)].append(mt)
                        (lck_late if is_late else lck_early)[(mouse,geno,region,cond)].append(mt)
                # per-trial lick traces for RT terciles (cap memory: store per trial)
                for ev_name, pred in (("hit_lick", lambda t: t.outcome=="Hit"),
                                       ("fa_lick", lambda t: t.outcome=="FA")):
                    trials = [t for t in s.trials if pred(t) and t.absolute_reaction_time is not None
                              and np.isfinite(t.absolute_reaction_time) and t.reaction_time is not None
                              and np.isfinite(t.reaction_time)]
                    if trials:
                        evs = np.array([t.absolute_reaction_time for t in trials], float)
                        tax, peth = extract_peth(sig, ts, evs, window=LICK_WIN, baseline_window=LICK_BL, normalize="subtract")
                        for k, t in enumerate(trials):
                            rt_trials[(mouse,geno,region,ev_name)]["rt"].append(float(t.reaction_time))
                            rt_trials[(mouse,geno,region,ev_name)]["tr"].append(peth[k].astype(np.float32))
            # per-mouse RT
            rt_fa += [t.reaction_time for t in s.trials if t.outcome=="FA" and t.reaction_time]
            rt_hit += [t.reaction_time for t in s.trials if t.outcome=="Hit" and t.reaction_time]
        sdt = calculate_sdt_metrics(np.array(outs_all), np.array(css_all, dtype=float))
        from collections import Counter
        c = Counter(outs_all)
        meta_rows.append(dict(mouse=mouse, genotype=geno, n_sessions=len(slist),
            n_trials=len(outs_all), n_Hit=c.get("Hit",0), n_Miss=c.get("Miss",0),
            n_FA=c.get("FA",0), n_CR=c.get("CR",0), n_Abort=c.get("Abort",0),
            d_prime=sdt.get("d_prime"), sdt_hit_rate=sdt.get("sdt_hit_rate"),
            sdt_fa_rate=sdt.get("sdt_fa_rate"),
            median_rt_fa=float(np.median(rt_fa)) if rt_fa else np.nan,
            median_rt_hit=float(np.median(rt_hit)) if rt_hit else np.nan))

    def _permouse(d):  # list of per-session means -> per-mouse mean trace
        return {k: np.nanmean(np.vstack(v), axis=0).astype(np.float32) for k, v in d.items() if v}
    def _rt_terciles(rtd):
        out = {}
        for key, dd in rtd.items():
            rt = np.array(dd["rt"]); tr = np.array(dd["tr"], dtype=np.float32)
            if len(rt) < 9:
                continue
            q1, q2 = np.quantile(rt, [1/3, 2/3])
            for name, mask in (("fast", rt<=q1), ("mid", (rt>q1)&(rt<=q2)), ("slow", rt>q2)):
                if mask.sum() >= 3:
                    out[key+(name,)] = np.nanmean(tr[mask], axis=0).astype(np.float32)
        return out

    out = {
        "change": {"time": chg_time["t"], "per_mouse": _permouse(chg),
                   "early": _permouse(chg_early), "late": _permouse(chg_late)},
        "lick": {"time": lck_time["t"], "per_mouse": _permouse(lck),
                 "early": _permouse(lck_early), "late": _permouse(lck_late)},
        "rt_tercile": {"time": lck_time["t"], "traces": _rt_terciles(rt_trials)},
        "meta": pd.DataFrame(meta_rows), "mice": BULK,
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("per-mouse change keys:", len(out["change"]["per_mouse"]), "lick keys:", len(out["lick"]["per_mouse"]))
    print("rt_tercile keys:", len(out["rt_tercile"]["traces"]))
    print(out["meta"][["mouse","genotype","n_sessions","n_FA","d_prime"]].to_string(index=False))
    print("Saved:", OUT_PKL)

if __name__ == "__main__":
    main()
