"""Decisive test of the one surviving result: does D1>D2 detection selectivity
survive (a) a MOTOR control and (b) REGION-matching?

Motor control: detection selectivity (Hit-minus-Miss change-aligned) computed in
an EARLY post-change window that PRECEDES the Hit lick (sensory/decision, motor-
free), vs the full window (motor-contaminated). Plus a lick-template-subtraction
variant (subtract the FA-lick pure-motor template, placed at the Hit lick time,
from the Hit change trace). Region-matched: tested WITHIN DMS and WITHIN VMS.

Reads bulk_extract.pkl (per-mouse means; unit=MOUSE). Bulk-8m only.
"""
import os, pickle, sys
import numpy as np
import pandas as pd

WT = r"e:/python_analysis/git_repos/_wt_intersectional_mos"
MAIN = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023"
sys.path.insert(0, os.path.join(WT, "src"))
from visdetect_photom.analysis.group_statistics import permutation_test

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUT = MAIN + "/FIGURES/intersectional_mos/narrative/motor_residualized_sensitivity.png"

EARLY = (0.0, 0.4)   # pre-lick sensory window (Hit licks ~>=0.8s post-change)
FULL = (0.0, 2.0)

def _peakabs(tr, t, win):
    m = (t >= win[0]) & (t <= win[1]); seg = tr[m]
    if seg.size == 0 or np.all(np.isnan(seg)): return np.nan
    return float(seg[np.nanargmax(np.abs(seg))])

def _mean(tr, t, win):
    m = (t >= win[0]) & (t <= win[1]); seg = tr[m]
    return float(np.nanmean(seg)) if seg.size else np.nan

def cliffs_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0: return np.nan
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (a.size * b.size)

def main():
    d = pickle.load(open(PKL, "rb"))
    ct = d["change"]["time"]; cpm = d["change"]["per_mouse"]
    lt = d["lick"]["time"]; lpm = d["lick"]["per_mouse"]
    meta = d["meta"].set_index("mouse")

    rows = []
    for (mouse, geno, region, cond), tr in cpm.items():
        if cond != "change_hit":
            continue
        miss = cpm.get((mouse, geno, region, "change_miss"))
        if miss is None:
            continue
        n = min(len(tr), len(miss))
        hit, miss, tc = tr[:n], miss[:n], ct[:n]
        diff = hit - miss
        full_sel = _peakabs(diff, tc, FULL)
        early_sel = _peakabs(diff, tc, EARLY)
        early_sel_mean = _mean(diff, tc, EARLY)
        # template-subtraction variant: subtract FA-lick motor template placed at Hit lick time
        resid_sel = np.nan
        fa = lpm.get((mouse, geno, region, "fa_lick"))
        rt_hit = meta.loc[mouse, "median_rt_hit"] if mouse in meta.index else np.nan
        if fa is not None and np.isfinite(rt_hit):
            # motor template on change-time axis: fa(lick-aligned) shifted so lick(t=0)->change+rt_hit
            motor_pred = np.interp(tc, lt + rt_hit, fa, left=0.0, right=0.0)
            hit_resid = hit - motor_pred
            resid_sel = _peakabs(hit_resid - miss, tc, FULL)
        rows.append(dict(mouse=mouse, genotype=geno, region=region, median_rt_hit=rt_hit,
                         full_sel=full_sel, early_sel=early_sel, early_sel_mean=early_sel_mean,
                         resid_sel=resid_sel))
    df = pd.DataFrame(rows)
    # collapse VLS-extra: keep DMS & VMS (region-matched targets); VLS descriptive
    print("median Hit RT per mouse (justifies early window <0.4s pre-lick):")
    print(meta["median_rt_hit"].round(3).to_string())
    print("\nper-mouse selectivity (full vs early/motor-free vs template-resid):")
    print(df.round(3).to_string(index=False))

    def test(region, col):
        sub = df[df.region == region]
        d1 = sub[sub.genotype == "D1"][col].dropna().values
        d2 = sub[sub.genotype == "D2"][col].dropna().values
        if len(d1) < 1 or len(d2) < 1: return None
        p = permutation_test(d1, d2)["p"] if len(d1) >= 2 and len(d2) >= 2 else np.nan
        return dict(region=region, metric=col, d1_mean=np.mean(d1), d2_mean=np.mean(d2),
                    diff=np.mean(d1) - np.mean(d2), perm_p=p, cliffs_d=cliffs_d(d1, d2),
                    n_d1=len(d1), n_d2=len(d2))

    print("\n=== WITHIN-REGION D1 vs D2 (region-matched) ===")
    stat_rows = []
    for region in ["DMS", "VMS"]:
        for col in ["full_sel", "early_sel", "resid_sel"]:
            r = test(region, col)
            if r: stat_rows.append(r); print(r)
    sdf = pd.DataFrame(stat_rows)

    # ── figure ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Does D1>D2 detection selectivity survive MOTOR control + REGION-matching?\n"
                 "early_sel = Hit-Miss in 0-0.4s (pre-lick, sensory) ; full_sel = 0-2s (motor-contaminated) ; "
                 "resid = FA-motor-template subtracted. Unit=MOUSE; bulk-8m.", fontsize=10)
    metrics = ["full_sel", "early_sel", "resid_sel"]
    mlabels = ["full (0-2s)", "early (0-0.4s, motor-free)", "template-resid"]
    colors = {"D1": "#2ca02c", "D2": "#1f77b4"}
    for ai, region in enumerate(["DMS", "VMS"]):
        ax = axes[0][ai]
        sub = df[df.region == region]
        x = np.arange(len(metrics))
        for gi, geno in enumerate(["D1", "D2"]):
            g = sub[sub.genotype == geno]
            for mi, col in enumerate(metrics):
                vals = g[col].dropna().values
                xpos = mi + (gi - 0.5) * 0.32
                ax.scatter([xpos] * len(vals), vals, color=colors[geno], s=55,
                           label=geno if mi == 0 else None, zorder=3, edgecolor="k", lw=0.4)
                if len(vals): ax.hlines(np.mean(vals), xpos - 0.13, xpos + 0.13, color=colors[geno], lw=2.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(mlabels, fontsize=8)
        ax.set_title(f"{region}  (D1 n={ (df[(df.region==region)&(df.genotype=='D1')]).shape[0] }, "
                     f"D2 n={ (df[(df.region==region)&(df.genotype=='D2')]).shape[0] })", fontsize=10)
        ax.set_ylabel("detection selectivity (Hit-Miss, Δz)"); ax.legend(fontsize=8)
        # annotate p for early_sel
        r = test(region, "early_sel")
        if r and np.isfinite(r["perm_p"]):
            ax.text(1, ax.get_ylim()[1]*0.9, f"early: D1-D2={r['diff']:.2f}\nperm p={r['perm_p']:.3f}, Cliff d={r['cliffs_d']:.2f}",
                    ha="center", fontsize=7.5)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("\nSaved:", OUT)
    sdf.to_csv(MAIN + "/FIGURES/intersectional_mos/narrative/motor_residualized_sensitivity.csv", index=False)

if __name__ == "__main__":
    main()
