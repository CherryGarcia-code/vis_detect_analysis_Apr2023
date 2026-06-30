"""bulk_learning: early-vs-late session learning of change-evoked & pre-lick signals.

BULK 8m cohort. Unit = MOUSE. Per genotype x region, compare early vs late
session mean traces (mean +/- SEM over mice). Quantify early->late change in
peak per mouse; D1 vs D2 (N=mice). REGION separation kept (DMS/VMS/VLS).
Traces are session-z-scored dF/F, baseline-subtracted (Delta z).
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUTDIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG = os.path.join(OUTDIR, "bulk_learning.png")

with open(PKL, "rb") as f:
    d = pickle.load(f)

tc = d["change"]["time"]  # 600, -2..4
tl = d["lick"]["time"]    # 500, -2..3

GENO_COLOR = {"D1": "#1f77b4", "D2": "#d62728"}


def peak_signed(trace, time, lo, hi):
    """Abs-max within [lo,hi] preserving sign (project extract_peak convention)."""
    m = (time >= lo) & (time <= hi)
    seg = np.asarray(trace)[m]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return np.nan
    idx = np.nanargmax(np.abs(seg))
    return seg[idx]


# ---- Build per-mouse early/late traces for each (cond, dataset) ----
# datasets: ('change','change_hit'), ('lick','fa_lick'), ('lick','hit_lick')
# post-event peak windows: change-aligned 0..2s ; lick-aligned -1..0.3 for fa (pre-lick go), 0..1 for hit
SPECS = [
    ("change", "change_hit", tc, (0.0, 2.0), "Change-evoked (Hit)", "Time from change (s)"),
    ("lick",   "fa_lick",    tl, (-1.0, 0.3), "Pre-FA ramp (FA lick)", "Time from FA lick (s)"),
    ("lick",   "hit_lick",   tl, (0.0, 1.0),  "Hit-lick motor", "Time from hit lick (s)"),
]


def collect(dataset, cond):
    """Return {(geno,region): {mouse: {'early':trace,'late':trace}}}."""
    out = defaultdict(lambda: defaultdict(dict))
    for split in ["early", "late"]:
        for (m, g, r, c), v in d[dataset][split].items():
            if c == cond:
                out[(g, r)][m][split] = np.asarray(v, float)
    return out


# Regions to show as multi-mouse panels (>=2 mice in a geno). DMS & VMS qualify.
PANEL_REGIONS = ["DMS", "VMS"]

# ---- Figure layout: rows = SPECS (3), cols = PANEL_REGIONS (2) for traces,
#      plus a final summary column for per-mouse early->late peak deltas ----
nrows = len(SPECS)
fig, axes = plt.subplots(nrows, 3, figsize=(16, 12))

summary = {}  # (spec_label, geno, region) -> list of (mouse, early_pk, late_pk)

for ri, (dataset, cond, taxis, pkwin, title, xlab) in enumerate(SPECS):
    data = collect(dataset, cond)
    # ---- trace panels for DMS, VMS ----
    for ci, region in enumerate(PANEL_REGIONS):
        ax = axes[ri, ci]
        for g in ["D1", "D2"]:
            key = (g, region)
            if key not in data:
                continue
            mice = sorted(data[key].keys())
            for split, ls, alpha in [("early", "--", 0.55), ("late", "-", 1.0)]:
                traces = [data[key][m][split] for m in mice if split in data[key][m]]
                if len(traces) == 0:
                    continue
                arr = np.vstack(traces)
                mean = np.nanmean(arr, axis=0)
                n = arr.shape[0]
                sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
                lab = f"{g} {split} (n={n})"
                ax.plot(taxis, mean, ls, color=GENO_COLOR[g], alpha=alpha, lw=2, label=lab)
                if n > 1:
                    ax.fill_between(taxis, mean - sem, mean + sem, color=GENO_COLOR[g], alpha=0.12)
            # record per-mouse peaks for summary
            for m in mice:
                ep = peak_signed(data[key][m].get("early"), taxis, *pkwin) if "early" in data[key][m] else np.nan
                lp = peak_signed(data[key][m].get("late"), taxis, *pkwin) if "late" in data[key][m] else np.nan
                summary.setdefault((title, g, region), []).append((m, ep, lp))
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvspan(pkwin[0], pkwin[1], color="gold", alpha=0.08)
        ax.set_title(f"{title}  |  {region}", fontsize=10)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel("Delta z-dF/F", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    # ---- summary col (ci=2): per-mouse early->late peak delta, D1 vs D2 (pool DMS+VMS panel regions) ----
    ax = axes[ri, 2]
    geno_deltas = {"D1": [], "D2": []}
    geno_pts = {"D1": [], "D2": []}  # (early,late,region)
    for g in ["D1", "D2"]:
        for region in PANEL_REGIONS + ["VLS"]:  # include VLS mice as descriptive single points
            recs = summary.get((title, g, region), [])
            for (m, ep, lp) in recs:
                if np.isnan(ep) or np.isnan(lp):
                    continue
                geno_deltas[g].append(lp - ep)
                geno_pts[g].append((ep, lp, region))
    # paired-style strip: plot early->late delta per mouse
    xpos = {"D1": 0, "D2": 1}
    for g in ["D1", "D2"]:
        vals = geno_deltas[g]
        if not vals:
            continue
        jitter = (np.random.RandomState(1).rand(len(vals)) - 0.5) * 0.25
        ax.scatter(np.full(len(vals), xpos[g]) + jitter, vals,
                   color=GENO_COLOR[g], s=45, alpha=0.8, edgecolor="k", lw=0.5, zorder=3)
        mean = np.mean(vals); sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.errorbar(xpos[g], mean, yerr=sem, fmt="_", color="k", ms=28, lw=2, zorder=4, capsize=6)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["D1", "D2"])
    ax.set_xlim(-0.6, 1.6)
    # Mann-Whitney D1 vs D2 on the delta (across-mouse, region-pooled descriptive)
    a, b = geno_deltas["D1"], geno_deltas["D2"]
    stat_txt = ""
    if len(a) >= 1 and len(b) >= 1:
        try:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            stat_txt = f"D1 n={len(a)} mean d={np.mean(a):+.3f}\nD2 n={len(b)} mean d={np.mean(b):+.3f}\nMWU p={p:.3f}"
        except Exception:
            stat_txt = f"D1 n={len(a)}, D2 n={len(b)}"
    ax.set_title(f"early->late peak delta\n{title}", fontsize=9)
    ax.set_ylabel("late - early peak (Delta z)", fontsize=8)
    ax.text(0.98, 0.02, stat_txt, transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.tick_params(labelsize=7)

fig.suptitle("BULK 8m: early vs late session learning (per-mouse; D1 blue / D2 red)\n"
             "dashed=early, solid=late; shaded=SEM over mice; gold=peak window",
             fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTPNG, dpi=120)
print("SAVED:", OUTPNG)

# ===================== STDOUT NUMBERS =====================
print("\n========== PER-MOUSE EARLY->LATE PEAK (signed abs-max in window) ==========")
for (title, dataset, cond, taxis, pkwin, xlab) in [
    (s[4], s[0], s[1], s[2], s[3], s[5]) for s in SPECS
]:
    print(f"\n### {title}  (window {pkwin[0]}..{pkwin[1]}s) ###")
    for g in ["D1", "D2"]:
        gdeltas = []
        for region in ["DMS", "VMS", "VLS"]:
            recs = summary.get((title, g, region), [])
            for (m, ep, lp) in recs:
                tag = ""
                if region == "VLS":
                    tag = " [single-mouse region, descriptive]"
                print(f"  {g} {region:3s} {m}: early={ep:+.3f}  late={lp:+.3f}  delta={lp-ep:+.3f}{tag}")
                if not (np.isnan(ep) or np.isnan(lp)):
                    gdeltas.append(lp - ep)
        if gdeltas:
            print(f"  --> {g} region-pooled: N={len(gdeltas)} mouse-region traces, "
                  f"mean delta={np.mean(gdeltas):+.3f} +/- {np.std(gdeltas, ddof=1)/np.sqrt(len(gdeltas)) if len(gdeltas)>1 else 0:.3f} SEM")
    # D1 vs D2 test
    a = [lp-ep for region in ["DMS","VMS","VLS"] for (m,ep,lp) in summary.get((title,"D1",region),[]) if not(np.isnan(ep) or np.isnan(lp))]
    b = [lp-ep for region in ["DMS","VMS","VLS"] for (m,ep,lp) in summary.get((title,"D2",region),[]) if not(np.isnan(ep) or np.isnan(lp))]
    if a and b:
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  D1 vs D2 early->late delta: MWU U={U:.1f} p={p:.4f}  (N_D1={len(a)}, N_D2={len(b)} mouse-region)")

# DMS-only focus (most mice, cleanest region)
print("\n========== DMS-ONLY FOCUS (change-evoked Hit; N=mice) ==========")
recs_d1 = summary.get(("Change-evoked (Hit)", "D1", "DMS"), [])
recs_d2 = summary.get(("Change-evoked (Hit)", "D2", "DMS"), [])
for lab, recs in [("D1 DMS", recs_d1), ("D2 DMS", recs_d2)]:
    deltas = [lp-ep for (m,ep,lp) in recs if not(np.isnan(ep) or np.isnan(lp))]
    grew = sum(1 for x in deltas if x > 0)
    print(f"  {lab}: N={len(deltas)} mice, mean delta={np.mean(deltas):+.3f}, {grew}/{len(deltas)} grew (late>early)")
