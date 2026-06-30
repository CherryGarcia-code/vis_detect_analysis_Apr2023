"""bulk_hit_vs_fa: Hit-lick vs FA-lick go-signal, per genotype x region (BULK 8m).

Unit = MOUSE. Per-mouse lick-aligned traces -> genotype mean +/- SEM over mice.
Quantify pre-lick ramp = mean Delta z in [-0.5, 0) per mouse; compare D1 vs D2,
and Hit vs FA. Traces are session-z-scored dF/F, baseline-subtracted (Delta z).
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUT_DIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "bulk_hit_vs_fa.png")

RAMP_WIN = (-0.5, 0.0)   # pre-lick ramp window
GENO_COLORS = {"D1": "#1f77b4", "D2": "#d62728"}
COND_STYLE = {"hit_lick": "-", "fa_lick": "--"}
REGIONS = ["DMS", "VMS", "VLS"]
CONDS = ["hit_lick", "fa_lick"]

d = pickle.load(open(PKL, "rb"))
lick = d["lick"]
t = lick["time"]
pm = lick["per_mouse"]

ramp_mask = (t >= RAMP_WIN[0]) & (t < RAMP_WIN[1])

def stack(region, geno, cond):
    """Return (mice_list, NxT array) of per-mouse traces."""
    rows, mice = [], []
    for (mouse, g, reg, c), tr in pm.items():
        if reg == region and g == geno and c == cond:
            rows.append(tr); mice.append(mouse)
    if not rows:
        return [], np.empty((0, t.size))
    order = np.argsort(mice)
    mice = [mice[i] for i in order]
    arr = np.array([rows[i] for i in order])
    return mice, arr

def mean_sem(arr):
    n = arr.shape[0]
    m = np.nanmean(arr, axis=0)
    if n > 1:
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)
    else:
        sem = np.zeros_like(m)
    return m, sem, n

# ---- Compute per-mouse pre-lick ramp scalars ----
ramp = {}  # (region, geno, cond) -> (mice, values)
for region in REGIONS:
    for geno in ["D1", "D2"]:
        for cond in CONDS:
            mice, arr = stack(region, geno, cond)
            if arr.shape[0] == 0:
                ramp[(region, geno, cond)] = ([], np.array([]))
                continue
            vals = np.nanmean(arr[:, ramp_mask], axis=1)
            ramp[(region, geno, cond)] = (mice, vals)

# ---- Figure: rows = regions, cols = [traces D1, traces D2, ramp bars] ----
fig, axes = plt.subplots(len(REGIONS), 3, figsize=(15, 11),
                         gridspec_kw={"width_ratios": [1, 1, 0.9]})
if len(REGIONS) == 1:
    axes = axes[None, :]

print("=" * 78)
print("PRE-LICK RAMP  Delta z, mean in [%.1f, %.1f) s   (unit = MOUSE)" % RAMP_WIN)
print("=" * 78)

summary_lines = []
for ri, region in enumerate(REGIONS):
    # ---- trace panels (one per genotype) ----
    for gi, geno in enumerate(["D1", "D2"]):
        ax = axes[ri, gi]
        for cond in CONDS:
            mice, arr = stack(region, geno, cond)
            if arr.shape[0] == 0:
                continue
            m, sem, n = mean_sem(arr)
            ax.plot(t, m, COND_STYLE[cond], color=GENO_COLORS[geno],
                    lw=2, label="%s (N=%d)" % (cond.split("_")[0], n))
            ax.fill_between(t, m - sem, m + sem, color=GENO_COLORS[geno], alpha=0.18)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvspan(RAMP_WIN[0], RAMP_WIN[1], color="gold", alpha=0.12, zorder=0)
        ax.set_xlim(-1.5, 1.0)
        ax.set_title("%s  %s   (Hit solid / FA dashed)" % (region, geno), fontsize=11)
        ax.set_xlabel("Time from lick (s)")
        if gi == 0:
            ax.set_ylabel("Delta z-dF/F")
        ax.legend(fontsize=8, loc="upper left")

    # ---- ramp bar panel ----
    axb = axes[ri, 2]
    xpos, labels, bars = [], [], []
    pos = 0
    geno_means = {}
    for geno in ["D1", "D2"]:
        for cond in CONDS:
            mice, vals = ramp[(region, geno, cond)]
            if vals.size == 0:
                pos += 1
                continue
            mu = np.nanmean(vals)
            sem = np.nanstd(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            axb.bar(pos, mu, width=0.7, color=GENO_COLORS[geno],
                    alpha=0.55 if cond == "hit_lick" else 0.25,
                    edgecolor=GENO_COLORS[geno], lw=1.5,
                    hatch="" if cond == "hit_lick" else "//")
            axb.errorbar(pos, mu, yerr=sem, color="k", capsize=3, lw=1)
            # jitter individual mice
            jit = (np.random.RandomState(0).rand(vals.size) - 0.5) * 0.25
            axb.scatter(np.full(vals.size, pos) + jit, vals, s=22,
                        color="k", zorder=5, alpha=0.7)
            labels.append("%s\n%s" % (geno, cond.split("_")[0]))
            xpos.append(pos)
            geno_means[(geno, cond)] = (mice, vals, mu, sem)
            pos += 1
        pos += 0.4  # gap between genotypes
    axb.axhline(0, color="grey", lw=0.6)
    axb.set_xticks(xpos)
    axb.set_xticklabels(labels, fontsize=8)
    axb.set_title("%s  pre-lick ramp [-0.5,0)" % region, fontsize=11)
    axb.set_ylabel("Delta z (per mouse)")

    # ---- print stats for this region ----
    print("\n--- %s ---" % region)
    for geno in ["D1", "D2"]:
        for cond in CONDS:
            mice, vals = ramp[(region, geno, cond)]
            if vals.size == 0:
                print("  %s %-8s : no mice" % (geno, cond))
                continue
            mu = np.nanmean(vals)
            sem = np.nanstd(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            print("  %s %-8s : N=%d  ramp=%+.3f +/- %.3f   vals=%s"
                  % (geno, cond, vals.size, mu, sem,
                     np.array2string(vals, precision=3)))

# ---- Cross-genotype contrasts (pooled within region; hit_lick focus) ----
print("\n" + "=" * 78)
print("D1 vs D2 pre-lick ramp contrasts (descriptive; small N)")
print("=" * 78)
for region in REGIONS:
    for cond in CONDS:
        m1, v1 = ramp[(region, "D1", cond)]
        m2, v2 = ramp[(region, "D2", cond)]
        if v1.size == 0 or v2.size == 0:
            continue
        diff = np.nanmean(v1) - np.nanmean(v2)
        print("  %s %-8s : D1(N=%d)=%+.3f  D2(N=%d)=%+.3f  D1-D2=%+.3f"
              % (region, cond, v1.size, np.nanmean(v1),
                 v2.size, np.nanmean(v2), diff))

# Hit vs FA within genotype (paired over mice where both exist)
print("\n" + "=" * 78)
print("Hit vs FA pre-lick ramp (paired over mice, within region x geno)")
print("=" * 78)
for region in REGIONS:
    for geno in ["D1", "D2"]:
        mh, vh = ramp[(region, geno, "hit_lick")]
        mf, vf = ramp[(region, geno, "fa_lick")]
        if vh.size == 0 or vf.size == 0:
            continue
        # align by mouse
        common = [mm for mm in mh if mm in mf]
        if not common:
            continue
        ih = [mh.index(mm) for mm in common]
        iff = [mf.index(mm) for mm in common]
        dpair = vh[ih] - vf[iff]
        print("  %s %s : N=%d mice  Hit-FA(paired)=%+.3f +/- %.3f"
              % (region, geno, len(common), np.nanmean(dpair),
                 np.nanstd(dpair, ddof=1) / np.sqrt(len(common)) if len(common) > 1 else 0.0))

fig.suptitle("BULK 8m: Hit-lick vs FA-lick go-signal (lick-aligned, Delta z)  |  unit = mouse",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_PNG, dpi=130)
print("\nSAVED:", OUT_PNG)
