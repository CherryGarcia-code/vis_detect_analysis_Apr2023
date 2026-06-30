"""bulk_outcome_geometry: outcome geometry + temporal expectation in bulk-8m photometry.

change-aligned change_hit vs change_miss vs anticip_cr, per genotype x region.
Per-mouse mean +/- SEM over MICE (N=mice). Quantify:
  (1) Detection selectivity = Hit-minus-Miss peak amplitude in [0,2]s post-change.
  (2) Temporal-expectation ramp = anticip_cr slope in [-1,0]s pre-change (Delta z / s).
Contrast D1 vs D2 within region.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUT_DIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "bulk_outcome_geometry.png")

with open(PKL, "rb") as f:
    d = pickle.load(f)
ch = d["change"]
t = np.asarray(ch["time"])
pm = ch["per_mouse"]

# windows
POST = (t >= 0.0) & (t <= 2.0)        # detection peak window
PRE  = (t >= -1.0) & (t <= 0.0)        # temporal-expectation ramp window

COND = {"change_hit": "Hit", "change_miss": "Miss", "anticip_cr": "CR (catch)"}
CCOL = {"change_hit": "#1b7837", "change_miss": "#b2182b", "anticip_cr": "#2166ac"}
GENO = ["D1", "D2"]
GCOL = {"D1": "#762a83", "D2": "#1b9e77"}
# regions ordered; DMS & VMS have D1&D2; VLS only n=1 each (flagged)
REGIONS = ["DMS", "VMS", "VLS"]

def mice_for(geno, region, cond):
    return sorted({k[0] for k in pm if k[1] == geno and k[2] == region and k[3] == cond})

def trace(mouse, geno, region, cond):
    return np.asarray(pm[(mouse, geno, region, cond)], dtype=float)

def stack(geno, region, cond):
    """matrix [n_mice, T] of per-mouse traces."""
    ms = mice_for(geno, region, cond)
    if not ms:
        return None, []
    return np.vstack([trace(m, geno, region, cond) for m in ms]), ms

def sem(mat):
    n = mat.shape[0]
    if n <= 1:
        return np.zeros(mat.shape[1])
    return np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)

# ---- per-mouse scalar metrics ----
def hit_minus_miss_peak(mouse, geno, region):
    """signed peak (abs-max preserving sign) of Hit-Miss difference in [0,2]s."""
    h = trace(mouse, geno, region, "change_hit")
    m = trace(mouse, geno, region, "change_miss")
    diff = (h - m)[POST]
    return diff[np.argmax(np.abs(diff))]

def cr_pre_slope(mouse, geno, region):
    """OLS slope of anticip_cr over [-1,0]s (Delta z per second)."""
    c = trace(mouse, geno, region, "anticip_cr")[PRE]
    tt = t[PRE]
    sl = np.polyfit(tt, c, 1)[0]
    return sl

# build metric tables
sel_rows = []   # (geno, region, mouse, value)
ramp_rows = []
for region in REGIONS:
    for geno in GENO:
        for mouse in mice_for(geno, region, "change_hit"):
            sel_rows.append((geno, region, mouse, hit_minus_miss_peak(mouse, geno, region)))
        for mouse in mice_for(geno, region, "anticip_cr"):
            ramp_rows.append((geno, region, mouse, cr_pre_slope(mouse, geno, region)))

def grp(rows, geno, region):
    return np.array([r[3] for r in rows if r[0] == geno and r[1] == region], float)

# =========================== FIGURE ===========================
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.42, wspace=0.32)

# Rows 1-2: trace overlays. Row1 = DMS (D1,D2), Row2 = VMS (D1,D2). Cols 0-1.
# Cols 2-3 of rows1-2: the two scalar summaries.
trace_axes = {}
for ri, region in enumerate(["DMS", "VMS"]):
    for gi, geno in enumerate(GENO):
        ax = fig.add_subplot(gs[ri, gi])
        trace_axes[(region, geno)] = ax
        for cond in ["change_hit", "change_miss", "anticip_cr"]:
            mat, ms = stack(geno, region, cond)
            if mat is None:
                continue
            mu = np.nanmean(mat, axis=0)
            se = sem(mat)
            ax.plot(t, mu, color=CCOL[cond], lw=2.0, label=f"{COND[cond]} (n={len(ms)})")
            if mat.shape[0] > 1:
                ax.fill_between(t, mu - se, mu + se, color=CCOL[cond], alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
        ax.axvspan(-1, 0, color="#2166ac", alpha=0.05)
        ax.set_xlim(-2, 4)
        ax.set_title(f"{region}  {geno}", fontsize=12, fontweight="bold", color=GCOL[geno])
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        if gi == 0:
            ax.set_ylabel("Delta z-dF/F")
        ax.set_xlabel("Time from change (s)")

# share y within each region row for honest comparison
for region in ["DMS", "VMS"]:
    axs = [trace_axes[(region, g)] for g in GENO]
    lo = min(a.get_ylim()[0] for a in axs); hi = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(lo, hi)

# ---- Col 2 rows 1-2: detection selectivity (Hit-Miss peak) per region ----
def dot_panel(ax, rows, region, ylabel, title, hline0=True):
    xs = {"D1": 0, "D2": 1}
    if hline0:
        ax.axhline(0, color="grey", lw=0.7, ls="-", alpha=0.6)
    summ = {}
    for geno in GENO:
        vals = grp(rows, geno, region)
        if len(vals) == 0:
            continue
        x = xs[geno]
        jit = (np.random.RandomState(1).rand(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), x) + jit, vals, s=55, color=GCOL[geno],
                   edgecolor="k", lw=0.6, zorder=3, alpha=0.9)
        mu = np.mean(vals)
        ax.hlines(mu, x - 0.22, x + 0.22, color=GCOL[geno], lw=3, zorder=4)
        if len(vals) > 1:
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(x, mu, yerr=se, color=GCOL[geno], capsize=4, lw=2, zorder=4)
        summ[geno] = vals
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"D1\n(n={len(grp(rows,'D1',region))})",
                                               f"D2\n(n={len(grp(rows,'D2',region))})"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, fontweight="bold")
    return summ

ax_sel_dms = fig.add_subplot(gs[0, 2])
dot_panel(ax_sel_dms, sel_rows, "DMS", "Hit-Miss peak (Delta z)", "DMS detection selectivity")
ax_sel_vms = fig.add_subplot(gs[1, 2])
dot_panel(ax_sel_vms, sel_rows, "VMS", "Hit-Miss peak (Delta z)", "VMS detection selectivity")

ax_ramp_dms = fig.add_subplot(gs[0, 3])
dot_panel(ax_ramp_dms, ramp_rows, "DMS", "CR pre-change slope (Delta z/s)", "DMS temporal-expectation ramp")
ax_ramp_vms = fig.add_subplot(gs[1, 3])
dot_panel(ax_ramp_vms, ramp_rows, "VMS", "CR pre-change slope (Delta z/s)", "VMS temporal-expectation ramp")

# ---- Row 3: VLS overlays (single mouse each, flagged) + grand difference summary ----
ax_vls_d1 = fig.add_subplot(gs[2, 0])
ax_vls_d2 = fig.add_subplot(gs[2, 1])
for ax, geno in [(ax_vls_d1, "D1"), (ax_vls_d2, "D2")]:
    for cond in ["change_hit", "change_miss", "anticip_cr"]:
        mat, ms = stack(geno, "VLS", cond)
        if mat is None:
            continue
        ax.plot(t, np.nanmean(mat, 0), color=CCOL[cond], lw=1.8, label=f"{COND[cond]}")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7); ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
    ax.set_xlim(-2, 4); ax.set_xlabel("Time from change (s)")
    nd1 = len(mice_for(geno, "VLS", "change_hit"))
    ax.set_title(f"VLS  {geno}  (n={nd1}, single-mouse)", fontsize=10, color=GCOL[geno])
    ax.legend(fontsize=7, frameon=False)
ax_vls_d1.set_ylabel("Delta z-dF/F")

# ---- Row 3 cols 2-3: pooled DMS+VMS difference traces D1 vs D2 ----
ax_pool_sel = fig.add_subplot(gs[2, 2])
for geno in GENO:
    # pool DMS+VMS per-mouse Hit-Miss difference traces
    dmats = []
    for region in ["DMS", "VMS"]:
        for mouse in mice_for(geno, region, "change_hit"):
            dmats.append(trace(mouse, geno, region, "change_hit") - trace(mouse, geno, region, "change_miss"))
    dmats = np.vstack(dmats)
    mu = np.nanmean(dmats, 0); se = sem(dmats)
    ax_pool_sel.plot(t, mu, color=GCOL[geno], lw=2, label=f"{geno} (n={dmats.shape[0]})")
    ax_pool_sel.fill_between(t, mu - se, mu + se, color=GCOL[geno], alpha=0.18, lw=0)
ax_pool_sel.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7); ax_pool_sel.axhline(0, color="grey", lw=0.6)
ax_pool_sel.set_xlim(-2, 4); ax_pool_sel.set_xlabel("Time from change (s)")
ax_pool_sel.set_ylabel("Hit-Miss (Delta z)")
ax_pool_sel.set_title("Detection-selectivity trace\n(DMS+VMS pooled per mouse)", fontsize=10, fontweight="bold")
ax_pool_sel.legend(fontsize=8, frameon=False)

ax_pool_cr = fig.add_subplot(gs[2, 3])
for geno in GENO:
    cmats = []
    for region in ["DMS", "VMS"]:
        for mouse in mice_for(geno, region, "anticip_cr"):
            cmats.append(trace(mouse, geno, region, "anticip_cr"))
    cmats = np.vstack(cmats)
    mu = np.nanmean(cmats, 0); se = sem(cmats)
    ax_pool_cr.plot(t, mu, color=GCOL[geno], lw=2, label=f"{geno} (n={cmats.shape[0]})")
    ax_pool_cr.fill_between(t, mu - se, mu + se, color=GCOL[geno], alpha=0.18, lw=0)
ax_pool_cr.axvspan(-1, 0, color="#2166ac", alpha=0.06)
ax_pool_cr.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7); ax_pool_cr.axhline(0, color="grey", lw=0.6)
ax_pool_cr.set_xlim(-2, 4); ax_pool_cr.set_xlabel("Time from (expected) change (s)")
ax_pool_cr.set_ylabel("CR trace (Delta z)")
ax_pool_cr.set_title("Temporal-expectation ramp (CR)\n(DMS+VMS pooled per mouse)", fontsize=10, fontweight="bold")
ax_pool_cr.legend(fontsize=8, frameon=False)

fig.suptitle("Bulk-8m outcome geometry: detection selectivity (Hit vs Miss) & CR temporal-expectation ramp  |  per-mouse mean +/- SEM over mice",
             fontsize=13, fontweight="bold", y=0.995)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("SAVED:", OUT)

# =========================== STATS / NUMBERS ===========================
print("\n===== PER-MOUSE METRICS =====")
def report(rows, label, unit):
    print(f"\n--- {label} ({unit}) ---")
    for region in REGIONS:
        line = f"  {region}: "
        for geno in GENO:
            v = grp(rows, geno, region)
            if len(v) == 0:
                line += f"{geno} n=0  "
            elif len(v) == 1:
                line += f"{geno} n=1 val={v[0]:+.4f}  "
            else:
                line += f"{geno} n={len(v)} mean={np.mean(v):+.4f}+/-{np.std(v,ddof=1)/np.sqrt(len(v)):.4f}  "
        print(line)
        d1 = grp(rows, "D1", region); d2 = grp(rows, "D2", region)
        if len(d1) >= 1 and len(d2) >= 1:
            print(f"        D1 vals={np.round(d1,4).tolist()}  D2 vals={np.round(d2,4).tolist()}")
            if len(d1) >= 2 and len(d2) >= 2:
                try:
                    U, p = stats.mannwhitneyu(d1, d2, alternative="two-sided")
                    print(f"        MWU U={U:.1f} p={p:.3f} (NOTE: tiny N, descriptive only)")
                except Exception as e:
                    print("        MWU err", e)

report(sel_rows, "Detection selectivity = Hit-Miss peak [0,2]s", "Delta z")
report(ramp_rows, "Temporal-expectation = CR slope [-1,0]s", "Delta z/s")

# pooled DMS+VMS comparison (still per-mouse unit)
print("\n===== POOLED DMS+VMS (per-mouse unit) =====")
for rows, lab in [(sel_rows, "Hit-Miss peak"), (ramp_rows, "CR pre slope")]:
    for geno in GENO:
        vals = np.array([r[3] for r in rows if r[0] == geno and r[1] in ("DMS", "VMS")], float)
        print(f"  {lab} {geno}: n={len(vals)} mean={np.mean(vals):+.4f} +/- {np.std(vals,ddof=1)/np.sqrt(len(vals)):.4f}  vals={np.round(vals,4).tolist()}")
    d1 = np.array([r[3] for r in rows if r[0]=="D1" and r[1] in ("DMS","VMS")],float)
    d2 = np.array([r[3] for r in rows if r[0]=="D2" and r[1] in ("DMS","VMS")],float)
    U,p = stats.mannwhitneyu(d1,d2,alternative="two-sided")
    # rank-biserial effect size
    rbc = 1 - 2*U/(len(d1)*len(d2))
    print(f"    POOLED D1 vs D2: MWU U={U:.1f} p={p:.3f} rank-biserial r={rbc:+.3f}  (N_D1={len(d1)} mice, N_D2={len(d2)} mice)")

# fraction of mice with positive detection selectivity / positive CR ramp
sel_pos = sum(1 for r in sel_rows if r[3] > 0); print(f"\n  Mice with POSITIVE Hit-Miss peak: {sel_pos}/{len(sel_rows)}")
ramp_pos = sum(1 for r in ramp_rows if r[3] > 0); print(f"  Mice with POSITIVE CR pre-change ramp: {ramp_pos}/{len(ramp_rows)}")
print("\nDONE")
