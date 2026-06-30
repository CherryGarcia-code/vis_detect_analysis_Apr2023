"""BULK neural psychometric: change-evoked Delta z amplitude vs log2(change_size).

Per (mouse x region): peak Delta z in (0,1.5s] for each of 5 change sizes.
Per-mouse Spearman rho(peak, log2 cs). Aggregate rho per genotype x region (mean +/- SEM over mice, N=mice).
Plot per-genotype mean amplitude-vs-change_size curves (per region).
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUT_DIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
OUT_PNG = os.path.join(OUT_DIR, "bulk_psychometric.png")
os.makedirs(OUT_DIR, exist_ok=True)

CS = [1.25, 1.35, 1.5, 2.0, 4.0]
CS_KEYS = [f"change_hit_cs{c}" for c in CS]
LOG2CS = np.log2(np.array(CS))
WIN = (0.0, 1.5)   # peak search window (s) post-change

with open(PKL, "rb") as f:
    d = pickle.load(f)

t = d["change"]["time"]
pm = d["change"]["per_mouse"]
win_mask = (t > WIN[0]) & (t <= WIN[1])

def peak_dz(trace):
    """Signed peak (abs-max) Delta z in window — preserves suppression sign."""
    seg = trace[win_mask]
    seg = seg[~np.isnan(seg)]
    if seg.size == 0:
        return np.nan
    return seg[np.argmax(np.abs(seg))]

# Collect per (mouse,geno,region): amplitude vector over 5 change sizes
groups = {}  # (mouse,geno,region) -> {cs: peak}
for (m, g, r, c), tr in pm.items():
    if c in CS_KEYS:
        cs_val = float(c.replace("change_hit_cs", ""))
        groups.setdefault((m, g, r), {})[cs_val] = peak_dz(tr)

# Build per-mouse curves (require all 5 CS) and per-mouse Spearman rho
records = []  # dict per (mouse,geno,region)
for (m, g, r), cdict in sorted(groups.items()):
    amps = np.array([cdict.get(c, np.nan) for c in CS])
    if np.isnan(amps).any():
        n_have = int(np.sum(~np.isnan(amps)))
        print(f"SKIP rho (incomplete): {m} {g} {r} have {n_have}/5 -> {amps}")
        # still keep partial curve for plotting if >=2 points but no rho
        rho = np.nan
        if n_have >= 3:
            valid = ~np.isnan(amps)
            rho, _ = spearmanr(LOG2CS[valid], amps[valid])
    else:
        rho, _ = spearmanr(LOG2CS, amps)
    records.append(dict(mouse=m, geno=g, region=r, amps=amps, rho=rho))

# Print per-mouse table
print("\n=== Per-mouse peak Delta z by change size, and Spearman rho(peak, log2 cs) ===")
print(f"{'mouse':8s} {'geno':4s} {'region':6s} " + " ".join(f"cs{c:<5}" for c in CS) + "  rho")
for rec in records:
    a = rec["amps"]
    astr = " ".join((f"{x:6.3f}" if not np.isnan(x) else "   nan") for x in a)
    print(f"{rec['mouse']:8s} {rec['geno']:4s} {rec['region']:6s} {astr}  {rec['rho']:+.3f}")

# Aggregate rho per genotype x region (N = mice with finite rho)
print("\n=== Aggregate Spearman rho per genotype x region (mean +/- SEM over mice) ===")
from collections import defaultdict
agg_rho = defaultdict(list)
for rec in records:
    if np.isfinite(rec["rho"]):
        agg_rho[(rec["geno"], rec["region"])].append(rec["rho"])
agg_rho_summary = {}
for key, vals in sorted(agg_rho.items()):
    vals = np.array(vals)
    n = len(vals)
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    agg_rho_summary[key] = (mean, sem, n, vals)
    semstr = f"{sem:.3f}" if np.isfinite(sem) else "nan"
    print(f"{key[0]} {key[1]:6s}: mean rho = {mean:+.3f} +/- {semstr} (N={n} mice) values={np.round(vals,3).tolist()}")

# Genotype-pooled across regions (each mouse counted once per region present)
print("\n=== Aggregate rho per genotype (pooled regions, N=mouse-region groups) ===")
agg_geno = defaultdict(list)
for rec in records:
    if np.isfinite(rec["rho"]):
        agg_geno[rec["geno"]].append(rec["rho"])
for g, vals in sorted(agg_geno.items()):
    vals = np.array(vals)
    n = len(vals)
    sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    semstr = f"{sem:.3f}" if np.isfinite(sem) else "nan"
    print(f"{g}: mean rho = {vals.mean():+.3f} +/- {semstr} (N={n} mouse-region groups)")

# ---- Per-genotype mean amplitude-vs-change_size curves (per region) ----
# Group complete curves by (geno,region)
curve_groups = defaultdict(list)  # (geno,region) -> list of amps arrays (complete only)
for rec in records:
    if not np.isnan(rec["amps"]).any():
        curve_groups[(rec["geno"], rec["region"])].append(rec["amps"])

print("\n=== Group-mean amplitude curves (complete-curve mice only) ===")
for key, arrs in sorted(curve_groups.items()):
    A = np.vstack(arrs)
    mean = A.mean(0)
    print(f"{key[0]} {key[1]:6s} (N={A.shape[0]}): " + " ".join(f"{v:6.3f}" for v in mean))

# ----------------------------- PLOT -----------------------------
GENO_COLOR = {"D1": "#1f77b4", "D2": "#d62728"}
REGIONS = ["DMS", "VMS", "VLS"]

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.32)

# Row 1: amplitude vs change size, one panel per region
for ci, region in enumerate(REGIONS):
    ax = fig.add_subplot(gs[0, ci])
    any_data = False
    for geno in ["D1", "D2"]:
        # individual mice (thin)
        for rec in records:
            if rec["geno"] == geno and rec["region"] == region and not np.isnan(rec["amps"]).any():
                ax.plot(LOG2CS, rec["amps"], color=GENO_COLOR[geno], alpha=0.25, lw=1.0)
                any_data = True
        # group mean +/- SEM
        key = (geno, region)
        if key in curve_groups:
            A = np.vstack(curve_groups[key])
            mean = A.mean(0)
            n = A.shape[0]
            sem = A.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            ax.errorbar(LOG2CS, mean, yerr=sem, color=GENO_COLOR[geno], lw=2.4,
                        marker="o", ms=6, capsize=3,
                        label=f"{geno} (N={n})")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(LOG2CS)
    ax.set_xticklabels([str(c) for c in CS])
    ax.set_xlabel("Change size (TF ratio)")
    if ci == 0:
        ax.set_ylabel("Peak Δz in (0,1.5s]")
    ax.set_title(f"{region}: change-evoked amplitude vs evidence")
    if any_data:
        ax.legend(fontsize=8, frameon=False)

# Row 2 panel A: per-mouse rho dots by genotype x region
axr = fig.add_subplot(gs[1, 0])
xt = []
xl = []
xpos = 0
for geno in ["D1", "D2"]:
    for region in REGIONS:
        key = (geno, region)
        if key in agg_rho_summary:
            mean, sem, n, vals = agg_rho_summary[key]
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.25
            axr.scatter(np.full(len(vals), xpos) + jitter, vals,
                        color=GENO_COLOR[geno], s=45, alpha=0.7, zorder=3)
            axr.plot([xpos - 0.2, xpos + 0.2], [mean, mean], color="k", lw=2.5, zorder=4)
            xt.append(xpos)
            xl.append(f"{geno}\n{region}\nN={n}")
            xpos += 1
axr.axhline(0, color="0.6", lw=0.8, ls="--")
axr.set_xticks(xt)
axr.set_xticklabels(xl, fontsize=8)
axr.set_ylabel("Spearman rho\n(peak vs log2 cs)")
axr.set_ylim(-1.05, 1.05)
axr.set_title("Per-mouse evidence-scaling rho")

# Row 2 panel B: genotype-pooled rho summary bars
axb = fig.add_subplot(gs[1, 1])
gx = 0
for g in ["D1", "D2"]:
    if g in agg_geno:
        vals = np.array(agg_geno[g])
        n = len(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        axb.bar(gx, mean, yerr=sem, color=GENO_COLOR[g], alpha=0.6, capsize=4, width=0.6)
        jit = (np.random.RandomState(1).rand(n) - 0.5) * 0.25
        axb.scatter(np.full(n, gx) + jit, vals, color=GENO_COLOR[g], edgecolor="k", s=40, zorder=3)
        axb.text(gx, 1.0, f"N={n}", ha="center", fontsize=8)
        gx += 1
axb.axhline(0, color="0.6", lw=0.8, ls="--")
axb.set_xticks([0, 1])
axb.set_xticklabels(["D1", "D2"])
axb.set_ylabel("Spearman rho")
axb.set_ylim(-1.05, 1.15)
axb.set_title("Evidence-scaling rho by genotype\n(pooled regions)")

# Row 2 panel C: text summary
axt = fig.add_subplot(gs[1, 2])
axt.axis("off")
lines = ["NEURAL PSYCHOMETRIC (BULK 8m)", "Peak Δz (0,1.5s] vs log2(change size)", ""]
for g in ["D1", "D2"]:
    if g in agg_geno:
        vals = np.array(agg_geno[g]); n = len(vals)
        sem = vals.std(ddof=1)/np.sqrt(n) if n>1 else float('nan')
        lines.append(f"{g}: rho={vals.mean():+.2f}+/-{sem:.2f} (N={n})")
lines.append("")
lines.append("By region (mean rho, N mice):")
for key in sorted(agg_rho_summary):
    mean, sem, n, _ = agg_rho_summary[key]
    lines.append(f"  {key[0]} {key[1]}: {mean:+.2f} (N={n})")
lines.append("")
lines.append("Positive rho = response grows")
lines.append("with stronger sensory evidence")
axt.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

fig.suptitle("Bulk striatal neural psychometric: does change-evoked D1/D2 response scale with evidence?",
             fontsize=13, y=0.98)
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print("\nSAVED:", OUT_PNG)
print("EXISTS:", os.path.exists(OUT_PNG))
