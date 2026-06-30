"""sens_mixedeffects: SENSITIVITY mixed-effects, D1>D2 detection selectivity, REGIONS SEPARATE.

BULK rows only. MixedLM detect_sel ~ C(genotype)*C(region), groups=mouse (random intercept).
Robustness: per-mouse-per-region mean detect_sel, permutation D1-vs-D2 within DMS, within VMS.
Second sensitivity readout: evidence_slope D1 vs D2 (per-mouse).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

CSV = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/per_session_metrics.csv"
OUT = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/narrative"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
b = df[df.cohort == "bulk"].copy()

# ---- per-mouse-per-region means (robustness unit = MOUSE) ----
def per_mouse_region(metric):
    d = b.dropna(subset=[metric])
    return d.groupby(["genotype", "region", "mouse"])[metric].mean().reset_index()

pmr_sel = per_mouse_region("detect_sel")
pmr_slope = per_mouse_region("evidence_slope")

print("=== per-mouse-per-region detect_sel ===")
print(pmr_sel.to_string(index=False))

def perm_test(x, y, n=10000):
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = np.mean(x) - np.mean(y)
    pool = np.concatenate([x, y]); nx = len(x)
    cnt = 0
    for _ in range(n):
        np.random.shuffle(pool)
        if abs(np.mean(pool[:nx]) - np.mean(pool[nx:])) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)

def cliffs_delta(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    gt = sum(1 for a in x for c in y if a > c)
    lt = sum(1 for a in x for c in y if a < c)
    return (gt - lt) / (len(x) * len(y))

# ---- robustness: permutation D1 vs D2 within DMS and VMS ----
print("\n=== Robustness: per-mouse permutation D1 vs D2 (detect_sel) ===")
robust = {}
for reg in ["DMS", "VMS", "VLS"]:
    d1 = pmr_sel[(pmr_sel.region == reg) & (pmr_sel.genotype == "D1")].detect_sel.values
    d2 = pmr_sel[(pmr_sel.region == reg) & (pmr_sel.genotype == "D2")].detect_sel.values
    line = f"{reg}: D1 n={len(d1)} mean={np.mean(d1):.4f}  D2 n={len(d2)} mean={np.mean(d2):.4f}"
    if len(d1) >= 2 and len(d2) >= 2:
        diff, p = perm_test(d1, d2)
        delta = cliffs_delta(d1, d2)
        line += f"  | D1-D2 diff={diff:.4f}  perm p={p:.4f}  Cliff's d={delta:.3f}"
    else:
        line += "  | DESCRIPTIVE ONLY (<2 mice/genotype)"
    robust[reg] = line
    print(line)

print("\n=== evidence_slope per-mouse D1 vs D2 (sensitivity readout #2) ===")
slope_res = {}
for reg in ["DMS", "VMS", "VLS"]:
    d1 = pmr_slope[(pmr_slope.region == reg) & (pmr_slope.genotype == "D1")].evidence_slope.values
    d2 = pmr_slope[(pmr_slope.region == reg) & (pmr_slope.genotype == "D2")].evidence_slope.values
    line = f"{reg}: D1 n={len(d1)} mean={np.nanmean(d1) if len(d1) else np.nan:.4f}  D2 n={len(d2)} mean={np.nanmean(d2) if len(d2) else np.nan:.4f}"
    if len(d1) >= 2 and len(d2) >= 2:
        diff, p = perm_test(d1, d2)
        line += f"  | diff={diff:.4f}  perm p={p:.4f}"
    else:
        line += "  | DESCRIPTIVE ONLY"
    slope_res[reg] = line
    print(line)

# ---- MixedLM: detect_sel ~ C(genotype)*C(region), groups=mouse ----
import statsmodels.formula.api as smf
# formal model excludes VLS (~1 mouse/genotype)
md = b.dropna(subset=["detect_sel"])
md = md[md.region.isin(["DMS", "VMS"])].copy()
md["genotype"] = pd.Categorical(md["genotype"], categories=["D2", "D1"])  # D2 ref => positive = D1>D2
md["region"] = pd.Categorical(md["region"], categories=["DMS", "VMS"])

print("\n=== MixedLM detect_sel ~ C(genotype)*C(region), groups=mouse (VLS excluded) ===")
print(f"N sessions={len(md)}, N mice={md.mouse.nunique()}")
model = smf.mixedlm("detect_sel ~ C(genotype) * C(region)", md, groups=md["mouse"])
res = model.fit(reml=False, method="lbfgs")
print(res.summary())

params = res.params
pvals = res.pvalues
# overall D1 effect (at DMS reference) and within-VMS via combination
geno_term = "C(genotype)[T.D1]"
inter_term = "C(genotype)[T.D1]:C(region)[T.VMS]"
b_d1_dms = params.get(geno_term, np.nan)        # D1-D2 within DMS
p_d1_dms = pvals.get(geno_term, np.nan)
b_inter = params.get(inter_term, np.nan)
b_d1_vms = b_d1_dms + b_inter                     # D1-D2 within VMS
print(f"\nModel D1-D2 within DMS: beta={b_d1_dms:.4f}  p={p_d1_dms:.4f}")
print(f"Model interaction (D1-D2 VMS minus DMS): beta={b_inter:.4f}  p={pvals.get(inter_term, np.nan):.4f}")
print(f"Model implied D1-D2 within VMS: beta={b_d1_vms:.4f}")

# Refit with VMS as reference to get clean within-VMS contrast + p
md_v = md.copy()
md_v["region"] = pd.Categorical(md_v["region"].astype(str), categories=["VMS", "DMS"])
res_v = smf.mixedlm("detect_sel ~ C(genotype) * C(region)", md_v, groups=md_v["mouse"]).fit(reml=False, method="lbfgs")
b_d1_vms2 = res_v.params.get(geno_term, np.nan)
p_d1_vms2 = res_v.pvalues.get(geno_term, np.nan)
print(f"Model (VMS-ref) D1-D2 within VMS: beta={b_d1_vms2:.4f}  p={p_d1_vms2:.4f}")

# ---- FIGURE ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
regions = ["DMS", "VMS", "VLS"]
colors = {"D1": "#1f77b4", "D2": "#d62728"}

for ax, reg in zip(axes, regions):
    sub = pmr_sel[pmr_sel.region == reg]
    for i, geno in enumerate(["D1", "D2"]):
        vals = sub[sub.genotype == geno].detect_sel.values
        x = i + np.random.uniform(-0.06, 0.06, len(vals))
        ax.scatter(x, vals, color=colors[geno], s=80, zorder=3,
                   edgecolor="k", linewidth=0.5, label=f"{geno} (n={len(vals)})")
        if len(vals):
            m = np.mean(vals)
            ax.hlines(m, i - 0.25, i + 0.25, color=colors[geno], lw=3, zorder=4)
            if len(vals) >= 2:
                # bootstrap 95% CI
                boots = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(2000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                ax.vlines(i, lo, hi, color=colors[geno], lw=1.5, zorder=2)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["D1", "D2"])
    ax.set_title(reg + ("" if reg != "VLS" else "  (descriptive: 1 mouse/geno)"))
    ax.set_ylabel("detect_sel (Hit-Miss change peak)")
    ax.legend(fontsize=8, loc="best")
    # annotate model/perm contrast
    if reg == "DMS":
        ax.text(0.5, 0.97, f"MixedLM D1-D2 b={b_d1_dms:.3f}, p={p_d1_dms:.3f}\n{robust['DMS'].split('|')[-1].strip()}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    elif reg == "VMS":
        ax.text(0.5, 0.97, f"MixedLM D1-D2 b={b_d1_vms2:.3f}, p={p_d1_vms2:.3f}\n{robust['VMS'].split('|')[-1].strip()}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

fig.suptitle("Sensitivity: D1 vs D2 detection selectivity (detect_sel), regions separate\n"
             "BULK cohort, unit=mouse; MixedLM (random intercept by mouse) + per-mouse permutation",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outpath = os.path.join(OUT, "sens_mixedeffects.png")
fig.savefig(outpath, dpi=130)
print(f"\nSaved figure -> {outpath}")
print("FILE_EXISTS:", os.path.exists(outpath))
