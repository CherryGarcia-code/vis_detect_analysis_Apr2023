"""TWO-AXIS DISSOCIATION (narrative centerpiece).

BULK only. Per mouse, primary region = DMS where available else VMS.
Aggregate sessions to per-mouse:
  SENSITIVITY axis  = mean detect_sel (neural Hit-minus-Miss change peak)
  IMPULSIVITY axis  = mean prefa_ramp (neural pre-FA-lick ramp)
  behavioral: mean d_prime, mean fa_rate_beh
Plot each mouse in sensitivity-vs-impulsivity space, colored by genotype.
Test D1 vs D2 on each axis via permutation (N=mice).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/per_session_metrics.csv"
OUTDIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/narrative"
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG = os.path.join(OUTDIR, "two_axis_dissociation.png")

RNG = np.random.default_rng(42)

def perm_test(a, b, n=10000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    obs = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pooled)
        d = np.mean(pooled[:na]) - np.mean(pooled[na:])
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)

def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp

df = pd.read_csv(CSV)
bulk = df[df["cohort"] == "bulk"].copy()

# Primary region per mouse: DMS where available else VMS
rows = []
for mouse, g in bulk.groupby("mouse"):
    regions = set(g["region"].unique())
    if "DMS" in regions:
        prim = "DMS"
    elif "VMS" in regions:
        prim = "VMS"
    else:
        continue  # VLS-only -> skip (descriptive only)
    sub = g[g["region"] == prim]
    genotype = sub["genotype"].iloc[0]
    rows.append({
        "mouse": mouse,
        "genotype": genotype,
        "region": prim,
        "n_sessions": len(sub),
        "sensitivity": sub["detect_sel"].mean(skipna=True),
        "impulsivity_coding": sub["prefa_ramp"].mean(skipna=True),
        "d_prime": sub["d_prime"].mean(skipna=True),
        "fa_rate_beh": sub["fa_rate_beh"].mean(skipna=True),
        "n_detect_sel": sub["detect_sel"].notna().sum(),
        "n_prefa": sub["prefa_ramp"].notna().sum(),
    })

per_mouse = pd.DataFrame(rows).sort_values(["genotype", "mouse"]).reset_index(drop=True)
print("=== PER-MOUSE TABLE (BULK, primary region) ===")
print(per_mouse.to_string(index=False))

d1 = per_mouse[per_mouse["genotype"] == "D1"]
d2 = per_mouse[per_mouse["genotype"] == "D2"]
print(f"\nN mice: D1={len(d1)} ({list(d1['mouse'])}), D2={len(d2)} ({list(d2['mouse'])})")

results = {}
for axis, label in [("sensitivity", "SENSITIVITY (detect_sel)"),
                    ("impulsivity_coding", "IMPULSIVITY-coding (prefa_ramp)"),
                    ("d_prime", "behavioral d_prime"),
                    ("fa_rate_beh", "behavioral fa_rate")]:
    obs, p = perm_test(d1[axis].values, d2[axis].values)
    dd = cohens_d(d1[axis].values, d2[axis].values)
    results[axis] = (obs, p, dd)
    print(f"\n[{label}] D1 mean={np.nanmean(d1[axis]):.4f} (n={d1[axis].notna().sum()}), "
          f"D2 mean={np.nanmean(d2[axis]):.4f} (n={d2[axis].notna().sum()})")
    print(f"    D1-D2 diff={obs:.4f}, perm p={p:.4f}, Cohen's d={dd:.3f}")

# ── FIGURE ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"D1": "#1f77b4", "D2": "#d62728"}

# Panel A: neural two-axis space
ax = axes[0]
for _, r in per_mouse.iterrows():
    ax.scatter(r["sensitivity"], r["impulsivity_coding"], s=180,
               color=colors.get(r["genotype"], "gray"), edgecolor="k", zorder=3)
    ax.annotate(r["mouse"].replace("BG_", ""), (r["sensitivity"], r["impulsivity_coding"]),
                fontsize=8, ha="center", va="center", color="white", zorder=4, weight="bold")
for gt in ["D1", "D2"]:
    sub = per_mouse[per_mouse["genotype"] == gt]
    if len(sub):
        ax.scatter(np.nanmean(sub["sensitivity"]), np.nanmean(sub["impulsivity_coding"]),
                   marker="X", s=320, color=colors[gt], edgecolor="k", linewidth=2,
                   zorder=5, label=f"{gt} mean (n={len(sub)})")
ax.set_xlabel("SENSITIVITY axis\nmean detect_sel (neural Hit-Miss change peak)")
ax.set_ylabel("IMPULSIVITY-coding axis\nmean prefa_ramp (neural pre-FA ramp)")
ax.set_title("Neural two-axis dissociation (BULK, per mouse)")
ax.axhline(0, color="gray", lw=0.6, ls=":"); ax.axvline(0, color="gray", lw=0.6, ls=":")
ax.legend(frameon=False, loc="best")

# annotate stats
s_obs, s_p, s_d = results["sensitivity"]
i_obs, i_p, i_d = results["impulsivity_coding"]
ax.text(0.02, 0.98,
        f"Sens: D1-D2={s_obs:+.3f}, p={s_p:.3f}, d={s_d:.2f}\n"
        f"Impuls: D1-D2={i_obs:+.3f}, p={i_p:.3f}, d={i_d:.2f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

# Panel B: behavioral two-axis space
ax = axes[1]
for _, r in per_mouse.iterrows():
    ax.scatter(r["d_prime"], r["fa_rate_beh"], s=180,
               color=colors.get(r["genotype"], "gray"), edgecolor="k", zorder=3)
    ax.annotate(r["mouse"].replace("BG_", ""), (r["d_prime"], r["fa_rate_beh"]),
                fontsize=8, ha="center", va="center", color="white", zorder=4, weight="bold")
for gt in ["D1", "D2"]:
    sub = per_mouse[per_mouse["genotype"] == gt]
    if len(sub):
        ax.scatter(np.nanmean(sub["d_prime"]), np.nanmean(sub["fa_rate_beh"]),
                   marker="X", s=320, color=colors[gt], edgecolor="k", linewidth=2,
                   zorder=5, label=f"{gt} mean (n={len(sub)})")
ax.set_xlabel("SENSITIVITY axis (behavioral)\nmean d_prime")
ax.set_ylabel("IMPULSIVITY axis (behavioral)\nmean fa_rate_beh")
ax.set_title("Behavioral two-axis space (BULK, per mouse)")
ax.legend(frameon=False, loc="best")
d_obs, d_p, d_d = results["d_prime"]
f_obs, f_p, f_d = results["fa_rate_beh"]
ax.text(0.02, 0.98,
        f"d': D1-D2={d_obs:+.3f}, p={d_p:.3f}, d={d_d:.2f}\n"
        f"FA: D1-D2={f_obs:+.3f}, p={f_p:.3f}, d={f_d:.2f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

fig.suptitle("D1 = sensitivity / D2 = impulsivity?  Two-axis dissociation (per-mouse, BULK)",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPNG, dpi=140)
print(f"\nSaved figure: {OUTPNG}")
print(f"Figure exists: {os.path.exists(OUTPNG)}")

# Summary verdict numbers
print("\n=== VERDICT NUMBERS ===")
print(f"Sensitivity (detect_sel): D1>{('>' if s_obs>0 else '<')}D2 by {s_obs:+.4f}, p={s_p:.4f}")
print(f"Impulsivity (prefa_ramp): D1 vs D2 diff {i_obs:+.4f}, p={i_p:.4f} "
      f"(narrative predicts D2>D1 i.e. negative diff)")
