"""int_reward: REWARD signal on good intersectional D1 cells.

Lick-aligned. Hit (rewarded) vs FA (unrewarded). Post-lick window.
Is there a Hit>FA post-lick reward-like signal in cortically-innervated D1?
Quantify post-lick AUC (0,1.5s) Hit vs FA per cell (subject x roi, good fiber only).

n=1 mouse per region -> DESCRIPTIVE / within-animal. Cells = subject x roi.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/good_d1_extract.pkl"
OUT_DIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
OUT_PNG = os.path.join(OUT_DIR, "int_reward.png")
os.makedirs(OUT_DIR, exist_ok=True)

with open(PKL, "rb") as f:
    d = pickle.load(f)

meta = d["lick"]["meta"].reset_index(drop=True)
traces = d["lick"]["traces"]
t = d["lick"]["time"]

# good fiber only (drops BG_028 G2 weak)
good = meta["fiber_quality"] == "good"
meta = meta[good].reset_index(drop=True)
traces = traces[good.values]

# region / cell-input labels
REGION = {"BG_027": "VMS", "BG_028": "DMS"}
INPUT = {"G0": "ipsi(IT+PT)", "G2": "contra(IT-only)"}

# windows
i0 = int(np.argmin(np.abs(t - 0.0)))      # lick (t=0)
i15 = int(np.argmin(np.abs(t - 1.5)))     # +1.5 s
i2 = int(np.argmin(np.abs(t - 2.0)))      # +2.0 s (display window end)
dt = float(np.median(np.diff(t)))
print(f"i0={i0} t={t[i0]:.3f}  i15={i15} t={t[i15]:.3f}  i2={i2} t={t[i2]:.3f}  dt={dt:.4f}")


def auc(arr_2d):
    """Mean post-lick AUC (0..1.5s) per row; trapezoid in z*s, nan-safe."""
    seg = arr_2d[:, i0:i15 + 1]
    # trapezoid along time, ignoring nan by nanmean*duration
    return np.nanmean(seg, axis=1) * ((i15 - i0) * dt)


# define cells = subject x roi (good only)
cells = []
for subj in ["BG_028", "BG_027"]:  # DMS first then VMS
    for roi in ["G0", "G2"]:
        sel = (meta.subject == subj) & (meta.roi == roi)
        if sel.sum() == 0:
            continue
        cells.append((subj, roi))

print("cells:", cells)

# per-cell stats
rows = []
boot_rng = np.random.default_rng(42)
N_BOOT = 1000
cell_traces = {}  # (subj,roi) -> dict outcome -> mean trace
for (subj, roi) in cells:
    region = REGION[subj]
    inp = INPUT[roi]
    base = (meta.subject == subj) & (meta.roi == roi)
    res = {"subj": subj, "roi": roi, "region": region, "input": inp}
    cell_traces[(subj, roi)] = {}
    auc_by_outcome = {}
    for oc in ["Hit", "FA"]:
        sel = base & (meta.outcome == oc)
        idx = np.where(sel.values)[0]
        tr = traces[idx]
        cell_traces[(subj, roi)][oc] = (np.nanmean(tr, axis=0), np.nanstd(tr, axis=0) / np.sqrt(np.maximum((~np.isnan(tr)).sum(0), 1)))
        a = auc(tr)
        a = a[~np.isnan(a)]
        auc_by_outcome[oc] = a
        res[f"n_{oc}"] = len(a)
        res[f"auc_{oc}_mean"] = float(np.mean(a))
        res[f"auc_{oc}_sem"] = float(np.std(a) / np.sqrt(len(a)))
    hit = auc_by_outcome["Hit"]
    fa = auc_by_outcome["FA"]
    diff = float(np.mean(hit) - np.mean(fa))
    res["auc_diff_Hit_minus_FA"] = diff
    # pooled-trial SD for effect size (Cohen's d on trials, descriptive)
    sp = np.sqrt((np.var(hit, ddof=1) * (len(hit) - 1) + np.var(fa, ddof=1) * (len(fa) - 1)) / (len(hit) + len(fa) - 2))
    res["cohens_d"] = float(diff / sp) if sp > 0 else np.nan
    # bootstrap CI of the diff over trials (within-animal, descriptive)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        hb = boot_rng.choice(hit, size=len(hit), replace=True)
        fb = boot_rng.choice(fa, size=len(fa), replace=True)
        boot[b] = hb.mean() - fb.mean()
    res["diff_ci_lo"] = float(np.percentile(boot, 2.5))
    res["diff_ci_hi"] = float(np.percentile(boot, 97.5))
    res["boot_p_diff_gt0"] = float(np.mean(boot <= 0))  # one-sided prob diff<=0
    rows.append(res)

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 50)
print("\n=== PER-CELL POST-LICK AUC (0..1.5s), Delta z * s ===")
print(df[["subj", "roi", "region", "input", "n_Hit", "n_FA",
          "auc_Hit_mean", "auc_FA_mean", "auc_diff_Hit_minus_FA",
          "diff_ci_lo", "diff_ci_hi", "cohens_d", "boot_p_diff_gt0"]].to_string(index=False))

# ---------------- FIGURE ----------------
ncell = len(cells)
fig = plt.figure(figsize=(5 * ncell, 8.5))
gs = fig.add_gridspec(2, ncell, height_ratios=[1.5, 1.0], hspace=0.38, wspace=0.30)

COL = {"Hit": "#2c7fb8", "FA": "#d95f02"}
disp0, disp1 = int(np.argmin(np.abs(t + 0.5))), i2  # show -0.5 .. +2.0

for ci, (subj, roi) in enumerate(cells):
    region = REGION[subj]
    inp = INPUT[roi]
    ax = fig.add_subplot(gs[0, ci])
    for oc in ["Hit", "FA"]:
        mu, sem = cell_traces[(subj, roi)][oc]
        ax.plot(t[disp0:disp1 + 1], mu[disp0:disp1 + 1], color=COL[oc], lw=2,
                label=f"{oc} (n={int(df.loc[(df.subj==subj)&(df.roi==roi), f'n_{oc}'].values[0])})")
        ax.fill_between(t[disp0:disp1 + 1], (mu - sem)[disp0:disp1 + 1], (mu + sem)[disp0:disp1 + 1],
                        color=COL[oc], alpha=0.22, lw=0)
    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.7)
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5)
    ax.axvspan(0, 1.5, color="gold", alpha=0.10, lw=0)
    ax.set_title(f"{subj}  D1.{region}  {roi} {inp}", fontsize=10)
    ax.set_xlabel("Time from lick (s)")
    if ci == 0:
        ax.set_ylabel("Delta z-dF/F (baseline-sub)")
    ax.legend(fontsize=8, loc="upper right", frameon=False)

# bottom: AUC bars per cell
ax2 = fig.add_subplot(gs[1, :])
xlabels = []
x = 0
width = 0.36
for (subj, roi) in cells:
    r = df[(df.subj == subj) & (df.roi == roi)].iloc[0]
    ax2.bar(x - width / 2, r.auc_Hit_mean, width, yerr=r.auc_Hit_sem, color=COL["Hit"],
            capsize=3, label="Hit" if x == 0 else None)
    ax2.bar(x + width / 2, r.auc_FA_mean, width, yerr=r.auc_FA_sem, color=COL["FA"],
            capsize=3, label="FA" if x == 0 else None)
    # annotate diff + bootstrap p
    top = max(r.auc_Hit_mean + r.auc_Hit_sem, r.auc_FA_mean + r.auc_FA_sem)
    ax2.text(x, top + 0.02, f"d={r.auc_diff_Hit_minus_FA:+.3f}\np={r.boot_p_diff_gt0:.3f}",
             ha="center", va="bottom", fontsize=8)
    xlabels.append(f"{subj}\nD1.{REGION[subj]}\n{roi} {INPUT[roi].split('(')[1][:-1]}")
    x += 1
ax2.set_xticks(range(len(cells)))
ax2.set_xticklabels(xlabels, fontsize=8)
ax2.axhline(0, color="grey", lw=0.6)
ax2.set_ylabel("Post-lick AUC 0..1.5s\n(Delta z * s)")
ax2.set_title("Reward-like signal: post-lick AUC, Hit (rewarded) vs FA (unrewarded)  [within-animal, descriptive; n=1 mouse/region]", fontsize=9)
ax2.legend(fontsize=9, frameon=False)

fig.suptitle("Intersectional good D1 (6f, MOs-recipient): post-lick reward signal  (Hit>FA?)", fontsize=12, y=0.995)
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print("\nSAVED:", OUT_PNG)
print("exists:", os.path.exists(OUT_PNG))
