"""
int_hit_vs_fa: Hit-lick vs FA-lick go-signal on good D1 fibers (intersectional 6f cohort).
Per cell (BG_028 D1.DMS G0; BG_027 D1.VMS G0 & G2), lick-aligned mean Hit vs FA trace,
SEM over sessions. Quantify pre-lick mean [-0.5, 0) Hit vs FA per cell.
n=1 mouse/cell -> DESCRIPTIVE, within-animal framing.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/good_d1_extract.pkl"
OUTDIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG = os.path.join(OUTDIR, "int_hit_vs_fa.png")

PRE_WIN = (-0.5, 0.0)   # pre-lick go-ramp window
RNG = np.random.default_rng(42)

with open(PKL, "rb") as f:
    d = pickle.load(f)

lk = d["lick"]
meta = lk["meta"].reset_index(drop=True)
traces = lk["traces"]
t = np.asarray(lk["time"])
good_mice = d["good_mice"]  # subject -> region

# good fibers only
mask_good = (meta["fiber_quality"] == "good").values
meta_g = meta[mask_good].reset_index(drop=True)
tr_g = traces[mask_good]

# region label per row
region_map = {("BG_028", "G0"): "DMS", ("BG_027", "G0"): "VMS", ("BG_027", "G2"): "VMS"}

# Define the three good cells (subject, roi)
cells = [
    ("BG_028", "G0", "BG_028  D1.DMS  G0"),
    ("BG_027", "G0", "BG_027  D1.VMS  G0"),
    ("BG_027", "G2", "BG_027  D1.VMS  G2"),
]

# pre-lick window indices
pre_idx = np.where((t >= PRE_WIN[0]) & (t < PRE_WIN[1]))[0]

def session_means(meta_sub, tr_sub):
    """Return dict session_id -> mean trace (over trials in that session)."""
    out = {}
    for sid, idx in meta_sub.groupby("session_id").groups.items():
        rows = [meta_sub.index.get_loc(i) for i in idx]
        out[sid] = np.nanmean(tr_sub[rows], axis=0)
    return out

def grand_and_sem_over_sessions(sess_means):
    """sess_means: dict sid->trace. Grand mean & SEM over sessions."""
    arr = np.vstack(list(sess_means.values()))  # [nsess, T]
    gm = np.nanmean(arr, axis=0)
    nsess = arr.shape[0]
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(nsess) if nsess > 1 else np.zeros_like(gm)
    return gm, sem, nsess, arr

colors = {"Hit": "#1b7837", "FA": "#762a83"}  # green Hit, purple FA

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
summary_rows = []

for ci, (subj, roi, label) in enumerate(cells):
    cell_mask = (meta_g["subject"] == subj) & (meta_g["roi"] == roi)
    meta_c = meta_g[cell_mask].copy()
    tr_c = tr_g[cell_mask.values]
    meta_c = meta_c.reset_index(drop=True)

    ax = axes[0, ci]
    cell_pre = {}
    cell_sessarr = {}
    for oc in ["Hit", "FA"]:
        sel = (meta_c["outcome"] == oc).values
        if sel.sum() == 0:
            continue
        meta_o = meta_c[sel].reset_index(drop=True)
        tr_o = tr_c[sel]
        sm = session_means(meta_o, tr_o)
        gm, sem, nsess, arr = grand_and_sem_over_sessions(sm)
        ax.plot(t, gm, color=colors[oc], lw=2, label=f"{oc} (n={nsess} sess, {sel.sum()} tr)")
        ax.fill_between(t, gm - sem, gm + sem, color=colors[oc], alpha=0.22, lw=0)
        # per-session pre-lick mean
        pre_per_sess = np.nanmean(arr[:, pre_idx], axis=1)
        cell_pre[oc] = pre_per_sess
        cell_sessarr[oc] = sm

    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.7)
    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.axvspan(PRE_WIN[0], PRE_WIN[1], color="gold", alpha=0.15)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time from lick (s)")
    ax.set_ylabel("Delta z-dF/F" if ci == 0 else "")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-2, 3)

    # ---- bottom row: pre-lick window quantification (per-session points) ----
    axb = axes[1, ci]
    hit_pre = cell_pre.get("Hit", np.array([]))
    fa_pre = cell_pre.get("FA", np.array([]))
    # jittered scatter
    for j, (oc, vals) in enumerate([("Hit", hit_pre), ("FA", fa_pre)]):
        x = np.full(len(vals), j) + RNG.normal(0, 0.06, len(vals))
        axb.scatter(x, vals, s=14, color=colors[oc], alpha=0.55, edgecolors="none")
        if len(vals):
            mu = np.nanmean(vals)
            se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            axb.plot([j - 0.25, j + 0.25], [mu, mu], color="k", lw=2.5)
            axb.errorbar(j, mu, yerr=se, color="k", capsize=4, lw=1.5)
    axb.axhline(0, color="grey", ls=":", lw=0.8)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["Hit", "FA"])
    axb.set_title(f"Pre-lick [{PRE_WIN[0]},{PRE_WIN[1]}) per session", fontsize=10)
    axb.set_ylabel("Mean Delta z-dF/F" if ci == 0 else "")
    axb.set_xlim(-0.6, 1.6)

    # paired-by-session comparison (sessions with BOTH Hit and FA)
    sm_hit = cell_sessarr.get("Hit", {})
    sm_fa = cell_sessarr.get("FA", {})
    common = sorted(set(sm_hit.keys()) & set(sm_fa.keys()))
    paired_diff = []
    for sid in common:
        h = np.nanmean(sm_hit[sid][pre_idx])
        f = np.nanmean(sm_fa[sid][pre_idx])
        paired_diff.append(h - f)
        axb.plot([0, 1], [h, f], color="grey", alpha=0.25, lw=0.6, zorder=0)
    paired_diff = np.asarray(paired_diff)

    # stats: Wilcoxon signed-rank on paired sessions (descriptive)
    wil_p = np.nan
    if len(paired_diff) >= 5:
        try:
            from scipy.stats import wilcoxon
            stat, wil_p = wilcoxon(paired_diff)
        except Exception:
            wil_p = np.nan

    hit_mu = np.nanmean(hit_pre) if len(hit_pre) else np.nan
    fa_mu = np.nanmean(fa_pre) if len(fa_pre) else np.nan
    # Cohen's d on paired diff
    d_paired = (np.nanmean(paired_diff) / np.nanstd(paired_diff, ddof=1)) if len(paired_diff) > 1 and np.nanstd(paired_diff, ddof=1) > 0 else np.nan

    summary_rows.append(dict(
        cell=label, region=region_map[(subj, roi)],
        n_sess_hit=len(hit_pre), n_sess_fa=len(fa_pre), n_sess_paired=len(paired_diff),
        hit_pre_mean=hit_mu, fa_pre_mean=fa_mu,
        diff_hit_minus_fa=hit_mu - fa_mu if not (np.isnan(hit_mu) or np.isnan(fa_mu)) else np.nan,
        paired_diff_mean=np.nanmean(paired_diff) if len(paired_diff) else np.nan,
        cohens_d_paired=d_paired, wilcoxon_p=wil_p,
    ))
    txt = f"Hit={hit_mu:.3f}  FA={fa_mu:.3f}\nHit-FA(paired)={np.nanmean(paired_diff):.3f}\nd={d_paired:.2f}  p={wil_p:.3g}"
    axb.text(0.97, 0.03, txt, transform=axb.transAxes, ha="right", va="bottom",
             fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

fig.suptitle("Intersectional 6f D1 (good fibers): Hit-lick vs FA-lick go-signal\n"
             "Lick-aligned mean +/- SEM over sessions | n=1 mouse/cell -> DESCRIPTIVE (within-animal)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUTPNG, dpi=130)
print("SAVED:", OUTPNG)

sdf = pd.DataFrame(summary_rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n===== PRE-LICK [-0.5,0) SUMMARY (per cell) =====")
print(sdf.to_string(index=False))

print("\n===== INTERPRETATION NUMBERS =====")
for _, r in sdf.iterrows():
    sign = "Hit>FA" if r["diff_hit_minus_fa"] > 0 else "FA>Hit"
    print(f"{r['cell']}: Hit={r['hit_pre_mean']:.3f} FA={r['fa_pre_mean']:.3f} "
          f"-> {sign} by {abs(r['diff_hit_minus_fa']):.3f} | paired d={r['cohens_d_paired']:.2f} "
          f"p={r['wilcoxon_p']:.3g} (n_paired_sess={r['n_sess_paired']})")

# overall: are both Hit and FA positive (shared go-ramp)?
print("\nShared go-ramp check (both positive pre-lick?):")
for _, r in sdf.iterrows():
    both_pos = (r["hit_pre_mean"] > 0) and (r["fa_pre_mean"] > 0)
    print(f"  {r['cell']}: Hit>0={r['hit_pre_mean']>0}  FA>0={r['fa_pre_mean']>0}  both_positive={both_pos}")
