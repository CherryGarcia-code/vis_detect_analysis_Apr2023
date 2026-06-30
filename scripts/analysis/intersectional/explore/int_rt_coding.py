"""int_rt_coding: RT coding on good D1 intersectional MOs-recipient cohort (6f).

Lick-aligned FA and Hit trials (fiber_quality=='good'). Within each cell x outcome,
split trials into RT terciles; plot mean lick-aligned trace per tercile. Test whether
pre-lick ramp amplitude scales with RT. Per-trial scatter of pre-lick mean [-0.5,0)s vs
reaction_time with Spearman (DESCRIPTIVE; n=1 mouse per cell; pseudo-replication caveat).

Cohort: BG_027 (D1.VMS, G0 ipsi IT+PT, G2 contra IT-only), BG_028 (D1.DMS, G0 ipsi).
Traces are session-z-scored dF/F, baseline-subtracted (Delta z). NOT pooled with bulk-8m.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/good_d1_extract.pkl"
OUT_DIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
OUT_PNG = os.path.join(OUT_DIR, "int_rt_coding.png")
os.makedirs(OUT_DIR, exist_ok=True)

PRE_LICK_WIN = (-0.5, 0.0)   # pre-lick ramp window, seconds rel to lick
BASE_WIN = (-2.0, -1.5)      # for sanity (already baseline-subtracted but reaffirm)

# ---- load ----
with open(PKL, "rb") as f:
    d = pickle.load(f)

lk = d["lick"]
t = np.asarray(lk["time"], dtype=float)
meta = lk["meta"].reset_index(drop=True)
traces = np.asarray(lk["traces"], dtype=np.float32)
assert traces.shape[0] == len(meta)

good = meta["fiber_quality"] == "good"
meta = meta[good].reset_index(drop=True)
traces = traces[good.values]

REGION = {"BG_027": "VMS", "BG_028": "DMS"}
def cell_label(r):
    return f'{r["subject"]} {REGION[r["subject"]]} {r["roi"]}'
meta["cell"] = meta.apply(cell_label, axis=1)

# pre-lick window indices
pre_mask = (t >= PRE_LICK_WIN[0]) & (t < PRE_LICK_WIN[1])
print("pre-lick window bins:", pre_mask.sum(), "time", t[pre_mask][0], "..", t[pre_mask][-1])

cells = ["BG_027 VMS G0", "BG_027 VMS G2", "BG_028 DMS G0"]
outcomes = ["FA", "Hit"]
TERC_LABELS = ["short", "mid", "long"]
TERC_COLORS = {"short": "#2c7fb8", "mid": "#7fcdbb", "long": "#d95f0e"}

# Storage for headline stats
results = {}  # (cell, outcome) -> dict

# ---- compute per-trial pre-lick mean ----
prelick_mean_all = traces[:, pre_mask].mean(axis=1)
meta = meta.copy()
meta["prelick_mean"] = prelick_mean_all

# ---- figure: rows = cells, cols = [FA terciles, Hit terciles, FA scatter, Hit scatter] ----
nrows = len(cells)
fig, axes = plt.subplots(nrows, 4, figsize=(18, 4.2 * nrows), squeeze=False)

for ri, cell in enumerate(cells):
    cm = meta[meta["cell"] == cell]
    for ci, oc in enumerate(outcomes):
        ax = axes[ri][ci]
        sub = cm[(cm["outcome"] == oc) & cm["reaction_time"].notna()].copy()
        # tercile split by RT within cell x outcome
        rts = sub["reaction_time"].values
        if len(sub) < 9:
            ax.set_title(f"{cell}\n{oc}: n={len(sub)} (too few)")
            continue
        q1, q2 = np.quantile(rts, [1/3, 2/3])
        def terc(rt):
            if rt <= q1: return "short"
            if rt <= q2: return "mid"
            return "long"
        sub["terc"] = [terc(x) for x in rts]
        idx = sub.index.values
        for tl in TERC_LABELS:
            rows = sub[sub["terc"] == tl]
            if len(rows) == 0:
                continue
            tr = traces[rows.index.values]
            mean = np.nanmean(tr, axis=0)
            sem = np.nanstd(tr, axis=0) / np.sqrt(max(len(rows), 1))
            ax.plot(t, mean, color=TERC_COLORS[tl], lw=1.8,
                    label=f"{tl} RT (med {rows['reaction_time'].median():.2f}s, n={len(rows)})")
            ax.fill_between(t, mean - sem, mean + sem, color=TERC_COLORS[tl], alpha=0.18)
        ax.axvline(0, color="k", lw=1.0, ls="--")
        ax.axvspan(PRE_LICK_WIN[0], PRE_LICK_WIN[1], color="grey", alpha=0.10)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlim(-2, 2)
        ax.set_title(f"{cell}  |  {oc}  (lick-aligned)")
        ax.set_xlabel("time from lick (s)")
        ax.set_ylabel("Delta z-dF/F")
        ax.legend(fontsize=7, loc="upper left")

        # tercile pre-lick amplitudes
        terc_amp = {}
        for tl in TERC_LABELS:
            rows = sub[sub["terc"] == tl]
            terc_amp[tl] = float(np.nanmean(traces[rows.index.values][:, pre_mask])) if len(rows) else np.nan
        results[(cell, oc)] = {
            "n": len(sub),
            "terc_amp": terc_amp,
            "q1": float(q1), "q2": float(q2),
        }

    # ---- scatter columns (3=FA, 4=Hit) ----
    for ci2, oc in enumerate(outcomes):
        ax = axes[ri][2 + ci2]
        sub = cm[(cm["outcome"] == oc) & cm["reaction_time"].notna()].copy()
        if len(sub) < 5:
            ax.set_title(f"{cell} {oc} scatter: n={len(sub)}")
            continue
        x = sub["reaction_time"].values
        y = sub["prelick_mean"].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        rho, p = spearmanr(x, y)
        ax.scatter(x, y, s=5, alpha=0.25, color=TERC_COLORS["mid"], edgecolors="none")
        # binned median trend
        try:
            nb = 8
            qs = np.quantile(x, np.linspace(0, 1, nb + 1))
            qs = np.unique(qs)
            bx, by = [], []
            for i in range(len(qs) - 1):
                m = (x >= qs[i]) & (x <= qs[i + 1]) if i == len(qs) - 2 else (x >= qs[i]) & (x < qs[i + 1])
                if m.sum() >= 3:
                    bx.append(np.median(x[m]))
                    by.append(np.median(y[m]))
            ax.plot(bx, by, "-o", color="#d95f0e", lw=2, ms=4, label="binned median")
        except Exception as e:
            print("bin trend err", cell, oc, e)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{cell} {oc}: pre-lick vs RT\nSpearman rho={rho:.3f}, p={p:.1e}, n={len(x)}")
        ax.set_xlabel("reaction_time (s)")
        ax.set_ylabel("pre-lick mean [-0.5,0)s (Delta z)")
        ax.legend(fontsize=7)
        results.setdefault((cell, oc), {})["rho"] = float(rho)
        results[(cell, oc)]["p"] = float(p)
        results[(cell, oc)]["n_scatter"] = int(len(x))

fig.suptitle("RT coding, good D1 intersectional MOs-recipient cohort (6f) -- DESCRIPTIVE, n=1 mouse/cell, pseudo-replication caveat\n"
             "lick-aligned FA & Hit, trials split by RT tercile; pre-lick ramp window shaded [-0.5,0)s",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=130)
print("SAVED:", OUT_PNG)

# ---- print key numbers ----
print("\n===== KEY NUMBERS =====")
for (cell, oc), r in sorted(results.items()):
    line = f"{cell} | {oc}: n={r.get('n','?')}"
    if "terc_amp" in r:
        ta = r["terc_amp"]
        line += (f" | prelick amp short={ta['short']:.3f} mid={ta['mid']:.3f} long={ta['long']:.3f}"
                 f" | long-short delta={ta['long']-ta['short']:+.3f}"
                 f" | RT cuts q1={r['q1']:.2f} q2={r['q2']:.2f}")
    if "rho" in r:
        line += f" | Spearman rho={r['rho']:+.3f} p={r['p']:.2e} (n_trials={r['n_scatter']})"
    print(line)

# Direction summary across cells for FA pre-lick ramp vs RT
print("\n--- FA pre-lick ramp vs RT (per cell, descriptive) ---")
for cell in cells:
    r = results.get((cell, "FA"), {})
    if "rho" in r and "terc_amp" in r:
        ta = r["terc_amp"]
        print(f"  {cell}: rho={r['rho']:+.3f} p={r['p']:.1e}; long-short tercile delta={ta['long']-ta['short']:+.3f} Delta z")
print("\n--- Hit pre-lick ramp vs RT (per cell, descriptive) ---")
for cell in cells:
    r = results.get((cell, "Hit"), {})
    if "rho" in r and "terc_amp" in r:
        ta = r["terc_amp"]
        print(f"  {cell}: rho={r['rho']:+.3f} p={r['p']:.1e}; long-short tercile delta={ta['long']-ta['short']:+.3f} Delta z")
