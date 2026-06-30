"""int_psychometric (good D1 intersectional, descriptive, n=1 mouse/cell).

NEURAL PSYCHOMETRIC: change-aligned Hit trials, fiber_quality=='good'.
Per cell (subject x roi/hemisphere) compute peak Delta z in (0, 1.5s) per trial,
then mean +/- SEM across trials per change_size, and Spearman rho (peak vs change_size)
on go trials (change_size > 1.0). Does the cortically-innervated D1 change response
scale with sensory evidence?
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PKL = r'e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/good_d1_extract.pkl'
OUTDIR = r'e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore'
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG = os.path.join(OUTDIR, 'int_psychometric.png')

PEAK_WIN = (0.0, 1.5)   # s post-change for peak Delta z
GO_SIZES = [1.25, 1.35, 1.5, 2.0, 4.0]
ALL_SIZES = [1.0] + GO_SIZES

with open(PKL, 'rb') as f:
    d = pickle.load(f)

meta = d['change']['meta'].reset_index(drop=True)
traces = d['change']['traces']
time = d['change']['time']

# GOOD fiber + Hit only
mask = (meta['fiber_quality'] == 'good') & (meta['outcome'] == 'Hit')
idx = np.where(mask.values)[0]
meta_g = meta.loc[idx].reset_index(drop=True)
tr_g = traces[idx, :]
print('Good Hit trials:', len(meta_g))

# peak Delta z in window (signed abs-max to allow suppression, but Hit expect activation)
wmask = (time >= PEAK_WIN[0]) & (time <= PEAK_WIN[1])
seg = tr_g[:, wmask]
# signed abs-max per trial (NaN-safe: all-NaN rows -> NaN peak)
allnan = np.isnan(seg).all(axis=1)
abs_seg = np.where(np.isnan(seg), -np.inf, np.abs(seg))
amax_i = np.argmax(abs_seg, axis=1)
peak = seg[np.arange(seg.shape[0]), amax_i]
peak[allnan] = np.nan
print('Trials with all-NaN peak window (dropped from stats):', int(allnan.sum()))
meta_g = meta_g.copy()
meta_g['peak'] = peak

# define cells: subject x roi (hemisphere distinguishes BG_027 G0/G2)
meta_g['cell'] = meta_g['subject'] + ' ' + d['good_mice'][meta_g['subject'].iloc[0]] if False else None
def cell_label(row):
    region = d['good_mice'][row['subject']]
    return f"{row['subject']} {region} {row['roi']} ({row['hemisphere']})"
meta_g['cell'] = meta_g.apply(cell_label, axis=1)
cells = sorted(meta_g['cell'].unique())
print('Cells:', cells)

# colormap for change sizes
cmap = plt.cm.viridis
size_colors = {s: cmap(i / (len(ALL_SIZES) - 1)) for i, s in enumerate(ALL_SIZES)}

fig, axes = plt.subplots(2, len(cells), figsize=(5.0 * len(cells), 8.4))
if len(cells) == 1:
    axes = axes.reshape(2, 1)

summary_rows = []
for ci, cell in enumerate(cells):
    cm = meta_g[meta_g['cell'] == cell]
    # ---- Top: mean trace per change_size ----
    axt = axes[0, ci]
    sub_idx_in_g = cm.index.values  # index into meta_g/tr_g
    for s in ALL_SIZES:
        sel = cm[cm['change_size'] == s]
        if len(sel) < 3:
            continue
        rows = sel.index.values
        mtr = np.nanmean(tr_g[rows, :], axis=0)
        axt.plot(time, mtr, color=size_colors[s], lw=1.6,
                 label=f"{s:g}x (n={len(sel)})",
                 ls='--' if s == 1.0 else '-')
    axt.axvline(0, color='k', lw=0.8, ls=':')
    axt.axvspan(PEAK_WIN[0], PEAK_WIN[1], color='gray', alpha=0.10)
    axt.set_xlim(-1, 3)
    axt.set_title(cell, fontsize=10)
    axt.set_xlabel('Time from change (s)')
    axt.set_ylabel('Delta z (baseline-sub)')
    axt.legend(fontsize=7, frameon=False, ncol=2)

    # ---- Bottom: neural psychometric (mean peak +/- SEM vs change_size) ----
    axb = axes[1, ci]
    xs, ys, es, ns = [], [], [], []
    for s in ALL_SIZES:
        sel = cm[cm['change_size'] == s]
        if len(sel) < 3:
            continue
        v = sel['peak'].values
        xs.append(s); ys.append(np.nanmean(v))
        es.append(np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))))
        ns.append(len(sel))
    xs = np.array(xs); ys = np.array(ys); es = np.array(es)
    # go-only points (>1.0) for the curve + stat
    go_m = np.array([x > 1.0 for x in xs])
    # plot catch point separately
    if (~go_m).any():
        axb.errorbar(xs[~go_m], ys[~go_m], yerr=es[~go_m], fmt='s', color='crimson',
                     ms=8, capsize=3, label='catch 1.0x (SDT-FA lick)')
    axb.errorbar(xs[go_m], ys[go_m], yerr=es[go_m], fmt='o-', color='navy',
                 ms=7, capsize=3, lw=1.8, label='go (Hit)')
    # Spearman on go trials (trial-level)
    go_trials = cm[cm['change_size'] > 1.0]
    rho, p = spearmanr(go_trials['change_size'].values, go_trials['peak'].values,
                       nan_policy='omit')
    n_go_valid = int(np.sum(~np.isnan(go_trials['peak'].values)))
    axb.set_xscale('log')
    axb.set_xticks(ALL_SIZES)
    axb.set_xticklabels([f'{s:g}' for s in ALL_SIZES], fontsize=8)
    axb.minorticks_off()
    axb.set_xlabel('change_size (TF ratio, log)')
    axb.set_ylabel('Peak Delta z in (0,1.5s)')
    axb.set_title(f"Spearman rho={rho:.3f}, p={p:.1e}\n(go trials, n={n_go_valid})",
                  fontsize=9)
    axb.legend(fontsize=7, frameon=False, loc='best')
    axb.grid(alpha=0.25)

    print(f"\n=== {cell} ===")
    print(f"  Spearman (go, trial-level) rho={rho:.4f} p={p:.3e} n_go={n_go_valid}")
    for s, y, e, n in zip(xs, ys, es, ns):
        print(f"    cs={s:>4g}  meanPeak={y:+.4f}  SEM={e:.4f}  n={n}")
    summary_rows.append(dict(cell=cell, rho=rho, p=p, n_go=len(go_trials),
                             peak_4x=ys[xs == 4.0][0] if (xs == 4.0).any() else np.nan,
                             peak_125x=ys[xs == 1.25][0] if (xs == 1.25).any() else np.nan))

fig.suptitle('Intersectional good-D1 neural psychometric: change-aligned Hit peak vs evidence (DESCRIPTIVE, n=1 mouse/cell)',
             fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUTPNG, dpi=130)
print('\nSaved:', OUTPNG)

sdf = pd.DataFrame(summary_rows)
print('\nSUMMARY:')
print(sdf.to_string(index=False))
# fold-change 1.25x -> 4x
for _, r in sdf.iterrows():
    if not np.isnan(r['peak_125x']) and r['peak_125x'] != 0:
        print(f"  {r['cell']}: peak 1.25x={r['peak_125x']:+.3f} -> 4x={r['peak_4x']:+.3f} "
              f"(delta={r['peak_4x']-r['peak_125x']:+.3f})")
