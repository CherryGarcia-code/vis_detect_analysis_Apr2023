"""IMPULSIVITY axis: neural prefa_ramp D1 vs D2 by region (bulk), + C1 brake AUROC."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
PSM = 'results/explore_cache/per_session_metrics.csv'
C1 = 'FIGURES/C1_fa_suppression_corrected/c1_auroc_stats.csv'
OUTDIR = 'FIGURES/intersectional_mos/narrative'
os.makedirs(OUTDIR, exist_ok=True)

def perm_test(a, b, n=10000):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = np.mean(a) - np.mean(b)
    pool = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(np.mean(pool[:na]) - np.mean(pool[na:])) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)

# ---- Bulk prefa_ramp per mouse per region ----
df = pd.read_csv(PSM)
bulk = df[df.cohort == 'bulk'].copy()
bulk = bulk.dropna(subset=['prefa_ramp'])
# per-mouse-per-region mean prefa_ramp
pm = (bulk.groupby(['region', 'genotype', 'mouse'])['prefa_ramp']
      .mean().reset_index())

regions = ['DMS', 'VMS', 'VLS']
print('=== BULK prefa_ramp: per-mouse-per-region mean ===')
ramp_results = {}
for reg in regions:
    sub = pm[pm.region == reg]
    d1 = sub[sub.genotype == 'D1']['prefa_ramp'].values
    d2 = sub[sub.genotype == 'D2']['prefa_ramp'].values
    print(f'\n[{reg}] D1 n={len(d1)} mice mean={np.mean(d1) if len(d1) else np.nan:.4f}; '
          f'D2 n={len(d2)} mice mean={np.mean(d2) if len(d2) else np.nan:.4f}')
    print(f'   D1 mice: {sub[sub.genotype=="D1"][["mouse","prefa_ramp"]].values.tolist()}')
    print(f'   D2 mice: {sub[sub.genotype=="D2"][["mouse","prefa_ramp"]].values.tolist()}')
    if len(d1) >= 2 and len(d2) >= 2:
        obs, p = perm_test(d1, d2)
        # rank-biserial via MWU
        from scipy.stats import mannwhitneyu
        try:
            U, _ = mannwhitneyu(d1, d2, alternative='two-sided')
            rb = 1 - 2 * U / (len(d1) * len(d2))
        except Exception:
            rb = np.nan
        print(f'   perm D1-D2 diff={obs:+.4f}, p={p:.4f}, rank-biserial={rb:+.3f}')
        ramp_results[reg] = dict(d1=d1, d2=d2, obs=obs, p=p, rb=rb)
    else:
        print('   descriptive only (insufficient mice for test)')
        ramp_results[reg] = dict(d1=d1, d2=d2, obs=np.nan, p=np.nan, rb=np.nan)

# ---- C1 brake AUROC: behavioral_fa / scheme3 ----
c1 = pd.read_csv(C1)
brake = c1[(c1.track == 'behavioral_fa') & (c1.scheme == 'scheme3')].copy()
print('\n\n=== C1 brake AUROC (behavioral_fa/scheme3): >0.5 brake, <0.5 go-ramp ===')
print('(withhold vs FA waiting-period; AUROC>0.5 = brake/suppression, <0.5 = pre-FA go-ramp)')
for pb in ['pooled', 'less', 'more']:
    print(f'\n-- prof_bin={pb} --')
    for reg in ['DMS', 'VMS', 'VLS']:
        for gt in ['D1', 'D2']:
            r = brake[(brake.region == reg) & (brake.genotype == gt) & (brake.prof_bin == pb)]
            if len(r):
                r = r.iloc[0]
                tag = 'BRAKE' if r.auroc_mean > 0.5 else 'GO-RAMP'
                sig = '*' if r.excludes_chance else ' '
                print(f'   {reg:4s} {gt}: AUROC={r.auroc_mean:.3f}{sig} n={int(r.n_mice)} -> {tag}')

# ============ FIGURE ============
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: prefa_ramp D1 vs D2 by region
ax = axes[0]
xpos = np.arange(len(regions))
w = 0.35
cols = {'D1': '#1f77b4', 'D2': '#d62728'}
for i, gt in enumerate(['D1', 'D2']):
    means, sems, offs = [], [], []
    for j, reg in enumerate(regions):
        vals = ramp_results[reg][f'd{i+1}']
        means.append(np.mean(vals) if len(vals) else np.nan)
        sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        offs.append(j + (i - 0.5) * w)
    ax.bar(offs, means, w, yerr=sems, capsize=4, color=cols[gt], alpha=0.75, label=gt)
    # scatter individual mice
    for j, reg in enumerate(regions):
        vals = ramp_results[reg][f'd{i+1}']
        if len(vals):
            ax.scatter(np.full(len(vals), j + (i - 0.5) * w), vals,
                       color='k', s=22, zorder=5)
ax.axhline(0, color='gray', lw=0.8)
ax.set_xticks(xpos); ax.set_xticklabels(regions)
ax.set_ylabel('Pre-FA-lick ramp  [-0.5,0)s  (Δ z-dF/F)')
ax.set_title('Neural impulsivity-coding: prefa_ramp\nD1 vs D2 by region (bulk; N=mice)')
ax.legend(title='genotype')
for reg in regions:
    rr = ramp_results[reg]
    if not np.isnan(rr['p']):
        j = regions.index(reg)
        ymax = max([np.mean(rr['d1']) if len(rr['d1']) else 0,
                    np.mean(rr['d2']) if len(rr['d2']) else 0])
        ax.text(j, ymax * 1.15 + 0.02, f"p={rr['p']:.3f}\nrb={rr['rb']:+.2f}",
                ha='center', fontsize=8)

# Panel B: brake AUROC by genotype x region x prof_bin
ax = axes[1]
pbins = ['less', 'more']  # early vs late proficiency
groups = []
for reg in ['DMS', 'VMS', 'VLS']:
    for gt in ['D1', 'D2']:
        groups.append((reg, gt))
gx = np.arange(len(groups))
bw = 0.38
pb_col = {'less': '#7fbf7b', 'more': '#1b7837'}
for k, pb in enumerate(pbins):
    vals, ns, excl = [], [], []
    for (reg, gt) in groups:
        r = brake[(brake.region == reg) & (brake.genotype == gt) & (brake.prof_bin == pb)]
        if len(r):
            vals.append(r.iloc[0].auroc_mean); ns.append(int(r.iloc[0].n_mice))
            excl.append(bool(r.iloc[0].excludes_chance))
        else:
            vals.append(np.nan); ns.append(0); excl.append(False)
    bars = ax.bar(gx + (k - 0.5) * bw, vals, bw, color=pb_col[pb], alpha=0.8,
                  label=f'{pb} ({"early" if pb=="less" else "late"})')
    for b, v, e in zip(bars, vals, excl):
        if e and not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.003, '*', ha='center', fontsize=12)
ax.axhline(0.5, color='k', lw=1.0, ls='--', label='chance (0.5)')
ax.set_xticks(gx)
ax.set_xticklabels([f'{r}\n{g}' for (r, g) in groups], fontsize=8)
ax.set_ylabel('Waiting-period AUROC (withhold vs FA)')
ax.set_ylim(0.42, 0.60)
ax.set_title('C1 brake: behavioral_fa/scheme3\n>0.5=brake (suppress), <0.5=go-ramp; *=excludes chance')
ax.legend(fontsize=8, loc='upper right')

fig.tight_layout()
out = os.path.join(OUTDIR, 'impulsivity_axis.png')
fig.savefig(out, dpi=130)
print(f'\nSAVED: {out}')
print('EXISTS:', os.path.exists(out))
