"""learning_trajectories: BULK. How do SENSITIVITY (detect_sel, d_prime) and
IMPULSIVITY (prefa_ramp, fa_rate_beh) change across learning (sess_frac), D1 vs D2.

Unit = MOUSE. Per-mouse OLS slope vs sess_frac, then D1 vs D2 permutation test (N=mice).
MixedLM metric ~ sess_frac*genotype + (sess_frac|mouse) as confirmation.
Regions kept separate; each mouse uses its PRIMARY region (most sessions).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/per_session_metrics.csv"
OUT = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/narrative"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

df = pd.read_csv(CSV)
b = df[df.cohort == 'bulk'].copy()

# primary region per mouse = region with most sessions
prim = (b.groupby(['mouse', 'region']).size().reset_index(name='n')
        .sort_values('n', ascending=False).drop_duplicates('mouse'))
prim_map = dict(zip(prim.mouse, prim.region))
print("PRIMARY REGION per mouse:", prim_map)

# neural metrics: restrict to each mouse's primary region
neural_df = b[b.apply(lambda r: r.region == prim_map[r.mouse], axis=1)].copy()
# behavioral metrics: region-independent -> dedupe per mouse+session
beh_df = b.drop_duplicates(['mouse', 'session_id']).copy()

METRICS = {
    'detect_sel':   ('SENSITIVITY: detect_sel (neural Hit-Miss)', neural_df),
    'd_prime':      ("SENSITIVITY: d'", beh_df),
    'prefa_ramp':   ('IMPULSIVITY: prefa_ramp (neural)', neural_df),
    'fa_rate_beh':  ('IMPULSIVITY: fa_rate_beh', beh_df),
}

def perm_test(x, y, n=10000):
    """two-sided permutation test on difference of means; returns p, observed diff."""
    x = np.asarray(x); y = np.asarray(y)
    obs = np.mean(x) - np.mean(y)
    pool = np.concatenate([x, y]); nx = len(x)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        d = np.mean(pool[:nx]) - np.mean(pool[nx:])
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1), obs

results = {}
per_mouse_slopes = {}

for met, (label, data) in METRICS.items():
    d = data.dropna(subset=[met, 'sess_frac']).copy()
    slopes = {}
    for m, g in d.groupby('mouse'):
        if len(g) < 4 or g.sess_frac.nunique() < 3:
            continue
        # OLS slope of metric vs sess_frac
        sl = np.polyfit(g.sess_frac.values, g[met].values, 1)[0]
        geno = g.genotype.iloc[0]
        slopes[m] = (sl, geno)
    per_mouse_slopes[met] = slopes
    d1 = [v[0] for v in slopes.values() if v[1] == 'D1']
    d2 = [v[0] for v in slopes.values() if v[1] == 'D2']
    # one-sample sign: are slopes different from 0 (pooled, per genotype) - report mean
    p_d1d2, obs = perm_test(d1, d2) if (len(d1) >= 2 and len(d2) >= 2) else (np.nan, np.nan)
    # MixedLM confirmation
    try:
        md = smf.mixedlm(f"{met} ~ sess_frac * C(genotype)", d,
                         groups=d['mouse'], re_formula="~sess_frac")
        mf = md.fit(reorder=False, method='lbfgs', maxiter=200)
        coefs = mf.params.to_dict()
        pvals = mf.pvalues.to_dict()
    except Exception as e:
        coefs, pvals = {}, {}
        print(f"  MixedLM FAILED for {met}: {e}")
    results[met] = dict(label=label, d1_slopes=d1, d2_slopes=d2,
                        p_d1d2=p_d1d2, obs=obs, coefs=coefs, pvals=pvals)
    print(f"\n=== {label} ===")
    print(f"  D1 mice slopes (n={len(d1)}): {[round(x,4) for x in d1]} mean={np.mean(d1):.4f}")
    print(f"  D2 mice slopes (n={len(d2)}): {[round(x,4) for x in d2]} mean={np.mean(d2):.4f}")
    print(f"  D1 vs D2 slope perm p={p_d1d2:.4f}  obs_diff={obs:.4f}")
    if pvals:
        for k in ['sess_frac', 'sess_frac:C(genotype)[T.D2]']:
            if k in coefs:
                print(f"  MixedLM {k}: beta={coefs[k]:.4f} p={pvals[k]:.4f}")

# ── FIGURE ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
GCOL = {'D1': '#1f77b4', 'D2': '#d62728'}
bins = np.linspace(0, 1, 6)
bin_centers = (bins[:-1] + bins[1:]) / 2

for j, (met, (label, data)) in enumerate(METRICS.items()):
    ax = axes[0, j]
    d = data.dropna(subset=[met, 'sess_frac']).copy()
    d['bin'] = pd.cut(d.sess_frac, bins, include_lowest=True, labels=bin_centers)
    for geno in ['D1', 'D2']:
        gg = d[d.genotype == geno]
        # per-mouse mean within bin, then mean+sem over mice
        pm = gg.groupby(['bin', 'mouse'])[met].mean().reset_index()
        agg = pm.groupby('bin')[met].agg(['mean', 'sem', 'count']).reset_index()
        agg = agg.dropna(subset=['mean'])
        xc = agg['bin'].astype(float).values
        ax.errorbar(xc, agg['mean'].values, yerr=agg['sem'].fillna(0).values,
                    marker='o', color=GCOL[geno], label=f'{geno} (n={gg.mouse.nunique()})',
                    capsize=3, lw=2)
    ax.set_xlabel('sess_frac (0=first,1=last)')
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', ls=':', lw=0.8)

    # bottom row: per-mouse slope summary
    ax2 = axes[1, j]
    sl = per_mouse_slopes[met]
    for i, geno in enumerate(['D1', 'D2']):
        vals = [v[0] for v in sl.values() if v[1] == geno]
        xj = np.full(len(vals), i) + RNG.uniform(-0.08, 0.08, len(vals))
        ax2.scatter(xj, vals, color=GCOL[geno], s=60, zorder=3)
        if vals:
            ax2.hlines(np.mean(vals), i - 0.2, i + 0.2, color=GCOL[geno], lw=3)
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['D1', 'D2'])
    pv = results[met]['p_d1d2']
    ax2.set_title(f'per-mouse slope vs sess_frac\nD1vD2 perm p={pv:.3f}', fontsize=9)
    ax2.set_ylabel('OLS slope')

fig.suptitle('BULK: Sensitivity & Impulsivity across learning (D1 vs D2). Unit=mouse, primary region.',
             fontsize=13, y=1.0)
fig.tight_layout()
fp = os.path.join(OUT, 'learning_trajectories.png')
fig.savefig(fp, dpi=130, bbox_inches='tight')
print("\nSAVED:", fp, "exists:", os.path.exists(fp))
