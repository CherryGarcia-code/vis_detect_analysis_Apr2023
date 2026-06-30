"""INTERSECTIONAL corroboration (DESCRIPTIVE, n=2 D1 mice, 3 fibers).
Cortically-innervated (MOs-recipient) D1 SPN cohort. NEVER pooled with bulk.
(a) sensitivity (detect_sel, d_prime) & impulsivity (prefa_ramp, fa_rate_beh)
    across learning (sess_frac) per fiber.
(b) within-mouse coupling detect_sel~d_prime and prefa_ramp~fa_rate.
Does cortically-innervated D1 show same learning/coupling directions as bulk D1?
DESCRIPTIVE ONLY. No population inference (n=2 mice).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/per_session_metrics.csv'
OUT = r'e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/narrative'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
i = df[df.cohort == 'intersectional'].copy()
i['fiber'] = i.mouse + '_' + i.region
fibers = sorted(i.fiber.unique())
COLORS = {'BG_027_VMS_G0': '#1b9e77', 'BG_027_VMS_G2': '#7570b3', 'BG_028_DMS_G0': '#d95f02'}

print('=== INTERSECTIONAL D1 (MOs-recipient) — DESCRIPTIVE, n=2 mice, 3 fibers ===')
print('Fibers:', fibers)

# ---- per-fiber Spearman: metric vs learning (sess_frac), and coupling ----
def sp(g, x, y):
    s = g[[x, y]].dropna()
    if len(s) < 5:
        return np.nan, np.nan, len(s)
    r, p = spearmanr(s[x], s[y])
    return r, p, len(s)

learn_metrics = ['detect_sel', 'd_prime', 'prefa_ramp', 'fa_rate_beh']
print('\n--- (a) Trajectory vs learning (sess_frac), per fiber: Spearman rho (p, n_sess) ---')
traj = {}
for f in fibers:
    g = i[i.fiber == f]
    traj[f] = {}
    line = f'{f}: '
    for m in learn_metrics:
        r, p, n = sp(g, 'sess_frac', m)
        traj[f][m] = r
        line += f'{m} rho={r:+.2f}(p={p:.3f},n={n})  '
    print(line)

print('\n--- (b) Within-fiber coupling: Spearman rho (p, n_sess) ---')
coup = {}
for f in fibers:
    g = i[i.fiber == f]
    rs, ps, ns = sp(g, 'd_prime', 'detect_sel')
    ri, pi, ni = sp(g, 'fa_rate_beh', 'prefa_ramp')
    coup[f] = {'sens': rs, 'imp': ri}
    print(f'{f}: detect_sel~d_prime rho={rs:+.2f}(p={ps:.3f},n={ns})  '
          f'prefa_ramp~fa_rate rho={ri:+.2f}(p={pi:.3f},n={ni})')

# ---- direction-consistency check vs bulk-D1 narrative ----
# Narrative (bulk D1): sensitivity grows with learning (detect_sel up, d_prime up),
# impulsivity falls (fa_rate down), sensitivity coupling positive (detect_sel~d_prime +).
print('\n--- Direction consistency with bulk-D1 narrative (expected signs) ---')
print('  expected: detect_sel vs learn +, d_prime vs learn +, fa_rate vs learn -, detect_sel~d_prime +')
for f in fibers:
    ds = '+' if traj[f]['detect_sel'] > 0 else '-'
    dp = '+' if traj[f]['d_prime'] > 0 else '-'
    fr = '+' if traj[f]['fa_rate_beh'] > 0 else '-'
    cs = '+' if coup[f]['sens'] > 0 else '-'
    print(f'  {f}: detect_sel/learn {ds}, d_prime/learn {dp}, fa_rate/learn {fr}, coupling {cs}')

# ================= FIGURE =================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

def smooth(x, y, w=7):
    o = np.argsort(x)
    x, y = np.array(x)[o], np.array(y)[o]
    if len(y) < w:
        return x, y
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode='same')
    return x, ys

# Row 1: trajectories vs learning
panel_a = [('detect_sel', 'detect_sel (Hit-Miss change, SENSITIVITY)'),
           ('d_prime', "d' (behavioral SENSITIVITY)"),
           ('fa_rate_beh', 'fa_rate_beh (behavioral IMPULSIVITY)')]
for ax, (m, title) in zip(axes[0], panel_a):
    for f in fibers:
        g = i[i.fiber == f][['sess_frac', m]].dropna()
        if len(g) < 3:
            continue
        ax.scatter(g.sess_frac, g[m], s=14, alpha=0.35, color=COLORS[f])
        xs, ys = smooth(g.sess_frac.values, g[m].values)
        ax.plot(xs, ys, color=COLORS[f], lw=2, label=f)
    ax.set_xlabel('learning (sess_frac: 0=first, 1=last)')
    ax.set_ylabel(m)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.legend(fontsize=7)

# Row 2: prefa_ramp trajectory + 2 coupling panels
ax = axes[1, 0]
for f in fibers:
    g = i[i.fiber == f][['sess_frac', 'prefa_ramp']].dropna()
    if len(g) < 3:
        continue
    ax.scatter(g.sess_frac, g.prefa_ramp, s=14, alpha=0.35, color=COLORS[f])
    xs, ys = smooth(g.sess_frac.values, g.prefa_ramp.values)
    ax.plot(xs, ys, color=COLORS[f], lw=2, label=f)
ax.set_xlabel('learning (sess_frac)')
ax.set_ylabel('prefa_ramp')
ax.set_title('prefa_ramp (pre-FA-lick ramp, neural IMPULSIVITY)', fontsize=10)
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.legend(fontsize=7)

ax = axes[1, 1]
for f in fibers:
    g = i[i.fiber == f][['d_prime', 'detect_sel']].dropna()
    ax.scatter(g.d_prime, g.detect_sel, s=16, alpha=0.5, color=COLORS[f],
               label=f'{f} (rho={coup[f]["sens"]:+.2f})')
ax.set_xlabel("d' (behavioral sensitivity)")
ax.set_ylabel('detect_sel (neural sensitivity)')
ax.set_title('SENSITIVITY coupling: neural~behavioral', fontsize=10)
ax.legend(fontsize=7)

ax = axes[1, 2]
for f in fibers:
    g = i[i.fiber == f][['fa_rate_beh', 'prefa_ramp']].dropna()
    ax.scatter(g.fa_rate_beh, g.prefa_ramp, s=16, alpha=0.5, color=COLORS[f],
               label=f'{f} (rho={coup[f]["imp"]:+.2f})')
ax.set_xlabel('fa_rate_beh (behavioral impulsivity)')
ax.set_ylabel('prefa_ramp (neural impulsivity)')
ax.set_title('IMPULSIVITY coupling: neural~behavioral', fontsize=10)
ax.legend(fontsize=7)

fig.suptitle('Intersectional MOs-recipient D1 cohort — DESCRIPTIVE (n=2 mice, 3 fibers; NO population inference; not pooled with bulk)',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, 'int_corroboration.png'), dpi=130)
print('\nSaved:', os.path.join(OUT, 'int_corroboration.png'))
