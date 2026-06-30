"""bulk_rt_coding: Does the pre-lick photometry ramp scale with reaction time?

rt_tercile lick traces (fast/mid/slow), per genotype x region x event (fa_lick, hit_lick).
Per-mouse-averaged mean trace per tercile (mean over mice, N=mice). Quantify pre-lick
peak amplitude per tercile; test monotonic fast->slow trend per genotype x region x event.

Discipline: 6f intersectional cohort NOT involved here (bulk-8m pkl only). Behavioral FA =
anticipatory lick (impulsivity). REGION kept separate. Unit = MOUSE.
Traces = session-z-scored dF/F, baseline-subtracted (Delta z). Lick at t=0.
"""
import os, sys, pickle
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUTDIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "bulk_rt_coding.png")

d = pickle.load(open(PKL, 'rb'))
rt = d['rt_tercile']
t = np.asarray(rt['time'])          # -2 .. 3 s, 500 samples
tr = rt['traces']

TERCILES = ['fast', 'mid', 'slow']
TCOL = {'fast': '#1b9e77', 'mid': '#7570b3', 'slow': '#d95f02'}

# Pre-lick window: ramp that builds toward the lick. Lick at t=0.
# Use [-0.5, 0] s as the pre-lick window (the immediate ramp before action).
PRE0, PRE1 = -0.5, 0.0
pre_mask = (t >= PRE0) & (t < PRE1)
# Peak window: allow capturing peak shortly around lick [-0.5, +0.2]
PK0, PK1 = -0.5, 0.2
pk_mask = (t >= PK0) & (t < PK1)

def scalar_pre(trace):
    """mean Delta z in pre-lick window."""
    return float(np.nanmean(trace[pre_mask]))

def scalar_peak(trace):
    """peak (max abs, signed) in peri-lick window."""
    seg = trace[pk_mask]
    if not np.isfinite(seg).any():
        return np.nan
    i = np.nanargmax(np.abs(seg))
    return float(seg[i])

# Organize: (geno,region,event) -> mouse -> {tercile: trace}
byc = defaultdict(lambda: defaultdict(dict))
for (m, g, r, e, terc), v in tr.items():
    byc[(g, r, e)][m][terc] = np.asarray(v, dtype=float)

# Cells with N>=2 mice having all three terciles (for stats); we will still PLOT N>=1.
def mice_with_all3(cell):
    out = []
    for m, dd in byc[cell].items():
        if all(tc in dd and np.isfinite(dd[tc]).any() for tc in TERCILES):
            out.append(m)
    return sorted(out)

# ---------- Quantification ----------
print("=" * 78)
print("PRE-LICK RAMP CODING OF REACTION TIME  (window pre=[-0.5,0)s, peak=[-0.5,0.2)s)")
print("Unit = MOUSE. Delta z (session-z dF/F, baseline-subtracted).")
print("=" * 78)

results = {}  # cell -> dict
for cell in sorted(byc):
    g, r, e = cell
    mice = mice_with_all3(cell)
    n = len(mice)
    # per-mouse scalar per tercile
    pre_by_terc = {tc: [] for tc in TERCILES}
    pk_by_terc = {tc: [] for tc in TERCILES}
    for m in mice:
        for tc in TERCILES:
            pre_by_terc[tc].append(scalar_pre(byc[cell][m][tc]))
            pk_by_terc[tc].append(scalar_peak(byc[cell][m][tc]))
    pre_mean = {tc: np.nanmean(pre_by_terc[tc]) if n else np.nan for tc in TERCILES}
    pk_mean = {tc: np.nanmean(pk_by_terc[tc]) if n else np.nan for tc in TERCILES}
    # monotonic trend test across terciles using per-mouse values (Spearman of tercile-rank vs scalar)
    # ranks: fast=0, mid=1, slow=2
    rho_pre = p_pre = rho_pk = p_pk = np.nan
    slow_minus_fast_pre = pk_slow_minus_fast = np.nan
    paired_p_pre = np.nan
    if n >= 2:
        xs, ys_pre, ys_pk = [], [], []
        for mi, m in enumerate(mice):
            for ti, tc in enumerate(TERCILES):
                xs.append(ti); ys_pre.append(scalar_pre(byc[cell][m][tc])); ys_pk.append(scalar_peak(byc[cell][m][tc]))
        xs = np.array(xs); ys_pre = np.array(ys_pre); ys_pk = np.array(ys_pk)
        m_ok = np.isfinite(ys_pre)
        if m_ok.sum() >= 3 and len(set(xs[m_ok])) > 1:
            rho_pre, p_pre = stats.spearmanr(xs[m_ok], ys_pre[m_ok])
        m_ok2 = np.isfinite(ys_pk)
        if m_ok2.sum() >= 3 and len(set(xs[m_ok2])) > 1:
            rho_pk, p_pk = stats.spearmanr(xs[m_ok2], ys_pk[m_ok2])
        # paired slow vs fast (per mouse), report mean diff + sign-consistency
        diffs_pre = np.array([scalar_pre(byc[cell][m]['slow']) - scalar_pre(byc[cell][m]['fast']) for m in mice])
        diffs_pk = np.array([scalar_peak(byc[cell][m]['slow']) - scalar_peak(byc[cell][m]['fast']) for m in mice])
        slow_minus_fast_pre = float(np.nanmean(diffs_pre))
        pk_slow_minus_fast = float(np.nanmean(diffs_pk))
    results[cell] = dict(mice=mice, n=n, pre_mean=pre_mean, pk_mean=pk_mean,
                         rho_pre=rho_pre, p_pre=p_pre, rho_pk=rho_pk, p_pk=p_pk,
                         slow_minus_fast_pre=slow_minus_fast_pre, pk_slow_minus_fast=pk_slow_minus_fast,
                         pre_by_terc=pre_by_terc, pk_by_terc=pk_by_terc)
    print(f"\n{g} {r} {e}  | N={n} mice {mice}")
    print(f"   pre-lick mean dz  fast={pre_mean['fast']:.3f} mid={pre_mean['mid']:.3f} slow={pre_mean['slow']:.3f}"
          f"  (slow-fast={slow_minus_fast_pre:.3f})")
    print(f"   peri-lick peak dz fast={pk_mean['fast']:.3f} mid={pk_mean['mid']:.3f} slow={pk_mean['slow']:.3f}"
          f"  (slow-fast={pk_slow_minus_fast:.3f})")
    if n >= 2:
        print(f"   trend Spearman(tercile-rank vs pre)  rho={rho_pre:.3f} p={p_pre:.3f}"
              f"  | peak rho={rho_pk:.3f} p={p_pk:.3f}")

# ---------- Figure ----------
# Plot cells with N>=2 mice for the main panels (richest, statistically meaningful).
main_cells = [c for c in sorted(byc) if results[c]['n'] >= 2]
# order: group by event then geno/region
order = sorted(main_cells, key=lambda c: (c[2], c[0], c[1]))

ncols = 4
nrows = int(np.ceil(len(order) / ncols)) + 1  # extra row for summary bar
fig = plt.figure(figsize=(4.2 * ncols, 3.4 * (nrows)))
gs = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.32)

def mouse_mean_trace(cell, tc):
    mice = results[cell]['mice']
    arr = np.vstack([byc[cell][m][tc] for m in mice])
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem

for idx, cell in enumerate(order):
    rrow, ccol = divmod(idx, ncols)
    ax = fig.add_subplot(gs[rrow, ccol])
    g, r, e = cell
    n = results[cell]['n']
    for tc in TERCILES:
        mean, sem = mouse_mean_trace(cell, tc)
        ax.plot(t, mean, color=TCOL[tc], lw=1.8, label=f"{tc}")
        ax.fill_between(t, mean - sem, mean + sem, color=TCOL[tc], alpha=0.18, lw=0)
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
    ax.axvspan(PRE0, PRE1, color='gold', alpha=0.10)
    ax.set_xlim(-1.5, 1.5)
    ax.set_title(f"{g} {r}  {e}\nN={n} mice", fontsize=9)
    ax.set_xlabel("time from lick (s)", fontsize=8)
    if ccol == 0:
        ax.set_ylabel("Δz dF/F", fontsize=8)
    rho = results[cell]['rho_pre']; p = results[cell]['p_pre']
    sf = results[cell]['slow_minus_fast_pre']
    ax.text(0.02, 0.97, f"ρ(pre)={rho:.2f}\np={p:.2f}\nslow−fast={sf:+.2f}",
            transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7, lw=0.4))
    if idx == 0:
        ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

# Summary bar row: slow-minus-fast pre-lick amplitude per cell (per-mouse points)
axb = fig.add_subplot(gs[nrows - 1, :])
labels = []
xpos = []
for i, cell in enumerate(order):
    g, r, e = cell
    res = results[cell]
    mice = res['mice']
    diffs = [scalar_pre(byc[cell][m]['slow']) - scalar_pre(byc[cell][m]['fast']) for m in mice]
    axb.bar(i, np.nanmean(diffs), color='#999999', width=0.6, zorder=1)
    axb.scatter([i] * len(diffs), diffs, color='k', s=22, zorder=3)
    labels.append(f"{g}\n{r}\n{e.replace('_lick','')}")
    xpos.append(i)
axb.axhline(0, color='k', lw=0.8)
axb.set_xticks(xpos); axb.set_xticklabels(labels, fontsize=7)
axb.set_ylabel("pre-lick Δz\n(slow − fast)", fontsize=8)
axb.set_title("Slow-RT minus Fast-RT pre-lick ramp amplitude (per-mouse points; bar=mean over mice)",
              fontsize=9)

fig.suptitle("Bulk-8m: does pre-lick striatal ramp scale with reaction time? (RT terciles, lick-aligned)",
             fontsize=12, y=0.995)
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print("\nSaved:", OUT)

# ---------- Headline numbers ----------
print("\n" + "=" * 78)
print("HEADLINE SUMMARY (cells with N>=2 mice)")
print("=" * 78)
# Tally sign of slow-fast pre-lick and peak by genotype
for g in ['D1', 'D2']:
    cells_g = [c for c in order if c[0] == g]
    sf_pre = [results[c]['slow_minus_fast_pre'] for c in cells_g]
    sf_pk = [results[c]['pk_slow_minus_fast'] for c in cells_g]
    print(f"{g}: cells={[ (c[1],c[2]) for c in cells_g]}")
    print(f"    slow-fast PRE: {[f'{x:+.3f}' for x in sf_pre]}  (mean {np.nanmean(sf_pre):+.3f})")
    print(f"    slow-fast PEAK:{[f'{x:+.3f}' for x in sf_pk]}  (mean {np.nanmean(sf_pk):+.3f})")
# Best monotonic cell
best = max(order, key=lambda c: abs(results[c]['rho_pre']) if np.isfinite(results[c]['rho_pre']) else -1)
print(f"\nStrongest pre-lick monotonic trend: {best} rho={results[best]['rho_pre']:.3f} p={results[best]['p_pre']:.3f} N={results[best]['n']}")
