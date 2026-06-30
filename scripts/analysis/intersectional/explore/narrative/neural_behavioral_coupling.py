"""neural_behavioral_coupling: within-mouse coupling of neural metrics to behavioral metrics across sessions (BULK).

(a) SENSITIVITY coupling = Spearman(detect_sel, d_prime) per mouse
(b) IMPULSIVITY coupling = Spearman(prefa_ramp, fa_rate_beh) per mouse
Aggregate per-mouse rho by genotype, test vs 0 (sign test + bootstrap).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
CSV = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/per_session_metrics.csv"
OUTDIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/narrative"
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "neural_behavioral_coupling.png")

MIN_SESS = 4  # minimum paired sessions to compute a within-mouse Spearman

df = pd.read_csv(CSV)
b = df[df.cohort == 'bulk'].copy()

GCOL = {'D1': '#1b7837', 'D2': '#762a83'}


def per_mouse_spearman(data, xcol, ycol):
    """Return list of dicts: per (mouse,genotype,region) within-mouse Spearman rho."""
    out = []
    for (mouse, geno, region), g in data.groupby(['mouse', 'genotype', 'region']):
        sub = g[[xcol, ycol, 'sess_frac']].dropna(subset=[xcol, ycol])
        n = len(sub)
        if n < MIN_SESS:
            continue
        # need variance in both
        if sub[xcol].nunique() < 2 or sub[ycol].nunique() < 2:
            continue
        rho, p = stats.spearmanr(sub[xcol], sub[ycol])
        if np.isnan(rho):
            continue
        out.append(dict(mouse=mouse, genotype=geno, region=region, n=n,
                        rho=rho, p=p, sub=sub.copy()))
    return out


def boot_ci(vals, nboot=10000):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return (np.nan, np.nan)
    bs = [np.mean(RNG.choice(vals, len(vals), replace=True)) for _ in range(nboot)]
    return (np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def sign_test(vals):
    """Two-sided sign test vs 0 (exact binomial). Returns p, n_pos, n."""
    vals = np.asarray(vals, float)
    vals = vals[vals != 0]
    n = len(vals)
    if n == 0:
        return np.nan, 0, 0
    n_pos = int(np.sum(vals > 0))
    res = stats.binomtest(n_pos, n, 0.5, alternative='two-sided')
    return res.pvalue, n_pos, n


sens = per_mouse_spearman(b, 'detect_sel', 'd_prime')
imp = per_mouse_spearman(b, 'prefa_ramp', 'fa_rate_beh')

print("=== SENSITIVITY coupling: Spearman(detect_sel, d_prime) per mouse ===")
for r in sens:
    print(f"  {r['mouse']} {r['genotype']} {r['region']}  n={r['n']:2d}  rho={r['rho']:+.3f}  p={r['p']:.3f}")
print("=== IMPULSIVITY coupling: Spearman(prefa_ramp, fa_rate_beh) per mouse ===")
for r in imp:
    print(f"  {r['mouse']} {r['genotype']} {r['region']}  n={r['n']:2d}  rho={r['rho']:+.3f}  p={r['p']:.3f}")


def summarize(records, label):
    print(f"\n--- {label} group-level (N=mice) ---")
    res = {}
    for geno in ['D1', 'D2']:
        vals = [r['rho'] for r in records if r['genotype'] == geno]
        if not vals:
            print(f"  {geno}: no mice")
            res[geno] = None
            continue
        m = np.mean(vals)
        lo, hi = boot_ci(vals)
        sp, npos, nn = sign_test(vals)
        # one-sample wilcoxon as supplementary
        try:
            wp = stats.wilcoxon(vals).pvalue if len(vals) >= 1 and any(v != 0 for v in vals) else np.nan
        except Exception:
            wp = np.nan
        print(f"  {geno}: N={len(vals)} mice  mean_rho={m:+.3f}  CI=[{lo:+.3f},{hi:+.3f}]  "
              f"sign_test p={sp if not np.isnan(sp) else float('nan'):.3f} ({npos}/{nn} pos)  wilcoxon p={wp:.3f}")
        res[geno] = dict(vals=vals, mean=m, lo=lo, hi=hi, sp=sp, npos=npos, nn=nn)
    return res


sens_sum = summarize(sens, "SENSITIVITY coupling (detect_sel vs d_prime)")
imp_sum = summarize(imp, "IMPULSIVITY coupling (prefa_ramp vs fa_rate_beh)")

# ---------------- FIGURE ----------------
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1])

# Panel A: sensitivity coupling rho by genotype
axA = fig.add_subplot(gs[0, 0])
# Panel B: impulsivity coupling rho by genotype
axB = fig.add_subplot(gs[0, 1])


def strip_panel(ax, records, summ, title, xlab):
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
    for i, geno in enumerate(['D1', 'D2']):
        vals = [r['rho'] for r in records if r['genotype'] == geno]
        x = np.full(len(vals), i) + RNG.uniform(-0.08, 0.08, len(vals))
        ax.scatter(x, vals, color=GCOL[geno], s=70, zorder=3, edgecolor='k', linewidth=0.5)
        if summ.get(geno):
            m = summ[geno]['mean']; lo = summ[geno]['lo']; hi = summ[geno]['hi']
            ax.plot([i - 0.18, i + 0.18], [m, m], color=GCOL[geno], lw=3, zorder=4)
            ax.plot([i, i], [lo, hi], color=GCOL[geno], lw=1.5, zorder=2)
            sp = summ[geno]['sp']
            ax.text(i, ax.get_ylim()[1] if False else max(vals) + 0.06,
                    f"p={sp:.2f}\n({summ[geno]['npos']}/{summ[geno]['nn']})",
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['D1', 'D2'])
    ax.set_ylabel('within-mouse Spearman rho')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab)
    ax.set_ylim(-1.05, 1.15)


strip_panel(axA, sens, sens_sum, 'SENSITIVITY coupling\ndetect_sel vs d_prime', '')
strip_panel(axB, imp, imp_sum, 'IMPULSIVITY coupling\nprefa_ramp vs fa_rate_beh', '')

# Example within-mouse scatters: pick strongest |rho| mice
def pick_examples(records, k=2):
    return sorted(records, key=lambda r: -abs(r['rho']))[:k]

ex_sens = pick_examples(sens, 2)
ex_imp = pick_examples(imp, 2)

ax_ex = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
         fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

ex_specs = [(ex_sens[0] if len(ex_sens) > 0 else None, 'detect_sel', 'd_prime', 'SENS'),
            (ex_imp[0] if len(ex_imp) > 0 else None, 'prefa_ramp', 'fa_rate_beh', 'IMP'),
            (ex_sens[1] if len(ex_sens) > 1 else None, 'detect_sel', 'd_prime', 'SENS'),
            (ex_imp[1] if len(ex_imp) > 1 else None, 'prefa_ramp', 'fa_rate_beh', 'IMP')]

for ax, (rec, xc, yc, tag) in zip(ax_ex, ex_specs):
    if rec is None:
        ax.axis('off'); continue
    sub = rec['sub']
    sc = ax.scatter(sub[xc], sub[yc], c=sub['sess_frac'], cmap='viridis', s=55,
                    edgecolor='k', linewidth=0.4)
    # regression line
    if sub[xc].nunique() > 1:
        sl, ic = np.polyfit(sub[xc], sub[yc], 1)
        xs = np.linspace(sub[xc].min(), sub[xc].max(), 20)
        ax.plot(xs, sl * xs + ic, color=GCOL[rec['genotype']], lw=2)
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    ax.set_title(f"{tag}: {rec['mouse']} ({rec['genotype']},{rec['region']})\n"
                 f"rho={rec['rho']:+.2f} p={rec['p']:.2f} n={rec['n']}", fontsize=8.5)
    plt.colorbar(sc, ax=ax, label='sess_frac', fraction=0.046, pad=0.04)

# Summary text panel
axT = fig.add_subplot(gs[1, 2:])
axT.axis('off')
lines = ["NEURAL <-> BEHAVIORAL COUPLING (BULK, within-mouse across sessions)", ""]
for label, summ in [("SENSITIVITY (detect_sel~d')", sens_sum), ("IMPULSIVITY (prefa_ramp~FA rate)", imp_sum)]:
    lines.append(label + ":")
    for geno in ['D1', 'D2']:
        if summ.get(geno):
            s = summ[geno]
            lines.append(f"   {geno}: N={len(s['vals'])} mice  mean rho={s['mean']:+.2f}  "
                         f"CI[{s['lo']:+.2f},{s['hi']:+.2f}]  sign p={s['sp']:.2f}")
        else:
            lines.append(f"   {geno}: no mice (insufficient paired sessions)")
    lines.append("")
lines.append(f"(min {MIN_SESS} paired sessions/mouse; statistical unit = MOUSE)")
axT.text(0.0, 1.0, "\n".join(lines), va='top', ha='left', fontsize=9, family='monospace')

fig.suptitle("Neural-Behavioral Coupling across learning (BULK)", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print(f"\nSAVED: {OUT}")
print("EXISTS:", os.path.exists(OUT))
