"""bulk_reward_rpe: Lick-aligned Hit (rewarded) vs FA (unrewarded) post-lick signal.

REWARD / RPE angle. Post-lick window. Is there a Hit>FA positive signal
(reward/RPE-like), and is it larger in D1 (direct pathway)?
Quantify post-lick AUC (0,1.5s) Hit-minus-FA per mouse, D1 vs D2 (N=mice).

DISCIPLINE: bulk-8m only; unit = MOUSE; aggregate per-mouse traces to
genotype mean +/- SEM over mice. Keep REGION separation. Behavioral FA =
anticipatory lick. Traces are session-z-scored dF/F, baseline-subtracted (Delta z).
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PKL = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl"
OUTDIR = r"e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/explore"
OUTPNG = os.path.join(OUTDIR, "bulk_reward_rpe.png")
os.makedirs(OUTDIR, exist_ok=True)

# Windows
AUC_LO, AUC_HI = 0.0, 1.5    # requested peri-lick window (lick-transient dominated)
LATE_LO, LATE_HI = 1.5, 3.0  # sustained reward/consumption window (RPE divergence)
PLOT_LO, PLOT_HI = -2.0, 3.0

GENO_COLORS = {"D1": "#1b7837", "D2": "#762a83"}  # green=D1, purple=D2

with open(PKL, "rb") as f:
    d = pickle.load(f)

t = np.asarray(d["lick"]["time"])  # 500, [-2,3]
pm = d["lick"]["per_mouse"]

auc_mask = (t >= AUC_LO) & (t <= AUC_HI)
late_mask = (t >= LATE_LO) & (t <= LATE_HI)
dt = float(np.median(np.diff(t)))

# Collect per (mouse,geno,region): hit trace, fa trace
mouse_keys = {}
for (mouse, geno, region, cond), tr in pm.items():
    mouse_keys.setdefault((mouse, geno, region), {})[cond] = np.asarray(tr, float)

# Per-mouse-region records with both conditions
records = []  # dict per mouse-region
for (mouse, geno, region), cd in sorted(mouse_keys.items()):
    if "hit_lick" not in cd or "fa_lick" not in cd:
        continue
    hit = cd["hit_lick"]
    fa = cd["fa_lick"]
    diff = hit - fa
    # mean amplitude (Delta z) in post-lick window = AUC/window-duration; report mean to keep z-units
    hit_post = float(np.nanmean(hit[auc_mask]))
    fa_post = float(np.nanmean(fa[auc_mask]))
    diff_post = float(np.nanmean(diff[auc_mask]))
    diff_late = float(np.nanmean(diff[late_mask]))
    hit_late = float(np.nanmean(hit[late_mask]))
    fa_late = float(np.nanmean(fa[late_mask]))
    records.append(dict(mouse=mouse, geno=geno, region=region,
                        hit=hit, fa=fa, diff=diff,
                        hit_post=hit_post, fa_post=fa_post, diff_post=diff_post,
                        hit_late=hit_late, fa_late=fa_late, diff_late=diff_late))

print("=== Per mouse-region post-lick (0,1.5s) mean Delta-z ===")
print(f"{'mouse':8s} {'geno':4s} {'region':5s} {'Hit':>8s} {'FA':>8s} {'Hit-FA':>8s}")
for r in records:
    print(f"{r['mouse']:8s} {r['geno']:4s} {r['region']:5s} "
          f"{r['hit_post']:8.4f} {r['fa_post']:8.4f} {r['diff_post']:8.4f}")

# ---- Per-MOUSE aggregation: average across regions within a mouse (unit=mouse) ----
from collections import defaultdict
by_mouse = defaultdict(list)
for r in records:
    by_mouse[(r["mouse"], r["geno"])].append(r)

mouse_diff = {}  # (mouse,geno) -> mean Hit-FA peri (0,1.5) across that mouse's regions
mouse_late = {}  # (mouse,geno) -> mean Hit-FA late (1.5,3)
mouse_hit = {}
mouse_fa = {}
for (mouse, geno), rs in by_mouse.items():
    mouse_diff[(mouse, geno)] = float(np.mean([x["diff_post"] for x in rs]))
    mouse_late[(mouse, geno)] = float(np.mean([x["diff_late"] for x in rs]))
    mouse_hit[(mouse, geno)] = float(np.mean([x["hit_post"] for x in rs]))
    mouse_fa[(mouse, geno)] = float(np.mean([x["fa_post"] for x in rs]))

d1_diff = np.array([v for (m, g), v in mouse_diff.items() if g == "D1"])
d2_diff = np.array([v for (m, g), v in mouse_diff.items() if g == "D2"])
d1_mice = [m for (m, g) in mouse_diff if g == "D1"]
d2_mice = [m for (m, g) in mouse_diff if g == "D2"]

print()
print("=== Per-mouse Hit-FA post-lick (0,1.5s) mean Delta-z (regions averaged within mouse) ===")
print("D1 mice:", d1_mice)
print("D1 Hit-FA:", np.round(d1_diff, 4))
print("D2 mice:", d2_mice)
print("D2 Hit-FA:", np.round(d2_diff, 4))
print(f"N mice: D1={len(d1_diff)}, D2={len(d2_diff)}")

# Within-mouse: is Hit>FA overall? (paired across all mouse-regions)
all_hit = np.array([r["hit_post"] for r in records])
all_fa = np.array([r["fa_post"] for r in records])
all_diff = all_hit - all_fa
# per-mouse paired (regions averaged) for the within-subject test
mm = sorted(mouse_diff.keys())
pm_diff_vals = np.array([mouse_diff[k] for k in mm])
try:
    w_stat, w_p = stats.wilcoxon(pm_diff_vals)
except Exception as e:
    w_stat, w_p = np.nan, np.nan
print()
print(f"Within-subject Hit-FA > 0 (per-mouse, N={len(pm_diff_vals)}): "
      f"mean={pm_diff_vals.mean():.4f}, Wilcoxon p={w_p:.4f}")

# D1 vs D2 on Hit-FA (between mice)
u_stat, u_p = stats.mannwhitneyu(d1_diff, d2_diff, alternative="two-sided")
# rank-biserial effect size
n1, n2 = len(d1_diff), len(d2_diff)
rbc = 1.0 - (2.0 * u_stat) / (n1 * n2)
# Hedges-like Cohen d (small N, descriptive)
pooled_sd = np.sqrt(((n1 - 1) * d1_diff.var(ddof=1) + (n2 - 1) * d2_diff.var(ddof=1)) / (n1 + n2 - 2))
cohend = (d1_diff.mean() - d2_diff.mean()) / pooled_sd if pooled_sd > 0 else np.nan
print()
print(f"D1 vs D2 Hit-FA: D1 mean={d1_diff.mean():.4f} (SEM={stats.sem(d1_diff):.4f}), "
      f"D2 mean={d2_diff.mean():.4f} (SEM={stats.sem(d2_diff):.4f})")
print(f"Mann-Whitney U={u_stat:.1f}, p={u_p:.4f}, rank-biserial={rbc:.3f}, Cohen d={cohend:.3f}")

# ---- LATE window (1.5,3 s): sustained reward / RPE divergence ----
d1_late = np.array([v for (m, g), v in mouse_late.items() if g == "D1"])
d2_late = np.array([v for (m, g), v in mouse_late.items() if g == "D2"])
all_late = np.array([mouse_late[k] for k in mm])
try:
    wl_stat, wl_p = stats.wilcoxon(all_late)
except Exception:
    wl_stat, wl_p = np.nan, np.nan
ul_stat, ul_p = stats.mannwhitneyu(d1_late, d2_late, alternative="two-sided")
rbc_l = 1.0 - (2.0 * ul_stat) / (len(d1_late) * len(d2_late))
pooled_sd_l = np.sqrt(((len(d1_late) - 1) * d1_late.var(ddof=1) +
                       (len(d2_late) - 1) * d2_late.var(ddof=1)) /
                      (len(d1_late) + len(d2_late) - 2))
cohend_l = (d1_late.mean() - d2_late.mean()) / pooled_sd_l if pooled_sd_l > 0 else np.nan
print()
print("=== LATE window (1.5,3 s) Hit-FA per-mouse (sustained reward signal) ===")
print(f"  Overall: mean={all_late.mean():.4f} N={len(all_late)} Wilcoxon p={wl_p:.4f}")
print(f"  D1 mean={d1_late.mean():.4f} (SEM={stats.sem(d1_late):.4f}) N={len(d1_late)}")
print(f"  D2 mean={d2_late.mean():.4f} (SEM={stats.sem(d2_late):.4f}) N={len(d2_late)}")
print(f"  MWU U={ul_stat:.1f}, p={ul_p:.4f}, rank-biserial={rbc_l:.3f}, Cohen d={cohend_l:.3f}")

# =================== FIGURE ===================
regions_order = ["DMS", "VMS", "VLS"]
genos = ["D1", "D2"]

# group traces by (geno,region) for trace panels
trace_groups = defaultdict(lambda: {"hit": [], "fa": []})
for r in records:
    trace_groups[(r["geno"], r["region"])]["hit"].append(r["hit"])
    trace_groups[(r["geno"], r["region"])]["fa"].append(r["fa"])

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.32)

pmask = (t >= PLOT_LO) & (t <= PLOT_HI)

# Top row: per-region trace panels (Hit vs FA) split by genotype color, but
# one panel per (geno,region) combination that exists. Build panel list.
panel_combos = []
for geno in genos:
    for region in regions_order:
        if (geno, region) in trace_groups:
            panel_combos.append((geno, region))

# Place up to 4 trace panels in top row + reuse bottom-left for extra if needed.
# We have combos: D1-DMS, D1-VMS, D1-VLS, D2-DMS, D2-VMS, D2-VLS (6). Use 6 small axes.
n_panels = len(panel_combos)
# Use a 2x3 grid in the left/middle for traces, right column for summary
gs2 = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.40)

trace_axes_pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for idx, (geno, region) in enumerate(panel_combos[:6]):
    rpos, cpos = trace_axes_pos[idx]
    ax = fig.add_subplot(gs2[rpos, cpos])
    g = trace_groups[(geno, region)]
    hits = np.vstack(g["hit"])
    fas = np.vstack(g["fa"])
    nmice = hits.shape[0]
    hmean = np.nanmean(hits, axis=0)
    fmean = np.nanmean(fas, axis=0)
    if nmice > 1:
        hsem = stats.sem(hits, axis=0, nan_policy="omit")
        fsem = stats.sem(fas, axis=0, nan_policy="omit")
    else:
        hsem = np.zeros_like(hmean)
        fsem = np.zeros_like(fmean)
    c = GENO_COLORS[geno]
    ax.plot(t[pmask], hmean[pmask], color=c, lw=2.0, label=f"Hit (rew) n={nmice}")
    ax.fill_between(t[pmask], (hmean - hsem)[pmask], (hmean + hsem)[pmask], color=c, alpha=0.20)
    ax.plot(t[pmask], fmean[pmask], color=c, lw=2.0, ls="--", label=f"FA (unrew) n={nmice}")
    ax.fill_between(t[pmask], (fmean - fsem)[pmask], (fmean + fsem)[pmask], color=c, alpha=0.10)
    ax.axvspan(AUC_LO, AUC_HI, color="gold", alpha=0.12, zorder=0)
    ax.axvspan(LATE_LO, LATE_HI, color="dodgerblue", alpha=0.10, zorder=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{geno}  {region}", fontsize=11, fontweight="bold", color=c)
    ax.set_xlabel("Time from lick (s)", fontsize=8)
    ax.set_ylabel("Delta z-dF/F", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=7)

def _strip_panel(ax, d1v, d2v, title, ylab):
    xs1 = np.random.RandomState(0).normal(0, 0.06, size=len(d1v))
    xs2 = np.random.RandomState(1).normal(1, 0.06, size=len(d2v))
    ax.bar([0, 1], [d1v.mean(), d2v.mean()],
           yerr=[stats.sem(d1v), stats.sem(d2v)], capsize=5,
           color=[GENO_COLORS["D1"], GENO_COLORS["D2"]], alpha=0.5, width=0.6, zorder=1)
    ax.scatter(xs1, d1v, color=GENO_COLORS["D1"], edgecolor="k", s=55, zorder=3)
    ax.scatter(xs2, d2v, color=GENO_COLORS["D2"], edgecolor="k", s=55, zorder=3)
    for x, y, m in zip(xs1, d1v, d1_mice):
        ax.annotate(m.replace("BG_", ""), (x, y), fontsize=6, ha="left", va="bottom")
    for x, y, m in zip(xs2, d2v, d2_mice):
        ax.annotate(m.replace("BG_", ""), (x, y), fontsize=6, ha="left", va="bottom")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"D1\n(n={len(d1v)})", f"D2\n(n={len(d2v)})"], fontsize=9)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)

# Right column top: PERI window (0,1.5) Hit-FA -- lick-transient dominated, null
axb = fig.add_subplot(gs2[0, 3])
_strip_panel(axb, d1_diff, d2_diff,
             f"PERI (0,1.5s)  Hit-FA\nMWU p={u_p:.3f}, d={cohend:.2f}  [gold band]",
             "Hit-FA mean Delta z")

# Right column bottom: LATE window (1.5,3) Hit-FA -- sustained reward / RPE divergence
axl = fig.add_subplot(gs2[1, 3])
_strip_panel(axl, d1_late, d2_late,
             f"LATE (1.5,3s)  Hit-FA  (reward)\nMWU p={ul_p:.3f}, d={cohend_l:.2f}  [blue band]",
             "Hit-FA mean Delta z")

fig.suptitle("BULK 8m: Lick-aligned reward / RPE-like signal  (Hit=rewarded vs FA=unrewarded)\n"
             "gold = peri window (0,1.5s); blue = late reward window (1.5,3s); shading = SEM over mice; unit = mouse",
             fontsize=13, fontweight="bold")
fig.savefig(OUTPNG, dpi=130, bbox_inches="tight")
print()
print("SAVED:", OUTPNG)
print("EXISTS:", os.path.exists(OUTPNG))
