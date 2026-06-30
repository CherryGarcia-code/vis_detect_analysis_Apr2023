"""motor_vs_value: Is the pre-lick go-ramp sensory/value (locked to stimulus change)
or motor preparation (locked to the action/lick)?

For each genotype x region (per-mouse mean +/- SEM over mice) compare:
  - CHANGE-aligned Hit response (change_hit): sensory/value, stimulus-locked
  - LICK-aligned Hit response (hit_lick): motor, action-locked
  - LICK-aligned FA response (fa_lick): purely internal/motor (FA has no change)

Quantify per mouse: peak amplitude + peak latency, change- vs lick-aligned.
If the ramp is sharper/earlier locked to the lick than to the change -> motor.
If locked to the change -> sensory/value.

Statistical unit = MOUSE. Per-mouse traces already aggregated in pkl.
DMS/VMS: 2-3 mice/geno (descriptive + permutation paired test on latency).
VLS: 1 mouse/geno -> descriptive only.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PKL = r'e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/results/explore_cache/bulk_extract.pkl'
OUTDIR = r'e:/python_analysis/git_repos/vis_detect_analysis_Apr2023/FIGURES/intersectional_mos/narrative'
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, 'motor_vs_value.png')

d = pickle.load(open(PKL, 'rb'))
ct = d['change']['time']; cpm = d['change']['per_mouse']
lt = d['lick']['time'];   lpm = d['lick']['per_mouse']

# NOTE on metric choice: the change-aligned Hit response is a SLOW MONOTONIC RAMP
# (still climbing at +3.5s), so a peak-latency metric just clips the window edge and
# is uninformative for change-alignment. The motor-vs-value question is really about
# RISE TIMING/SHARPNESS: how tightly concentrated the rise is around each event.
# We therefore use a "rise-onset latency" (time to reach 50% of the local max within
# a symmetric window) PLUS a "peri-event slope" (mean derivative in [-0.25,+0.25]s).
# Tighter/earlier rise locked to the lick than to the change => motor.

def peak_amp_lat(time, trace, lo, hi):
    """abs-max peak within [lo,hi] window; return (amp_signed, latency)."""
    m = (time >= lo) & (time <= hi)
    seg = trace[m]; tt = time[m]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return np.nan, np.nan
    idx = np.nanargmax(np.abs(seg))
    return seg[idx], tt[idx]

def rise_onset_lat(time, trace, lo, hi):
    """Latency at which trace first crosses 50% of its (window) max above the
    pre-event baseline. Captures when the rise happens, robust to slow ramps."""
    m = (time >= lo) & (time <= hi)
    seg = trace[m]; tt = time[m]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return np.nan
    base = np.nanmean(trace[(time >= lo) & (time < min(0, hi))]) if np.any((time>=lo)&(time<0)) else 0.0
    pk = np.nanmax(seg)
    if pk - base <= 1e-6:
        return np.nan
    thr = base + 0.5*(pk - base)
    above = np.where(seg >= thr)[0]
    return tt[above[0]] if above.size else np.nan

def peri_slope(time, trace, half=0.25):
    """Mean derivative (per s) of trace in [-half,+half] around event = rise sharpness."""
    m = (time >= -half) & (time <= half)
    seg = trace[m]; tt = time[m]
    if seg.size < 2 or np.all(np.isnan(seg)):
        return np.nan
    return (seg[-1]-seg[0])/(tt[-1]-tt[0])

# Window for rise search: -1.0 to +3.0 s relative to event (covers change ramp).
WIN = (-1.0, 3.0)

GENO_REG = [('D1','DMS'),('D1','VMS'),('D1','VLS'),
            ('D2','DMS'),('D2','VMS'),('D2','VLS')]
COND_DEF = [
    ('change_hit', cpm, ct, 'Change-aligned Hit (sensory/value)', 'tab:blue'),
    ('hit_lick',   lpm, lt, 'Lick-aligned Hit (motor)',          'tab:green'),
    ('fa_lick',    lpm, lt, 'Lick-aligned FA (internal/motor)',  'tab:red'),
]

def mice_for(pm, geno, reg, cond):
    return sorted(m for (m,g,r,c) in pm if g==geno and r==reg and c==cond)

# Collect per-mouse metrics per (geno,reg,cond): amp, peak-lat, rise-onset-lat, peri-slope
peaks = {}  # (geno,reg,cond) -> dict mouse-> (amp, peak_lat, rise_lat, slope)
for (geno,reg) in GENO_REG:
    for cond, pm, time, _lbl, _col in COND_DEF:
        ms = mice_for(pm, geno, reg, cond)
        peaks[(geno,reg,cond)] = {}
        for m in ms:
            tr = pm[(m,geno,reg,cond)]
            a,pl = peak_amp_lat(time, tr, *WIN)
            rl = rise_onset_lat(time, tr, *WIN)
            sl = peri_slope(time, tr)
            peaks[(geno,reg,cond)][m] = (a, pl, rl, sl)

# ---- Figure: rows = geno x region (DMS,VMS for each geno), overlays of 3 conds ----
plot_combos = [('D1','DMS'),('D1','VMS'),('D2','DMS'),('D2','VMS')]
fig = plt.figure(figsize=(16,12))
gs = fig.add_gridspec(3, 4, height_ratios=[1,1,0.9], hspace=0.42, wspace=0.32)

def mean_sem(pm, time, geno, reg, cond):
    ms = mice_for(pm, geno, reg, cond)
    if not ms: return None,None,0,[]
    M = np.vstack([pm[(m,geno,reg,cond)] for m in ms])
    mu = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0, ddof=1)/np.sqrt(M.shape[0]) if M.shape[0]>1 else np.zeros_like(mu)
    return mu, sem, M.shape[0], ms

for j,(geno,reg) in enumerate(plot_combos):
    ax = fig.add_subplot(gs[0, j])
    for cond, pm, time, lbl, col in COND_DEF:
        mu,sem,n,ms = mean_sem(pm,time,geno,reg,cond)
        if mu is None: continue
        ax.plot(time, mu, color=col, lw=2, label=f'{lbl.split("(")[0].strip()} (n={n})')
        ax.fill_between(time, mu-sem, mu+sem, color=col, alpha=0.18)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlim(-1.5, 3)
    ax.set_title(f'{geno} {reg}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time from event (s)')
    if j==0: ax.set_ylabel(u'Δ z-dF/F (baseline-sub)')
    ax.legend(fontsize=6.5, loc='upper right')

# Row 2: peak-LATENCY summary per mouse, change vs lick (Hit), and fa_lick
# Paired comparison change_hit vs hit_lick latency per mouse (same mice).
def paired_perm(x, y, nperm=10000, seed=42):
    x=np.asarray(x,float); y=np.asarray(y,float)
    msk=~(np.isnan(x)|np.isnan(y)); x=x[msk]; y=y[msk]
    if len(x)<2: return np.nan, np.nan, len(x)
    diff=x-y; obs=np.mean(diff); rng=np.random.default_rng(seed)
    cnt=0
    for _ in range(nperm):
        signs=rng.choice([-1,1], size=len(diff))
        if abs(np.mean(diff*signs))>=abs(obs)-1e-12: cnt+=1
    return obs, (cnt+1)/(nperm+1), len(x)

RIDX = 2  # rise-onset latency index in tuple
lat_stats = {}  # (geno,reg) -> dict
for j,(geno,reg) in enumerate(plot_combos):
    ax = fig.add_subplot(gs[1, j])
    ch_lat = peaks[(geno,reg,'change_hit')]
    hl_lat = peaks[(geno,reg,'hit_lick')]
    fa_lat = peaks[(geno,reg,'fa_lick')]
    common = sorted(set(ch_lat)&set(hl_lat))
    xs=[]; ys=[]
    for m in common:
        xc=ch_lat[m][RIDX]; xl=hl_lat[m][RIDX]
        xs.append(xc); ys.append(xl)
        ax.plot([0,1],[xc,xl],'-o',color='gray',alpha=0.7,ms=4)
    # fa points
    fa_vals=[fa_lat[m][RIDX] for m in fa_lat if not np.isnan(fa_lat[m][RIDX])]
    for m in fa_lat:
        if not np.isnan(fa_lat[m][RIDX]):
            ax.plot([2],[fa_lat[m][RIDX]],'o',color='tab:red',alpha=0.8,ms=5)
    obs,p,n = paired_perm(xs, ys)
    lat_stats[(geno,reg)]=dict(obs=obs,p=p,n=n,
        ch_lat=[ch_lat[m][RIDX] for m in common],
        hl_lat=[hl_lat[m][RIDX] for m in common],
        fa_lat=fa_vals, mice=common)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['Change\nHit','Lick\nHit','Lick\nFA'], fontsize=8)
    ax.set_title(f'{geno} {reg}: rise-onset latency', fontsize=10)
    if j==0: ax.set_ylabel('Rise-onset latency (s, 50% of max)')
    sub = f'paired dChange-dLick={obs:+.2f}s\np={p:.3f} (n={n})' if not np.isnan(obs) else 'n<2'
    ax.text(0.02,0.98,sub,transform=ax.transAxes,va='top',fontsize=7.5,
            bbox=dict(boxstyle='round',fc='wheat',alpha=0.6))

# Row 3: peak-AMPLITUDE summary (Hit change vs Hit lick vs FA lick) per geno x region
ax_amp = fig.add_subplot(gs[2, :2])
labels=[]; pos=0; xticks=[]; xlabs=[]
cols={'change_hit':'tab:blue','hit_lick':'tab:green','fa_lick':'tab:red'}
for (geno,reg) in plot_combos:
    base=pos
    for ci,cond in enumerate(['change_hit','hit_lick','fa_lick']):
        amps=[v[0] for v in peaks[(geno,reg,cond)].values() if not np.isnan(v[0])]
        x=pos
        if amps:
            ax_amp.bar(x, np.mean(amps), width=0.7, color=cols[cond], alpha=0.6,
                       yerr=(np.std(amps,ddof=1)/np.sqrt(len(amps)) if len(amps)>1 else 0),
                       capsize=2)
            ax_amp.plot([x]*len(amps), amps, 'k.', ms=3)
        pos+=1
    xticks.append(base+1); xlabs.append(f'{geno}\n{reg}')
    pos+=0.8
ax_amp.set_xticks(xticks); ax_amp.set_xticklabels(xlabs, fontsize=8)
ax_amp.axhline(0,color='gray',lw=0.6)
ax_amp.set_ylabel('Peak amplitude (Δ z-dF/F)')
ax_amp.set_title('Peak amplitude: Change-Hit (blue) / Lick-Hit (green) / Lick-FA (red)', fontsize=10)

# Row 3 right: text interpretation
ax_txt = fig.add_subplot(gs[2, 2:]); ax_txt.axis('off')
lines=['INTERPRETATION (per-mouse, N=mice):','']
for (geno,reg) in plot_combos:
    s=lat_stats[(geno,reg)]
    cl=np.nanmean(s['ch_lat']) if s['ch_lat'] else np.nan
    hl=np.nanmean(s['hl_lat']) if s['hl_lat'] else np.nan
    fl=np.nanmean(s['fa_lat']) if s['fa_lat'] else np.nan
    verdict='?'
    if not np.isnan(s['obs']):
        # change latency LATER (more positive) than lick latency -> ramp tracks lick -> motor
        verdict='MOTOR-locked' if s['obs']>0.15 else ('SENSORY-locked' if s['obs']<-0.15 else 'mixed')
    lines.append(f'{geno} {reg} (n={s["n"]}): chgLat={cl:+.2f}s lickLat={hl:+.2f}s faLat={fl:+.2f}s')
    lines.append(f'    dChange-dLick={s["obs"]:+.2f}s p={s["p"]:.3f} -> {verdict}')
ax_txt.text(0.0,1.0,'\n'.join(lines),va='top',ha='left',fontsize=9,family='monospace',
            transform=ax_txt.transAxes)

fig.suptitle('Motor vs Value: pre-lick go-ramp aligned to stimulus change vs to the lick (BULK)',
             fontsize=14, fontweight='bold')
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print('SAVED', OUT, 'exists=', os.path.exists(OUT))

# ---- print key numbers ----
print('\n=== RISE-ONSET LATENCY / PERI-SLOPE / AMP per genotype x region (per-mouse) ===')
for (geno,reg) in GENO_REG:
    for cond in ['change_hit','hit_lick','fa_lick']:
        vals=peaks[(geno,reg,cond)]
        rls=[v[2] for v in vals.values() if not np.isnan(v[2])]
        sls=[v[3] for v in vals.values() if not np.isnan(v[3])]
        amps=[v[0] for v in vals.values() if not np.isnan(v[0])]
        if not rls: continue
        print(f'{geno} {reg:3s} {cond:11s} n={len(rls)} riseLat={np.mean(rls):+.3f}s periSlope={np.mean(sls):+.3f}/s amp={np.mean(amps):+.3f}')

print('\n=== PAIRED change_hit vs hit_lick RISE-ONSET LATENCY (per-mouse) ===')
for (geno,reg) in plot_combos:
    s=lat_stats[(geno,reg)]
    print(f'{geno} {reg}: dChange-minus-dLick = {s["obs"]:+.3f}s  p={s["p"]:.4f}  n={s["n"]} mice={s["mice"]}')

# Aggregate across DMS+VMS (the powered regions) for an overall sign test of the paired effect
all_diff=[];
for (geno,reg) in plot_combos:
    s=lat_stats[(geno,reg)]
    for a,b in zip(s['ch_lat'], s['hl_lat']):
        if not (np.isnan(a) or np.isnan(b)): all_diff.append(a-b)
all_diff=np.array(all_diff)
print(f'\nPooled (all mice/geno/reg DMS+VMS) change-minus-lick latency diff: mean={np.mean(all_diff):+.3f}s n_mice_rows={len(all_diff)} pos_frac={np.mean(all_diff>0):.2f}')
obs,p,n=paired_perm(all_diff, np.zeros_like(all_diff))
print(f'  one-sample perm vs 0: obs={obs:+.3f} p={p:.4f}')
