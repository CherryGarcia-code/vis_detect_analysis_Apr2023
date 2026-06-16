"""G1 — TF-Pulse Evidence Encoding (D1 vs D2 baseline-TF kernel + pulse-triggered).

D1 and D2 are DIFFERENT animals: all comparisons are GROUP-LEVEL.
The kernel reflects neural response convolved with GCaMP kinetics (timescale = upper bound).

Usage:
    py scripts/analysis/photometry/09_tf_pulse_encoding.py
    py scripts/analysis/photometry/09_tf_pulse_encoding.py --no-qc
    py scripts/analysis/photometry/09_tf_pulse_encoding.py --state-filter Engaged --state-results-dir results/hmm/BG_013
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import GENOTYPE_COLORS, MIN_PHOTOM_CSV_BYTES, get_roi_region
from visdetect_photom.core.qc import compute_session_roi_qc, check_behavioral_engagement, merge_hemispheres
from visdetect_photom.core.staging import load_staging_manifest, excluded_mice
from visdetect_photom.analysis.group_utils import get_genotype
from visdetect_photom.analysis.group_statistics import pushpull_sign_contrast, format_stats_table
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider, filter_trials_by_state
from visdetect_photom.analysis.tf_kernel import (
    lag_grid, build_region_design, fit_trf, kernel_timescale, shuffle_null,
    pulse_triggered_average,
)
from visdetect_photom.core.stimulus import fast_slow_pulse_times

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _subject_full(sid):
    s = str(sid)
    return f"BG_{s.zfill(3)}" if (not s.startswith("BG_") and s.isdigit()) else s


def _region_sources(session, subj, use_qc):
    if use_qc:
        merged = merge_hemispheres(session, qc_results=compute_session_roi_qc(session))
        return {r: (m["signal"], m["timestamps"]) for r, m in merged.items()}
    by = defaultdict(list)
    for roi, tr in session.photometry_data.items():
        region = get_roi_region(roi, subj)
        if region:
            by[region.rsplit("_", 1)[0]].append((tr.signal, tr.timestamps))
    out = {}
    for r, trs in by.items():
        if len(trs) == 1:
            out[r] = trs[0]
        elif len(trs) >= 2:
            n = min(len(s) for s, _ in trs)
            # ROIs within a session share one timestamp axis (same processed_df in
            # session.py), so averaging signals on trs[0]'s clock is sample-aligned.
            out[r] = (np.mean([s[:n] for s, _ in trs], axis=0), trs[0][1][:n])
    return out


def collect(session_files, *, use_qc, state_provider, keep_states, max_sessions, excluded,
            validate_anchor):
    lags = lag_grid()
    # per (genotype, region): {subject: list of session kernels} (per-mouse replication)
    kern = defaultdict(lambda: defaultdict(list))
    # per (genotype, region): all segments pooled across mice (existence test + null)
    seg_pool = defaultdict(list)
    pta = defaultdict(lambda: defaultdict(lambda: {"fast": [], "slow": []}))
    ptv = {"t": None}
    qc_rows = []   # per session×region trial-disposition counts (effective-N / alignment QC)
    n = 0
    for sf in session_files:
        if max_sessions and n >= max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception:
            continue
        geno = get_genotype(sess.subject_id)
        if geno == "Unknown":
            continue
        subj = _subject_full(sess.subject_id)
        if (subj in excluded or f"BG_{str(sess.subject_id).zfill(3)}" in excluded
                or str(sess.subject_id) in excluded):
            continue
        if use_qc and not check_behavioral_engagement(sess)["pass"]:
            continue
        keep = None
        if state_provider is not None and keep_states is not None:
            keep = filter_trials_by_state(sess, state_provider, keep_states)
            if len(keep) == 0:
                logging.warning(f"  {subj}: state filter kept 0 trials — session skipped.")
                continue
        sources = _region_sources(sess, subj, use_qc)
        for region, (sig, ts) in sources.items():
            segs, counts = build_region_design(sess, sig, ts, state_keep=keep,
                                               validate=validate_anchor, return_counts=True)
            qc_rows.append({"subject_id": subj, "genotype": geno, "region": region, **counts})
            if len(segs) >= 1:
                _, k = fit_trf(segs, lags=lags)
                if np.any(np.isfinite(k)):
                    kern[(geno, region)][subj].append(k)
                seg_pool[(geno, region)].extend(segs)
            # pulse-triggered companion
            fast_t, slow_t = [], []
            for tr in sess.trials:
                if keep is not None and tr.trial_index not in keep:
                    continue
                f, s = fast_slow_pulse_times(tr)
                fast_t.append(f); slow_t.append(s)
            fast_t = np.concatenate(fast_t) if fast_t else np.array([])
            slow_t = np.concatenate(slow_t) if slow_t else np.array([])
            for label, times in (("fast", fast_t), ("slow", slow_t)):
                res = pulse_triggered_average(sig, ts, times)
                if res is not None:
                    ptv["t"], mean, _ = res
                    pta[(geno, region)][subj][label].append(mean)
        n += 1
        if n % 20 == 0:
            logging.info(f"  processed {n}")
    return lags, kern, seg_pool, pta, ptv["t"], qc_rows


def _per_mouse_mean(subj_map):
    """{subject: [arrays]} -> list of (subject, mean_array)."""
    return [(s, np.nanmean(np.array(a), axis=0)) for s, a in subj_map.items() if a]


def main():
    ap = argparse.ArgumentParser(description="G1: TF-pulse evidence encoding")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "G1_tf_pulse_encoding"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None)
    ap.add_argument("--state-results-dir", default=None)
    ap.add_argument("--max_sessions", type=int, default=None)
    ap.add_argument("--n-shuffles", type=int, default=200,
                    help="circular-shift resamples for the pooled shuffle-null band")
    ap.add_argument("--pooled-max-seg", type=int, default=5000,
                    help="cap on pooled segments for the existence-test kernel+null; point "
                         "and null share this sample size (coherent band, bounded runtime)")
    ap.add_argument("--validate-anchor", action="store_true", default=False,
                    help="re-enable the change-anchor gate (OFF by default: it cancels the "
                         "photometry onset algebraically and its 50ms tolerance is below the "
                         "50ms stimulus quantization, so it spuriously drops well-aligned trials)")
    args = ap.parse_args()

    use_qc = not args.no_qc
    out = Path(args.output_dir)
    files = io.find_all_sessions(args.root_dir, recursive=True, min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(files)} session files.")

    excl = excluded_mice(load_staging_manifest())
    if excl:
        logging.info(f"Excluding mice (staging all-Excluded): {sorted(excl)}")

    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir"); sys.exit(1)
        provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
        logging.info(f"State filter: keep {keep_states} (from {args.state_results_dir})")
    else:
        provider, keep_states = PooledStateProvider(), ["All"]

    if not args.validate_anchor:
        logging.info("Change-anchor gate DISABLED (default) — see --validate-anchor.")
    lags, kern, seg_pool, pta, ptv_t, qc_rows = collect(
        files, use_qc=use_qc, state_provider=provider, keep_states=keep_states,
        max_sessions=args.max_sessions, excluded=excl, validate_anchor=args.validate_anchor)
    if not kern:
        logging.error("No kernels computed."); sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    regions = sorted({r for (_, r) in kern})

    # ── effective-N / alignment QC (per session×region) ──
    if qc_rows:
        qc_df = pd.DataFrame(qc_rows)
        qc_df.to_csv(out / "G1_alignment_qc.csv", index=False)
        agg = qc_df.groupby("genotype")[["n_seen", "n_validate_drop", "n_empty_window",
                                         "n_too_short", "n_kept", "n_pulses"]].sum()
        logging.info(f"Effective-N (trials summed over sessions):\n{agg}")

    # ── pooled existence kernel + shuffle-null band per (genotype, region) ──
    pooled = {}  # (geno, region) -> {kernel, lo, hi, peak_lag, n_seg, n_total}
    _prng = np.random.default_rng(42)
    for region in regions:
        for geno in ("D1", "D2"):
            segs = seg_pool.get((geno, region), [])
            n_total = len(segs)
            if n_total < 2:
                continue
            # Cap segments for the existence test so the pooled kernel AND its null
            # share one sample size (coherent band) and runtime stays bounded as data grows.
            use_segs = segs
            if n_total > args.pooled_max_seg:
                idx = _prng.choice(n_total, args.pooled_max_seg, replace=False)
                use_segs = [segs[i] for i in idx]
            _, pk = fit_trf(use_segs, lags=lags)
            _, lo, hi = shuffle_null(use_segs, lags=lags, n_shuffles=args.n_shuffles)
            pooled[(geno, region)] = {"kernel": pk, "lo": lo, "hi": hi,
                                      "peak_lag": kernel_timescale(lags, pk)["peak_lag"],
                                      "n_seg": len(use_segs), "n_total": n_total}

    # ── per-mouse kernel summary (incl. amplitude at the pooled peak lag, item 4) ──
    rows, stat_rows = [], []
    for region in regions:
        for geno in ("D1", "D2"):
            pl = pooled.get((geno, region), {}).get("peak_lag")
            pl_idx = int(np.argmin(np.abs(lags - pl))) if (pl is not None and np.isfinite(pl)) else None
            for subj, mean_k in _per_mouse_mean(kern.get((geno, region), {})):
                ts_ = kernel_timescale(lags, mean_k)
                sp_at = (float(mean_k[pl_idx]) if (pl_idx is not None
                         and np.isfinite(mean_k[pl_idx])) else np.nan)
                rows.append({"subject_id": subj, "genotype": geno, "region": region, **ts_,
                             "pooled_peak_lag": pl, "signed_peak_at_pooled_lag": sp_at})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "G1_kernels.csv", index=False)

    # D1-vs-D2 contrast on the stabilized (pooled-lag) amplitude when available
    for region in regions:
        sub = metrics[metrics["region"] == region]
        col = ("signed_peak_at_pooled_lag"
               if sub["signed_peak_at_pooled_lag"].notna().any() else "signed_peak")
        d1 = sub[sub["genotype"] == "D1"][col].dropna().values
        d2 = sub[sub["genotype"] == "D2"][col].dropna().values
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "metric": col})
        stat_rows.append(res)
    if stat_rows:
        format_stats_table(stat_rows, save_path=str(out / "G1_stats.csv"))

    # ── figures: per region (2×2) ──
    jit = np.random.default_rng(0)
    gate = "anchor-gate ON" if args.validate_anchor else "anchor-gate OFF"
    for region in regions:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(f"G1 — TF-pulse encoding — {region}\n"
                     f"(D1/D2 different animals; kernel = neural ⊛ GCaMP; {gate})", fontsize=11)

        # [0,0] per-mouse-mean kernel ± SEM (mouse-level replication)
        ax = axes[0, 0]
        for geno in ("D1", "D2"):
            km = _per_mouse_mean(kern.get((geno, region), {}))
            if not km:
                continue
            K = np.array([k for _, k in km])
            mean = np.nanmean(K, axis=0)
            sem = np.nanstd(K, axis=0, ddof=0) / np.sqrt(max(K.shape[0], 1))
            c = GENOTYPE_COLORS[geno]
            ax.plot(lags, mean, color=c, lw=1.5, label=f"{geno} ({K.shape[0]} mice)")
            ax.fill_between(lags, mean - sem, mean + sem, color=c, alpha=0.2)
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6); ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("Lag (s): TF → dF/F"); ax.set_ylabel("kernel weight")
        ax.set_title("Per-mouse-mean TRF kernel ± SEM"); ax.legend(fontsize=8); sns.despine(ax=ax)

        # [0,1] pooled kernel vs shuffle-null band (existence test)
        ax = axes[0, 1]
        for geno in ("D1", "D2"):
            pj = pooled.get((geno, region))
            if pj is None:
                continue
            c = GENOTYPE_COLORS[geno]
            seg_lbl = (f"{pj['n_seg']}/{pj['n_total']}" if pj["n_total"] > pj["n_seg"]
                       else f"{pj['n_seg']}")
            ax.plot(lags, pj["kernel"], color=c, lw=1.6, label=f"{geno} pooled ({seg_lbl} seg)")
            ax.fill_between(lags, pj["lo"], pj["hi"], color=c, alpha=0.15)
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6); ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("Lag (s): TF → dF/F"); ax.set_ylabel("pooled kernel weight")
        ax.set_title(f"Pooled kernel vs shuffle-null ({args.n_shuffles}×)\nreal where line exits the band")
        ax.legend(fontsize=8); sns.despine(ax=ax)

        # [1,0] fast/slow pulse-triggered (fast solid / slow dashed)
        ax = axes[1, 0]
        if ptv_t is not None:
            for geno in ("D1", "D2"):
                c = GENOTYPE_COLORS[geno]
                for label, style in (("fast", "-"), ("slow", "--")):
                    traces = [np.nanmean(np.array(v[label]), axis=0)
                              for v in pta.get((geno, region), {}).values() if v[label]]
                    if traces:
                        m = np.nanmean(np.array(traces), axis=0)
                        ax.plot(ptv_t, m, color=c, ls=style, lw=1.3, label=f"{geno} {label}")
            ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.set_xlabel("Time from pulse (s)"); ax.set_ylabel("z-dF/F (pre-pulse)")
            ax.set_title("Fast vs slow pulse-triggered"); ax.legend(fontsize=7); sns.despine(ax=ax)

        # [1,1] per-mouse signed peak @ pooled lag (shows N + spread directly)
        ax = axes[1, 1]
        sub = metrics[metrics["region"] == region]
        for i, geno in enumerate(("D1", "D2")):
            vals = sub[sub["genotype"] == geno]["signed_peak_at_pooled_lag"].dropna().values
            if vals.size:
                x = np.full(vals.size, i, float) + jit.uniform(-0.08, 0.08, vals.size)
                ax.scatter(x, vals, color=GENOTYPE_COLORS[geno], s=40, alpha=0.85, zorder=3)
                ax.hlines(np.mean(vals), i - 0.2, i + 0.2, color=GENOTYPE_COLORS[geno], lw=2)
        ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["D1", "D2"]); ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("signed peak @ pooled lag"); ax.set_title("Per-mouse kernel peak (N, spread)")
        sns.despine(ax=ax)

        p = out / f"G1_{region}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        logging.info(f"Saved {p}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
