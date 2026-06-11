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
            out[r] = (np.mean([s[:n] for s, _ in trs], axis=0), trs[0][1][:n])
    return out


def collect(session_files, *, use_qc, state_provider, keep_states, max_sessions, excluded):
    lags = lag_grid()
    # per (genotype, region): {subject: list of kernels}; and pulse-triggered traces
    kern = defaultdict(lambda: defaultdict(list))
    pta = defaultdict(lambda: defaultdict(lambda: {"fast": [], "slow": []}))
    ptv = {"t": None}
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
        sources = _region_sources(sess, subj, use_qc)
        for region, (sig, ts) in sources.items():
            segs = build_region_design(sess, sig, ts, state_keep=keep, validate=True)
            if len(segs) >= 1:
                _, k = fit_trf(segs, lags=lags)
                if np.any(np.isfinite(k)):
                    kern[(geno, region)][subj].append(k)
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
    return lags, kern, pta, ptv["t"]


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
    else:
        provider, keep_states = PooledStateProvider(), ["All"]

    lags, kern, pta, ptv_t = collect(files, use_qc=use_qc, state_provider=provider,
                                     keep_states=keep_states, max_sessions=args.max_sessions,
                                     excluded=excl)
    if not kern:
        logging.error("No kernels computed."); sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)

    # ── per-mouse kernel summary + stats ──
    rows, stat_rows = [], []
    regions = sorted({r for (_, r) in kern})
    for region in regions:
        for geno in ("D1", "D2"):
            for subj, mean_k in _per_mouse_mean(kern.get((geno, region), {})):
                ts_ = kernel_timescale(lags, mean_k)
                rows.append({"subject_id": subj, "genotype": geno, "region": region, **ts_})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "G1_kernels.csv", index=False)

    for region in regions:
        sub = metrics[metrics["region"] == region]
        d1 = sub[sub["genotype"] == "D1"]["signed_peak"].values
        d2 = sub[sub["genotype"] == "D2"]["signed_peak"].values
        res = pushpull_sign_contrast(d1, d2)
        res.update({"region": region, "metric": "kernel_signed_peak"})
        stat_rows.append(res)
    if stat_rows:
        format_stats_table(stat_rows, save_path=str(out / "G1_stats.csv"))

    # ── figures: per region (kernel D1 vs D2 + pulse-triggered) ──
    for region in regions:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"G1 — TF-pulse encoding — {region}\n(D1/D2 different animals; kernel = neural ⊛ GCaMP)", fontsize=11)
        # kernel
        ax = axes[0]
        for geno in ("D1", "D2"):
            km = _per_mouse_mean(kern.get((geno, region), {}))
            if not km:
                continue
            K = np.array([k for _, k in km])
            mean = np.nanmean(K, axis=0)
            sem = np.nanstd(K, axis=0) / np.sqrt(max(K.shape[0], 1))
            c = GENOTYPE_COLORS[geno]
            ax.plot(lags, mean, color=c, lw=1.5, label=f"{geno} ({K.shape[0]} mice)")
            ax.fill_between(lags, mean - sem, mean + sem, color=c, alpha=0.2)
        ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6); ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
        ax.set_xlabel("Lag (s): TF → dF/F"); ax.set_ylabel("kernel weight"); ax.set_title("TRF kernel")
        ax.legend(fontsize=8); sns.despine(ax=ax)
        # pulse-triggered (fast solid / slow dashed)
        ax = axes[1]
        if ptv_t is not None:
            for geno in ("D1", "D2"):
                c = GENOTYPE_COLORS[geno]
                for label, style in (("fast", "-"), ("slow", "--")):
                    traces = [np.nanmean(np.array(v[label]), axis=0)
                              for v in pta.get((geno, region), {}).values() if v[label]]
                    if traces:
                        m = np.nanmean(np.array(traces), axis=0)
                        ax.plot(ptv_t, m, color=c, ls=style, lw=1.3,
                                label=f"{geno} {label}")
            ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.set_xlabel("Time from pulse (s)"); ax.set_ylabel("z-dF/F (pre-pulse)")
            ax.set_title("Fast vs slow pulse-triggered"); ax.legend(fontsize=7); sns.despine(ax=ax)
        p = out / f"G1_{region}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        logging.info(f"Saved {p}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
