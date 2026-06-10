"""C2 — D1/D2 Response Geometry (mode-aware push-pull + grading + commitment).

Per-region figures (rows = change / lick / anticipation blocks) + a cross-region
push-pull summary + stats CSVs. D1 and D2 are DIFFERENT animals: all push-pull
results are GROUP-LEVEL sign contrasts, never within-animal anticorrelation.

Usage:
    py scripts/analysis/photometry/08_d1_d2_geometry.py
    py scripts/analysis/photometry/08_d1_d2_geometry.py --no-qc
    py scripts/analysis/photometry/08_d1_d2_geometry.py --state-filter Engaged --state-results-dir results/hmm/BG_013
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import GENOTYPE_COLORS, MIN_PHOTOM_CSV_BYTES
from visdetect_photom.core.staging import load_staging_manifest, excluded_mice
from visdetect_photom.analysis.geometry import (
    build_geometry_dataset, run_pushpull_tests, run_grading,
)
from visdetect_photom.analysis.group_statistics import format_stats_table
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

TRACE_EPOCHS = ["change_hit", "change_miss", "hit_lick", "fa_lick",
                "anticipation_hit", "anticipation_miss", "anticipation_cr"]


def _aggregate(trace_list):
    """[(subj, trace)] -> grand mean/SEM over per-mouse traces (N=mice)."""
    if not trace_list:
        return None
    rows = np.array([tr for _, tr in trace_list])
    mean = np.nanmean(rows, axis=0)
    n = rows.shape[0]
    sem = np.nanstd(rows, axis=0, ddof=0) / np.sqrt(max(n, 1))
    return {"mean": mean, "sem": sem, "n_mice": n}


def _plot_traces(ax, t, agg_d1, agg_d2, title, xlabel="Time (s)"):
    for agg, geno in [(agg_d1, "D1"), (agg_d2, "D2")]:
        if agg is None:
            continue
        c = GENOTYPE_COLORS[geno]
        ax.plot(t, agg["mean"], color=c, lw=1.5, label=f"{geno} ({agg['n_mice']} mice)")
        ax.fill_between(t, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
                        color=c, alpha=0.2)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Δ z-dF/F", fontsize=8)
    ax.legend(fontsize=7)
    sns.despine(ax=ax)


def _build_region_figure(region, traces_by_group, time_axis, out_dir):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"C2 — D1/D2 Response Geometry — {region}\n"
                 f"(D1 vs D2 are different animals: group-level sign contrast)", fontsize=12)
    gs = gridspec.GridSpec(3, 3, hspace=0.5, wspace=0.35)

    def agg(epoch):
        return (_aggregate(traces_by_group.get(("D1", region, epoch), [])),
                _aggregate(traces_by_group.get(("D2", region, epoch), [])))

    panels = [
        (0, 0, "change_hit", "Change (Hit)", "Time from change (s)"),
        (0, 1, "change_miss", "Change (Miss)", "Time from change (s)"),
        (0, 2, "anticipation_cr", "Anticipation (CR)", "Time from change (s)"),
        (1, 0, "hit_lick", "Hit lick", "Time from lick (s)"),
        (1, 1, "fa_lick", "FA lick", "Time from lick (s)"),
        (1, 2, "anticipation_hit", "Anticipation (Hit)", "Time from change (s)"),
    ]
    for r, c, epoch, title, xl in panels:
        d1, d2 = agg(epoch)
        _plot_traces(fig.add_subplot(gs[r, c]), time_axis, d1, d2, title, xl)

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"C2_geometry_{region}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {p}")


def _build_summary_figure(pushpull_df, out_dir, metric="signed_auc"):
    if pushpull_df.empty:
        return
    epochs = sorted(pushpull_df["epoch"].unique())
    regions = sorted(pushpull_df["region"].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5), squeeze=False)
    fig.suptitle(f"C2 — Push-pull sign summary ({metric}, D1 vs D2)", fontsize=12)
    x = np.arange(len(epochs))
    for ai, region in enumerate(regions):
        ax = axes[0][ai]
        sub = pushpull_df[pushpull_df["region"] == region].set_index("epoch")
        d1 = [sub.loc[e, "d1_mean"] if e in sub.index else np.nan for e in epochs]
        d2 = [sub.loc[e, "d2_mean"] if e in sub.index else np.nan for e in epochs]
        ax.bar(x - 0.2, d1, 0.4, color=GENOTYPE_COLORS["D1"], label="D1")
        ax.bar(x + 0.2, d2, 0.4, color=GENOTYPE_COLORS["D2"], label="D2")
        for xi, e in enumerate(epochs):
            if e in sub.index and bool(sub.loc[e, "opposite_sign"]):
                ax.text(xi, ax.get_ylim()[1] * 0.9, "*", ha="center", fontsize=14)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(epochs, rotation=45, ha="right", fontsize=7)
        ax.set_title(region, fontsize=10); ax.set_ylabel(metric, fontsize=8)
        ax.legend(fontsize=7); sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "C2_pushpull_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser(description="C2: D1/D2 Response Geometry")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "C2_d1_d2_geometry"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None,
                    help="comma-separated behavioral states to keep (default: pooled)")
    ap.add_argument("--state-results-dir", default=None,
                    help="HMM results dir for --state-filter")
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    use_qc = not args.no_qc
    out_dir = Path(args.output_dir)

    sessions_files = io.find_all_sessions(args.root_dir, recursive=True,
                                          min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(sessions_files)} session files.")

    manifest = load_staging_manifest()
    excl = excluded_mice(manifest)
    if excl:
        logging.info(f"Excluding mice (staging all-Excluded): {sorted(excl)}")

    state_provider, keep_states = None, None
    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir")
            sys.exit(1)
        state_provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
        logging.info(f"State filter: keep {keep_states}")
    else:
        state_provider = PooledStateProvider()
        keep_states = ["All"]

    sessions, n = [], 0
    for sf in sessions_files:
        if args.max_sessions and n >= args.max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.warning(f"skip {sf.get('trials','?')}: {e}")
            continue
        if f"BG_{str(sess.subject_id).zfill(3)}" in excl or sess.subject_id in excl:
            continue
        sessions.append(sess)
        n += 1
        if n % 20 == 0:
            logging.info(f"  loaded {n}")

    per_mouse, traces_by_group, time_axis = build_geometry_dataset(
        sessions, use_qc=use_qc, state_provider=state_provider, keep_states=keep_states)
    if time_axis is None:
        logging.error("No data extracted.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_mouse.to_csv(out_dir / "C2_geometry_metrics.csv", index=False)

    pushpull = run_pushpull_tests(per_mouse, metric="signed_auc")
    grading = run_grading(per_mouse, metric="signed_auc")
    if not pushpull.empty:
        pushpull.to_csv(out_dir / "C2_pushpull_stats.csv", index=False)
    if not grading.empty:
        grading.to_csv(out_dir / "C2_grading.csv", index=False)

    regions = sorted({k[1] for k in traces_by_group})
    for region in regions:
        _build_region_figure(region, traces_by_group, time_axis, out_dir)
    _build_summary_figure(pushpull, out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
