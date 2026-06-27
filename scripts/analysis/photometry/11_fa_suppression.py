"""C1 — FA suppression-failure (MOs-D2 brake): waiting-period prediction of
withhold-vs-lick from bulk D1/D2 signal.

Two tracks (behavioral_fa primary, sdt_fa control) x two window schemes
(scheme1 baseline-onset fixed, scheme3 hazard-time-matched). Per-mouse delta
(withhold-lick) + single-trial AUROC, group push-pull sign contrast, and a
coarse proficiency split. D1 and D2 are DIFFERENT animals: push-pull is a
GROUP-LEVEL sign contrast.

Usage:
    py scripts/analysis/photometry/11_fa_suppression.py
    py scripts/analysis/photometry/11_fa_suppression.py --no-qc
    py scripts/analysis/photometry/11_fa_suppression.py --max_sessions 10
    py scripts/analysis/photometry/11_fa_suppression.py --state-filter Engaged --state-results-dir results/hmm/BG_013
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import io
from visdetect_photom.core.session import load_session_from_files
from visdetect_photom.core.constants import GENOTYPE_COLORS, MIN_PHOTOM_CSV_BYTES
from visdetect_photom.core.staging import load_staging_manifest, excluded_mice
from visdetect_photom.analysis.state_provider import PooledStateProvider, HMMStateProvider
from visdetect_photom.analysis.suppression import (
    build_suppression_dataset, compute_delta_and_auroc, run_suppression_stats,
    assign_proficiency_bins,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

TRACKS = ["behavioral_fa", "sdt_fa"]
SCHEMES = ["scheme1", "scheme3"]


def _load_sessions(args, excl):
    files = io.find_all_sessions(args.root_dir, recursive=True,
                                 min_photom_bytes=MIN_PHOTOM_CSV_BYTES)
    logging.info(f"Discovered {len(files)} session files.")
    sessions, n = [], 0
    for sf in files:
        if args.max_sessions and n >= args.max_sessions:
            break
        try:
            sess = load_session_from_files(sf)
        except Exception as e:
            logging.warning(f"skip {sf.get('trials', '?')}: {e}")
            continue
        if f"BG_{str(sess.subject_id).zfill(3)}" in excl or sess.subject_id in excl:
            continue
        sessions.append(sess)
        n += 1
    logging.info(f"Loaded {len(sessions)} sessions.")
    return sessions


def _qualifying_n(per_trial_df):
    if per_trial_df.empty:
        return pd.DataFrame()
    g = per_trial_df.copy()
    g["finite"] = np.isfinite(g["scalar"].astype(float))
    return (g.groupby(["track", "scheme", "region", "genotype", "group"])["finite"]
             .agg(n_total="size", n_finite="sum").reset_index())


def _plot_delta_summary(pushpull_df, out_dir):
    if pushpull_df.empty:
        return
    regions = sorted(pushpull_df["region"].unique())
    keys = pushpull_df[["track", "scheme"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * max(len(regions), 1), 5),
                             squeeze=False)
    fig.suptitle("C1 — waiting-period Δ(withhold−lick), D1 vs D2 (group-level)", fontsize=12)
    x = np.arange(len(keys))
    for ai, region in enumerate(regions):
        ax = axes[0][ai]
        sub = pushpull_df[pushpull_df["region"] == region].set_index(["track", "scheme"])
        d1 = [sub.loc[tuple(k), "d1_mean"] if tuple(k) in sub.index else np.nan for k in keys]
        d2 = [sub.loc[tuple(k), "d2_mean"] if tuple(k) in sub.index else np.nan for k in keys]
        ax.bar(x - 0.2, d1, 0.4, color=GENOTYPE_COLORS["D1"], label="D1")
        ax.bar(x + 0.2, d2, 0.4, color=GENOTYPE_COLORS["D2"], label="D2")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["/".join(k) for k in keys], rotation=45, ha="right", fontsize=7)
        ax.set_title(region, fontsize=10)
        ax.set_ylabel("Δ z-dF/F (withhold−lick)", fontsize=8)
        ax.legend(fontsize=7)
        sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "C1_delta_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {p}")


def main():
    ap = argparse.ArgumentParser(description="C1: FA suppression-failure (MOs-D2 brake)")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "C1_fa_suppression"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--state-filter", default=None,
                    help="comma-separated behavioral states to keep (default: pooled)")
    ap.add_argument("--state-results-dir", default=None)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    use_qc = not args.no_qc
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_staging_manifest()
    excl = excluded_mice(manifest)
    if excl:
        logging.info(f"Excluding mice (staging all-Excluded): {sorted(excl)}")

    if args.state_filter:
        if not args.state_results_dir:
            logging.error("--state-filter requires --state-results-dir")
            sys.exit(1)
        state_provider = HMMStateProvider(args.state_results_dir)
        keep_states = [s.strip() for s in args.state_filter.split(",")]
    else:
        state_provider = PooledStateProvider()
        keep_states = ["All"]

    sessions = _load_sessions(args, excl)
    if not sessions:
        logging.error("No sessions loaded.")
        sys.exit(1)

    prof_bins = assign_proficiency_bins(sessions, manifest)

    all_trials, all_pushpull, all_auroc = [], [], []
    for track in TRACKS:
        for scheme in SCHEMES:
            df = build_suppression_dataset(
                sessions, track=track, scheme=scheme, use_qc=use_qc,
                state_provider=state_provider, keep_states=keep_states, manifest=manifest)
            if df.empty:
                continue
            df["prof_bin"] = df["recording_id"].map(prof_bins)
            all_trials.append(df)

            # pooled (primary)
            pm = compute_delta_and_auroc(df)
            pp, au = run_suppression_stats(pm)
            for frame, store in ((pp, all_pushpull), (au, all_auroc)):
                if not frame.empty:
                    frame = frame.copy()
                    frame["track"], frame["scheme"], frame["prof_bin"] = track, scheme, "pooled"
                    store.append(frame)
            # proficiency split (robustness)
            for b in ("less", "more"):
                pmb = compute_delta_and_auroc(df[df["prof_bin"] == b])
                ppb, aub = run_suppression_stats(pmb)
                for frame, store in ((ppb, all_pushpull), (aub, all_auroc)):
                    if not frame.empty:
                        frame = frame.copy()
                        frame["track"], frame["scheme"], frame["prof_bin"] = track, scheme, b
                        store.append(frame)

    if not all_trials:
        logging.error("No waiting-period scalars extracted.")
        sys.exit(1)

    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.to_csv(out_dir / "c1_per_trial_scalars.csv", index=False)
    _qualifying_n(trials_df).to_csv(out_dir / "c1_qualifying_n.csv", index=False)
    if all_pushpull:
        pd.concat(all_pushpull, ignore_index=True).to_csv(out_dir / "c1_pushpull_stats.csv", index=False)
    if all_auroc:
        pd.concat(all_auroc, ignore_index=True).to_csv(out_dir / "c1_auroc_stats.csv", index=False)

    if all_pushpull:
        pooled_pp = pd.concat(all_pushpull, ignore_index=True)
        _plot_delta_summary(pooled_pp[pooled_pp["prof_bin"] == "pooled"], out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
