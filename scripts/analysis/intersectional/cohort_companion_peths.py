"""Companion PETH panels per cell for the intersectional cohort (BG_027-030).

Per cell (subject): a change-aligned panel (change_hit vs change_miss) and a
SEPARATE lick-aligned panel (hit_lick vs fa_lick). Trial-pooled (per-session
mean traces averaged across sessions) — illustrative, not the statistic.
Alignments are never mixed on one panel.
"""
import argparse, logging, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))
from visdetect_photom.core import cohort
from visdetect_photom.analysis.state_provider import PooledStateProvider
from visdetect_photom.analysis.geometry import compute_geometry_metrics_for_session

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
CHANGE_EPOCHS = [("change_hit", "Hit"), ("change_miss", "Miss")]
LICK_EPOCHS = [("hit_lick", "Hit lick"), ("fa_lick", "FA lick")]


def _norm(s):
    s = str(s)
    return s if s.startswith("BG_") else f"BG_{s.zfill(3)}"


def main():
    ap = argparse.ArgumentParser(description="Intersectional cohort companion PETHs")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--root_dir", default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--no-qc", action="store_true", default=False)
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()
    use_qc = not args.no_qc
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sessions = cohort.load_cohort_sessions("intersectional_mos", args.root_dir,
                                           max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded."); sys.exit(1)

    sp, keep = PooledStateProvider(), ["All"]
    # accumulate per (subject, epoch) -> list of per-session mean traces
    acc = defaultdict(list)
    time_axis = None
    for sess in sessions:
        _, traces, t = compute_geometry_metrics_for_session(
            sess, use_qc=use_qc, state_provider=sp, keep_states=keep)
        if t is not None and time_axis is None:
            time_axis = t
        subj = _norm(sess.subject_id)
        for (region, epoch), tr in traces.items():
            acc[(subj, epoch)].append(tr)
    if time_axis is None:
        logging.error("No traces extracted."); sys.exit(1)

    subjects = sorted({k[0] for k in acc})
    fig, axes = plt.subplots(len(subjects), 2, figsize=(11, 3 * max(len(subjects), 1)),
                             squeeze=False)
    fig.suptitle("Intersectional cohort — companion PETHs (trial-pooled; illustrative)", fontsize=12)
    for ri, subj in enumerate(subjects):
        for ci, (epochs, xl, title) in enumerate(
                [(CHANGE_EPOCHS, "Time from change (s)", "change-aligned"),
                 (LICK_EPOCHS, "Time from lick (s)", "lick-aligned")]):
            ax = axes[ri][ci]
            for epoch, label in epochs:
                trs = acc.get((subj, epoch))
                if not trs:
                    continue
                m = np.nanmean(np.array(trs), axis=0)
                ax.plot(time_axis, m, lw=1.4, label=label)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5, alpha=0.4)
            ax.set_title(f"{subj} — {title}", fontsize=9)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("Δ z-dF/F", fontsize=8)
            ax.legend(fontsize=7); sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "cohort_companion_peths.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    logging.info(f"Saved {p}")


if __name__ == "__main__":
    main()
