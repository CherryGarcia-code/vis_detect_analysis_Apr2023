"""C2 response geometry for the intersectional MOs-recipient cohort (BG_027-030).

Session is the statistical unit: per-session geometry metrics (signed_auc +
latencies), summarized per cell with session-bootstrap CIs, in a 2x2. NEVER
pooled with bulk-8m. D1 vs D2 = cell-level sign contrast only.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import cohort
from visdetect_photom.core.constants import GENOTYPE_COLORS
from visdetect_photom.analysis.state_provider import PooledStateProvider
from visdetect_photom.analysis.geometry import compute_geometry_metrics_for_session

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
SUMMARY_EPOCHS = ["change_hit", "change_miss", "hit_lick", "fa_lick"]
SUMMARY_VALUES = ("signed_auc", "peak_latency", "onset_latency")

def _plot_geometry_2x2(cell_summary, out_dir, value="signed_auc"):
    genos, regions = ["D1", "D2"], ["DMS", "VMS"]
    epochs = SUMMARY_EPOCHS
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    fig.suptitle(f"C2 geometry (intersectional MOs-recipient) — {value} by epoch\n"
                 "(session-unit; n=1 mouse/cell, within-animal)", fontsize=11)
    x = np.arange(len(epochs))
    for ri, geno in enumerate(genos):
        for ci, region in enumerate(regions):
            ax = axes[ri][ci]
            sub = cell_summary[(cell_summary["genotype"] == geno)
                               & (cell_summary["region"] == region)].set_index("epoch")
            means = [sub.loc[e, f"{value}_mean"] if e in sub.index else np.nan for e in epochs]
            ax.bar(x, means, color=GENOTYPE_COLORS[geno])
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels(epochs, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{geno} · {region}", fontsize=9); ax.set_ylabel(value, fontsize=8)
            sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cohort_c2_geometry_2x2.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    logging.info(f"Saved {p}")

def main():
    ap = argparse.ArgumentParser(description="C2 — intersectional MOs-recipient cohort")
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
    logging.info(f"Loaded {len(sessions)} cohort sessions.")

    sp, keep = PooledStateProvider(), ["All"]
    rows = []
    for sess in sessions:
        srows, _, _ = compute_geometry_metrics_for_session(
            sess, use_qc=use_qc, state_provider=sp, keep_states=keep)
        for r in srows:
            r["recording_id"] = sess.recording_id
            rows.append(r)
    if not rows:
        logging.error("No geometry metrics extracted."); sys.exit(1)
    per_session = pd.DataFrame(rows)
    per_session.to_csv(out_dir / "cohort_c2_session_metrics.csv", index=False)

    # pooled epochs only (change_size NaN); summarize per cell x epoch over sessions
    pooled = per_session[per_session["change_size"].isna()
                         & per_session["epoch"].isin(SUMMARY_EPOCHS)]
    cell_rows = []
    for epoch, g in pooled.groupby("epoch"):
        summ = cohort.summarize_sessions_by_cell(g, value_cols=SUMMARY_VALUES)
        summ["epoch"] = epoch
        cell_rows.append(summ)
    cell_summary = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    cell_summary.to_csv(out_dir / "cohort_c2_cell_summary.csv", index=False)
    if not cell_summary.empty:
        _plot_geometry_2x2(cell_summary, out_dir)
    logging.info("Done.")

if __name__ == "__main__":
    main()
