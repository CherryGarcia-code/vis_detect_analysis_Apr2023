"""C1 for the intersectional MOs-recipient cohort (BG_027-030), in a 2x2.

Session is the statistical unit (n=1 mouse/cell): per-cell pooled AUROC/delta
(compute_delta_and_auroc) PLUS per-session distribution with session-bootstrap
CIs (summarize_sessions_by_cell). Trial-pooled companion PETHs are illustrative.
NEVER pooled with bulk-8m. D1 vs D2 reported as a cell-level sign contrast only.
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
from visdetect_photom.analysis.suppression import (
    build_suppression_dataset, compute_delta_and_auroc,
    compute_session_delta_and_auroc,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
TRACKS = ["behavioral_fa", "sdt_fa"]
SCHEMES = ["scheme1", "scheme3"]

def _plot_brake_2x2(cell_summary, out_dir):
    """2x2 grid: rows = genotype (D1/D2), cols = region (DMS/VMS); bar = AUROC
    (behavioral_fa/scheme3) with session-bootstrap CI; chance line at 0.5."""
    sub = cell_summary[(cell_summary["track"] == "behavioral_fa")
                       & (cell_summary["scheme"] == "scheme3")]
    genos, regions = ["D1", "D2"], ["DMS", "VMS"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), squeeze=False)
    fig.suptitle("C1 brake (intersectional MOs-recipient) — withhold-vs-FA AUROC\n"
                 "(session-unit; n=1 mouse/cell, within-animal)", fontsize=11)
    for ri, geno in enumerate(genos):
        for ci, region in enumerate(regions):
            ax = axes[ri][ci]
            row = sub[(sub["genotype"] == geno) & (sub["region"] == region)]
            if len(row):
                r = row.iloc[0]
                lo = r["auroc_mean"] - r["auroc_ci_lo"]; hi = r["auroc_ci_hi"] - r["auroc_mean"]
                ax.bar([0], [r["auroc_mean"]], color=GENOTYPE_COLORS[geno],
                       yerr=[[max(lo,0)], [max(hi,0)]], capsize=4)
                ax.set_title(f"{geno} · {region} ({r['subject_id']}, {int(r['n_sessions'])} sess)", fontsize=8)
            else:
                ax.set_title(f"{geno} · {region} (no data)", fontsize=8)
            ax.axhline(0.5, color="k", ls="--", lw=0.8)
            ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_ylabel("AUROC", fontsize=8)
            sns.despine(ax=ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cohort_c1_brake_2x2.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    logging.info(f"Saved {p}")

def main():
    ap = argparse.ArgumentParser(description="C1 — intersectional MOs-recipient cohort")
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
    all_sess_scalars, cell_rows, qual_rows = [], [], []
    for track in TRACKS:
        for scheme in SCHEMES:
            df = build_suppression_dataset(sessions, track=track, scheme=scheme,
                                           use_qc=use_qc, state_provider=sp, keep_states=keep)
            if df.empty:
                continue
            df["track"], df["scheme"] = track, scheme

            per_session = compute_session_delta_and_auroc(df)
            if not per_session.empty:
                per_session["track"], per_session["scheme"] = track, scheme
                all_sess_scalars.append(per_session)
                summ = cohort.summarize_sessions_by_cell(per_session)
                # pooled per-cell point estimate (all trials), one row/cell at n=1/cell
                pooled = compute_delta_and_auroc(df)[
                    ["subject_id", "genotype", "region", "delta", "auroc"]].rename(
                    columns={"delta": "delta_pooled", "auroc": "auroc_pooled"})
                merged = summ.merge(pooled, on=["subject_id", "genotype", "region"], how="left")
                merged["track"], merged["scheme"] = track, scheme
                cell_rows.append(merged)

            g = df.copy(); g["finite"] = np.isfinite(g["scalar"].astype(float))
            qn = (g.groupby(["track","scheme","region","genotype","group"])["finite"]
                    .agg(n_total="size", n_finite="sum").reset_index())
            qual_rows.append(qn)

    if not all_sess_scalars:
        logging.error("No waiting-period scalars extracted."); sys.exit(1)
    pd.concat(all_sess_scalars, ignore_index=True).to_csv(
        out_dir / "cohort_c1_session_scalars.csv", index=False)
    cell_summary = pd.concat(cell_rows, ignore_index=True)
    cell_summary.to_csv(out_dir / "cohort_c1_cell_summary.csv", index=False)
    pd.concat(qual_rows, ignore_index=True).to_csv(
        out_dir / "cohort_c1_qualifying_n.csv", index=False)
    _plot_brake_2x2(cell_summary, out_dir)
    logging.info("Done.")

if __name__ == "__main__":
    main()
