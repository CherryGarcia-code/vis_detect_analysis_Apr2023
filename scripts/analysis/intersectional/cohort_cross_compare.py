"""Rank-based bulk-vs-intersectional comparison (secondary, caveated).

Compares ONLY indicator-invariant quantities (AUROC here; extend to sign /
latency similarly). Magnitude (dF/F) is never compared across indicators —
enforced by cohort.assert_rank_based.
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))
from visdetect_photom.core import cohort
from visdetect_photom.core.constants import GENOTYPE_COLORS

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def build_cross_compare(cohort_c1_csv, bulk_auroc_csv,
                        track="behavioral_fa", scheme="scheme3"):
    cohort.assert_rank_based("auroc")  # guard: rank-based only
    coh = pd.read_csv(cohort_c1_csv)
    if "track" in coh.columns:
        coh = coh[(coh["track"] == track) & (coh["scheme"] == scheme)]
    bulk = pd.read_csv(bulk_auroc_csv)
    # The real bulk c1_auroc_stats.csv has MANY rows per genotype x region
    # (one per track x scheme x prof_bin). Filter to the cohort-matched slice
    # BEFORE selecting, but only on columns that actually exist (synthetic test
    # CSVs carry just genotype/region/auroc_mean and must still work).
    if "prof_bin" in bulk.columns:
        bulk = bulk[bulk["prof_bin"] == "pooled"]
    if "track" in bulk.columns:
        bulk = bulk[bulk["track"] == track]
    if "scheme" in bulk.columns:
        bulk = bulk[bulk["scheme"] == scheme]
    out = []
    for geno in ("D1", "D2"):
        for region in ("DMS", "VMS"):
            m = cohort.match_cohort_cells(geno, region)
            if not m["intersectional"]:
                continue
            cs = m["intersectional"][0]
            crow = coh[(coh["genotype"] == geno) & (coh["region"] == region)]
            brow = bulk[(bulk["genotype"] == geno) & (bulk["region"] == region)]
            a_c = float(crow.iloc[0]["auroc_mean"]) if len(crow) else np.nan
            a_b = float(brow.iloc[0]["auroc_mean"]) if len(brow) else np.nan
            out.append({"genotype": geno, "region": region,
                        "intersectional_subject": cs, "bulk_subjects": ",".join(m["bulk"]),
                        "auroc_cohort": a_c, "auroc_bulk": a_b,
                        "delta_auroc": a_c - a_b})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser(description="Rank-based bulk-vs-intersectional comparison")
    rr = Path(__file__).resolve().parents[3]
    ap.add_argument("--cohort_c1", default=str(rr / "FIGURES" / "intersectional_mos" / "cohort_c1_cell_summary.csv"))
    ap.add_argument("--bulk_auroc", default=str(rr / "FIGURES" / "C1_fa_suppression" / "c1_auroc_stats.csv"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = build_cross_compare(args.cohort_c1, args.bulk_auroc)
    df.to_csv(out_dir / "cohort_cross_compare.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r.genotype}·{r.region}" for r in df.itertuples()]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["auroc_bulk"], 0.4, label="bulk-8m", color="#999999")
    ax.bar(x + 0.2, df["auroc_cohort"], 0.4, label="MOs-recipient 6f", color="#d62728")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_ylabel("brake AUROC (rank-based; magnitudes not compared)")
    ax.set_title("Cross-cohort brake AUROC (caveated: 6f vs 8m, n=1/cell)")
    ax.legend(); sns.despine(ax=ax)
    fig.savefig(out_dir / "cohort_cross_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Done.")

if __name__ == "__main__":
    main()
