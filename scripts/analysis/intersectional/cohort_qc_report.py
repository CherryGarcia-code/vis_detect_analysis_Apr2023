"""6f QC calibration report for the intersectional cohort (BG_027-030).

Applies the SAME 8m-tuned QC (compute_session_roi_qc) and only REPORTS the
per-ROI metrics + pass flag, so 6f pass-rates and metric distributions per cell
are visible. An indicator-aware threshold is introduced ONLY if these data
demand it, and then documented in constants -- never silently changed here.
"""
import argparse, logging, sys
from pathlib import Path
import pandas as pd

_repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo_root / "src"))

from visdetect_photom.core import cohort
from visdetect_photom.core.qc import compute_session_roi_qc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _norm_subject(subject_id):
    """Normalize subject_id to BG_### format for grouping/reporting."""
    s = str(subject_id)
    if s.startswith("BG_"):
        return s
    return f"BG_{s.zfill(3)}" if s.isdigit() else s


def main():
    rr = _repo_root
    ap = argparse.ArgumentParser(description="Intersectional cohort QC report")
    ap.add_argument("--root_dir", default=str(rr / "photom_data" / "intrsct_GCaMP6f"))
    ap.add_argument("--output_dir", default=str(rr / "FIGURES" / "intersectional_mos"))
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = cohort.load_cohort_sessions("intersectional_mos", args.root_dir,
                                           max_sessions=args.max_sessions)
    if not sessions:
        logging.error("No cohort sessions loaded from %s", args.root_dir)
        sys.exit(1)

    rows = []
    for sess in sessions:
        qc = compute_session_roi_qc(sess)
        for roi, metrics in qc.items():
            row = {
                "subject_id": _norm_subject(getattr(sess, "subject_id", "")),
                "recording_id": getattr(sess, "recording_id", "")
                or getattr(sess, "session_id", ""),
                "session_id": getattr(sess, "session_id", ""),
                "roi": roi,
            }
            for k, v in metrics.items():
                # fail_reasons is a list -> join to a CSV-friendly string
                row[k] = ";".join(map(str, v)) if isinstance(v, (list, tuple)) else v
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "cohort_qc_report.csv"
    df.to_csv(out_csv, index=False)
    logging.info("Wrote %s (%d roi-rows from %d sessions).",
                 out_csv, len(df), len(sessions))

    # Brief, report-only pass-rate summary to the log (no threshold change).
    if "pass" in df.columns and not df.empty:
        for region, g in df.groupby("region" if "region" in df.columns else "roi"):
            rate = float(g["pass"].mean())
            logging.info("  pass-rate [%s]: %.0f%% (%d/%d ROIs)",
                         region, 100 * rate, int(g["pass"].sum()), len(g))


if __name__ == "__main__":
    main()
