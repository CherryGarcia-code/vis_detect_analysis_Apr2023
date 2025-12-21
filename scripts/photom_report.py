"""
Photometry Report Generator.

This script generates PDF reports visualizing aggregated session data, including learning curves 
and region-specific activity patterns.

Usage:
    python -m scripts.photom_report <summary_csv> [--out <out_pdf>]

Arguments:
    summary_csv : Path to the aggregate summary CSV file (e.g., all_sessions_manifest.csv).
    --out       : Path to the output PDF report.

Example:
    python -m scripts.photom_report pdf_output/all_sessions_manifest.csv --out report.pdf
"""
import os
import argparse
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _compute_region_means_per_event(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()
    def roi_to_region(roi_name: str):
        if isinstance(roi_name, str):
            if ("G0" in roi_name) or ("G2" in roi_name):
                return "DMS"
            if ("G4" in roi_name) or ("G5" in roi_name):
                return "VLS"
        return None
    def region_mean(row, event_prefix: str, region: str):
        vals = []
        for k, v in row.items():
            if isinstance(k, str) and k.startswith(event_prefix):
                if roi_to_region(k) == region and pd.notna(v):
                    vals.append(float(v))
        return float(np.nanmean(vals)) if vals else np.nan
    for ev in ["hit_", "miss_", "change_"]:
        for region in ["DMS", "VLS"]:
            df[f"{ev[:-1]}_{region}_mean"] = df.apply(lambda r: region_mean(r, ev, region), axis=1)
    return df


def _load_data(agg_csv: str, beh_csv: str) -> pd.DataFrame:
    agg_df = pd.read_csv(agg_csv)
    agg_df["mouse_id"] = agg_df["mouse_id"].astype(str)
    agg_df["session_date"] = agg_df["session_date"].astype(str)
    beh_df = pd.read_csv(beh_csv) if os.path.exists(beh_csv) else pd.DataFrame(columns=["mouse_id","session_date"]) 
    if not beh_df.empty:
        beh_df["mouse_id"] = beh_df["mouse_id"].astype(str)
        beh_df["session_date"] = beh_df["session_date"].astype(str)
    df = _compute_region_means_per_event(agg_df).merge(beh_df, on=["mouse_id","session_date"], how="left")
    return df


def _summarize_effects(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    # Per genotype × region: mean hit peak
    for region in ["DMS", "VLS"]:
        col = f"hit_{region}_mean"
        if col in df.columns:
            grp = df.groupby("genotype")[col].mean(numeric_only=True)
            vals = {k: float(v) for k, v in grp.dropna().to_dict().items()}
            if vals:
                lines.append(f"Mean Hit peak — {region}: " + ", ".join([f"{g}={v:.3f}" for g,v in vals.items()]))
            if set(["Drd1","A2a"]).issubset(set(vals.keys())):
                diff = vals.get("Drd1", np.nan) - vals.get("A2a", np.nan)
                if np.isfinite(diff):
                    lines.append(f"Δ(Drd1−A2a) Hit — {region}: {diff:.3f}")

    # Correlations: FA vs Hit peaks
    if "fa_rate" in df.columns:
        for region in ["DMS","VLS"]:
            col = f"hit_{region}_mean"
            if col in df.columns:
                sub = df[["fa_rate", col]].dropna()
                if not sub.empty and sub[col].nunique() > 1:
                    r = float(sub.corr(numeric_only=True).loc["fa_rate", col])
                    lines.append(f"Corr(FA, Hit {region}) = {r:.3f}")

    # Correlations: d' vs Change peaks
    if "d_prime" in df.columns:
        for region in ["DMS","VLS"]:
            col = f"change_{region}_mean"
            if col in df.columns:
                sub = df[["d_prime", col]].dropna()
                if not sub.empty and sub[col].nunique() > 1:
                    r = float(sub.corr(numeric_only=True).loc["d_prime", col])
                    lines.append(f"Corr(d', Change {region}) = {r:.3f}")

    # Optional: RT vs Hit peak
    if "median_rt_s" in df.columns:
        for region in ["DMS","VLS"]:
            col = f"hit_{region}_mean"
            if col in df.columns:
                sub = df[["median_rt_s", col]].dropna()
                if not sub.empty and sub[col].nunique() > 1:
                    r = float(sub.corr(numeric_only=True).loc["median_rt_s", col])
                    lines.append(f"Corr(RT, Hit {region}) = {r:.3f}")

    if not lines:
        lines.append("Insufficient data for statistical summary.")
    return lines


def _add_text_page(pdf: PdfPages, title: str, paragraphs: List[str]):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
    fig.suptitle(title, fontsize=18, y=0.98)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')
    y = 0.95
    for p in paragraphs:
        ax.text(0.02, y, p, fontsize=11, va='top', ha='left', wrap=True)
        y -= 0.06
        if y < 0.1:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis('off')
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, image_path: str, caption: str):
    if not os.path.exists(image_path):
        return
    img = plt.imread(image_path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.05, 0.10, 0.9, 0.8])
    ax.imshow(img)
    ax.axis('off')
    fig.text(0.5, 0.04, caption, ha='center', fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def build_report(out_pdf: str, group_dir: str, agg_csv: str, beh_csv: str):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    df = _load_data(agg_csv, beh_csv)
    summary_lines = _summarize_effects(df)

    with PdfPages(out_pdf) as pdf:
        # Title page
        title = "Photometry summary report"
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        _add_text_page(pdf, title, [f"Generated: {date_str}", f"Sessions: {df[['mouse_id','session_date']].drop_duplicates().shape[0]}"])

        # Summary stats page
        _add_text_page(pdf, "Key effects (genotype × region × behavior)", summary_lines)

        # Append key figures if present
        figs = [
            ("genotype_region_violin.png", "Distribution of peak z-dF/F by genotype and region (per event)"),
            ("genotype_region_bar.png", "Mean peak z-dF/F by genotype and region (per event)"),
            ("session_scatter_by_genotype_region.png", "Session-wise responses across learning (per event)"),
            ("within_session_drift.png", "Within-session drift (trial quartiles) by genotype and region"),
            ("corr_fa_vs_hit_DMS.png", "Correlation: FA rate vs Hit peak — DMS"),
            ("corr_fa_vs_hit_VLS.png", "Correlation: FA rate vs Hit peak — VLS"),
            ("corr_rt_vs_hit_DMS.png", "Correlation: RT vs Hit peak — DMS"),
            ("corr_rt_vs_hit_VLS.png", "Correlation: RT vs Hit peak — VLS"),
            ("corr_psy_vs_change_DMS.png", "Correlation: psychometric slope vs Change peak — DMS"),
            ("corr_psy_vs_change_VLS.png", "Correlation: psychometric slope vs Change peak — VLS"),
            ("corr_dprime_vs_change_DMS.png", "Correlation: d' vs Change peak — DMS"),
            ("corr_dprime_vs_change_VLS.png", "Correlation: d' vs Change peak — VLS"),
        ]
        for fname, caption in figs:
            _add_image_page(pdf, os.path.join(group_dir, fname), caption)


def main():
    parser = argparse.ArgumentParser(description="Generate a compact PDF report with key photometry group plots and summaries")
    parser.add_argument("--out", dest="out_pdf", default=None, help="Output PDF path (default: pdf_output/photometry_report.pdf)")
    parser.add_argument("--group-dir", dest="group_dir", default=None, help="Directory with group plots (default: pdf_output/group_plots)")
    parser.add_argument("--agg-csv", dest="agg_csv", default=None, help="Aggregated photometry CSV (default: pdf_output/photom_summary_all.csv)")
    parser.add_argument("--beh-csv", dest="beh_csv", default=None, help="Behavior metrics CSV (default: pdf_output/behavior_summary_all.csv)")
    args = parser.parse_args()

    out_pdf = args.out_pdf or os.path.join(REPO_ROOT, "pdf_output", "photometry_report.pdf")
    group_dir = args.group_dir or os.path.join(REPO_ROOT, "pdf_output", "group_plots")
    agg_csv = args.agg_csv or os.path.join(REPO_ROOT, "pdf_output", "photom_summary_all.csv")
    beh_csv = args.beh_csv or os.path.join(REPO_ROOT, "pdf_output", "behavior_summary_all.csv")

    build_report(out_pdf, group_dir, agg_csv, beh_csv)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()
