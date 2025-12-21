"""
Group Analysis for Photometry Data.

This script aggregates session summaries to perform group-level analysis, such as comparing 
responses across regions (DMS vs VLS) or tracking changes over time.

Usage:
    python -m scripts.photom_group_analysis <summary_dir> [--out <out_dir>]

Arguments:
    summary_dir : Directory containing session summary CSVs (e.g., output of photometry_analysis).
    --out       : Output directory for group analysis results.

Example:
    python -m scripts.photom_group_analysis pdf_output --out group_analysis_results
"""
import os
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _roi_to_region(roi_name: str) -> Optional[str]:
    # Match the convention used in photometry_analysis
    if any(token in roi_name for token in ["G0", "G2"]):
        return "DMS"
    if any(token in roi_name for token in ["G4", "G5"]):
        return "VLS"
    return None


def load_agg(agg_csv: str) -> pd.DataFrame:
    if not os.path.exists(agg_csv):
        raise FileNotFoundError(f"Aggregate CSV not found: {agg_csv}")
    df = pd.read_csv(agg_csv)
    return df


def melt_agg_to_long(agg_df: pd.DataFrame) -> pd.DataFrame:
    # Identify event columns (hit_, miss_, change_)
    value_cols = [c for c in agg_df.columns if isinstance(c, str) and (c.startswith("hit_") or c.startswith("miss_") or c.startswith("change_"))]
    if not value_cols:
        return pd.DataFrame()
    df_long = agg_df.melt(
        id_vars=["mouse_id", "session_date", "genotype", "regions"],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    # Split metric into event + roi key
    def split_metric(s: str):
        # e.g., hit_zscored_G2_clean_signal_dff
        parts = s.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, s
    df_long[["event", "roi_key"]] = df_long["metric"].apply(lambda s: pd.Series(split_metric(s)))
    # Map roi -> region
    df_long["region"] = df_long["roi_key"].apply(_roi_to_region)
    # Tidy types
    df_long["mouse_id"] = df_long["mouse_id"].astype(str)
    return df_long.dropna(subset=["value", "region"])  # keep rows with values and mapped regions


def _ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def plot_violin_by_genotype_region(df_long: pd.DataFrame, out_dir: str) -> str:
    _ensure_out(out_dir)
    sns.set_context('talk')
    # Use per-mouse means across sessions to avoid overweighting long training
    mouse_means = (
        df_long.groupby(["mouse_id", "genotype", "region", "event"])  # per mouse/event/region
              .agg(value=("value", "mean")).reset_index()
    )
    g = sns.catplot(
        data=mouse_means,
        x="genotype",
        y="value",
        hue="region",
        col="event",
        kind="violin",
        inner=None,
        cut=0,
        height=4,
        aspect=0.9,
    )
    # Overlay per-mouse points
    sns.stripplot(
        data=mouse_means,
        x="genotype",
        y="value",
        hue="region",
        dodge=True,
        ax=None,
    )
    for ax in g.axes.flatten():
        ax.set_ylabel("peak z-dF/F (0–1s)")
        sns.despine(ax=ax)
    # Reduce duplicate legends
    try:
        g._legend.remove()  # type: ignore[attr-defined]
    except Exception:
        pass
    out_png = os.path.join(out_dir, "genotype_region_violin.png")
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    return out_png


def plot_bar_with_ci(df_long: pd.DataFrame, out_dir: str) -> str:
    _ensure_out(out_dir)
    sns.set_context('talk')
    # Aggregate to per-mouse first, then summary across mice for CI bars
    mouse_means = (
        df_long.groupby(["mouse_id", "genotype", "region", "event"]).agg(value=("value", "mean")).reset_index()
    )
    summary = (
        mouse_means.groupby(["genotype", "region", "event"]).agg(
            mean=("value", "mean"),
            sem=("value", "sem"),
            n=("value", "count"),
        ).reset_index()
    )
    g = sns.catplot(
        data=summary,
        x="genotype",
        y="mean",
        hue="region",
        col="event",
        kind="bar",
        height=4,
        aspect=0.9,
        ci=None,
    )
    # Add error bars manually (SEM)
    for (genotype, event), df_sub in summary.groupby(["genotype", "event"]):
        pass  # seaborn handles bars; error bars are tricky across facets; skip for now to keep simple
    for ax in g.axes.flatten():
        ax.set_ylabel("mean peak z-dF/F (0–1s)")
        sns.despine(ax=ax)
    try:
        g._legend.remove()  # type: ignore[attr-defined]
    except Exception:
        pass
    out_png = os.path.join(out_dir, "genotype_region_bar.png")
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    return out_png


def plot_session_scatter(df_long: pd.DataFrame, out_dir: str) -> str:
    _ensure_out(out_dir)
    sns.set_context('talk')
    # Order sessions within mouse for x-axis (ordinal)
    df = df_long.copy()
    # Convert to sortable date if possible
    def _parse_date(s):
        try:
            return pd.to_datetime(str(s), format="%Y%m%d")
        except Exception:
            return pd.NaT
    df["session_date_parsed"] = df["session_date"].apply(_parse_date)
    df = df.dropna(subset=["session_date_parsed"]).sort_values(["mouse_id", "session_date_parsed"])  
    df["session_index"] = df.groupby("mouse_id").cumcount() + 1

    g = sns.relplot(
        data=df,
        x="session_index",
        y="value",
        hue="genotype",
        style="region",
        col="event",
        kind="scatter",
        height=4,
        aspect=0.9,
    )
    for ax in g.axes.flatten():
        ax.set_xlabel("session index")
        ax.set_ylabel("peak z-dF/F (0–1s)")
        sns.despine(ax=ax)
    out_png = os.path.join(out_dir, "session_scatter_by_genotype_region.png")
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    return out_png


def _parse_genotype_file(path: str) -> Dict[str, Dict[str, object]]:
    """Parse mouse_genotypes_and_procedeures.txt into mapping BG_XXX -> {genotype, regions[]}"""
    mapping: Dict[str, Dict[str, object]] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('BG_'):
                continue
            parts = line.replace(';', '').split(',')
            left = parts[0]
            regions_part = (parts[1] if len(parts) > 1 else '').strip()
            tokens = left.split()
            if len(tokens) >= 2:
                mouse = tokens[0]  # e.g., BG_016
                genotype = tokens[1]
            else:
                continue
            regions = [r.strip() for r in regions_part.split('&')] if regions_part else []
            mapping[mouse] = {"genotype": genotype, "regions": regions}
    return mapping


def _compute_peak_zdf_over_window(df: pd.DataFrame, roi_cols: List[str], start_s: float, end_s: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df is None or df.empty:
        return out
    idx = pd.to_numeric(df.index, errors="coerce")
    mask = (idx >= start_s) & (idx <= end_s)
    for c in roi_cols:
        if c in df.columns:
            vals = pd.to_numeric(df.loc[mask, c], errors="coerce")
            out[c] = float(vals.max(skipna=True))
    return out


def _roi_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith("zscored_") and c.endswith("_clean_signal_dff")]


def plot_fa_session_scatter(out_dir: str) -> Optional[str]:
    """Compute FA early (<=3s) and late (>3s) peak z-dF/F per session and plot scatter by genotype/region.

    This reprocesses sessions directly from photom_data to avoid requiring FA columns in aggregates.
    """
    # Lazy import helpers to avoid hard dep when not needed
    try:
        from scripts.photometry_analysis import find_all_sessions  # type: ignore
        from vis_detect_helpers_v9 import process_session_data, extract_photom_windows_from_session_s  # type: ignore
    except Exception as e:
        print("FA scatter: failed to import helpers:", e)
        return None

    geno_map = _parse_genotype_file(os.path.join(REPO_ROOT, 'photom_data', 'mouse_genotypes_and_procedeures.txt'))

    # Discover sessions
    sessions = find_all_sessions(os.path.join(REPO_ROOT, 'photom_data'), recursive=True)
    if not sessions:
        print("FA scatter: no sessions discovered")
        return None

    rows: List[Dict[str, object]] = []
    peak_window = (0.0, 1.0)

    for i, s in enumerate(sessions):
        try:
            sess_df = process_session_data([s["photom"]], [s["photom_io"]], [s["session_settings"]], [s["trials"]])
        except Exception:
            continue
        if sess_df is None or sess_df.empty:
            continue

        mouse_id = str(sess_df.get("mouse_id", pd.Series([None])).iloc[0])
        session_date = str(sess_df.get("session_date", pd.Series([None])).iloc[0])
        # Genotype
        mouse_key = f"BG_{int(float(mouse_id)):03d}" if mouse_id and mouse_id != 'None' else None
        genotype = geno_map.get(mouse_key, {}).get("genotype") if mouse_key else None

        # Extract FA windows
        try:
            early_FA, late_FA, _, _ = extract_photom_windows_from_session_s(sess_df, "FA")
        except Exception:
            continue

        def region_mean_from_windows(windows: List[pd.DataFrame]) -> Dict[str, float]:
            # concat peaks per trial, then mean per region
            peaks = []
            for w in windows:
                try:
                    p = _compute_peak_zdf_over_window(w, _roi_cols(w), *peak_window)
                except Exception:
                    p = {}
                if p:
                    peaks.append(p)
            if not peaks:
                return {}
            # average ROI peaks across trials per region
            region_vals: Dict[str, List[float]] = {"DMS": [], "VLS": []}
            for p in peaks:
                for k, v in p.items():
                    reg = _roi_to_region(k)
                    if reg in region_vals and pd.notna(v):
                        region_vals[reg].append(float(v))
            out: Dict[str, float] = {}
            for reg, vs in region_vals.items():
                if vs:
                    out[reg] = float(np.nanmean(vs))
            return out

        for label, windows in [("FA<=3s", early_FA), ("FA>3s", late_FA)]:
            reg_means = region_mean_from_windows(windows)
            for region, value in reg_means.items():
                rows.append({
                    "mouse_id": mouse_id,
                    "session_date": session_date,
                    "genotype": genotype,
                    "region": region,
                    "event": label,
                    "value": value,
                })

    if not rows:
        print("FA scatter: no FA rows computed")
        return None

    df = pd.DataFrame(rows)
    # Build session index per mouse
    def _parse_date(s):
        try:
            return pd.to_datetime(str(s), format="%Y%m%d")
        except Exception:
            return pd.NaT
    df["session_date_parsed"] = df["session_date"].apply(_parse_date)
    df = df.dropna(subset=["session_date_parsed"]).sort_values(["mouse_id", "session_date_parsed"]).reset_index(drop=True)
    df["session_index"] = df.groupby("mouse_id").cumcount() + 1

    _ensure_out(out_dir)
    sns.set_context('talk')
    g = sns.relplot(
        data=df,
        x="session_index",
        y="value",
        hue="genotype",
        style="region",
        col="event",
        kind="scatter",
        height=4,
        aspect=0.9,
    )
    for ax in g.axes.flatten():
        ax.set_xlabel("session index")
        ax.set_ylabel("peak z-dF/F (0–1s)")
        sns.despine(ax=ax)
    out_png = os.path.join(out_dir, "session_scatter_FA_early_late.png")
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    return out_png


def maybe_plot_within_session_drift(agg_df: pd.DataFrame, out_dir: str) -> List[str]:
    """
    If per-trial exports are present (pdf_output/per_trial/**/per_trial_metrics_*.csv),
    generate within-session drift plots: trial quartile vs peak z-dF/F, faceted by event and genotype.
    Returns a list of saved paths (empty if no per-trial files).
    """
    from glob import glob

    per_trial_root = os.path.join(REPO_ROOT, "pdf_output", "per_trial")
    files = glob(os.path.join(per_trial_root, "**", "per_trial_metrics_*.csv"), recursive=True)
    if not files:
        return []
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        return []
    df_pt = pd.concat(dfs, ignore_index=True)
    # Add genotype via merge on mouse_id, session_date from agg_df (drop duplicates first)
    meta = agg_df[["mouse_id", "session_date", "genotype"]].drop_duplicates()
    df_pt["mouse_id"] = df_pt["mouse_id"].astype(str)
    df_pt["session_date"] = df_pt["session_date"].astype(str)
    meta["mouse_id"] = meta["mouse_id"].astype(str)
    meta["session_date"] = meta["session_date"].astype(str)
    df_pt = df_pt.merge(meta, on=["mouse_id", "session_date"], how="left")

    # Keep only events we trust: hit and change
    df_pt = df_pt[df_pt["event"].isin(["hit", "change"])].copy()
    if df_pt.empty:
        return []

    # Quartile index within each session by trial order
    df_pt = df_pt.sort_values(["mouse_id", "session_date", "trial_index"]).reset_index(drop=True)
    # Compute percentile rank within session and map to 1..4 quartiles
    df_pt["_rank_pct"] = df_pt.groupby(["mouse_id", "session_date"])['trial_index'].rank(method='first', pct=True)
    df_pt["trial_quartile"] = np.ceil(df_pt["_rank_pct"] * 4).astype(int)
    df_pt["trial_quartile"] = df_pt["trial_quartile"].clip(1, 4)
    df_pt.drop(columns=["_rank_pct"], inplace=True)

    # Melt region means
    keep_cols = [c for c in ["DMS_mean", "VLS_mean"] if c in df_pt.columns]
    if not keep_cols:
        return []
    long = df_pt.melt(id_vars=["mouse_id", "session_date", "genotype", "event", "trial_quartile"], value_vars=keep_cols, var_name="region", value_name="value")
    long["region"] = long["region"].str.replace("_mean", "", regex=False)
    long = long.dropna(subset=["value"]).copy()
    if long.empty:
        return []

    # Aggregate per mouse/session within quartile
    long_agg = long.groupby(["genotype", "event", "region", "mouse_id", "session_date", "trial_quartile"]).agg(value=("value", "mean")).reset_index()

    sns.set_context('talk')
    g = sns.relplot(
        data=long_agg,
        x="trial_quartile",
        y="value",
        hue="region",
        col="event",
        row="genotype",
        kind="line",
        facet_kws={'sharey': False},
    )
    for ax in g.axes.flatten():
        ax.set_xlabel("trial quartile (within session)")
        ax.set_ylabel("peak z-dF/F (0–1s)")
        sns.despine(ax=ax)
    out_png = os.path.join(out_dir, "within_session_drift.png")
    g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(g.figure)
    return [out_png]


def _compute_region_means_per_event(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    From wide agg_df, compute per-row region means per event (hit/miss/change) using ROI mapping.
    Returns df with columns like hit_DMS_mean, hit_VLS_mean, etc.
    """
    df = agg_df.copy()
    def region_mean(row, event_prefix: str, region: str):
        vals = []
        for k, v in row.items():
            if isinstance(k, str) and k.startswith(event_prefix):
                if _roi_to_region(k) == region and pd.notna(v):
                    vals.append(float(v))
        return float(np.nanmean(vals)) if vals else np.nan
    for ev in ["hit_", "miss_", "change_"]:
        for region in ["DMS", "VLS"]:
            df[f"{ev[:-1]}_{region}_mean"] = df.apply(lambda r: region_mean(r, ev, region), axis=1)
    return df


def maybe_plot_correlations(agg_csv: str, out_dir: str) -> List[str]:
    """
    If behavior_summary_all.csv exists, merge with photom_summary_all.csv and produce correlation plots:
    - FA rate vs hit peak (DMS/VLS)
    - Median RT vs hit peak
    - Psychometric slope vs change peak
    Returns list of saved plot paths.
    """
    beh_csv = os.path.join(REPO_ROOT, "pdf_output", "behavior_summary_all.csv")
    if not os.path.exists(beh_csv):
        return []
    agg_df = pd.read_csv(agg_csv)
    beh_df = pd.read_csv(beh_csv)
    # Normalize keys
    agg_df["mouse_id"] = agg_df["mouse_id"].astype(str)
    agg_df["session_date"] = agg_df["session_date"].astype(str)
    beh_df["mouse_id"] = beh_df["mouse_id"].astype(str)
    beh_df["session_date"] = beh_df["session_date"].astype(str)
    # Compute region means
    agg_means = _compute_region_means_per_event(agg_df)
    df = agg_means.merge(beh_df, on=["mouse_id", "session_date"], how="inner")
    if df.empty:
        return []

    paths: List[str] = []
    sns.set_context('talk')

    # FA rate vs hit peak
    for region in ["DMS", "VLS"]:
        col = f"hit_{region}_mean"
        if col in df.columns:
            g = sns.lmplot(data=df, x="fa_rate", y=col, hue="genotype", scatter_kws={"alpha":0.6})
            plt.xlabel("FA rate")
            plt.ylabel(f"Hit peak z-dF/F (0–1s) — {region}")
            out_png = os.path.join(out_dir, f"corr_fa_vs_hit_{region}.png")
            g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
            plt.close(g.figure)
            paths.append(out_png)

    # Median RT vs hit peak (seconds)
    for region in ["DMS", "VLS"]:
        col = f"hit_{region}_mean"
        if col in df.columns and "median_rt_s" in df.columns:
            sub = df[pd.notna(df["median_rt_s"])].copy()
            if not sub.empty:
                g = sns.lmplot(data=sub, x="median_rt_s", y=col, hue="genotype", scatter_kws={"alpha":0.6})
                plt.xlabel("Median RT (s)")
                plt.ylabel(f"Hit peak z-dF/F (0–1s) — {region}")
                out_png = os.path.join(out_dir, f"corr_rt_vs_hit_{region}.png")
                g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
                plt.close(g.figure)
                paths.append(out_png)

    # Psychometric slope vs change peak
    for region in ["DMS", "VLS"]:
        col = f"change_{region}_mean"
        if col in df.columns and "psy_slope" in df.columns:
            sub = df[pd.notna(df["psy_slope"])].copy()
            if not sub.empty:
                g = sns.lmplot(data=sub, x="psy_slope", y=col, hue="genotype", scatter_kws={"alpha":0.6})
                plt.xlabel("Psychometric slope (Hit vs change category)")
                plt.ylabel(f"Change peak z-dF/F (0–1s) — {region}")
                out_png = os.path.join(out_dir, f"corr_psy_vs_change_{region}.png")
                g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
                plt.close(g.figure)
                paths.append(out_png)

    # d' vs change peak
    for region in ["DMS", "VLS"]:
        col = f"change_{region}_mean"
        if col in df.columns and "d_prime" in df.columns:
            sub = df[pd.notna(df["d_prime"])].copy()
            if not sub.empty:
                g = sns.lmplot(data=sub, x="d_prime", y=col, hue="genotype", scatter_kws={"alpha":0.6})
                plt.xlabel("d' (SDT)")
                plt.ylabel(f"Change peak z-dF/F (0–1s) — {region}")
                out_png = os.path.join(out_dir, f"corr_dprime_vs_change_{region}.png")
                g.figure.savefig(out_png, dpi=200, bbox_inches='tight')
                plt.close(g.figure)
                paths.append(out_png)

    return paths

def main():
    parser = argparse.ArgumentParser(description="Group-level photometry comparisons by genotype and region")
    parser.add_argument("--agg-csv", dest="agg_csv", default=None, help="Path to aggregated CSV (default: pdf_output/photom_summary_all.csv)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory for group plots (default: pdf_output/group_plots)")
    parser.add_argument("--fa-scatter", action="store_true", help="Also compute and plot FA early/late session scatter")
    args = parser.parse_args()

    agg_csv = args.agg_csv or os.path.join(REPO_ROOT, "pdf_output", "photom_summary_all.csv")
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "pdf_output", "group_plots")
    os.makedirs(out_dir, exist_ok=True)

    agg_df = load_agg(agg_csv)
    df_long = melt_agg_to_long(agg_df)
    if df_long.empty:
        print("No data in aggregated CSV to plot.")
        return

    paths: List[str] = []
    try:
        paths.append(plot_violin_by_genotype_region(df_long, out_dir))
    except Exception as e:
        print("Violin plot failed:", e)
    try:
        paths.append(plot_bar_with_ci(df_long, out_dir))
    except Exception as e:
        print("Bar plot failed:", e)
    try:
        paths.append(plot_session_scatter(df_long, out_dir))
    except Exception as e:
        print("Scatter plot failed:", e)

    # Optional: within-session drift plots if per-trial exports exist
    try:
        paths.extend(maybe_plot_within_session_drift(agg_df, out_dir))
    except Exception as e:
        print("Within-session drift plot failed:", e)

    # Optional: correlations vs behavior metrics
    try:
        paths.extend(maybe_plot_correlations(agg_csv, out_dir))
    except Exception as e:
        print("Correlation plots failed:", e)

    if args.fa_scatter:
        try:
            fa_png = plot_fa_session_scatter(out_dir)
            if fa_png:
                paths.append(fa_png)
        except Exception as e:
            print("FA scatter plot failed:", e)

    for p in paths:
        print("Saved:", p)


if __name__ == "__main__":
    main()
