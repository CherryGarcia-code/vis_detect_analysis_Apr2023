"""
Reusable inferential statistics functions for group-level photometry analysis.

All functions return dicts with test statistic, p-value, and effect size.
Non-parametric by default (appropriate for neural data with small N).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, List, Optional, Tuple, Callable, Any


# ── Two-sample tests ─────────────────────────────────────────

def mannwhitney_with_effect_size(x: np.ndarray, y: np.ndarray,
                                  alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Mann-Whitney U test with rank-biserial correlation as effect size.

    rank_biserial_r = 1 - (2*U) / (n1*n2)
    Interpretation: |r| < 0.3 small, 0.3-0.5 medium, > 0.5 large.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return {"U": np.nan, "p": np.nan, "rank_biserial_r": np.nan,
                "n1": len(x), "n2": len(y)}

    U, p = sp_stats.mannwhitneyu(x, y, alternative=alternative)
    n1, n2 = len(x), len(y)
    r = 1 - (2 * U) / (n1 * n2)

    return {"U": float(U), "p": float(p), "rank_biserial_r": float(r),
            "n1": n1, "n2": n2}


def wilcoxon_with_effect_size(x: np.ndarray, y: np.ndarray,
                               alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Wilcoxon signed-rank test for paired samples with matched-pairs rank-biserial r.

    r = W / sum_of_ranks (matched-pairs effect size).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 5:
        return {"W": np.nan, "p": np.nan, "matched_r": np.nan, "n": len(x)}

    diff = x - y
    # Remove zeros (ties at zero)
    nonzero = diff != 0
    if nonzero.sum() < 5:
        return {"W": np.nan, "p": np.nan, "matched_r": np.nan, "n": int(nonzero.sum())}

    W, p = sp_stats.wilcoxon(x, y, alternative=alternative)
    n = int(nonzero.sum())
    sum_ranks = n * (n + 1) / 2
    r = float(W / sum_ranks) if sum_ranks > 0 else np.nan

    return {"W": float(W), "p": float(p), "matched_r": r, "n": n}


# ── Multi-group tests ────────────────────────────────────────

def kruskal_with_effect_size(*groups: np.ndarray) -> Dict[str, float]:
    """
    Kruskal-Wallis H test with epsilon-squared (η²_H) effect size.

    η²_H = (H - k + 1) / (N - k), where k = number of groups, N = total samples.
    """
    clean_groups = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        g = g[np.isfinite(g)]
        if len(g) > 0:
            clean_groups.append(g)

    if len(clean_groups) < 2:
        return {"H": np.nan, "p": np.nan, "eta_sq_H": np.nan,
                "k": len(clean_groups), "N": sum(len(g) for g in clean_groups)}

    H, p = sp_stats.kruskal(*clean_groups)
    k = len(clean_groups)
    N = sum(len(g) for g in clean_groups)
    eta_sq = (H - k + 1) / (N - k) if N > k else np.nan

    return {"H": float(H), "p": float(p), "eta_sq_H": float(eta_sq),
            "k": k, "N": N}


# ── Correlation ──────────────────────────────────────────────

def spearman_with_ci(x: np.ndarray, y: np.ndarray,
                     n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    """
    Spearman rank correlation with bootstrap 95% CI.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 5:
        return {"rho": np.nan, "p": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "n": len(x)}

    rho, p = sp_stats.spearmanr(x, y)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i] = sp_stats.spearmanr(x[idx], y[idx]).statistic

    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    return {"rho": float(rho), "p": float(p),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n": n}


# ── Bootstrap ────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, func: Callable = np.nanmean,
                 n_boot: int = 1000, seed: int = 42,
                 ci: float = 95.0) -> Dict[str, float]:
    """
    Bootstrap confidence interval for any summary statistic.

    Args:
        data: 1D array of observations.
        func: Summary function (default: np.nanmean).
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        ci: Confidence level (default: 95%).

    Returns:
        Dict with observed, ci_lo, ci_hi, n.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) < 2:
        return {"observed": func(data) if len(data) > 0 else np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "n": len(data)}

    observed = float(func(data))
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = func(data[idx])

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_stats, [alpha, 100 - alpha])

    return {"observed": observed, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


# ── Permutation test ─────────────────────────────────────────

def permutation_test(x: np.ndarray, y: np.ndarray,
                     stat_func: Callable = None,
                     n_perm: int = 10000, seed: int = 42) -> Dict[str, float]:
    """
    Two-sample permutation test for difference in means (or custom statistic).

    Args:
        x, y: Two groups.
        stat_func: Function(x, y) -> float. Default: difference of means.
        n_perm: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with observed, p (two-sided), n1, n2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if stat_func is None:
        stat_func = lambda a, b: np.mean(a) - np.mean(b)

    if len(x) < 2 or len(y) < 2:
        return {"observed": np.nan, "p": np.nan, "n1": len(x), "n2": len(y)}

    observed = float(stat_func(x, y))
    combined = np.concatenate([x, y])
    n1 = len(x)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_stat = stat_func(combined[:n1], combined[n1:])
        if abs(perm_stat) >= abs(observed):
            count += 1

    p = (count + 1) / (n_perm + 1)  # +1 to include the observed

    return {"observed": observed, "p": float(p), "n1": n1, "n2": len(y)}


# ── Formatting ───────────────────────────────────────────────

def format_stats_table(results: List[Dict[str, Any]],
                       save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convert a list of stats result dicts into a formatted DataFrame.

    Each dict should have at minimum: 'comparison', 'test', 'statistic', 'p', 'effect_size'.
    Additional keys are preserved as columns.

    Args:
        results: List of dicts from the test functions above, enriched with 'comparison' and 'test' keys.
        save_path: If provided, saves to CSV.

    Returns:
        DataFrame with formatted results.
    """
    df = pd.DataFrame(results)

    # Add significance stars
    if 'p' in df.columns:
        df['sig'] = df['p'].apply(_sig_stars)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def _sig_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ── Timing analysis ──────────────────────────────────────────

def extract_peak_latency(trace: np.ndarray, time_axis: np.ndarray,
                          peak_window: Tuple[float, float] = (0.0, 1.5)) -> float:
    """
    Extract latency (in seconds) of the peak (abs-max) response within peak_window.

    Robust to fiber placement, expression level, gain — only depends on when
    the signal reaches its maximum, not how large it is.
    """
    mask = (time_axis >= peak_window[0]) & (time_axis <= peak_window[1])
    if not np.any(mask):
        return np.nan
    segment = trace[mask]
    t_segment = time_axis[mask]
    valid_mask = np.isfinite(segment)
    if valid_mask.sum() == 0:
        return np.nan
    idx = np.argmax(np.abs(segment[valid_mask]))
    return float(t_segment[valid_mask][idx])


def extract_onset_latency(trace: np.ndarray, time_axis: np.ndarray,
                           threshold_n_std: float = 2.0,
                           baseline_window: Tuple[float, float] = (-2.0, 0.0),
                           search_window: Tuple[float, float] = (0.0, 2.0),
                           n_consecutive: int = 3) -> float:
    """
    Extract onset latency: first time the trace exceeds threshold_n_std * baseline_std
    for n_consecutive bins after event onset.

    This is robust to amplitude scaling — it only asks "when does the signal
    first reliably deviate from baseline?"
    """
    bl_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
    search_mask = (time_axis >= search_window[0]) & (time_axis <= search_window[1])

    bl_vals = trace[bl_mask]
    bl_vals = bl_vals[np.isfinite(bl_vals)]
    if len(bl_vals) < 5:
        return np.nan
    bl_mean = np.nanmean(bl_vals)
    bl_std = np.nanstd(bl_vals)
    if bl_std < 1e-6:
        return np.nan

    threshold = bl_mean + threshold_n_std * bl_std
    search_trace = trace[search_mask]
    search_times = time_axis[search_mask]

    # Find first stretch of n_consecutive bins above threshold
    above = np.abs(search_trace - bl_mean) > (threshold_n_std * bl_std)
    above[~np.isfinite(search_trace)] = False

    count = 0
    for i, is_above in enumerate(above):
        if is_above:
            count += 1
            if count >= n_consecutive:
                onset_idx = i - n_consecutive + 1
                return float(search_times[onset_idx])
        else:
            count = 0

    return np.nan


# ── Mode-aware response extraction (C2) ──────────────────────

def _window_segment(trace, time_axis, window):
    """Return (finite values, their times) inside the window."""
    trace = np.asarray(trace, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    mask = (time_axis >= window[0]) & (time_axis <= window[1])
    seg = trace[mask]
    t_seg = time_axis[mask]
    finite = np.isfinite(seg)
    return seg[finite], t_seg[finite]


def extract_activation(trace, time_axis, window):
    """Peak positive deflection in window (>0), else nan (pure suppression/flat)."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    m = float(np.max(seg))
    return m if m > 0 else np.nan


def extract_suppression(trace, time_axis, window):
    """Peak negative deflection in window (<0), else nan (pure activation/flat)."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    m = float(np.min(seg))
    return m if m < 0 else np.nan


def extract_signed_peak(trace, time_axis, window):
    """Abs-max deflection in window, preserving sign (captures activation OR suppression)."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    return float(seg[np.argmax(np.abs(seg))])


def extract_signed_auc(trace, time_axis, window):
    """Mean (net signed response) over window."""
    seg, _ = _window_segment(trace, time_axis, window)
    if seg.size == 0:
        return np.nan
    return float(np.mean(seg))


def extract_ramp_slope(trace, time_axis, window):
    """Slope (signal-units/s) of a degree-1 fit over window; offset-invariant."""
    seg, t_seg = _window_segment(trace, time_axis, window)
    if seg.size < 2:
        return np.nan
    return float(np.polyfit(t_seg, seg, 1)[0])


def pushpull_sign_contrast(d1_vals, d2_vals, n_perm=10000, seed=42):
    """Group-level D1-vs-D2 sign contrast (NOT within-animal anticorrelation).

    Returns per-genotype mean + bootstrap 95% CI, each sign, an `opposite_sign`
    flag (signs differ AND both CIs exclude 0), permutation p on (meanD1-meanD2),
    and rank-biserial effect size.
    """
    d1 = np.asarray(d1_vals, float); d1 = d1[np.isfinite(d1)]
    d2 = np.asarray(d2_vals, float); d2 = d2[np.isfinite(d2)]
    out = {"d1_n": int(d1.size), "d2_n": int(d2.size),
           "d1_mean": float(np.mean(d1)) if d1.size else np.nan,
           "d2_mean": float(np.mean(d2)) if d2.size else np.nan}

    d1_ci = bootstrap_ci(d1) if d1.size >= 2 else {"ci_lo": np.nan, "ci_hi": np.nan}
    d2_ci = bootstrap_ci(d2) if d2.size >= 2 else {"ci_lo": np.nan, "ci_hi": np.nan}
    out.update({"d1_ci_lo": d1_ci["ci_lo"], "d1_ci_hi": d1_ci["ci_hi"],
                "d2_ci_lo": d2_ci["ci_lo"], "d2_ci_hi": d2_ci["ci_hi"]})

    def _excl_zero(ci):
        return np.isfinite(ci["ci_lo"]) and (ci["ci_lo"] > 0 or ci["ci_hi"] < 0)

    out["d1_sign"] = int(np.sign(out["d1_mean"])) if np.isfinite(out["d1_mean"]) else 0
    out["d2_sign"] = int(np.sign(out["d2_mean"])) if np.isfinite(out["d2_mean"]) else 0
    out["opposite_sign"] = bool(_excl_zero(d1_ci) and _excl_zero(d2_ci)
                                and out["d1_sign"] != out["d2_sign"])

    if d1.size >= 2 and d2.size >= 2:
        out["p"] = permutation_test(d1, d2, n_perm=n_perm, seed=seed)["p"]
        out["rank_biserial_r"] = mannwhitney_with_effect_size(d1, d2)["rank_biserial_r"]
    else:
        out["p"] = np.nan
        out["rank_biserial_r"] = np.nan
    return out


# ── AUROC (single-trial discriminability) ────────────────────

def auroc_score(scores, labels) -> float:
    """Area under ROC for score discriminating positive class (label==1) from
    negative (label==0). AUROC = P(score_pos > score_neg) via the Mann-Whitney U
    statistic: U / (n_pos * n_neg). Non-finite scores are dropped. Returns NaN if
    either class is empty.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    U, _ = sp_stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(U / (pos.size * neg.size))
