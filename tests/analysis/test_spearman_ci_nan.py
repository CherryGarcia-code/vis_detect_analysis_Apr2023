"""Regression test: spearman_with_ci bootstrap CI must not be poisoned by NaN
bootstrap replicates.

At small n, some bootstrap resamples are constant (e.g. all identical x or y),
which makes scipy's spearmanr return NaN for that replicate. Using np.percentile
over an array containing any NaN returns NaN for BOTH bounds, even though the
point estimate rho and the bulk of replicates are finite. The fix uses
np.nanpercentile so the CI survives those degenerate resamples.
"""
import numpy as np

from visdetect_photom.analysis.group_statistics import spearman_with_ci


def test_ci_finite_despite_constant_resamples():
    res = spearman_with_ci(x=[1, 2, 3, 4, 5], y=[2, 1, 4, 3, 5],
                           n_boot=1000, seed=42)
    assert np.isfinite(res["rho"])
    assert np.isfinite(res["ci_lo"])
    assert np.isfinite(res["ci_hi"])
    assert res["ci_lo"] <= res["ci_hi"]
