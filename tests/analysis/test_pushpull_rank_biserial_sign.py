"""Regression test: rank_biserial_r in pushpull_sign_contrast must agree in sign
with (d1_mean - d2_mean).

Convention: observed = meanD1 - meanD2 (positive => D1>D2). The rank-biserial
effect size reported in the same output row must therefore be POSITIVE when D1>D2
and NEGATIVE when D2>D1, so the two effect-direction signals never disagree.
"""
import numpy as np

from visdetect_photom.analysis.group_statistics import pushpull_sign_contrast


def test_rank_biserial_positive_when_d1_greater():
    # D1 clearly larger than D2 in every pair.
    res = pushpull_sign_contrast([2, 3, 4, 5], [-2, -3, -4, -5], n_perm=2000, seed=42)
    assert res["d1_mean"] > res["d2_mean"]
    assert np.isfinite(res["rank_biserial_r"])
    assert res["rank_biserial_r"] > 0  # agrees in sign with d1_mean - d2_mean


def test_rank_biserial_negative_when_d2_greater():
    # Reversed: D2 clearly larger than D1.
    res = pushpull_sign_contrast([-2, -3, -4, -5], [2, 3, 4, 5], n_perm=2000, seed=42)
    assert res["d1_mean"] < res["d2_mean"]
    assert np.isfinite(res["rank_biserial_r"])
    assert res["rank_biserial_r"] < 0
