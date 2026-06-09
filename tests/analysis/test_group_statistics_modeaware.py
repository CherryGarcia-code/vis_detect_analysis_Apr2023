import numpy as np
import pytest
from visdetect_photom.analysis.group_statistics import (
    extract_activation, extract_suppression, extract_signed_peak,
    extract_signed_auc, extract_ramp_slope, pushpull_sign_contrast,
)

T = np.linspace(-2.0, 4.0, 600)          # 100 Hz over [-2, 4]
WIN = (0.0, 1.5)


def _bump(amp, center=0.5, width=0.3):
    return amp * np.exp(-((T - center) ** 2) / (2 * width ** 2))


def test_activation_positive_bump():
    assert extract_activation(_bump(2.0), T, WIN) == pytest.approx(2.0, abs=0.05)

def test_activation_pure_dip_is_nan():
    assert np.isnan(extract_activation(_bump(-2.0), T, WIN))

def test_suppression_negative_dip():
    assert extract_suppression(_bump(-3.0), T, WIN) == pytest.approx(-3.0, abs=0.05)

def test_suppression_pure_bump_is_nan():
    assert np.isnan(extract_suppression(_bump(2.0), T, WIN))

def test_signed_peak_preserves_sign():
    assert extract_signed_peak(_bump(-3.0), T, WIN) == pytest.approx(-3.0, abs=0.05)
    assert extract_signed_peak(_bump(2.0), T, WIN) == pytest.approx(2.0, abs=0.05)

def test_signed_auc_sign():
    assert extract_signed_auc(_bump(2.0), T, WIN) > 0
    assert extract_signed_auc(_bump(-2.0), T, WIN) < 0

def test_ramp_slope_known_slope():
    trace = 3.0 * T  # slope 3 per second
    assert extract_ramp_slope(trace, T, (-1.5, 0.0)) == pytest.approx(3.0, abs=0.01)

def test_empty_window_returns_nan():
    assert np.isnan(extract_activation(_bump(2.0), T, (10.0, 11.0)))

def test_pushpull_opposite_sign_flagged():
    d1 = np.array([1.8, 2.1, 1.9, 2.3])
    d2 = np.array([-1.7, -2.0, -1.6, -2.2])
    res = pushpull_sign_contrast(d1, d2, n_perm=2000, seed=42)
    assert res["opposite_sign"] is True
    assert res["d1_sign"] == 1 and res["d2_sign"] == -1
    assert res["p"] < 0.05

def test_pushpull_same_sign_not_flagged():
    d1 = np.array([1.8, 2.1, 1.9, 2.3])
    d2 = np.array([1.5, 1.7, 1.6, 1.9])
    res = pushpull_sign_contrast(d1, d2, n_perm=2000, seed=42)
    assert res["opposite_sign"] is False
