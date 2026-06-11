import numpy as np
from visdetect_photom.analysis.group_statistics import auroc_score


def test_auroc_perfect_separation_positive():
    # positive class (1) clearly higher than negative (0) -> AUROC ~ 1.0
    scores = np.array([5.0, 6.0, 7.0, 1.0, 2.0, 3.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert auroc_score(scores, labels) == 1.0

def test_auroc_perfect_separation_negative():
    # positive class lower than negative -> AUROC ~ 0.0
    scores = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert auroc_score(scores, labels) == 0.0

def test_auroc_chance_when_interleaved():
    # pos=[1,4] vs neg=[2,3]: exactly 2/4 pos-neg pairs have pos>neg -> AUROC=0.5
    # (original plan used [1,2,3,4] with labels [1,0,1,0] which gives pos=[1,3] vs
    # neg=[2,4] -> 1/4 wins -> AUROC=0.25, not 0.5; corrected here)
    scores = np.array([1.0, 4.0, 2.0, 3.0])
    labels = np.array([1,   1,   0,   0])
    assert auroc_score(scores, labels) == 0.5

def test_auroc_nan_when_one_class_empty():
    assert np.isnan(auroc_score(np.array([1.0, 2.0]), np.array([1, 1])))

def test_auroc_ignores_nonfinite():
    scores = np.array([5.0, np.nan, 7.0, 1.0, 2.0])
    labels = np.array([1, 1, 1, 0, 0])
    assert auroc_score(scores, labels) == 1.0
