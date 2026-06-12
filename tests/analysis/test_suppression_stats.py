import numpy as np
import pandas as pd
from visdetect_photom.analysis.suppression import (
    compute_delta_and_auroc, run_suppression_stats,
)


def _per_trial(subject, genotype, region, lick_vals, withhold_vals):
    rows = []
    for v in lick_vals:
        rows.append({"subject_id": subject, "genotype": genotype, "region": region,
                     "track": "behavioral_fa", "scheme": "scheme1", "group": "lick",
                     "scalar": v})
    for v in withhold_vals:
        rows.append({"subject_id": subject, "genotype": genotype, "region": region,
                     "track": "behavioral_fa", "scheme": "scheme1", "group": "withhold",
                     "scalar": v})
    return rows


def test_compute_delta_and_auroc_brake_direction():
    # withhold higher than lick -> delta > 0 and AUROC > 0.5 (activity predicts withholding)
    rows = _per_trial("BG_013", "D1", "DMS",
                      lick_vals=list(np.arange(0, 10) * 0.1),
                      withhold_vals=list(1.0 + np.arange(0, 10) * 0.1))
    pm = compute_delta_and_auroc(pd.DataFrame(rows))
    r = pm.iloc[0]
    assert r["delta"] > 0
    assert r["auroc"] > 0.5

def test_compute_delta_skips_below_min_n():
    rows = _per_trial("BG_013", "D1", "DMS", lick_vals=[0.1, 0.2], withhold_vals=[1.0])
    pm = compute_delta_and_auroc(pd.DataFrame(rows))
    assert pm.empty  # < MIN_TRIALS_PER_GROUP

def test_run_suppression_stats_pushpull_opposite_sign():
    # 2 D1 mice: withhold>lick (delta>0); 2 D2 mice: withhold<lick (delta<0)
    rows = []
    for s in ("BG_013", "BG_020"):
        rows += _per_trial(s, "D1", "DMS",
                           lick_vals=list(np.zeros(10)),
                           withhold_vals=list(np.ones(10)))
    for s in ("BG_016", "BG_018"):
        rows += _per_trial(s, "D2", "DMS",
                           lick_vals=list(np.ones(10)),
                           withhold_vals=list(np.zeros(10)))
    pm = compute_delta_and_auroc(pd.DataFrame(rows))
    pp, au = run_suppression_stats(pm)
    row = pp[pp["region"] == "DMS"].iloc[0]
    assert row["d1_sign"] == 1 and row["d2_sign"] == -1
    assert set(au["genotype"]) == {"D1", "D2"}
