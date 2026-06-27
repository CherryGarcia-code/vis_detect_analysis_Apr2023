"""Session-unit (per-recording) delta/AUROC for the n=1-mouse/cell cohort.

Grouped by recording_id (the true per-recording unit), NOT session_id, which is
only date-granular and would pool same-day recordings.
"""
import numpy as np
import pandas as pd

from visdetect_photom.analysis.suppression import compute_session_delta_and_auroc


def _rows(recording_id, lick, withhold, session_id="s2026-01-01"):
    r = [{"subject_id": "BG_029", "genotype": "D2", "region": "DMS",
          "track": "behavioral_fa", "scheme": "scheme1", "group": "lick",
          "trial_index": i, "scalar": v, "session_id": session_id,
          "recording_id": recording_id}
         for i, v in enumerate(lick)]
    r += [{"subject_id": "BG_029", "genotype": "D2", "region": "DMS",
           "track": "behavioral_fa", "scheme": "scheme1", "group": "withhold",
           "trial_index": 100 + i, "scalar": v, "session_id": session_id,
           "recording_id": recording_id}
          for i, v in enumerate(withhold)]
    return r


def test_session_unit_emits_one_row_per_recording():
    # Two distinct recordings sharing the SAME date-granular session_id, to prove
    # grouping is by recording_id (not session_id).
    rows = (_rows("recA", np.zeros(10), np.ones(10), session_id="sameday")
            + _rows("recB", np.ones(10), np.zeros(10), session_id="sameday"))
    out = compute_session_delta_and_auroc(pd.DataFrame(rows))

    # One row per recording, even though both share session_id "sameday".
    assert set(out["recording_id"]) == {"recA", "recB"}
    assert len(out) == 2

    a = out[out["recording_id"] == "recA"].iloc[0]
    assert a["delta"] > 0 and a["auroc"] > 0.5      # withhold>lick in recA
    assert a["n_lick"] == 10 and a["n_withhold"] == 10

    b = out[out["recording_id"] == "recB"].iloc[0]
    assert b["delta"] < 0 and b["auroc"] < 0.5      # withhold<lick in recB
