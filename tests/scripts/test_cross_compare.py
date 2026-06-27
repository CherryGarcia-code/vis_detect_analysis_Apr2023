# tests/scripts/test_cross_compare.py
import importlib.util
from pathlib import Path
import pandas as pd

_P = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "intersectional" / "cohort_cross_compare.py"
_S = importlib.util.spec_from_file_location("cohort_cross_compare", _P)
mod = importlib.util.module_from_spec(_S); _S.loader.exec_module(mod)

def test_build_cross_compare_matches_and_is_rank_based(tmp_path):
    coh = tmp_path / "coh.csv"; bulk = tmp_path / "bulk.csv"
    pd.DataFrame([{"subject_id":"BG_029","genotype":"D2","region":"DMS",
                   "track":"behavioral_fa","scheme":"scheme3","auroc_mean":0.62}]).to_csv(coh, index=False)
    pd.DataFrame([{"genotype":"D2","region":"DMS","auroc_mean":0.55}]).to_csv(bulk, index=False)
    out = mod.build_cross_compare(str(coh), str(bulk))
    r = out[(out["genotype"]=="D2") & (out["region"]=="DMS")].iloc[0]
    assert r["auroc_cohort"] == 0.62 and r["auroc_bulk"] == 0.55
    assert abs(r["delta_auroc"] - 0.07) < 1e-9
    assert r["intersectional_subject"] == "BG_029"
