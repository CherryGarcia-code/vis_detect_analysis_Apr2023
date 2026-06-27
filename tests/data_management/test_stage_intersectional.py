import importlib.util, os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "stage_intersectional_cohort",
    Path(__file__).resolve().parents[2] / "scripts" / "data_management" / "stage_intersectional_cohort.py")
stage_mod = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(stage_mod)

def test_normalize_filename_fixes_stale_subject_and_underscore():
    # stale BG_027 photom inside a BG_030 folder -> corrected to BG_030
    assert stage_mod.normalize_filename("BG_027__photom_2024-12-04T12_01_00.csv", "BG_030") \
        == "BG_030__photom_2024-12-04T12_01_00.csv"
    assert stage_mod.normalize_filename("BG_027__photom_IO_2024-12-04T12_01_00.csv", "BG_030") \
        == "BG_030__photom_IO_2024-12-04T12_01_00.csv"
    # single underscore -> double underscore, correct subject untouched otherwise
    assert stage_mod.normalize_filename("BG_030_trials.json", "BG_030") == "BG_030__trials.json"
    # already-correct name is unchanged
    assert stage_mod.normalize_filename("BG_030__session_settings.json", "BG_030") \
        == "BG_030__session_settings.json"

def test_stage_is_idempotent(tmp_path):
    src = tmp_path / "ceph" / "BG_030"; src.mkdir(parents=True)
    (src / "BG_027__photom_2024-12-04T12_01_00.csv").write_text("x" * 100)
    (src / "BG_030__trials.json").write_text("{}")
    dest = tmp_path / "dest" / "BG_030"
    r1 = stage_mod.stage("BG_030", src, dest, dry_run=False)
    r2 = stage_mod.stage("BG_030", src, dest, dry_run=False)
    assert (dest / "BG_030__photom_2024-12-04T12_01_00.csv").exists()
    assert r2["copied"] == 0  # second run copies nothing new
