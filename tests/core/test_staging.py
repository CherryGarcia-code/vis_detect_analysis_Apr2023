import pandas as pd
from types import SimpleNamespace
from visdetect_photom.core.staging import (
    load_staging_manifest, get_session_stage, excluded_mice,
)


def _manifest(tmp_path):
    df = pd.DataFrame([
        {"subject_id": "BG_013", "session_name": "013_20231205", "stage": "Learning"},
        {"subject_id": "BG_013", "session_name": "013_20231206", "stage": "Expert"},
        {"subject_id": "BG_014", "session_name": "014_20231219", "stage": "Excluded"},
        {"subject_id": "BG_014", "session_name": "014_20231221", "stage": "Excluded"},
    ])
    p = tmp_path / "staging_manifest.csv"
    df.to_csv(p, index=False)
    return p


def test_load_missing_returns_none():
    assert load_staging_manifest("nope/missing.csv") is None


def test_get_session_stage_matches_session_id(tmp_path):
    m = load_staging_manifest(str(_manifest(tmp_path)))
    sess = SimpleNamespace(subject_id="013", session_id="013_20231206")
    assert get_session_stage(sess, m) == "Expert"


def test_get_session_stage_unknown_when_absent(tmp_path):
    m = load_staging_manifest(str(_manifest(tmp_path)))
    sess = SimpleNamespace(subject_id="999", session_id="999_20990101")
    assert get_session_stage(sess, m) == "Unknown"


def test_excluded_mice_includes_all_excluded(tmp_path):
    m = load_staging_manifest(str(_manifest(tmp_path)))
    excl = excluded_mice(m)
    assert "BG_014" in excl
    assert "BG_013" not in excl
