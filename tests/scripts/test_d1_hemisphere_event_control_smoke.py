import os, subprocess, sys, pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "photom_data", "intrsct_GCaMP6f")
SCRIPT = os.path.join(REPO, "scripts", "analysis", "intersectional",
                      "d1_hemisphere_event_control.py")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_d1_hemisphere_event_control_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--max_sessions", "4",
         "--root_dir", DATA, "--output_dir", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "d1_hemisphere_event_control.png").exists()
    assert (out / "d1_hemisphere_event_control_amplitudes.csv").exists()
    assert (out / "d1_hemisphere_event_control_ratios.csv").exists()
