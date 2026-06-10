import os
import subprocess
import sys
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "scripts", "analysis", "photometry", "08_d1_d2_geometry.py")
DATA = os.path.join(REPO, "photom_data")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="photom_data/ not present")
def test_script_runs_on_small_subset(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--max_sessions", "3", "--output_dir", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "C2_geometry_metrics.csv").exists()
