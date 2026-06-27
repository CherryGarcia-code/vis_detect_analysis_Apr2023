import os, subprocess, sys, pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "photom_data", "intrsct_GCaMP6f")
C1 = os.path.join(REPO, "scripts", "analysis", "intersectional", "c1_cohort_suppression.py")
C2 = os.path.join(REPO, "scripts", "analysis", "intersectional", "c2_cohort_geometry.py")
QC = os.path.join(REPO, "scripts", "analysis", "intersectional", "cohort_qc_report.py")


@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_c1_cohort_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, C1, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_c1_session_scalars.csv").exists()
    assert (out / "cohort_c1_cell_summary.csv").exists()


@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_c2_cohort_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, C2, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_c2_cell_summary.csv").exists()


@pytest.mark.skipif(not os.path.isdir(DATA), reason="cohort data not present")
def test_cohort_qc_report_runs(tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, QC, "--max_sessions", "4",
                           "--root_dir", DATA, "--output_dir", str(out)],
                          cwd=REPO, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "cohort_qc_report.csv").exists()
