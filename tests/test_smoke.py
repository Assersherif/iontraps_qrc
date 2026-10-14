"""End-to-end smoke tests. Metrics from --quick are NOT meaningful; these only check wiring."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "raw" / "NF-UNSW-NB15.parquet"


def test_parallel_matches_serial():
    from iontraps_qrc import parallel
    assert parallel.selftest(workers=2)


@pytest.mark.skipif(not DATA.exists(), reason="needs data/raw/NF-UNSW-NB15.parquet")
@pytest.mark.parametrize("workers", [0, 2])
def test_quick_cell(tmp_path, workers):
    cmd = [sys.executable, "-W", "ignore", str(REPO / "scripts" / "run_sweep.py"), "--quick",
           "--only", "6q_a2a_L6", "--out", str(tmp_path), "--workers", str(workers)]
    subprocess.run(cmd, check=True, cwd=tmp_path, capture_output=True)
    res = json.loads((tmp_path / "results_6q_a2a_L6.json").read_text())["results"]
    assert 0.0 <= res["supervised_quantum"] <= 1.0
    assert (tmp_path / "sweep_summary.json").exists()
