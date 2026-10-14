#!/usr/bin/env python3
"""Run the all-to-all sweep (qubits x L) on an NF-UNSW-NB15 parquet.

    python scripts/run_sweep.py --data data/raw/NF-UNSW-NB15-v2.parquet
    python scripts/run_sweep.py --data <parquet> --qubits 6,8 --L 6 --out results/my_run
    python scripts/run_sweep.py --quick          # wiring check, tiny sizes, do not quote numbers
    python scripts/run_sweep.py --qubits 12,16 --L 6,8,10 --workers 24   # parallel simulator

The reservoir, shots, sample sizes, split seed, readout and detector are the
ones in iontraps_qrc.core / iontraps_qrc.sweep. This wrapper only sets what
a new file cannot be known to the core:

  * the parquet path (--data, or env NFUNSW_PARQUET);
  * --log1p-all: log1p EVERY numeric column, as needed for the v2 export
    (SRC_TO_DST_SECOND_BYTES reaches 1.6e15 and overflows float32 in
    StandardScaler). TCP_FLAGS stays one-hot. Default: on for files whose
    name contains "v2", off otherwise.
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq

from iontraps_qrc import core, sweep

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="NF-UNSW-NB15 parquet (default: data/raw/NF-UNSW-NB15.parquet)")
    ap.add_argument("--log1p-all", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--qubits", help="comma list, e.g. 6,8,10 (split half input / half memory)")
    ap.add_argument("--L", help="comma list of step counts, e.g. 6,8,10")
    ap.add_argument("--out", help="output directory (default results/sweep_alltoall)")
    ap.add_argument("--only", help="run a single cell, e.g. 8q_a2a_L8")
    ap.add_argument("--workers", type=int, default=0,
                    help="N>0: simulate circuits on N processes (bit-identical to serial)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = Path(a.data) if a.data else None
    if data:
        core.NFUNSW_PARQUET_CANDIDATES.insert(0, str(data))
    log1p_all = a.log1p_all if a.log1p_all is not None else bool(data and "v2" in data.name)
    if log1p_all:
        skip = set(core.NFUNSW_CATEGORICAL) | {core.NFUNSW_TARGET, core.NFUNSW_ATTACK}
        core.NFUNSW_HEAVY_TAILED = [c for c in pq.ParquetFile(data).schema_arrow.names if c not in skip]
    if a.qubits:
        sweep.QUBITS = [(n, n // 2, n // 2) for n in map(int, a.qubits.split(","))]
    if a.L:
        sweep.L_VALUES = [int(v) for v in a.L.split(",")]
    if a.out:
        sweep.OUT_DIR = Path(a.out)
    elif a.quick:
        sweep.OUT_DIR = REPO / "results" / "quick_check"
    if a.workers > 0:
        from iontraps_qrc import parallel
        parallel.install(core, workers=a.workers)

    print(f"[run] data={data or 'default'} log1p_all={log1p_all} qubits={sweep.QUBITS} "
          f"L={sweep.L_VALUES} out={sweep.OUT_DIR}")
    if a.dry_run:
        sweep.dry_run()
    else:
        sweep.run_all(quick=a.quick, only=a.only)


if __name__ == "__main__":
    main()
