#!/bin/bash
# Full grid used in the report: 6, 8, 10, 12, 16 qubits x L = 6, 8, 10
# (14 qubits come from the 13 Oct run). 16 qubits is last because it is the
# most expensive. Needs the v2 parquet (see data/README.md). Resumable: a stage
# with a DONE marker is skipped.
#   PYTHON=python DATA=data/raw/NF-UNSW-NB15-v2.parquet WORKERS=24 bash scripts/rerun_all.sh
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}; DATA=${DATA:-data/raw/NF-UNSW-NB15-v2.parquet}; WORKERS=${WORKERS:-24}
stage() {   # label qubits Ls
  out="results/runs/$1"; mkdir -p "$out"
  [ -f "$out/DONE" ] && { echo "skip $1"; return; }
  PYTHONUNBUFFERED=1 "$PY" -W ignore scripts/run_sweep.py --data "$DATA" --qubits "$2" --L "$3" \
      --workers "$WORKERS" --out "$out" > "$out/run.log" 2>&1 && touch "$out/DONE"
}
stage 6-8-10-12q 6,8,10,12 6,8,10
stage 16q 16 6,8,10
