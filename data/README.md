# Data

`data/raw/` is git-ignored (parquet files are large). Expected files:

| file | used for |
|---|---|
| `NF-UNSW-NB15.parquet` (v1, 12 columns) | default loader, smoke tests. Present locally. |
| `NF-UNSW-NB15-v2.parquet` (37 features, repeated rows removed) | the committed results in `results/`. **Not in this repo**; it was built outside it (`prepare_v2_parquet.py`, not preserved here). Needed to reproduce `results/`. |

Point the code at a file with `--data PATH` (scripts) or the env var `NFUNSW_PARQUET`.
For v2, `--log1p-all` is applied automatically (file name contains `v2`).
