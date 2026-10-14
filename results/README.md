# Results

Grid: 6, 8, 10, 12, 14, 16 qubits x L = 6, 8, 10 on NF-UNSW-NB15-v2, 512 shots, split seed 42, zero-day family = Exploits.

| file | content |
|---|---|
| `sweep_summary_full_grid.json` | all 18 cells (AUCs, diagnostics, `source` run) — the main table |
| `grid_effects_check.json` | paired difference quantum − classical with 4000-resample bootstrap 95% intervals |
| `rerun_comparison.json` | 14 Oct rerun vs 13 Oct run: all 12 shared cells bit-identical; 3 new cells (L=10) |
| `cells/results_<N>q_a2a_L<L>.json` | full per-cell output (spec, budget, options) for 14 of 18 cells |

Missing per-cell files: `14q_a2a_L{6,8,10}` (13 Oct run, not rerun) and `16q_a2a_L10`. Their numbers are
still in `sweep_summary_full_grid.json`.

Quantum columns of the report table are the `*_denoised` fields (only features above their own shot noise).
`iontraps_qrc.report.compare_with_report` checks all 18 cells against the report table: 18/18 match at 3 decimals.
