# m-separation / PAG orientation baselines 

Established: 2026-07-14 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 99d25765 (the commit that added this file; the measured commit was not written down)

Sparse and stress criterion benches (run with `--test` in `gate_pag.sh`):

- `antecedent-graph` bench `mseparation`: `msep_admg_sparse_200`, `msep_admg_stress_80`,
  `msep_pag_sparse_100`, `msep_pag_stress_60`
- `antecedent-discovery` bench `pag_orientation`: `pag_orient_sparse_40`,
  `pag_orient_stress_120`

These establish local regression baselines for exit criteria; absolute
timings are machine-dependent.

Numeric wall-time gate: none published (`--test` smoke only).
