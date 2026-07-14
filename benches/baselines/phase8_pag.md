# m-separation / PAG orientation baselines (Phase 8)

Sparse and stress criterion benches (run with `--test` in `gate_phase8.sh`):

- `causal-graph` bench `mseparation`: `msep_admg_sparse_200`, `msep_admg_stress_80`,
  `msep_pag_sparse_100`, `msep_pag_stress_60`
- `causal-discovery` bench `pag_orientation`: `pag_orient_sparse_40`,
  `pag_orient_stress_120`

These establish local regression baselines for Phase 8 exit criteria; absolute
timings are machine-dependent.
