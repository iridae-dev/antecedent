# m-separation / PAG orientation baselines 

Sparse and stress criterion benches (run with `--test` in `gate_pag.sh`):

- `antecedent-graph` bench `mseparation`: `msep_admg_sparse_200`, `msep_admg_stress_80`,
  `msep_pag_sparse_100`, `msep_pag_stress_60`
- `antecedent-discovery` bench `pag_orientation`: `pag_orient_sparse_40`,
  `pag_orient_stress_120`

These establish local regression baselines for exit criteria; absolute
timings are machine-dependent.
