# Phase 9 regime / mediation baselines

Criterion benches (run with `--test` in `gate_phase9.sh`):

- `causal-discovery` bench `rpcmci`: `rpcmci_sparse_120`, `rpcmci_stress_240`
- `causal-estimate` bench `temporal_mediation`: `mediation_sparse_200`,
  `mediation_stress_800`

**Budgets (local regression, Apple M1 class):**

| Case | Soft latency budget |
|------|---------------------|
| rpcmci_sparse_120 | < 500 ms / iter typical |
| rpcmci_stress_240 | < 2 s / iter typical |
| mediation_sparse_200 | < 5 ms / iter typical |
| mediation_stress_800 | < 20 ms / iter typical |

Memory: multi-env sample plans must not clone sibling environment series
(see `causal-data` `MultiEnvSamplePlan` unit test). Absolute timings are
machine-dependent; gate only smoke-runs `--test`.
