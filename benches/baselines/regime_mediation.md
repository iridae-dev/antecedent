# regime / mediation baselines

Criterion benches (run with `--test` in `gate_context.sh`):

- `antecedent-discovery` bench `rpcmci`: `rpcmci_sparse_120`, `rpcmci_stress_240`
- `antecedent-estimate` bench `temporal_mediation`: `mediation_sparse_200`,
  `mediation_stress_800`

**Budgets (local regression, Apple M1 class):**

| Case | Soft latency budget |
|------|---------------------|
| rpcmci_sparse_120 | < 500 ms / iter (asserted in bench) |
| rpcmci_stress_240 | < 2 s / iter (asserted in bench) |
| mediation_sparse_200 | ~5 ms / iter typical; asserted gate **10 ms** (2× headroom for `--test` noise) |
| mediation_stress_800 | ~20 ms / iter typical; asserted gate **40 ms** (2× headroom for `--test` noise) |

Memory: multi-env sample plans must not clone sibling environment series
(see `antecedent-data` `MultiEnvSamplePlan` unit test; J-PCMCI+ emits
`jpcmci_plus.multi_env_plan` diagnostic). Soft latency budgets above are
asserted on a single timed iteration when Criterion runs with `--test`.
