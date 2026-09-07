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

## 1.2 Bayesian mediation reference

`bayesian_mediation_n800_draws512` fits independent Gaussian mediator/outcome
posteriors on two prepared designs and composes direct, mediated, total and
requested effects. The posterior workspace is reused. The timed loop includes
both fits and composition; design preparation is outside it. The retained
payload assertion is exactly 512 × 4 values, independent of row count; this is
an output-size check, not a peak-allocation measurement.

Measured locally on macOS 26.5.2 arm64, 2026-09-07, optimized Criterion build,
10 samples, 1 s warm-up and 1 s measurement:

| Case | Mean (95% bootstrap interval) | Optional local mean gate |
|------|-------------------------------|--------------------------|
| bayesian_mediation_n800_draws512 | 178.4 µs (172.7–184.8 µs) | ≤ **5 ms** |

The short local measurement is a regression reference, not a portable latency
guarantee. `gate_context.sh` executes this case and its output assertion.
`GATE_CRITERION_MEANS=1 bash scripts/gate_hot_path_baselines.sh` additionally
checks locally available Criterion means against the declared gate.
