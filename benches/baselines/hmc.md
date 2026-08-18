# HMC GLM baselines

Owner: `antecedent-prob` / `fit_hmc_glm`

## Criteria

- Repeated HMC fits of the same shape must reuse `LaplaceWorkspace` scratch: after a
  warm `ws.prepare(n, dim, n_draws * n_chains)`, `ws.grow_count` must stay flat across
  every subsequent fit (asserted in the Criterion bench after the timed loop).
- Bench target: `hmc_gaussian_n30_p1` — n=30 intercept-only known-σ² Gaussian,
  2 chains × 20 warmup × 40 post-warmup draws × 4 leapfrog steps. The `--test`
  smoke is **not** a publication gate (ESS ≥ 100 / R̂ ≤ 1.01 live in unit tests).

## Measured mean

- `hmc_gaussian_n30_p1`: **101.24 µs** (Criterion mean, CI 100.32–102.20 µs).
- Established: 2026-08-18. Machine class: Apple M1 Max.

## How to refresh

```bash
cargo bench -p antecedent-prob --bench hmc
```

Record the Criterion mean and update this file with the run date and machine class. The
bench aborts (assert) if the workspace grows across fits — a refresh that trips the
assert is a reuse regression, not a new baseline.

Gate: mean ≤ **121.49 µs** (20% over 101.24 µs).
