# MCMC diagnostics baselines

Established: 2026-08-18 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 59295cec (the commit that added this file; the measured commit was not written down)

Owner: `antecedent-prob` / `mcmc_summary`

## Criteria

- One-pass `mcmc_summary` (max rank∪folded R̂, min bulk ESS, min tail ESS) over a
  packed multi-chain draw buffer. Standing decision: Geyer / rank-normalized
  statistics stay exact; no FFT autocorrelation.
- Bench target: `mcmc_summary_c4_n256_p2` — 4 chains × 256 draws × 2 parameters,
  AR(1) ρ=0.5 synthetic draws, column-major
  `samples[(chain * n_draws + draw) * n_params + param]`.

## Measured mean

- `mcmc_summary_c4_n256_p2`: **158.26 µs** (Criterion mean, CI 153.47–167.90 µs).
- Established: 2026-08-18. Machine class: Apple M1 Max.

## How to refresh

```bash
cargo bench -p antecedent-prob --bench mcmc_stats
```

Record the Criterion mean and update this file with the run date and machine class.

Gate: mean ≤ **189.91 µs** (20% over 158.26 µs).
