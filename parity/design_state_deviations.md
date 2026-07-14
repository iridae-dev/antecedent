# Design / incremental-state deviations

Intentional limitations relative to DESIGN.md §19–20 and
`parity/design_state.toml`.

## 1. Design evaluation models

EIG uses a discrete graph-posterior soft-evidence model (not full Bayesian
optimal design over arbitrary discovery search). Effect-width reduction uses
linear-Gaussian / OLS Gram scaling. These are production algorithms for the
shipped surface, not stubs.

## 2. Incremental models selected

Linear OLS sufficient statistics, streaming covariance, and lag-index cache keys
are `done`. Particle-filter state and full graph-score incremental caches remain
waived. Tracked as `intentional_deviation` on
`design_state.incremental.particle_graph_score`.

## 3. No action dispatch

Decision analysis returns EU / regret / chance constraints only. The library
does not execute external actions or own approval workflows (DESIGN §1 / §19.4).
