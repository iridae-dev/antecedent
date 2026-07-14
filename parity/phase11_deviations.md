# Phase 11 deviations

Intentional limitations relative to the full DESIGN.md §19–20 sketch.

## 1. Design evaluation models

EIG uses a discrete graph-posterior soft-evidence model (not full Bayesian
optimal design over arbitrary discovery search). Effect-width reduction uses
linear-Gaussian / OLS Gram scaling. These are production algorithms for the
Phase 11 exit criteria, not stubs.

## 2. Incremental models selected

Phase 11 ships linear OLS sufficient statistics, streaming covariance, and
lag-index cache keys. Particle-filter state and full graph-score incremental
caches remain deferred.

## 3. No action dispatch

Decision analysis returns EU / regret / chance constraints only. The library
does not execute external actions or own approval workflows (DESIGN §1 / §19.4).
