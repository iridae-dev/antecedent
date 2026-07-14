# ADR 0006: Initial Bayesian GLM backend

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.6, §14.5

## Decision

The initial Bayesian GLM uses a native Laplace approximation behind a
backend-neutral inference interface. External probabilistic-programming
adapters come later under optional features.

## Consequences

- Laplace MAP, Hessian factorization, and MVN approximation
 before HMC/SMC adapters.
- Priors and parametric restrictions are recorded as assumptions; they do not
 erase nonparametric non-identifiability.
