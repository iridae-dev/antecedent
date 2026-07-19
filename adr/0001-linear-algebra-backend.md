# ADR 0001: Linear algebra backend

- Status: Accepted
- Date: 2026-07-21

## Decision

`faer` is the default dense linear-algebra backend behind an operation-level
abstraction. Public APIs expose library-owned matrix views. Optional BLAS is
additive and never removes the default `faer` path from conformance testing.

## Consequences

- `causal-stats` always depends on `faer` (required, not a feature flag).
- Callers never take `faer` types as the public causal API surface.
- Optional BLAS, when added, is an additive feature and never removes `faer`.
- Backend swaps are measured on designated workloads.
