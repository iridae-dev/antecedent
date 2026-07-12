# ADR 0001: Linear algebra backend

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.1, §11.1

## Decision

`faer` is the default dense linear-algebra backend behind an operation-level
abstraction. Public APIs expose library-owned matrix views. Optional BLAS is
additive and never removes the default `faer` path from conformance testing.

## Consequences

- `causal-stats` depends on `faer` under the `faer` feature (default).
- Callers never take `faer` types as the public causal API surface.
- Backend swaps are measured on designated workloads.
