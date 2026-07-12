# ADR 0011: Performance posture

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.11, §2 (rules 11–22), §23, §28.8–28.9

## Decision

Correctness and performance are co-equal from Phase 0. There is no late
project-wide optimization phase. Hot paths require, from initial
implementation:

- prepared / batch APIs;
- reusable workspaces;
- memory plans;
- scalar reference implementations;
- optimized differential tests;
- benchmark gates and baselines.

## Consequences

- Feature PRs are incomplete without representative benchmarks and allocation
  assertions for designated hot paths.
- Optimizations must not silently change statistical semantics.
- Parallelism is explicit and bounded via `ExecutionContext`.
