# ADR 0016 — Phase 11 design and incremental state

- Status: Accepted
- Date: 2026-07-21

## Context

DESIGN.md Phase 11 requires experiment/measurement candidate ranking (EIG,
identification probability, effect-width, decision utility) and an embeddable
`CausalState` with event/invalidation and incremental sufficient statistics.

## Decision

- Add crates `causal-design` and `causal-state` (no mutual dependency); shared
  registry IDs / `CacheBudget` / Monte Carlo reports live in `causal-core`.
- Design ranking is batched Monte Carlo with common-random-number draws,
  adaptive stop on rank uncertainty, and explicit constraint-violation records
  (never silent drops).
- `CausalState::apply` invalidates and versions only; callers request
  `refresh_results`. Incremental OLS / streaming covariance match full
  recomputation on fixtures.
- Facade + Python bind the same Rust entry points.

## Consequences

`scripts/gate_phase11.sh` gates inventory honesty, conformance pins, and
criterion smokes. Phase 9 deferral #2 is resolved.
