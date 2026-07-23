# ADR 0016 — design and incremental state

- Status: Accepted
- Date: 2026-07-21

## Context

 requires experiment/measurement candidate ranking (EIG,
identification probability, effect-width, decision utility) and an embeddable
`CausalState` with event/invalidation and incremental sufficient statistics.

## Decision

- Add crates `antecedent-design` and `antecedent-state` (no mutual dependency); shared
  registry IDs / `CacheBudget` / Monte Carlo reports live in `antecedent-core`.
- Design ranking is batched Monte Carlo with common-random-number draws,
  adaptive stop on rank uncertainty, and explicit constraint-violation records
  (never silent drops).
- `CausalState::apply` invalidates and versions only; callers request
  `refresh_results`. Incremental OLS / streaming covariance / graph-score
  deltas / particle-filter steps match full recomputation on fixtures.
- Facade + Python bind the same Rust entry points.

## Consequences

`scripts/gate_design_state.sh` gates inventory honesty, conformance pins, and
criterion smokes. deferral #2 is resolved.
