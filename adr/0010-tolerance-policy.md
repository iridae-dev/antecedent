# ADR 0010: Numeric tolerance policy

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.10, §28.5

## Decision

There is **no** project-wide epsilon. Fixtures declare one of:

- `Exact`
- `StableFloat`
- `BackendSensitive`
- `ResidualBased`
- `MonteCarlo`
- `PosteriorDistribution`

## Consequences

- Conformance and property tests attach a tolerance class to each comparison.
- Changing a fixture's class requires an explicit justification.
