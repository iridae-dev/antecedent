# ADR 0005: Temporal indexing

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.5, §5.4, §6.1

## Decision

Temporal analysis uses stable `(VariableId, Lag)` identities and time-major
dense indexes for finite unfolding. Dense indexes are construction-time
artifacts and are **not** serialized. `Lag(0)` is contemporaneous; negative-lag
conventions are confined to import/export adapters.

## Consequences

- Graph and discovery hot paths use `DenseNodeId`, never string names.
- Serialized artifacts carry stable variable schemas and lag metadata, not
  dense node IDs.
