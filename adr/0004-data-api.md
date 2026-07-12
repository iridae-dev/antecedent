# ADR 0004: Library-owned data API

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.4, §5.2

## Decision

Public core APIs expose stable library-owned data views (`TableView`,
`ColumnView`, matrix views). Arrow is the preferred external and cross-language
physical representation. Arrow crate types are **not** the public causal API.
`causal-data` provides Arrow-backed implementations and adapters.

## Consequences

- Algorithms operate on typed slices, bitmaps, and prepared buffers after one
  dispatch at the column boundary.
- Materialization is explicit and recorded in execution diagnostics.
