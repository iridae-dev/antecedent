# ADR 0017 — parity closure and 1.0 preparation

- Status: Superseded by [0018](0018-release-version-history.md) (version-freeze
  and format-freeze clauses only; see that ADR)
- Date: 2026-07-21
- Updated: 2026-07-22 (retire deviation vocabulary; DESIGN.md retired)

## Context

1.0 preparation requires closing or explicitly scoping every parity manifest
item, stabilizing artifact schemas, completing the Python wheel matrix,
generating docs from conformance, stabilizing benchmark baselines, and
recording security/licensing/unsafe/dependency review — without treating
performance as a deferred rewrite (ADR 0011).

## Decision

> **Superseded (2026-07-29):** the version-freeze and format-freeze bullet
> immediately below is stale — three releases (0.2.0, 0.3.0, 0.4.0) have
> since shipped and the artifact format moved to `{ major: 0, minor: 2 }`.
> See [ADR 0018](0018-release-version-history.md) for the current version
> history and format freeze. The rest of this ADR's decisions are unaffected
> and remain Accepted.

- ~~Keep crate and Python package versions at **0.1.0**; freeze artifact
  `FormatVersion { major: 0, minor: 1 }` with an explicit migration registry.~~
  Superseded by [ADR 0018](0018-release-version-history.md).
- Inventories use only `pending` / `in_progress` / `done`. Permanent product
  contracts are marked `done` with an inline note (no `intentional_deviation` /
  `*_deviations.md`). Required 1.0 chapters are closed in inventories.
- Ship **DOT + JSON + GML + NetworkX** DAG interchange in `antecedent-io` as the
  string-graph surface for `pinned baseline.model_graph.parsing`.
- Ship full CPython 3.11–3.14 wheel CI (Linux x86_64/aarch64 manylinux, macOS
  arm64, Windows x86_64) with default `faer` and no system BLAS.
- Generate `docs/conformance/` from fixtures; index hot paths in
  `docs/hot_paths.md`; gate via `scripts/gate_release.sh`.
- Retire `DESIGN.md` in favor of `docs/architecture.md` and
  `docs/development.md`.

## Consequences

No required capability uses a waiver status. Release preparation evidence lives
under parity inventories, ADR, docs, CI, and the gate. A future 1.0.0 version
bump is a separate release decision. Any future work reopens as `pending`
inventory rows with inline notes.
