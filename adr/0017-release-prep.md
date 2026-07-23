# ADR 0017 — parity closure and 1.0 preparation

- Status: Accepted
- Date: 2026-07-21
- Updated: 2026-07-22 (retire deviation vocabulary; DESIGN.md retired)

## Context

1.0 preparation requires closing or explicitly scoping every parity manifest
item, stabilizing artifact schemas, completing the Python wheel matrix,
generating docs from conformance, stabilizing benchmark baselines, and
recording security/licensing/unsafe/dependency review — without treating
performance as a deferred rewrite (ADR 0011).

## Decision

- Keep crate and Python package versions at **0.1.0**; freeze artifact
  `FormatVersion { major: 0, minor: 1 }` with an explicit migration registry.
- Inventories use only `pending` / `in_progress` / `done`. Permanent product
  contracts are marked `done` with an inline note (no `intentional_deviation` /
  `*_deviations.md`). Required 1.0 chapters are closed in inventories.
- Ship **DOT + JSON + GML + NetworkX** DAG interchange in `causal-io` as the
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
