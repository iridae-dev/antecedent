# ADR 0017 — parity closure and 1.0 preparation

- Status: Accepted
- Date: 2026-07-21
- Updated: 2026-07-22 (retire deviation vocabulary)

## Context

DESIGN.md requires closing or explicitly scoping every parity manifest
item, stabilizing artifact schemas, completing the Python wheel matrix,
generating docs from conformance, stabilizing benchmark baselines, and
recording security/licensing/unsafe/dependency review — without treating Phase
12 as a deferred performance rewrite (DESIGN §28).

## Decision

- Keep crate and Python package versions at **0.1.0**; freeze artifact
  `FormatVersion { major: 0, minor: 1 }` with an explicit migration registry.
- Inventories use only `pending` / `in_progress` / `done`. Unfinished DESIGN
  chapters stay `pending` and on `TODO.md`. Permanent product contracts are
  written into DESIGN.md and marked `done` (no `intentional_deviation` /
  `*_deviations.md`).
- Ship **DOT + JSON + GML + NetworkX** DAG interchange in `causal-io` as the
  string-graph surface for `pinned baseline.model_graph.parsing`.
- Ship full CPython 3.11–3.14 wheel CI (Linux x86_64/aarch64 manylinux, macOS
  x86_64/arm64, Windows x86_64) with default `faer` and no system BLAS.
- Generate `docs/conformance/` from fixtures; index hot paths in
  `docs/hot_paths.md`; gate via `scripts/gate_release.sh`.

## Consequences

No required capability uses a waiver status. Release preparation evidence lives
under parity inventories, ADR, docs, CI, and the gate. A future 1.0.0 version
bump is a separate release decision. Full scope completion still requires
clearing `TODO.md` / `pending` rows.
