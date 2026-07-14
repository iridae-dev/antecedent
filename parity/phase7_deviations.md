# Phase 7 deviations

Intentional deferrals from DESIGN.md §32 (Phase 7) as reconciled against
`parity/gcm.toml` (authoritative for `status`).

## 1. Shapley / coalition attribution → Phase 10 — RESOLVED

Completed in Phase 10 (`causal-attribution` Shapley engine, distribution-change,
unit-change, robust variant). See `parity/phase10.toml` and ADR 0015.

## 2. Bayesian DAG posterior search remains deferred

Graph-posterior **model collections** consume supplied `WeightedGraphSamples`.
Bayesian DAG/DBN search stays deferred (see Phase 6 deviations).

## Verification

Every `phase = 7` row in `parity/gcm.toml` with `status = "done"` is backed by
conformance under `conformance/phase7/`, unit tests, and/or criterion benches
mapped in `scripts/gate_phase7.sh`. Phase 10 rows are gated by
`scripts/gate_phase10.sh`.
