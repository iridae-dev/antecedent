# Phase 9 deviations

Intentional deferrals from DESIGN.md §32 (Phase 9).

## 1. Mechanism-change / Shapley attribution across regimes → Phase 10 — RESOLVED

Completed in Phase 10. See `parity/phase10.toml` and ADR 0015.

## 2. Experiment design / CausalState → Phase 11 — RESOLVED

Completed in Phase 11. See `parity/phase11.toml` and ADR 0016.

## 3. FCI / RFCI

Still deferred (unchanged from Phase 8).

## 4. Full nonparametric natural-effect ID

Phase 9 ships the **linear temporal mediation** surface (identify + estimate;
NDE/NIE under linear SEM). Exotic nonparametric path-specific variants beyond
that surface remain deferred.

## 5. RPCMCI regime assignment mode

RPCMCI accepts an external [`RegimeAssignment`] (e.g. `two_regime_half_split`)
and fits one graph per regime. Full Tigramite-style unsupervised regime search
is not claimed; callers supply typed regime labels.

## Verification

`tigramite.discovery.jpcmci_plus`, `tigramite.discovery.rpcmci`,
`tigramite.effects`, and `dowhy.estimate.conditional` are `done` with evidence
mapped in `scripts/gate_phase9.sh` (integration tests that load
`conformance/phase9/*/expected.json`).
