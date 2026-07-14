# Phase 9 deviations

Intentional deferrals from DESIGN.md §32 (Phase 9).

## 1. Mechanism-change / Shapley attribution across regimes → Phase 10

RPCMCI ships typed regime assignments and per-regime graphs. Cross-regime
mechanism-change and Shapley-style attribution remain Phase 10.

## 2. Experiment design / CausalState → Phase 11

## 3. FCI / RFCI

Still deferred (unchanged from Phase 8).

## 4. Full nonparametric natural-effect ID

Phase 9 ships the **linear temporal mediation** surface (identify + estimate;
NDE/NIE under linear SEM). Exotic nonparametric path-specific variants beyond
that surface remain deferred.

## Verification

`tigramite.discovery.jpcmci_plus`, `tigramite.discovery.rpcmci`,
`tigramite.effects`, and `dowhy.estimate.conditional` are `done` with evidence
mapped in `scripts/gate_phase9.sh`.
