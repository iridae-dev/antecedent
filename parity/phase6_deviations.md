# Phase 6 deviations

Intentional deferrals from DESIGN.md §32 (Phase 6 deliverable list) as reconciled
against the tracked authority for Bayesian capability status,
`parity/bayesian.toml`. That manifest is authoritative for `status`; this
document explains gaps between the DESIGN.md §32 narrative and that inventory.

## 1. Full `causal-model` / PCM/SCM registry → Phase 7

Phase 6 ships Bayesian linear/GLM **fit objects** and g-computation inside
`causal-estimate`. The PCM/SCM mechanism registry, topological sampling plans,
and do-samplers remain Phase 7.

## 2. Bayesian DAG / DBN posterior search → later discovery work

DESIGN.md §13.7 graph posterior search is not in the Phase 6 deliverable list.
Phase 6 ships graph-weighted **effect envelopes** over supplied
`WeightedGraphSamples` only.

## 3. Hierarchical / BVAR / state-space / GP mechanisms

Listed in §14.4 as “after the base backend is stable” / optional. Deferred.

## 4. Stan / PyMC adapters

ADR 0006: native Laplace first; external adapters later.

## 5. MCMC chain diagnostics / SBC

§18.4 items that require multi-chain MCMC (ESS, divergences, SBC) wait for
HMC/SMC. Phase 6 ships PPC, prior sensitivity, and Laplace convergence/curvature
diagnostics.

## 6. Bayesian CI tests (§12)

Not listed under Phase 6 deliverables.

## Verification

Every `phase = 6` row in `parity/bayesian.toml` with `status = "done"` is backed
by conformance under `conformance/phase6/`, unit/integration harnesses, and/or
criterion benches mapped in `scripts/gate_phase6.sh`.
