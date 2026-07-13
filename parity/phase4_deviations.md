# Phase 4 deviations

Intentional deferrals from DESIGN.md §32 (Phase 4 deliverable list) as reconciled
against the tracked authority for capability status, `parity/dowhy.toml`.
`parity/dowhy.toml` is authoritative for `status`; this document explains any
gap between the DESIGN.md §32 narrative and that manifest for Phase 4.

## 1. Do-samplers → Phase 7

DESIGN.md §32 lists "do-samplers" under the Phase 4 deliverable narrative.
`parity/dowhy.toml`'s `dowhy.do_sampling` capability is tracked at
`status = "pending"`, `phase = 7`. Do-samplers ship with GCM/counterfactual
sampling infrastructure.

## 2. Conditional effects / effect modifiers → Phase 9

`dowhy.estimate.conditional` is `status = "pending"`, `phase = 9`. Phase 4 ships
unconditional ATE/ATT/ATC only.

## 3. General (nonparametric) mediation → Phase 9

Phase 4 ships the front-door two-stage / product-of-coefficients linear mediation
estimator. General mediation (Pearl nonparametric formula, NDE/NIE, multi-mediator
front-door, temporal mediation) remains Phase 9.

## Verification

All Phase 4 capabilities in `parity/dowhy.toml` (every row with `phase = 4`) are
`status = "done"` and backed by conformance under `conformance/phase4/` (estimators
and `refuters/`) plus unit/integration harnesses. Linear, partial-linear,
nonparametric, and Reisz sensitivity are implemented; Reisz is tracked under
`dowhy.refute.sensitivity` notes.
