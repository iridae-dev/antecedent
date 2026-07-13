# Phase 4 deviations

Intentional deferrals from DESIGN.md §32 (Phase 4 deliverable list) as reconciled
against the tracked authority for capability status, `parity/dowhy.toml`.
`parity/dowhy.toml` is authoritative for `status`; this document explains any
gap between the DESIGN.md §32 narrative and that manifest for Phase 4.

## 1. Do-samplers → Phase 7

DESIGN.md §32 lists "do-samplers" under the Phase 4 deliverable narrative.
`parity/dowhy.toml`'s `dowhy.do_sampling` capability (weighting, multivariate
weighting, kernel-density, MCMC do-sampling; DESIGN.md §26 "Do-sampling") is
tracked at `status = "pending"`, `phase = 4` in the DESIGN.md §32 wording but
`phase = 7` in the manifest, which groups it with GCM/counterfactual sampling
(DESIGN.md §32 Phase 7: "GCM and counterfactuals"). The manifest's `phase`
field is authoritative; do-samplers ship in Phase 7 alongside the PCM/SCM
sampling infrastructure they depend on.

## 2. Conditional effects / effect modifiers → Phase 9

`dowhy.estimate.conditional` (conditional effects / effect modifiers) is
`status = "pending"`, `phase = 9` in `parity/dowhy.toml`, matching DESIGN.md
§32 Phase 9 ("contextual, regime, effect, and mediation parity"). Phase 4
ships unconditional ATE/ATT/ATC only; `AverageEffectQuery.effect_modifiers`
exists in the query type but is rejected by every Phase 4 estimator (see
`causal_estimate::frontdoor::prepare_frontdoor_problem` and equivalent guards
in the other estimators).

## 3. General (nonparametric) mediation → Phase 9

Phase 4 ships the **front-door two-stage / product-of-coefficients** linear
mediation estimator (`identifier = "frontdoor"`, `estimator =
"frontdoor.two_stage"`; see `causal_estimate::frontdoor`), which covers
DoWhy's `dowhy.estimate.two_stage` capability (already `done`). It assumes a
single mediator, a linear structural model, and no direct treatment→outcome
edge. General mediation — Pearl's nonparametric mediation formula, natural
direct/indirect effects, multi-mediator front-door sets, and Tigramite's
linear *temporal* mediation — is deferred to Phase 9 (DESIGN.md §32 Phase 9
"linear temporal mediation"; DESIGN.md §26 "causal mediation", "linear
mediation", "direct, total, mediated, and conditional effects").

## 4. Reisz-representer sensitivity diagnostics → deferred

DESIGN.md §18.2 and §26 list "Reisz-representer diagnostics" alongside
linear, partial-linear, and nonparametric sensitivity analysis. Phase 4 ships
**linear and partial-linear sensitivity** only (tracked under
`dowhy.refute.sensitivity`, `status = "done"`, `phase = 4`); nonparametric
sensitivity and Reisz-representer-based diagnostics (automatic debiased
machine learning-style robustness bounds) are not yet implemented. No
DESIGN.md phase currently owns this item explicitly; it is tracked here as an
open deferral pending an owning phase, and `dowhy.refute.sensitivity`'s
`done` status reflects the linear/partial-linear subset shipped, not the full
DESIGN.md §18.2 list.

## Verification

All Phase 4 capabilities in `parity/dowhy.toml` (every row with `phase = 4`)
are `status = "done"` as of this writing, each backed by conformance fixtures
under `conformance/dowhy/` and `conformance/phase4/`. The four deferrals above
are the only Phase 4-adjacent gaps between the DESIGN.md §32 narrative and the
manifest.
