# front-door functional conformance fixture

**Suite path:** `conformance/estimate/frontdoor_functional`

Generated inline by `crates/antecedent/tests/estimate_conformance.rs` — no CSV, no
random draw. The test expands the exact 4000-row population table of a binary SCM
whose latent treatment–outcome confounder modifies the mediator's effect:

`U ~ Bern(.5)`, `P(T=1|U) = .1 + .5U`, `P(M=1|T) = .1 + .7T`,
`P(Y=1|M,U) = .05 + .9·M·U`, with `U` dropped from the data and the graph
(`T -> M -> Y`).

Truth by enumeration over `U`: `0.7 · 0.9 · E[U] = 0.315`.

Comparisons:

- `frontdoor.functional` must equal the enumerated truth to `1e-12` and record
  `frontdoor.functional.saturated_cells`.
- `frontdoor.linear_two_stage` must land on its own probability limit `0.3631`
  (the `P(t)·Var(M|t)`-weighted within-arm slope times `0.7`), stay more than `0.04`
  from the truth, and record `frontdoor.linear_path_product`.

The arms are unbalanced (`P(T=1) = 0.35`) and the within-arm slopes differ
(`0.771` vs `0.277`), so weighting `Σ_t'` uniformly (`0.3669`) also fails the pin.

## Expected summary

Top-level keys: `estimator, identifier, linear_path_product_limit, notes, reference, tolerance, true_effect` (7 fields).
