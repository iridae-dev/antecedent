# linear front-door two-stage conformance fixture

**Suite path:** `conformance/estimate/frontdoor`

Generated inline by `crates/antecedent/tests/estimate_conformance.rs` — no
CSV, no pinned baseline install. Clean-room synthetic SCM, deterministic from a fixed
`ExecutionContext` seed.

SCM: `U -> T -> M -> Y` with `U -> Y`, `U` unmeasured. `T = 3 + U + noise`
(uncentred), `M = 0.4·T + noise`, `Y = 5·M + U + noise`; the mediated effect is
`0.4 · 5 = 2`. The two path coefficients differ from each other and from their
product, so returning either one alone (0.4, 5), or the confounded `Y ~ T` slope
(about 3), fails.

Comparisons:

- `|estimate.ate - true_effect| < tolerance` with `tolerance = 0.06`, about five
  standard errors at `n = 4000` (finite-sample Monte Carlo check; not a
  `StableFloat` comparison).
- `se_analytic` against the closed-form delta-method SE of this linear Gaussian
  SCM, `sqrt(0.566 / n)`, within `|ln ratio| < 0.1`.
- The result records the `frontdoor.linear_path_product` restriction.

The recorded DoWhy block in `expected.json` was run on the fixture's earlier SCM
(`M = T`, `Y = 2M`) and is kept as provenance only; nothing is compared with it.
The linear SCM here has no treatment–mediator interaction, which is the only
setting in which the product of coefficients equals the front-door functional;
see `conformance/estimate/frontdoor_functional` for the case where it does not.

## Expected summary

Top-level keys: `estimator, generation, identifier, notes, reference, tolerance, true_effect` (7 fields).
