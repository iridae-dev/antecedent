# Cross-world effects on mixed and class graphs

**Suite path:** `conformance/estimate/cross_world_class_truth`

Enumerated count tables and the exact functionals they imply, for a counterfactual and a natural
direct effect on an ADMG, a temporal ADMG, a CPDAG and a PAG.

* `front_door`: `t -> m -> y` with `t <-> y`. The effect on the treated of setting `t` to 1 among the
  units observed at 0 is `sum_m P(m | t=1) P(y=1 | m, t=0) - P(y=1 | t=0) = 0.3`; the same table with
  no latent edge gives the natural direct effect `0.38` by the mediation formula.
* `unconfounded_chain`: `w -> x -> y`; the effect on the treated is the contrast `0.5`.
* `mediated_two_parents`: `c -> t`, `w -> m`, `t -> m`, `t -> y`, `m -> y`; the natural direct effect
  on a PAG whose edges are all visible is `0.38`.
* `two_mediators`: `t -> a`, `t -> b`, `a -> y`, `b -> y`; the effect along the edges that avoid the
  mediator `a` is `0.096` by the edge g-formula.
* `line`: a noiseless line with slope 2. The completions `x -> y` and `y -> x` of `x - y` give 2 and
  0, so the enumeration-weighted mean is 1 with the identified set `[0, 2]`.

## Expected summary

Top-level keys: `case, cases, line, oracle, schema_version, tables` (6 fields).
