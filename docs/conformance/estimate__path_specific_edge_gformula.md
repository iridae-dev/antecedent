# Path-specific effect with a complementary path (edge g-formula)

**Suite path:** `conformance/estimate/path_specific_edge_gformula`

Known-truth pin for `PathSpecificEffect` when a second directed path competes
with the selected one. The graph is `c -> t, c -> m, c -> y, t -> m, t -> y,
m -> y`; the query asks for the effect of `t` on `y` through `m`. The binary
SCM (see `reference.py`) has

- path-specific effect `E[Y(t=0, M(t=1))] - E[Y(t=0, M(t=0))] = 0.12`,
- total effect `0.356`,
- natural direct effect `0.156`.

`path_specific.natural` identifies this with the edge g-formula: each factor
`P(v | pa(v))` over the outcome's ancestors binds the treatment to the active
level on the `t -> m` edge (it starts a selected path) and to the control level
on `t -> y`, minus the all-control g-formula. An evaluator that binds one level
everywhere returns the total effect; one that swaps the levels returns the
natural direct effect. Before 1.9 the surgical-graph functional returned the
total effect here.

The contingency table is the exact SCM law. `reference.py` enumerates the
equally likely noise cells of the structural equations, so the counterfactual
truth comes from the SCM itself rather than from a g-formula, and the same
cells give the observational counts (divided by their common factor: 2,500
rows). It then recomputes the edge g-formula from the expanded table with
numpy and checks it against the structural truth. Run
`python3 conformance/estimate/path_specific_edge_gformula/reference.py --check`.

## Shared descendant that is not a recanting witness

The `shared_descendant` key holds a second case with the same schema. Graph
`c -> t, c -> w, c -> y, t -> a, t -> b, a -> w, b -> w, w -> y`; the query
selects the paths through `a`, so `t -> a -> w -> y` is selected and
`t -> b -> w -> y` is not. `w` lies on both, but the paths leave `t` through
different children, so it is not a recanting witness (Avin, Shpitser & Pearl
2005): `w` takes `a` from the active world and `b` from the control world
through one ordinary factor `P(w | a, b, c)`. Before this case was added the
identifier refused any node on both kinds of path. The binary SCM (thresholds
multiples of 1/4, see `reference.py`) has

- path-specific effect `E[Y(W(A(1), B(0)))] - E[Y(W(A(0), B(0)))] = 0.125`,
- total effect `0.0625`,
- effect along the unselected path `-0.0625`.

The truth comes from enumerating the 4^6 noise cells of the structural
equations; the 2,048-row table is the exact law (counts divided by their common
factor 2).

Consumers:

- `crates/antecedent/tests/path_specific_edge_gformula_pins.rs` pins the
  Frequentist plug-in at `0.12` (1e-9) and the Bayesian Dirichlet posterior
  mean within 0.02 of it (90% interval covers 0.12 and excludes 0.356), for
  explicit and accepted Dags, `none`/`cheap`/`full`, fresh and prepared; and
  the `shared_descendant` case the same way (0.125, interval excludes the
  total effect 0.0625).
- `crates/antecedent-estimate/src/functional_distribution.rs` checks the
  plug-in and that one Bayesian draw evaluates the edge g-formula on a single
  reweighted row law.
- `crates/antecedent/tests/v19_static_calibration.rs`
  (`path_specific_two_path_*_nominal_90_coverage`) samples from the same SCM
  for 400-replicate coverage.

## Expected summary

Top-level keys: `bayesian, case, columns, contingency_table, frequentist, graph, identification, query, schema_version, shared_descendant, truth` (11 fields).
