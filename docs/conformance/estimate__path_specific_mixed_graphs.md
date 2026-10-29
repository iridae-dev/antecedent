# Path-specific effect on a mixed graph, a class and a PAG

**Suite path:** `conformance/estimate/path_specific_mixed_graphs`

Known truth for `PathSpecificEffect` on a supplied `Admg`, `TemporalAdmg` (its contemporaneous
projection), `Cpdag` and `Pag`. Every case is a binary structural model whose noises are uniform
on four equally likely bins, so the observed law is an exact contingency table and the
counterfactual truth comes from the structural equations, not from a g-formula (`reference.py`).

The effect is `E[Y(A(1), B(0))] - E[Y(A(0), B(0))]`: the treatment `T` enters `A` at the active
level and `B` at the control level, selecting the path `T -> A -> Y` only.

- `district_admg` (`T -> A`, `T -> B`, `A -> Y`, `B -> Y`, latent `U` behind `A <-> Y`): the effect
  is `0.125`, the total effect `0.25`. The route identifies it by the district formula; a
  covariate-adjusted or total-effect functional is wrong here.
- `routed_markov` (`W -> T`, `T -> A`, `T -> B`, `A -> Y`, `B -> Y`, no latent): the effect is
  `0.15625`, the total effect `0.375`. A graph with every mark decided is identified by the edge
  g-formula; the same table is used for a Cpdag with every edge compelled and for a Pag whose
  edges out of `T` and into `Y` are visible.
- `collider_class` (the `(t, a, b, y)` marginal, class `A -> Y <- B`, `T - A`, `T - B`): three
  completions. `T -> A, T -> B` gives `0.15625`; `B -> T -> A` adjusts `B` and gives `0.1875`;
  `A -> T -> B` has no directed path through `A` and is one third unidentified mass.

Run `uv run --frozen --script conformance/estimate/path_specific_mixed_graphs/reference.py --check`
(add `--self-test` to confirm the check rejects mutated fixtures).

Consumer: `crates/antecedent/tests/path_specific_mixed_graph_route_evidence.rs`.

## Expected summary

Top-level keys: `bayesian, case, collider_class, district_admg, frequentist, oracle, routed_markov, schema_version` (8 fields).
