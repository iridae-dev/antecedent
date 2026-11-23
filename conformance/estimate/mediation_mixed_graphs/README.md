# Mediation decomposition on a mixed graph, a class and a PAG

Known truth for `MediationEffect` on a supplied or accepted `Admg`, `TemporalAdmg` (its
contemporaneous projection), `Cpdag` and `Pag`. Every case is a binary structural model whose
noises are uniform on four equally likely bins, so the observed law is an exact contingency table
and the effects come from evaluating the outcome equation under each treatment argument and each
mediator-treatment argument on every noise cell, not from a mediation formula (`reference.py`).

- Total effect `E[Y(1, M(1))] - E[Y(0, M(0))]`, natural direct effect `E[Y(1, M(0))] - E[Y(0, M(0))]`
  and the pure natural indirect effect `E[Y(0, M(1))] - E[Y(0, M(0))]`. An interaction makes the
  pure indirect effect differ from total minus direct.
- `covariate_latent` (`W -> T`, `W -> M`, `W -> Y`, `T -> M`, `T -> Y`, `M -> Y`, latent `U` behind
  `W <-> Y`): total `0.15625`, direct `0.15625`, indirect `0.0625`.
- `markov_parents` (`C -> T`, `W -> M`, `T -> M`, `T -> Y`, `M -> Y`): total `0.25`, direct
  `0.15625`, indirect `0.1875`. The same table serves a CPDAG with every edge compelled and a PAG
  whose marks are decided and visible.

Run `uv run --frozen --script conformance/estimate/mediation_mixed_graphs/reference.py --check`
(add `--self-test` to confirm the check rejects mutated fixtures).

Consumer: `crates/antecedent/tests/mediation_mixed_graph_route_evidence.rs`.
