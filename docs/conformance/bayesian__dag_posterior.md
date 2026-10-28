# dag_posterior

**Suite path:** `conformance/bayesian/dag_posterior`

Exact DAG posterior enumeration and structure/order MCMC on small Gaussian
SEMs. Facade composition: `discovery=ExactDagPosterior|OrderMcmc|StructureMcmc|CiScreenedPosterior`
+ `inference=Bayesian` mixes effect draws via `aggregate_effect_envelope`
(Python `analyze`); temporal analog uses `discovery=DbnPosterior` with
`PulseEffect`/`SustainedEffect`.

Consumed by
`crates/antecedent-discovery/tests/dag_posterior_conformance.rs`, which runs
every engine in `engines` on known SEMs (a chain, a collider, a lag-1 series)
and requires the fixture's minimum posterior mass on the true structure: the
chain skeleton, both collider arrows, and the lag-1 DBN edge. It also pins
`score_family` and `exact_max_nodes` to the implementation. This is internal
known-truth evidence; no external package generated these values.

The CI-screened engine's screened space on the collider leaves a single graph,
so its chains never move and its MCMC publication gate refuses that posterior;
the test checks the collider posterior with that gate off.

The minimum masses sit just below the exact BIC posterior: an independent
enumeration of all 25 three-node DAGs (weights from residual sums of squares and the
BIC penalty; uniform prior) gives skeleton mass 1 on the chain, 0.876 on each
collider arrow, and the exact engine must reproduce those marginals to 1e-6. The
sampling engines must stay within 0.06 of the enumeration, and the CI-screened
engine, whose screened space drops graphs, must keep the edges the full posterior
puts at least half its mass on.

## Expected summary

Top-level keys: `chain_fixture, collider_fixture, dbn_lag1_fixture, engines, exact_max_nodes, oracle, reference, score_family, tolerance_class` (9 fields).
