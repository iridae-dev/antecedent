# dag_posterior

**Suite path:** `conformance/bayesian/dag_posterior`

Exact DAG posterior enumeration and structure/order MCMC on small Gaussian
SEMs. Facade composition: `discovery=ExactDagPosterior|OrderMcmc|StructureMcmc|CiScreenedPosterior`
+ `inference=Bayesian` mixes effect draws via `aggregate_effect_envelope`
(Python `analyze`); temporal analog uses `discovery=DbnPosterior` with
`PulseEffect`/`SustainedEffect`.

Exercised by `antecedent-discovery` unit tests:
`exact_enumeration`, `structure_mcmc`, `order_mcmc`, `ci_screened_posterior`,
`dbn_posterior`; facade: `bayesian_exact_dag_posterior_effect_envelope`,
`manufacturing_dbn_posterior_bayesian_envelope`, Python
`test_graph_posterior_analyze`.

## Expected summary

Top-level keys: `chain_fixture, collider_fixture, dbn_lag1_fixture, engines, exact_max_nodes, reference, score_family, tolerance_class` (8 fields).
