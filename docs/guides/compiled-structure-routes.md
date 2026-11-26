# Discover, attribute and simulate as compiled queries

Use this page when you write Rust against the `antecedent` crate and want a structural
answer in the same form as any other study. The answer can be the graph a search returns.
It can be the node that explains a change. It can be the draws a fitted model generates.
You get a program with a stable identity. You get a prepared handle. You get a result whose
contract and claim you can read. The result carries that identity. The review gate is the
only way a discovered structure reaches a downstream study. Python users reach the same calls
through the bindings listed under [Python](#python).

These calls run and report what they ran. They are not cells of the
[support matrix](../support-matrix.md). They make no coverage claim. A result states the
method and its declared inference. A discovered graph is an estimate, not an established
structure.

## The three query kinds

| kind | built with | route cells | payload |
| --- | --- | --- | --- |
| `CausalQuery::Discover` | `StudyBuilder::discovery(DiscoverySpec, variables)` | `DiscoverStatic` (tabular), `DiscoverTemporal` (series, panel, event, multi-environment) | `DiscoveryPayload`: the search engine's own result |
| `CausalQuery::Attribute` | `StudyBuilder::graph(dag).query(AttributeQuery)` | `GcmAttributeDag` | `AttributionPayload`: the attribution and the per-node mechanism selection record |
| `CausalQuery::Simulate` | `StudyBuilder::graph(dag).query(SimulateQuery)` | `GcmSimulateDag` | `SimulationPayload`: draws, a policy value, unit effects, draws with their noise, a noise replay, counterfactual values of the factual rows, or a temporal rollout or panel replay, with the selection record |

The query encodes everything the answer depends on. For a search, that is the variables,
the method, and the bounds. The query also carries a digest of the executable
configuration. The digest covers the conditional-independence test and the
multiple-testing adjustment. It covers the background knowledge, the prior, and the score.
For attribution and simulation, the query encodes the target, its options, and its seed.
If any of these change, you get a different program.

`MechanismChange` and `UnitChange` use the same call. You supply a static `Dag`, or you
pass one you accepted. The call checks it as an attribution (`GcmDag`). It runs when you
estimate.

```rust
use antecedent::{CompiledExecution, DiscoverySpec, Study, accept_discovered};

let search = Study::tabular(data.clone())
    .discovery(DiscoverySpec::pc(params, "parcorr"), variables)
    .build()?;
let execution = CompiledExecution::run(&search, &ctx)?;   // prepare + estimate_retained
let (contract, claim) = execution.contract_and_claim(&ctx)?;
assert_eq!(&*contract.query_kind, "Discover");
```

`CompiledExecution` is the prepared handle together with its result. In Rust, call
`Study::prepare`. Then call `PreparedStudy::estimate_retained`. Those two calls do the same work.
`contract.query_kind` names the query kind (`"Discover"` here). `execution.result()` holds
the payload of the table above. The claim records that execution. The review gate accepts a
result only together with the claim that records it.

## Inference is declared by the method

A frequentist search evaluates its decision rule exactly. It publishes no sampling
uncertainty. The frequentist searches are:

- `PC`
- `FCI`
- `RFCI`
- `GES`
- `DirectLiNGAM`
- `NOTEARS`
- the PCMCI family

A structure posterior is a posterior under its declared prior and a Gaussian BIC score. The
posteriors are:

- exact enumeration
- exact edge marginals
- order MCMC
- structure MCMC
- the CI-screened posterior
- the DBN posterior

The contract's `inference` is whatever the method declares. A discovery study
refuses `StudyBuilder::inference`. It refuses refuters, validators, bootstrap, and estimator
options.

## Carry a discovered structure into a downstream study

A CPDAG or PAG stays a class. The only way a discovered structure reaches
another study is the review gate:

```rust
let (_, claim) = execution.contract_and_claim(&ctx)?;
let accepted = accept_discovered(execution.result(), &claim, true)?;   // or execution.accept_dag(&ctx)
let provenance = accepted.provenance().unwrap();   // method, inference, class, query digest,
                                                   // bounds, data snapshot, program, claim, review
let fit = Study::tabular(data).graph(accepted).query(attribute_query).build()?;
```

The gate refuses a result that no prepared handle produced. It refuses a claim that records
a different execution. Leftover marks are refused unless you accept them. Pass `true`, as
Python's `accept_discovered=True` does. A study prepared on the accepted graph carries
`StructureProvenance` in its contract, on `structure_provenance`. It also carries that
provenance in its program identity. The identification digest does not include the
provenance. The identification matches the same graph supplied by hand. The program identity
is different. In Rust, call `CompiledExecution::accept_dag`. It reads a fully oriented CPDAG
as the DAG it describes. A mechanism fit needs that DAG.

A structure posterior is accepted with `accept_discovered_posterior`. The posteriors are
exact enumeration, MCMC, and the CI-screened posterior. In Rust, call
`CompiledExecution::accept_posterior` for the same acceptance. The `AcceptedPosterior` goes
to a study through `StudyBuilder::accepted_graph_posterior`. Its atoms and weights become
the study's graph posterior. The contract and program identity carry the provenance. The
review label is `posterior`. As with a single graph, the identification digest does not.
In Rust, call `AcceptedPosterior::map_dag`. It keeps only the most probable DAG atom as an
accepted graph. The review label is `posterior_map`. A temporal posterior has no single
static DAG. `map_dag` refuses it. The atoms index the searched variables by position. A
posterior is accepted only from a search over the schema's columns in order.

## Mechanisms

An attribution or simulation study fits its mechanisms from a `MechanismProgram`. By
default that is the standard registry, choosing for each node by best held-out score. In
Rust, call `MechanismProgram::from_candidates` to declare the candidates. Call
`with_options` to declare the splits. `with_wrapped_prefit` runs a model whose mechanisms
you replaced with your own code. The wrapped variables enter the program label
(`gcm.registry.wrapped.<digest>`). Such a model is never refitted. On other rows the call
refuses. The program's label is part of the statistical program you run. Every result
publishes the per-node selection record. In Rust, call `MechanismProgram::with_prefit` to
reuse an existing fit when the call runs on the same rows. The match is by content digest.
Uncertainty from the mechanism fit is not quantified (see
[Mechanism selection](mechanism-selection.md)).

## Resources and cancellation

Before a search or a model fit allocates anything, it takes a lease from the
context's admission pool. The lease is sized from the rows, the variables or nodes, the
maximum lag, and the draws. A pool that is too small gives `CausalError::Resource` with a
next-attempt hint. A cancelled context gives `CausalError::Cancelled` before the search or
fit starts. It gives `CausalError::Cancelled` again after the search or fit ends. Simulation
draws come from the query's own seed.

## Python

Every `discover_*` binding runs these calls. `FittedGcm` attribution and simulation run
them. The discover-then-fit helpers run them. Their results expose `.contract` and
`.claim`. They expose `.inspect()` and `.route`. On `.route`, read `program_identity` and
`structure_provenance()`. Methods that return a list or tuple record the call on
`FittedGcm.last_route`. `antecedent.gcm.fit_gcm_discovered` goes through
`fit_gcm_accepted`. The fitted model keeps the provenance of the search that produced its
graph.

## Callbacks in simulation programs

A simulation program can hold batch callbacks (`antecedent.intervention_callbacks`). They
run in the process. The contract records a recoverable callback, declared from source, by
its recovery record. It records an in-process function by the canonical digest of the whole
query. That digest covers each callback's name, inputs, and output. It covers pinned
dependencies and implementation kind. Running locally is allowed. Decoding such a record
back into a query is what export or replay does. That decode is refused with
`policy_callback_not_portable`. The refusal names the next step: declare the callback from
self-contained source.

## Refusals

| Request | Outcome |
| --- | --- |
| `StudyBuilder::inference`, refuters, validators, bootstrap or estimator options on a discovery study | refused at build (`option_not_applicable`); the method declares its own inference |
| a result that no prepared handle stamped, or a claim that seals a different execution, handed to the review gate | refused |
| leftover marks at the review gate without acceptance | refused; pass `true` to accept them, as Python's `accept_discovered=True` does |
| `AcceptedPosterior::map_dag` on a temporal posterior | refused; a temporal posterior has no single static DAG |
| a wrapped pre-fitted model run on other rows | refused; such a model is never refitted |
| an admission pool too small for the rows, variables, lag and draws | `CausalError::Resource` with a next-attempt hint |
| a cancelled context | `CausalError::Cancelled` |
| decoding a query that holds an in-process callback (export or replay) | `policy_callback_not_portable`; see [other registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes) |

How refusals reach your code, and the fields to branch on, are in
[Refusals](../concepts/refusals.md).

## Examples

- [`examples/rust/gcm_do.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/gcm_do.rs)
  and
  [`examples/rust/intrinsic_influence.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/intrinsic_influence.rs)
  run a simulation and an attribution study through `CompiledExecution::run`.
- [`examples/rust/discover_then_estimate.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/discover_then_estimate.rs)
  and
  [`examples/python/discover_then_estimate.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/discover_then_estimate.py)
  run discovery once. They then estimate repeatedly from an accepted graph without
  rediscovering. Both write the accepted DAG by hand to stand in for a reviewer's decision.
  Neither passes through the review gate.

## Related behaviour

- `PreparedPolicy` compiles a program once against a fitted model and records that support.
  Read `support`, `rebind`, and the compilation counters. Its `value`, `sample`, and
  `counterfactual` run the simulation calls. `policy_support` is a support check. It
  computes no answer.
- Ranking root causes (`rank_root_causes`) orders an attribution's components. It accepts
  only an attribution one of these calls returned, or a compiled study returned. The ranking
  carries the contract of the answer it ranks.
- The free `antecedent::discovery::discover_*` functions are hidden engines
  (`#[doc(hidden)]`). The `antecedent::gcm` attribution, simulation, noise-replay,
  counterfactual-policy, and temporal-policy functions are hidden engines too. They are not
  the public way in. The Python bindings do not call them directly.

## Related

- [Give discovery background knowledge](discovery-background-knowledge.md)
- [Read graph uncertainty and unidentified mass](../graph_uncertainty.md)
- [Fit and query a structural causal model](gcm-estimands.md)
- [Choose and inspect mechanisms](mechanism-selection.md)
