# Discovery, attribution and simulation as compiled routes

Structure discovery, attribution on a fitted structural model and simulation
from one are ordinary queries. Each one compiles to a lowered program with a
stable identity, prepares to a handle and, once estimated, gives a result
stamped with that identity. The contract and claim can be read off it like any
other study. None of these routes is licensed. They run and say what they ran,
but no coverage record stands behind them, so they are not cells of the
support matrix and have no row in `parity/licensed_routes.toml`.

## The three query kinds

| kind | built with | route cells | payload |
| --- | --- | --- | --- |
| `CausalQuery::Discover` | `StudyBuilder::discovery(DiscoverySpec, variables)` | `DiscoverStatic` (tabular), `DiscoverTemporal` (series, panel, event, multi-environment) | `DiscoveryPayload`: the search engine's own result |
| `CausalQuery::Attribute` | `StudyBuilder::graph(dag).query(AttributeQuery)` | `GcmAttributeDag` | `AttributionPayload`: the attribution and the per-node mechanism selection record |
| `CausalQuery::Simulate` | `StudyBuilder::graph(dag).query(SimulateQuery)` | `GcmSimulateDag` | `SimulationPayload`: draws, a policy value, unit effects, draws with their noise, a noise replay, counterfactual values of the factual rows, or a temporal rollout or panel replay, with the selection record |

The query encodes everything the answer depends on: the variables, method and
bounds of a search (with a digest of the executable configuration: the
conditional-independence test, multiple-testing adjustment, background
knowledge, prior and score), and the target, its options and its seed for
attribution and simulation. If any of these change, the program changes too.

`MechanismChange` and `UnitChange` take the same retained path: a supplied or
accepted static `Dag` is sealed as a checked attribution operation (route cell
`GcmDag`) and runs at estimate.

```rust
use antecedent::{CompiledExecution, DiscoverySpec, Study, accept_discovered};

let search = Study::tabular(data.clone())
    .discovery(DiscoverySpec::pc(params, "parcorr"), variables)
    .build()?;
let execution = CompiledExecution::run(&search, &ctx)?;   // prepare + estimate_retained
let (contract, claim) = execution.contract_and_claim(&ctx)?;
assert_eq!(&*contract.query_kind, "Discover");
```

`CompiledExecution` is the prepared handle together with the result it stamped.
`Study::prepare` and `PreparedStudy::estimate_retained` do the same work in two
steps.

## Inference is declared by the method

A frequentist search (PC, FCI, RFCI, GES, DirectLiNGAM, NOTEARS, the PCMCI
family) evaluates its decision rule exactly and publishes no sampling
uncertainty. A structure posterior (exact enumeration, exact edge marginals,
order or structure MCMC, the CI-screened posterior, the DBN posterior) is a
posterior under its declared prior and Gaussian BIC score. The contract's
`inference` is whatever the method declares. A discovery study refuses
`StudyBuilder::inference`, refuters, validators, bootstrap and estimator options.

## From a discovered structure to a downstream study

A CPDAG or PAG stays a class. The only way a discovered structure reaches
another study is the review gate:

```rust
let (_, claim) = execution.contract_and_claim(&ctx)?;
let accepted = accept_discovered(execution.result(), &claim, true)?;   // or execution.accept_dag(&ctx)
let provenance = accepted.provenance().unwrap();   // method, inference, class, query digest,
                                                   // bounds, data snapshot, program, claim, review
let fit = Study::tabular(data).graph(accepted).query(attribute_query).build()?;
```

The gate refuses a result that no prepared handle stamped, or a claim that seals
a different execution. Leftover marks are refused unless you accept them, just
as in Python's `accept_discovered=True`. A study prepared on the accepted graph
carries `StructureProvenance` in its contract (`structure_provenance`) and in
its program identity. Its identification digest does not include the
provenance, so the identification is the same as it would be for the same graph
supplied by hand, while the program and lowered identity are different.
`CompiledExecution::accept_dag` additionally reads a fully oriented CPDAG as the
DAG it describes, which is the structure a mechanism fit needs.

A structure posterior (exact enumeration, MCMC, the CI-screened posterior) is accepted
with `accept_discovered_posterior` (or `CompiledExecution::accept_posterior`). The
`AcceptedPosterior` goes to a study through `StudyBuilder::accepted_graph_posterior`.
Its atoms and weights become the study's graph posterior, and the contract and program
identity carry the provenance (review `posterior`). As with a single graph, the
identification digest does not. `AcceptedPosterior::map_dag` keeps only the most
probable DAG atom as an accepted graph instead (review `posterior_map`). A temporal
posterior has no single static DAG, so `map_dag` refuses it. The atoms index the
searched variables by position, so a posterior is accepted only from a search over the
schema's columns in order.

## Mechanisms

An attribution or simulation study fits its mechanisms from a
`MechanismProgram`: by default the standard registry, choosing for each node by
best held-out score. `MechanismProgram::from_candidates` and `with_options`
let you declare the candidates and splits instead. `with_wrapped_prefit` runs a model
whose mechanisms the caller replaced with its own code. The wrapped variables enter the
program label (`gcm.registry.wrapped.<digest>`), and such a model is never refitted:
on other rows the route refuses. The program's label is bound
into the lowered statistical program, and every result publishes the per-node
selection record. `MechanismProgram::with_prefit` reuses an existing fit when
the route runs on the same rows, matched by content digest. Uncertainty from
the mechanism fit is not quantified (see [Mechanism selection](mechanism-selection.md)).

## Resources and cancellation

Before a search or a model fit allocates anything, it takes a lease from the
context's admission pool, sized from the rows, variables (or nodes), maximum lag
and draws. A pool that is too small gives `CausalError::Resource` with a
next-attempt hint. A cancelled context gives `CausalError::Cancelled` before
the search or fit starts and again after it ends. Simulation draws come from
the query's own seed.

## Python

Every `discover_*` binding, `FittedGcm` attribution and simulation, and the
discover-then-fit helpers run these routes. Their results expose `.contract`,
`.claim`, `.inspect()` and `.route` (`program_identity`,
`structure_provenance()`). Methods that return a list or tuple record their
route on `FittedGcm.last_route`. `antecedent.gcm.fit_gcm_discovered` goes
through `fit_gcm_accepted`, so the fitted model keeps the provenance of the
search that produced its graph.

## Callbacks in simulation programs

A simulation program can hold batch callbacks (`antecedent.intervention_callbacks`). They
run in the process exactly as before. The route's contract records a recoverable callback
(declared from source) by its verified recovery record. It records an in-process function
by the canonical digest of the whole query, which covers each callback's name, inputs,
output, pinned dependencies and implementation kind. Running locally is allowed.
Decoding such a record back into a query, which is what export or replay does, is
refused with `policy_callback_not_portable`, and the refusal names the next step:
declare the callback from self-contained source.

## What is not here

- `PreparedPolicy` still compiles and licenses a program once against a fitted model
  (`support`, `rebind`, the compilation counters). Its `value`, `sample` and
  `counterfactual` run the compiled simulation routes. `policy_support` is a licensing
  check and computes no answer.
- Ranking root causes (`rank_root_causes`) orders an attribution's components. It
  accepts only an attribution a compiled route or compiled study returned, so the
  ranking carries the contract of the answer it ranks.
- The free `antecedent::discovery::discover_*` functions and the `antecedent::gcm`
  attribution, simulation, noise-replay, counterfactual-policy and temporal-policy
  functions are hidden engines (`#[doc(hidden)]`). They are not the public way in, and
  the Python bindings do not call them directly.
