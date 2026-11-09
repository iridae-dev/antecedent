# Compiler crate graph

New internal types for a [compiler layer](compiler-layers.md) go in existing
crates and add no dependency edge. `PreparedStudy` / `PreparedAnalysis` stay the
workflow. `antecedent-expr` stays the identified-functional IR.

The normal `[dependencies]` edges below, including optional ones, are those of
each member `Cargo.toml`. No production cycle exists. There are no
`[build-dependencies]` on workspace crates. `scripts/check_architecture.py`
checks the production ownership boundaries in the static gate.

Dev-only edges, not part of the library graph:

- `antecedent-io` dev-depends on `antecedent-io` (`test-util`). A self-edge, not a cross-crate cycle.
- `antecedent-model` dev-depends on `antecedent-identify`. `antecedent-identify` does not depend on `antecedent-model`. Not a cycle.
- Several crates dev-depend on `antecedent-test-support`, a test-only crate with no workspace dependencies.

## Homes for new internal types

"Existing types" are already in that crate. "Home for new internal types" is
where a new internal type for that layer goes. No new crate.

| Layer | Existing types | Home for new internal types |
| --- | --- | --- |
| 1. Scientific problem | `CausalQuery`, schema, interventions, `ValueExpr`, assumptions in `antecedent-core`. Causal graphs in `antecedent-graph`. Observations in `antecedent-data`. | `antecedent-core`. Graphs stay in `antecedent-graph`. Storage stays in `antecedent-data`. |
| 2. Identification product | `IdentifiedEstimand`, `CausalExprArena`, `FunctionalProgram` in `antecedent-expr`. `IdentificationResult` and `Identifier` in `antecedent-identify`. | `antecedent-identify` for products and obligations of search. Functionals stay in `antecedent-expr`. |
| 3. Statistical program | Estimator `prepare` / `fit`, `PreparedEstimationProblem`, `ScoreTable`, `JointFitPlan` in `antecedent-estimate`. Posterior algorithms and `PriorSpec` in `antecedent-prob`. | `antecedent-estimate` for frequentist and Bayesian realization of an identified functional. `antecedent-prob` for posterior algorithms and priors. `antecedent-learn` stays prediction (`LearnerFactory`, `FittedPredictor`, `DesignView`) and does not become the statistical-program owner. `antecedent-model` stays generative mechanisms (`CompiledCausalModel`, `NodeProgram`), not the analysis statistical program. |
| 4. Evidence/decision program | `EvidenceLedger`, joint-law checks, mechanism-composition plans in `antecedent-contract`. `DecisionProblem`, `evaluate_decision`, `evaluate_evsi`, design ranking in `antecedent-design`. | Rules and refusals stay in `antecedent-contract`. Utility, EVSI, and ranking stay in `antecedent-design`. |
| 5. Physical plan | Records in `antecedent-contract` (`PhysicalExecutionPlanRecord`). Builder in `antecedent` (`PhysicalExecutionPlan`). Kernels in `antecedent-kernels`. Retained statistics in `antecedent-state`. | Executable operators stay in `antecedent`, the crate that already depends on estimate, kernels, data, and the plan records. Estimator-free records stay in `antecedent-contract`. `antecedent-kernels` stays semantics-free. `antecedent-state` stays caches and sufficient statistics. |
| 6. Claim/receipt | Identities and `ExecutionReceipt` in `antecedent-core`. `ClaimEnvelope` and `HandoffReceipt` in `antecedent-contract`. `CausalContract` and `StudyResult` in `antecedent`. Consume split across `antecedent-io-core` and `antecedent-io`. | Envelope types stay in `antecedent-contract`. Digest vocabulary stays in `antecedent-core`. Estimator-free consume stays in `antecedent-io-core`. Estimator-bound replay stays in `antecedent-io`. The prepared companion stays `CausalContract` on `antecedent`. |

## Allowed direction

An edge may point only at a crate the source already reaches, or along the
existing normal graph. New internal types do not add edges. In particular:

- `antecedent-expr` depends on `antecedent-core` and `antecedent-kernels`.
- `antecedent-contract` depends only on `antecedent-core`.
- Neither gains an estimator, model, design, or facade dependency.
- `antecedent-learn` is not imported by `antecedent-expr` or `antecedent-contract`. It provides prediction to `antecedent-estimate`, `antecedent-model`, `antecedent-validate` and `antecedent-io-core`.
- `antecedent-design` may keep `antecedent-io-core`. It does not gain `antecedent-estimate`, `antecedent-validate`, or `antecedent-io`.
- `antecedent-io-core` stays free of `antecedent-estimate`, `antecedent-validate`, `antecedent-design`, `antecedent-io`, and `antecedent`.
- `antecedent-kernels` stays free of causal crates above `antecedent-core`.

Normal edges (optional edges included):

```text
antecedent-core
antecedent-kernels          -> core
antecedent-contract         -> core
antecedent-graph            -> core
antecedent-learn-burn       -> core
antecedent-expr             -> core, kernels
antecedent-data             -> core, kernels
antecedent-prob             -> core, kernels, stats
antecedent-stats            -> core, kernels
antecedent-state            -> core, kernels, stats
antecedent-learn            -> core, kernels, stats, prob, learn-burn
antecedent-identify         -> core, data, expr, graph, kernels
antecedent-discovery        -> core, data, graph, kernels, prob, state, stats
antecedent-model            -> core, data, graph, kernels, learn, stats
antecedent-counterfactual   -> core, contract, data, graph, model
antecedent-attribution      -> core, data, graph, kernels, model, counterfactual, prob, stats
antecedent-io-core          -> core, contract, counterfactual, data, discovery, expr,
                               graph, identify, kernels, learn, model, prob, stats
antecedent-estimate         -> core, contract, data, expr, graph, identify, io-core,
                               kernels, learn, model, prob, stats
antecedent-validate         -> core, data, discovery, estimate, expr, graph, identify,
                               io-core, kernels, learn, prob, stats
antecedent-design           -> core, contract, graph, identify, io-core, kernels, prob, stats
antecedent-io               -> core, contract, data, discovery, estimate, expr, graph,
                               identify, io-core, kernels, learn, model, prob, stats,
                               validate
antecedent                  -> attribution, contract, core, counterfactual, data, design,
                               discovery, estimate, expr, graph, identify, io, io-core,
                               kernels, learn, model, prob, state, stats, validate
antecedent-py               -> antecedent
```

The Python extension (`antecedent-py`) depends only on the facade.

## Cycle check

Placing new internal types by layer adds no edge, so the normal graph stays acyclic.

Putting executable physical work in `antecedent-io-core` would require
`antecedent-io-core -> antecedent-estimate` and close
`antecedent-estimate -> antecedent-io-core`. Putting it in
`antecedent-design` would require `antecedent-design -> antecedent-estimate`,
which is forbidden below even though that single edge is acyclic.
Putting identification products in `antecedent-expr` that need
`antecedent-identify` would close `antecedent-identify -> antecedent-expr`.
Putting statistical programs in `antecedent-contract` or `antecedent-expr`
would import an estimator crate into a crate those estimators already reach.

## Edges a lower crate must not add

Each row is an import that the current normal graph already reaches in the
opposite direction. Adding it cycles.

| Importer | Must not import | Existing path that would close |
| --- | --- | --- |
| `antecedent-core` | any workspace crate | every crate depends on `core` |
| `antecedent-expr` | `antecedent-identify` | `identify -> expr` |
| `antecedent-expr` | `antecedent-estimate`, `antecedent-io-core`, `antecedent-design`, `antecedent-io`, `antecedent` | those crates depend on `expr` |
| `antecedent-contract` | `antecedent-estimate` | `estimate -> io-core -> contract` |
| `antecedent-contract` | `antecedent-io-core`, `antecedent-design`, `antecedent-counterfactual`, `antecedent-io`, `antecedent` | each depends on `contract` |
| `antecedent-identify` | `antecedent-estimate`, `antecedent-io-core`, `antecedent-design`, `antecedent-io`, `antecedent` | each depends on `identify` |
| `antecedent-learn` | `antecedent-estimate`, `antecedent-model`, `antecedent-io-core`, `antecedent` | `estimate -> learn`; `model -> learn`; `io-core -> learn` |
| `antecedent-learn-burn` | `antecedent-learn` | `learn -> learn-burn` (optional) |
| `antecedent-kernels` | `antecedent-stats`, `antecedent-prob`, `antecedent-data`, `antecedent-estimate` | each depends on `kernels` |
| `antecedent-model` | `antecedent-estimate` | `estimate -> io-core -> model` |
| `antecedent-model` | `antecedent-counterfactual`, `antecedent-attribution` | both depend on `model` |
| `antecedent-io-core` | `antecedent-estimate` | `estimate -> io-core` |
| `antecedent-io-core` | `antecedent-design` | `design -> io-core` |
| `antecedent-io-core` | `antecedent-validate`, `antecedent-io`, `antecedent` | each depends on `io-core` |
| `antecedent-estimate` | `antecedent-validate`, `antecedent-io`, `antecedent` | each depends on `estimate` |
| `antecedent-discovery` | `antecedent-io-core`, `antecedent-estimate` | `io-core -> discovery`; `estimate -> io-core -> discovery` |
| `antecedent-state` | `antecedent-discovery`, `antecedent-estimate` | `discovery -> state`; `estimate -> io-core -> discovery -> state` |
| `antecedent-counterfactual` | `antecedent-attribution`, `antecedent-io-core` | `attribution -> counterfactual`; `io-core -> counterfactual` |
| `antecedent-design` | `antecedent` | `antecedent -> design` |
| `antecedent-io` | `antecedent` | `antecedent -> io` |
| `antecedent` | `antecedent-py` | `antecedent-py -> antecedent` |

`antecedent-prob` must not import `antecedent-learn`: `learn -> prob`.
`antecedent-stats` must not import `antecedent-state` or `antecedent-learn`.

## Acyclic imports that are rejected

These edges do not cycle against the current normal graph. They are rejected
because they would put a higher layer's crate under a lower one, or would
break the estimator-free cuts.

| Edge | Why it does not cycle | Why it is rejected |
| --- | --- | --- |
| `antecedent-design -> antecedent-estimate` | `estimate` does not reach `design` | Design may depend on `io-core` and must not acquire estimate, validate, or estimator-bound IO. |
| `antecedent-design -> antecedent-validate` | `validate` does not reach `design` | Same cut. |
| `antecedent-design -> antecedent-io` | `io` does not depend on `design` | `antecedent-io` is the estimator-bound artifact crate. |
| `antecedent-contract -> antecedent-model` | `model` does not depend on `contract` | Contract stays free of model and estimator back-references. |
| `antecedent-contract -> antecedent-learn` | `learn` does not depend on `contract` | Same. Prediction machinery does not move into the contract crate. |
| `antecedent-expr -> antecedent-model` or `antecedent-learn` | neither depends on `expr` through a path that returns only if expr imported them; `model` does not depend on `expr` | Expression stays the identified-functional IR. Its edges are `core` and `kernels`. |
| `antecedent-io-core -> antecedent-design` | listed above; this one does cycle | Recorded in the cycle table. `io-core` also stays estimator-free and design-free so design can keep reading it. |

`antecedent-estimate -> antecedent-contract` is a normal edge and acyclic
(`contract` depends only on `core`). It does not make `antecedent-contract` a
home for statistical program types: those stay in `antecedent-estimate` and
`antecedent-prob`. Claim envelopes stay in `antecedent-contract`, which does
not import `antecedent-estimate`.
