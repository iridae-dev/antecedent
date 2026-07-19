# Architecture

Library for causal discovery, identification, estimation, SCMs, counterfactuals,
attribution, and validation. Rust owns computation; Python is a thin binding layer.

## Invariants

1. **Identify before estimate.** Estimators never choose confounders or assert identifiability.
2. **Graph classes stay distinct.** DAG, ADMG, CPDAG, PAG, and temporal variants are not aliases.
3. **Uncertainty sources stay distinct.** Parameter, sampling, graph, orientation, identification, mechanism, regime, and measurement uncertainty are separate fields — not collapsed into one CI.
4. **Bayesian inference does not erase non-identifiability.** Priors/restrictions are recorded as assumptions.
5. **One workflow for static and temporal.** Modality is compiled from data + query, not a second API.
6. **Discovered structure is evidence.** Review, constraints, and completion are explicit.
7. **Heavy work stays in Rust.** Python crosses at coarse operations only.
8. **Results are reproducible.** Schema, preprocessing, graph version, assumptions, config, seeds, backend versions, and warnings attach to artifacts.
9. **Parity is capability parity**, not Python API cloning.
10. **Performance is part of the feature.** Hot paths need benches, allocation profiles, and explicit memory/layout contracts before merge.

## Crates

```text
causal-core          ids, schemas, queries, interventions, provenance, plans, errors
causal-kernels       borrowed views + scalar/portable/arch kernels (no causal semantics)
causal-data          tabular / temporal / panel / multi-env views, sample planning, Arrow
causal-graph         DAG/ADMG/CPDAG/PAG, separation, overlays, temporal unfold
causal-expr          arena-backed causal-functional IR
causal-stats         regression, covariance, resampling, CI tests, faer LA backend
causal-prob          posteriors, priors, graph samples, inference backends
causal-discovery     PC/FCI/GES/LiNGAM/NOTEARS, PCMCI family, Bayesian DAG engines
causal-identify      adjustment, IV, front-door, mediation, ID/IDC, envelopes
causal-estimate      frequentist + Bayesian estimators for identified functionals
causal-model         SCMs, mechanisms, intervention overlays, sampling
causal-counterfactual  abduction–action–prediction
causal-attribution   anomaly / distribution / mechanism / path / Shapley
causal-validate      refuters, sensitivity, discovery stability, Bayesian checks
causal-design        EIG / VoI / experiment ranking (computation only)
causal-state         incremental caches, invalidation, sufficient statistics
causal-io            CBOR+Arrow artifacts, graph interchange, migration
causal               facade: CausalAnalysis planner + re-exports
```

Dependency edges point downward (no cycles). Facade (`causal`) sits on top.
Bayesian discovery may use `causal-prob` without pulling `causal-model`.

## Analysis pipeline

```text
data (+ optional discovery)
  → graph / GraphEvidence
  → identify → IdentifiedEstimand
  → estimate | posterior
  → validate (optional)
  → CausalAnalysisResult (+ plans, provenance, diagnostics)
```

Compilation produces an inspectable **logical** plan (semantics) and **physical**
plan (layouts, kernels, batching, parallelism, memory). Physical choices must not
change logical semantics.

`run()` auto-accepts only when the graph is fully specified for the query;
otherwise compilation returns `ReviewRequired`.

## Execution model

- `ExecutionContext` owns thread budget, RNG seeds, memory limits, and kernel policy.
- No private global thread pools; no recursive oversubscription.
- Workspaces and prepared designs are reused across bootstrap / draw batches.
- SIMD is an implementation detail behind library-owned views (`KernelPolicy`).
- Python callbacks are explicit slow paths; the physical plan marks them.

## Python package

```text
python/
  src/lib.rs          # PyO3 → causal._native
  causal/             # pure-Python wrappers + stubs
```

Algorithms stay in Rust. Bindings convert and release the GIL. Identification and
validation surface through `analyze(...).identification` / `.validation`, not
separate top-level modules.

## Where to look next

| Topic | Location |
|-------|----------|
| Artifact wire format | [artifacts.md](artifacts.md) |
| Hot-path benches / budgets | [hot_paths.md](hot_paths.md) |
| Capability inventories | [parity/](../parity/README.md) |
| ADRs | [adr/](../adr/README.md) |
| Conformance fixtures | [conformance/](conformance/README.md) |
| Security / unsafe / license review | [security_review.md](security_review.md) |
