# Causal Rust Library: Technical Design

Status: implementation design, revision 2

Performance posture: correctness and performance are co-equal implementation requirements. Hot-path data layout, allocation behavior, vectorization, parallel execution, and benchmark budgets are designed and tested during feature development; there is no late project-wide optimization pass.

Scope baseline: functional parity with pinned external estimate/discovery baselines (see parity/baselines/ and ADR 0009), excluding EconML integration, plus a Bayesian-first extension that preserves frequentist parity. The project is a library and Python extension, not a hosted service, workflow system, dashboard, or deployment platform.

## Status markers

Each major section is tagged so roadmap text is distinguishable from what exists:

- **built** — implemented substantially as described; remaining gaps, if any, are called out inline
- **partial** — meaningful implementation exists; open gaps are named here or in `TODO.md`
- **planned** — roadmap only; not yet built (DESIGN leads the code — do not delete unbuilt sections)

Subsection status inherits the parent unless noted.

## 1. Scope boundary [built]

The library implements causal computation. It owns:

- data schemas and memory views required for causal analysis;
- optimized scalar and SIMD-capable numerical kernels used by the causal algorithms;
- causal graph representations and graph algorithms;
- causal discovery for static and temporal data;
- symbolic identification and do-calculus;
- frequentist and Bayesian estimation;
- structural causal models, interventions, and counterfactuals;
- attribution of outcome and distribution changes;
- robustness checks, falsification, and sensitivity analysis;
- experiment-design and decision-analysis primitives;
- incremental model-state primitives;
- serialization of models, queries, results, and provenance;
- Rust APIs and Python bindings;
- benchmark fixtures and performance-regression baselines for supported workloads.

The library does not own:

- data ingestion services;
- job orchestration or distributed schedulers;
- model registries or approval workflows;
- user authentication or authorization;
- dashboards or alert routing;
- long-running process supervision;
- action execution against external systems;
- policy administration or organization-specific governance;
- cluster resource management;
- GPU fleet management or remote execution.

Applications may build those capabilities on top of the library.

Performance is within library scope. The project is responsible for avoiding avoidable copies, allocation-heavy hot loops, unsuitable graph representations, uncontrolled parallelism, and Python round trips. The project is not responsible for operating the surrounding service or selecting infrastructure on behalf of callers.

## 2. Non-negotiable design rules [built]

1. **Identification is separate from estimation.** An estimator never chooses confounders or silently asserts identifiability.
2. **Graph classes remain distinct.** DAG, ADMG, CPDAG, PAG, and temporal variants are not interchangeable aliases.
3. **Uncertainty sources remain distinct.** Parameter, sampling, graph, orientation, identification, mechanism, regime, and measurement uncertainty are represented separately.
4. **Bayesian inference does not erase non-identifiability.** Priors and parametric restrictions are recorded as additional assumptions, not reported as nonparametric identification.
5. **Static and temporal analysis use one user workflow.** Modality-specific behavior is compiled internally from data and query semantics.
6. **Discovered structure is evidence, not asserted truth.** Review, constraints, and graph completion are explicit operations.
7. **Heavy execution stays in Rust.** Python calls cross the language boundary at coarse-grained operations.
8. **Every result is reproducible.** Data schema, preprocessing, graph version, assumptions, algorithm configuration, random seeds, backend versions, and warnings are attached to artifacts.
9. **Parity is capability parity, not Python API parity.** Rust types and interfaces are native to Rust.
10. **No universal dynamic object model.** Traits are used at extension points; concrete enums and structs are used where the set of semantics is known.
11. **Performance is a functional requirement.** A feature is incomplete until its representative workloads have benchmarks, allocation profiles, and resource behavior recorded.
12. **Data movement is designed before algorithm implementation.** Hot-path inputs and outputs must have an explicit memory layout, ownership plan, and copy policy before the algorithm is accepted.
13. **No per-observation dynamic dispatch.** Trait-object calls, Python callbacks, boxing, hash lookup, and heap allocation are prohibited inside scalar inner loops unless the slow path is explicit in the API and benchmarked separately.
14. **Scalar and optimized implementations share one semantic contract.** Portable scalar kernels are the correctness reference. SIMD, parallel, BLAS, and architecture-specific paths must pass the same property, conformance, and tolerance tests.
15. **Do not defer hot-path architecture.** Algorithms expected to dominate runtime must use reusable workspaces, stable dense indexes, batched execution, and vectorization-friendly layouts from their first implementation.
16. **Do not optimize by changing statistical semantics.** Reordering, approximation, caching, parallel reduction, or reduced precision may not silently change sample selection, masking, conditioning order, randomization, stopping rules, or estimand definitions.
17. **Allocation behavior is part of the API contract for core kernels.** Repeated operations must expose workspace or batch APIs when scratch storage is material. Per-call scratch allocation is not accepted in high-frequency paths.
18. **Parallelism is explicit and bounded.** Core crates do not create private global pools, recursively oversubscribe, or select thread counts without the execution context.
19. **SIMD is an implementation strategy, not a public type.** Public APIs expose stable library-owned views. Optimized kernels select scalar, portable-vector, or architecture-specific implementations behind those views.
20. **Benchmarks gate merges.** Changes to designated hot paths must not regress the accepted baseline beyond the documented budget without an approved explanation and replacement baseline.
21. **Memory limits are enforced.** Algorithms with potentially superlinear storage expose bounds, streaming modes, or explicit refusal instead of relying on eventual allocation failure.
22. **Fast paths are visible.** Execution diagnostics record copies, materializations, backend selection, thread use, cache hits, and fallback paths when those choices materially affect performance.

## 3. Workspace layout [built]

Repository root (this workspace; not a nested `causal-rs/` directory):

```text
./
  Cargo.toml
  crates/
    causal-core/
    causal-data/
    causal-graph/
    causal-expr/
    causal-kernels/
    causal-stats/
    causal-prob/
    causal-discovery/
    causal-identify/
    causal-estimate/
    causal-model/
    causal-counterfactual/
    causal-attribution/
    causal-validate/
    causal-design/
    causal-state/
    causal-io/
    causal/
  python/                    # maturin package; workspace member
    Cargo.toml               # PyO3 cdylib (causal._native)
    pyproject.toml
    src/lib.rs               # Rust binding crate
    causal/                  # pure-Python package + stubs (python-source = ".")
      __init__.py
      *.py / *.pyi
      _native.*              # built extension module
      py.typed
    tests/
    examples/
  parity/
    *.toml                   # capability inventories (status vocabulary in §26)
    release.toml
    fixtures/
  conformance/
    estimate/ | discovery/ | gcm/ | pag/ | …
    gates/
  scripts/
    gate_*.sh                # release / inventory gates
  benches/
  fuzz/
  docs/
  adr/                      # ADRs 0001–0017 (see §33)
  provenance/
```

### 3.1 Crate responsibilities

#### `causal-core`

Contains identifiers, schemas, assumptions, queries, interventions, provenance, diagnostics, errors, execution policy, and common artifact envelopes. It must not depend on numerical, graph-algorithm, Arrow, or Python crates. Core identifiers are compact and copyable; user strings do not appear in hot graph or numerical structures.

#### `causal-data`

Owns stable library-defined tabular, temporal, panel, multi-environment, and event-indexed data views. It is responsible for type metadata, category domains, masks, lag-aligned sample planning, row selection, splitting, and Arrow adapters. It must expose borrowed typed column views and prepared samples so downstream algorithms do not repeatedly decode Arrow arrays or allocate design matrices.

#### `causal-graph`

Owns graph types, endpoint semantics, graph transformations, separation algorithms, paths, districts, equivalence-class operations, temporal unfolding, graph evidence, dense node indexes, and reusable traversal workspaces. It must not use user-facing names as graph keys in algorithmic paths.

#### `causal-expr`

Owns the symbolic probability and causal-functional intermediate representation used by identification and posterior evaluation. Expression nodes are interned or arena-backed where repeated cloning would otherwise dominate identification and simplification.

#### `causal-kernels`

Owns low-level borrowed matrix/vector views and scalar, portable-vector, and architecture-specific kernels. It is the only default crate permitted to contain reviewed SIMD-related `unsafe` code. It provides reduction, covariance, residualization, distance, contingency-table, sampling, and small-matrix helper kernels used by `causal-stats`. It contains no causal semantics.

#### `causal-stats`

Owns numerical algorithms, regressions, covariance estimators, resampling, nearest-neighbor search, multiple-testing correction, density and dependence measures, and the linear-algebra backend abstraction. It uses `faer` by default and delegates elementwise hot loops to `causal-kernels`.

#### `causal-prob`

Owns probability distributions, posterior samples, weighted graph samples, latent-state draws, inference diagnostics, prior specifications, and inference-backend interfaces. Draw storage is columnar and batch-oriented; one heap object per draw is prohibited.

#### `causal-discovery`

Owns static and temporal discovery algorithms, conditional-independence tests, score-based search, graph priors, posterior graph search, and discovery diagnostics. Candidate sets, conditioning sets, and orientation queues use compact indexed structures and reusable workspaces.

#### `causal-identify`

Owns adjustment identification, IV, front-door, mediation, ID/IDC, generalized adjustment, partial-identification results, and identification over graph classes. It operates on graph indexes and expression arenas rather than repeatedly cloning graphs or expression trees.

#### `causal-estimate`

Owns estimators for identified functionals: regression, weighting, matching, stratification, doubly robust, IV, regression discontinuity, temporal effects, Bayesian g-computation, and posterior functional evaluation. Fit objects retain prepared design information and sufficient statistics when repeated evaluation is expected.

#### `causal-model`

Owns probabilistic and structural causal models, causal mechanisms, mechanism assignment and fitting, observational/interventional sampling, and model validation. Models compile to a topological execution plan; sampling does not traverse the semantic graph for every generated row.

#### `causal-counterfactual`

Owns abduction-action-prediction, exogenous-noise inference, counterfactual worlds, nested counterfactual evaluation, and unit-level effects. Counterfactual worlds share immutable model structure and use intervention overlays rather than cloning models.

#### `causal-attribution`

Owns anomaly attribution, distribution-change attribution, mechanism-change attribution, path attribution, Shapley decompositions, feature relevance, and root-cause ranking. Exact combinatorial methods must expose explicit size limits and approximation alternatives.

#### `causal-validate`

Owns refuters, sensitivity analysis, overlap diagnostics, graph falsification, posterior predictive checks, prior sensitivity, simulation-based calibration, and discovery stability. Validation reuses the shared resampling engine and does not create nested thread pools.

#### `causal-design`

Owns expected-information-gain, value-of-information, experiment ranking, measurement design, intervention selection, utility and loss primitives, and constraints. Monte Carlo design evaluation is batched and reports its compute budget and Monte Carlo error.

#### `causal-state`

Owns incremental state updates, invalidation tracking, cached sufficient statistics, versioned causal artifacts, and reevaluation of registered queries. State caches are bounded, keyed by semantic versions, and independently discardable.

#### `causal-io`

Owns stable CBOR metadata serialization, Arrow IPC sections, import/export, DOT/GML/JSON, NetworkX-compatible exchange, model bundles, and schema migrations. It serializes versioned wire types, not internal Rust structs directly.

#### `causal`

High-level facade. It re-exports stable types and provides the common logical planner, physical execution planner, and analysis workflow.

### 3.2 Dependency direction

Edges below match current `Cargo.toml` dependency graphs. Arrows mean "is depended on by"
(`A <- B` ⇒ `B` depends on `A`). All edges point downward — no layering violations.

```text
causal-core
  <- causal-kernels
  <- causal-data          (+ causal-kernels)
  <- causal-graph
  <- causal-expr
  <- causal-prob          (+ causal-kernels)
  <- causal-stats         (+ causal-kernels)

causal-data + causal-graph + causal-stats
  <- causal-discovery
  # planned (Bayesian graph discovery, §13.7): also causal-prob

causal-graph + causal-expr + causal-data
  <- causal-identify

causal-data + causal-expr + causal-stats + causal-prob
  + causal-kernels + causal-graph + causal-identify
  <- causal-estimate

causal-data + causal-graph + causal-stats + causal-prob + causal-kernels
  <- causal-model

causal-model + causal-prob + causal-data + causal-graph
  <- causal-counterfactual

causal-model + causal-counterfactual + causal-prob
  + causal-data + causal-graph + causal-stats + causal-kernels
  <- causal-attribution

# Not "all analysis crates" — each owns a narrower surface matching §3.1:

causal-core + causal-data + causal-estimate + causal-expr
  + causal-identify + causal-discovery + causal-prob
  + causal-stats + causal-kernels
  <- causal-validate

causal-core + causal-kernels + causal-prob + causal-stats
  <- causal-design

causal-core + causal-stats
  <- causal-state

causal-core + causal-estimate + causal-graph + causal-identify + causal-prob
  <- causal-io

(all analysis crates above)
  <- causal                 # facade only
```

Circular dependencies are prohibited. Shared types move downward only when their semantics are genuinely shared. A perceived need for a high-level crate to be imported by a lower-level crate is resolved through a smaller interface type, not a cycle.

## 4. Core identity and schema types [built]

Identifiers are compact, copyable indexes. User-facing names and labels are stored once in schemas and dictionaries, not repeated throughout graphs, queries, or numerical structures.

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VariableId(u32);

#[repr(transparent)]
pub struct EnvironmentId(u32);

#[repr(transparent)]
pub struct RegimeId(u32);

#[repr(transparent)]
pub struct Lag(u32);
```

`Lag(0)` is contemporaneous. Historical nodes use positive lag values internally. Negative-lag conventions are confined to import/export adapters.

```rust
pub struct VariableSchema {
    pub id: VariableId,
    pub name: Arc<str>,
    pub value_type: ValueType,
    pub role_hints: SmallRoleSet,
    pub unit: Option<Arc<str>>,
    pub category_domain: Option<CategoryDomainId>,
    pub measurement: MeasurementSpec,
}

pub enum ValueType {
    Continuous,
    Count,
    Binary,
    Categorical,
    Ordinal,
    Vector { width: NonZeroU32, element: ScalarType },
}
```

Roles are hints and constraints, not graph truth:

```rust
pub enum RoleHint {
    TreatmentCandidate,
    OutcomeCandidate,
    InstrumentCandidate,
    Context,
    Selection,
    UnitId,
    Time,
    Environment,
    Regime,
}
```

Schema construction assigns dense variable IDs and validates uniqueness once. Algorithmic code receives `VariableId` and immutable schema references. Name lookup is allowed at API boundaries and diagnostics, not inside traversal or numerical loops.

Do not:

- use strings as graph node or matrix-column keys in hot paths;
- clone variable names into edges, estimands, diagnostics, or posterior draws;
- represent role hints as heap-allocated sets when the closed role set fits in a bit mask;
- serialize internal dense IDs without the corresponding stable schema table;
- make schema lookup require a hash map after IDs have been resolved.

## 5. Data model [built]

Completion criteria for this chapter: masked temporal MCI (`discovery.data.masks`) and
pinned discovery baseline-style vector-variable CI grouping (`discovery.data.vector_variables`) are shipped
and gated. Remaining data-path polish (Arrow fixed-size-list ingest, full pinned discovery baseline
`mask_type` enum) is incremental and does not reopen this chapter.

### 5.1 Concrete dataset types

Do not use one runtime enum as the primary API. Use separate concrete types so algorithm applicability is enforced by traits.

```rust
pub struct TabularData<S = OwnedColumnarStorage> { /* schema, storage, masks, weights */ }
pub struct TimeSeriesData<S = OwnedColumnarStorage> { /* storage, time index, series metadata */ }
pub struct PanelData<S = OwnedColumnarStorage> { /* unit partitions + time indexes */ }
pub struct MultiEnvironmentData<D> { /* environment partitions */ }
pub struct EventData<S = OwnedColumnarStorage> { /* irregular event times and marks */ }
```

Algorithms declare accepted data:

```rust
pub trait DiscoveryAlgorithm<D> {
    type Output;

    fn discover(
        &mut self,
        data: &D,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError>;
}
```

### 5.2 Stable library-owned data views

Public core APIs do not expose Arrow Rust types. They expose library-owned borrowed views:

```rust
pub trait TableView {
    fn schema(&self) -> &CausalSchema;
    fn row_count(&self) -> usize;
    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError>;
}

pub enum ColumnView<'a> {
    Float64(Float64View<'a>),
    Int64(Int64View<'a>),
    Boolean(BooleanView<'a>),
    Categorical(CategoricalView<'a>),
    Timestamp(TimestampView<'a>),
    FixedVector(FixedVectorView<'a>),
}
```

Arrow is the preferred external and cross-language physical representation. `causal-data` provides Arrow-backed implementations and adapters through the Arrow C Data Interface. Internal algorithms operate on typed slices, bitmaps, strided views, and prepared buffers after one dispatch at the column boundary.

Required storage support:

- primitive numeric buffers;
- booleans represented as bitmap-backed views where possible;
- dictionary-encoded categories;
- fixed-size lists for vector variables;
- null bitmaps;
- optional independent analysis masks;
- optional weights;
- timestamp and duration buffers.

Materialization is explicit and recorded in execution diagnostics. The planner selects borrowed, copied-contiguous, transposed, or chunked execution based on algorithm requirements and memory budget.

### 5.3 Categorical representation and contrasts

Categoricals use dictionary-encoded `u32` codes and a separate immutable domain:

```rust
#[repr(transparent)]
pub struct CategoryCode(u32);

pub struct CategoryDomain {
    pub levels: Arc<[CategoryLevel]>,
    pub ordered: bool,
    pub reference: Option<CategoryCode>,
    pub unknown_policy: UnknownCategoryPolicy,
}

pub struct CategoricalView<'a> {
    pub codes: &'a [CategoryCode],
    pub validity: ValidityView<'a>,
    pub domain: &'a CategoryDomain,
}
```

Missing values are represented by validity, not by a synthetic category. Unknown levels fail by default. Mapping to an `Other` level is allowed only when that level is declared in the fitted schema.

Raw category IDs are never treated as numerical magnitudes. Contrast coding occurs during design compilation:

```rust
pub enum Contrast {
    Treatment { reference: CategoryCode },
    SumToZero,
    Helmert,
    Polynomial,
    FullRankIndicator,
    Custom(ContrastMatrix),
}
```

Frequentist convenience APIs default to treatment coding only when the reference is explicitly declared. Bayesian GLMs require an explicit contrast in the initial implementation because coefficient priors depend on coding. Fitted artifacts store the exact level order, contrast matrix, generated columns, and reference level.

### 5.4 Temporal identity and indexing

Stationary temporal edges use positive lag magnitudes:

```rust
pub struct TemporalEdge {
    pub source: VariableId,
    pub target: VariableId,

    /// source[t-lag] -> target[t]
    pub lag: u32,
    pub marks: EdgeMarks,
}
```

`lag == 0` is contemporaneous. A contemporaneous self-edge is invalid; a lagged self-edge is valid.

Stable unfolded node identity is separate from dense storage indexing:

```rust
pub struct TemporalNodeKey {
    pub variable: VariableId,
    pub offset: i32,
}
```

Finite unfoldings use time-major dense indexes:

```text
dense_index = time_slice_index * variable_count + variable_index
slice_index = offset + history
```

Dense indexes are process-local implementation details and are never serialized. Public APIs and artifacts use `TemporalNodeKey`. `TemporalIndexer` owns validation and conversion; index arithmetic is not duplicated in algorithms.

```rust
pub struct TimeIndex {
    pub ordering: TimeOrdering,
    pub regularity: SamplingRegularity,
    pub duplicate_policy: DuplicateTimePolicy,
    pub storage: TimeStorage,
}

pub enum SamplingRegularity {
    Regular { interval: Duration },
    Irregular,
}
```

Integer lags are never interpreted as durations for irregular data. Irregular algorithms must use duration windows, explicit alignment policies, or native event models.

### 5.5 Sample planning and construction

CI tests, regressions, and temporal estimators share a two-stage sample API.

```rust
pub struct SampleRequest<'a> {
    pub x: &'a [NodeRef],
    pub y: &'a [NodeRef],
    pub z: &'a [NodeRef],
    pub reference: ReferencePointPolicy,
    pub missing: MissingPolicy,
    pub mask: MaskPolicy,
    pub weights: WeightPolicy,
}

pub struct SamplePlan {
    pub columns: Vec<PreparedColumn>,
    pub row_selector: PreparedRowSelector,
    pub output_layout: SampleLayout,
    pub cache_key: SampleCacheKey,
}

pub struct SampleWorkspace {
    pub row_indexes: Vec<u32>,
    pub values: AlignedBuffer<f64>,
    pub validity_words: Vec<u64>,
    pub scratch: AlignedBuffer<f64>,
}

pub struct PreparedSample<'a> {
    pub matrix: MatrixRef<'a>,
    pub partitions: SamplePartitions,
    pub selected_rows: RowSelectionRef<'a>,
    pub effective_n: usize,
    pub dropped: DropSummary,
}
```

`SamplePlan` is reusable across repeated tests with the same variables and policies. `SampleWorkspace` is caller- or execution-context-owned scratch space. A materialized owned sample is available for persistence and slow-path extensions, but high-frequency discovery code consumes borrowed prepared samples.

The same request with the same data version and policy must produce the same row set. Cache keys include node order, masks, missingness, weights, reference policy, data version, and relevant time-index version.

### 5.6 Splits

Provide explicit split strategies:

- random IID split;
- grouped split;
- cluster split;
- blocked temporal split;
- rolling-origin split;
- discovery/estimation split with temporal gap;
- environment holdout;
- regime holdout.

The planner never applies a random row split to temporal or panel data unless explicitly requested.

### 5.7 Data-path performance requirements

Required:

- numeric columns remain contiguous or chunk-described; algorithms do not iterate through per-cell enums;
- validity and analysis masks are combined in word-sized batches;
- lagged row maps are computed once per data/time-index version and reused;
- design matrices are produced column-wise into aligned buffers;
- row selections use compact `u32` or `usize` indexes selected by data size;
- conversion diagnostics report copied bytes, borrowed bytes, transpositions, and category remapping;
- memory-budget checks occur before materializing dense lag or posterior matrices.

Do not:

- use a row-oriented `Vec<Vec<Value>>` as canonical storage;
- convert categorical codes to `f64` before a model requires contrasts;
- rebuild missingness masks for every CI test;
- transpose or repack a matrix inside an estimator iteration;
- allocate a fresh design matrix for every conditioning set when a workspace can be reused;
- retain pandas or Python object references in Rust analysis artifacts;
- expose Arrow crate types as stable public causal APIs;
- silently copy a large Arrow or NumPy input on every analysis stage.

## 6. Graph model [partial]

### 6.1 Node forms and dense indexes

```rust
pub enum NodeRef {
    Static(VariableId),
    Lagged { variable: VariableId, lag: Lag },
    Context { variable: VariableId, environment: Option<EnvironmentId> },
}

#[repr(transparent)]
pub struct DenseNodeId(u32);
```

Static graphs accept only `Static`. Temporal graphs accept `Lagged`; context-aware graphs may include `Context` nodes. Graph construction resolves stable node identities to compact dense IDs once. User-facing names never participate in traversal, hashing, or adjacency lookup.

### 6.2 Endpoint and edge semantics

```rust
pub enum Endpoint {
    Tail,
    Arrow,
    Circle,
    Conflict, // pinned discovery baseline `x`; CPDAG conflicts use Conflict–Conflict (`x-x`)
}

pub struct MarkedEdge<N> {
    pub a: N,
    pub b: N,
    pub at_a: Endpoint,
    pub at_b: Endpoint,
}
```

Named graph types constrain valid endpoint combinations:

- `Dag`: tail-arrow only, acyclic;
- `Admg`: directed and bidirected, no directed cycles;
- `Cpdag`: directed, undirected, and conflict (`x-x`) equivalence-class marks;
- `Pag`: tail, arrow, circle, and conflict endpoints under ancestral-graph constraints;
- temporal variants add stationarity and lag constraints.

Invalid endpoint combinations cannot be inserted through safe constructors.

### 6.3 Internal graph storage

Graph semantics are independent of storage. The default indexed storage is hybrid:

- compact adjacency vectors for sparse traversal;
- optional adjacency bitsets for repeated membership and set operations;
- edge-mark arrays indexed by dense edge IDs;
- stable node and edge iteration order;
- reusable traversal workspaces containing visited bitsets, queues, parent buffers, and path stacks.

```rust
pub struct GraphWorkspace {
    pub visited: BitSet,
    pub frontier: Vec<DenseNodeId>,
    pub scratch_nodes: Vec<DenseNodeId>,
    pub predecessor: Vec<Option<DenseNodeId>>,
}
```

The implementation may choose bitset-only storage for small dense graphs and adjacency-only storage for very large sparse graphs, but this selection occurs when compiling or constructing the graph. Algorithms do not branch on storage representation for every edge access.

### 6.4 Graph evidence is separate from graph semantics

```rust
pub struct GraphEvidence<G> {
    pub graph: G,
    pub edge_evidence: IndexedEdgeEvidence,
    pub source: EvidenceSource,
    pub constraints: ConstraintLedger<G::Node>,
    pub diagnostics: Vec<GraphDiagnostic>,
}

pub struct EdgeEvidence {
    pub statistic: Option<f64>,
    pub p_value: Option<f64>,
    pub adjusted_p_value: Option<f64>,
    pub interval: Option<Interval>,
    pub selection_frequency: Option<f64>,
    pub posterior_probability: Option<f64>,
    pub separating_sets: Vec<ConditioningSet>,
    pub provenance: Vec<EvidenceRecord>,
}
```

Evidence uses edge IDs or sorted compact edge keys. A `HashMap` keyed by high-level node objects is not used for every evidence access in discovery hot paths. An expert edge may have no p-value. A Bayesian-discovery edge may have posterior probability but no p-value.

### 6.5 Required graph algorithms

Implement in this order:

1. adjacency, insertion, removal, and validation;
2. directed ancestry and reachability;
3. topological order;
4. bounded path search and path witnesses;
5. d-separation for DAGs;
6. districts and m-separation for ADMGs;
7. CPDAG orientation utilities;
8. PAG definite-status paths and m-separation;
9. latent projection;
10. graph mutilation under intervention;
11. moralization and ancestral subgraphs;
12. Markov blankets
   (**status:** shipped — `Dag::markov_blanket` and Richardson-moral
   `Admg::markov_blanket` with inducing-path / district closure);
13. temporal stationarity expansion;
14. finite temporal unfolding;
15. graph completions and equivalence-class sampling.

All separation algorithms return a witness when possible:

```rust
pub enum SeparationResult<N> {
    Separated {
        conditioning: Vec<N>,
        certificate: SeparationCertificate<N>,
    },
    Connected {
        active_path: Vec<PathStep<N>>,
    },
}
```

Witness construction is optional in batch APIs when only a boolean result is required. The boolean fast path must not allocate path objects.

### 6.6 Temporal graph rules

A temporal directed edge must not point from the future into the past. Contemporaneous directed edges remain acyclic within a time slice when the graph type requires a DAG or CPDAG. Temporal templates are stationary declarations; finite unfolding creates concrete time-indexed nodes using the indexing policy in section 5.4.

A summary graph is a visualization artifact and must not be accepted by identification routines without an explicit expansion policy.

### 6.7 Graph performance requirements

Required:

- batched ancestry and separation APIs for repeated queries;
- intervention/mutilation without cloning full graphs on hot paths
  (**status:** shipped — `GraphOverlay` / `DagView` hide incoming or outgoing
  edges without cloning adjacency; `Dag::mutilate` materializes when an owned
  `Dag` is required; SCM sampling overlays remain in `causal-model`);
- lazy temporal unfolding or windowed unfolding when the algorithm does not require the complete graph;
- delta queues for orientation rules so a rule revisits only affected triples or paths;
- compact conditioning and separating sets using sorted dense IDs or bitsets;
- deterministic traversal order independent of hash randomization;
- benchmarks for sparse, dense, temporal, and PAG workloads.

Do not:

- enumerate all simple paths for identification or separation when a polynomial graph criterion exists;
- clone a graph for every intervention, bootstrap replicate, or graph completion;
- store node names on every edge;
- use recursive traversal where graph depth can exhaust the stack;
- scan every graph edge after each local orientation change;
- eagerly unfold every time point for stationary algorithms;
- expose raw dense indexes in serialized artifacts.

## 7. Assumptions and provenance [built]

Assumptions are typed records with source and scope.

```rust
pub struct AssumptionSet {
    pub entries: Vec<AssumptionRecord>,
}

pub struct AssumptionRecord {
    pub assumption: Assumption,
    pub source: AssumptionSource,
    pub scope: AssumptionScope,
    pub status: AssumptionStatus,
}

pub enum Assumption {
    CausalMarkov,
    Faithfulness,
    CausalSufficiency,
    Consistency,
    Positivity,
    NoInterference,
    Stationarity,
    PiecewiseStationarity,
    NoSelectionBias,
    ExclusionRestriction { instrument: VariableId },
    Monotonicity,
    ParametricRestriction(ParametricAssumption),
    PriorRestriction(PriorAssumption),
    Custom { id: Arc<str>, description: Arc<str> },
}
```

Every identification and estimation result references the exact assumptions used. Assumptions can be declared, tested, contradicted, or untestable. An untestable assumption is not marked as validated.

## 8. Causal query model [built]

The public query model uses typed variants rather than free-form strings.

**Built today** (`causal-core::CausalQuery`):

```rust
pub enum CausalQuery {
    AverageEffect(AverageEffectQuery),
    ConditionalEffect(ConditionalEffectQuery),
    Mediation(MediationQuery),
    Counterfactual(CounterfactualQuery),
    ChangeAttribution(ChangeAttributionQuery),
    AnomalyAttribution(AnomalyAttributionQuery),
    TemporalEffect(TemporalEffectQuery),
    MechanismChange(MechanismChangeQuery),
    UnitChange(UnitChangeQuery),
    Distribution(InterventionalDistributionQuery),
    PathSpecific(PathSpecificEffectQuery),
}
```

`Distribution` and `PathSpecific` are first-class variants (not overloads of
`ChangeAttribution` / `Mediation`). Types validate and register in state; GCM
wrappers sample interventional distributions (`sample_interventional_distribution`)
and path *contribution* (`attribute_path_specific` → `path_decompose`). Minimal
IO wire forms exist for those two queries. **Identify / estimate algorithms** for
interventional distributions (IDC) and path-restricted natural effects remain
planned — coordinate with deep identification (§10); do not fork a second
`AutoIdentifier`. Nonparametric path-specific natural effects are `pending` on
`context.mediation.nonparametric` (see `TODO.md`).

### 8.1 Interventions

```rust
pub enum Intervention {
    Set { variable: VariableId, value: Value },
    Shift { variable: VariableId, delta: Value },
    Stochastic { variable: VariableId, policy: StochasticPolicy },
    Soft { variable: VariableId, mechanism: MechanismOverride },
    Sequence(InterventionSequence),
}
```

Temporal sequences specify start, duration, cadence, and post-intervention behavior.

```rust
pub enum TemporalPolicy {
    Pulse { at: TimeOffset },
    Sustained { from: TimeOffset, until: TimeOffset },
    // Planned: Dynamic { rule: DynamicRuleId },
}
```

### 8.2 Target population

```rust
pub enum TargetPopulation {
    AllObserved,
    Treated,
    Untreated,
    Environment(EnvironmentId),
    // Planned: Predicate(PredicateExpr), CustomDistribution(DistributionRef),
}
```

Target population is part of the query identity and serialization.

## 9. Symbolic causal-functional IR [built]

`causal-expr` represents identified functionals independently of any estimator. The semantic form is an arena-backed directed acyclic expression graph rather than recursively boxed trees.

```rust
#[repr(transparent)]
pub struct ExprId(u32);

pub enum ExprNode {
    Distribution {
        variables: VarSetId,
        conditioned_on: VarSetId,
        intervention: InterventionSetId,
        domain: DomainRef,
    },
    Product(ExprListId),
    SumOut { variables: VarSetId, expr: ExprId },
    IntegralOut { variables: VarSetId, expr: ExprId },
    Ratio { numerator: ExprId, denominator: ExprId },
    Expectation { function: OutcomeExprId, distribution: ExprId },
    Contrast { left: ExprId, right: ExprId, op: ContrastOp },
}

pub struct CausalExprArena {
    pub nodes: Vec<ExprNode>,
    pub var_sets: InternedVarSets,
    pub interventions: InternedInterventionSets,
    pub lists: InternedExprLists,
}
```

Requirements (**status:** shipped):

- structural equality and stable hashing;
- alpha-normalized variable ordering;
- interned sorted variable and intervention sets;
- simplification without changing semantics
  (`CausalExprArena::simplify` — bottom-up rewrite + memoization);
- pretty printing and LaTeX rendering;
- evaluation against empirical, parametric, or posterior distribution providers
  (`DistributionProvider` trait; `EmpiricalTableProvider` /
  `PosteriorDrawProvider`; parametric backends implement the trait);
- derivation traces linking rewrites to identification rules;
- compiled evaluators for repeated empirical or posterior evaluation
  (`CompiledEvaluator`; discrete `SumOut` path — `IntegralOut` rejected).

The arena may hash-cons repeated subexpressions. Simplification uses iterative worklists and memoization. Derivation metadata is stored separately from canonical semantic nodes so adding an explanation does not duplicate the expression graph.

Do not:

- recursively clone expression trees during ID/IDC;
- store unsorted duplicate variable vectors in every distribution node;
- use pretty-printed strings as equality or cache keys;
- evaluate an expression by repeatedly resolving variable names;
- recursively evaluate deep expressions without a compiled topological order;
- allow simplification rules to depend on pointer identity or insertion order.

## 10. Identification subsystem [partial]

### 10.1 Result model

```rust
pub struct IdentificationResult {
    pub status: IdentificationStatus,
    pub query: CausalQuery,
    pub estimands: Vec<IdentifiedEstimand>,
    pub derivation: DerivationTrace,
    pub required_assumptions: AssumptionSet,
    pub diagnostics: Vec<IdentificationDiagnostic>,
    pub performance: IdentificationPerformanceRecord,
}

pub enum IdentificationStatus {
    NonparametricallyIdentified,
    IdentifiedUnderParametricRestrictions,
    IdentifiedUnderPriorRestrictions,
    PartiallyIdentified,
    GraphDependent,
    NotIdentified,
}
```

A graph ensemble result is not reduced to a single status:

```rust
pub struct IdentificationEnvelope<G> {
    pub invariant: Option<IdentifiedEstimand>,
    pub cases: Vec<GraphIdentificationCase<G>>,
    pub identified_weight: ProbabilityMass,
    pub unidentified_weight: ProbabilityMass,
    pub critical_graph_features: Vec<GraphFeature>,
}
```

### 10.2 Algorithms

Implement:

- minimal, maximal, and all backdoor adjustment sets;
- efficient adjustment sets where defined;
- generalized adjustment for CPDAG/PAG classes;
- front-door identification;
- instrumental-variable candidate validation;
- mediation and natural-effect identification;
- ID algorithm for semi-Markovian models;
- IDC for conditional interventional distributions;
- hedge certificates for non-identifiability;
- identification under selection and transport extensions as later modules;
- temporal identification by explicit unfolding or stationary templates.

### 10.3 Identifier interface

```rust
pub trait Identifier<G> {
    fn prepare(
        &self,
        graph: &G,
        assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError>;

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError>;
}
```

`AutoIdentifier` tries applicable methods but returns all valid estimands and selection rationale. It does not silently choose an estimator.

### 10.4 Adjustment-set search

Adjustment search supports:

- exact enumeration for small graphs;
- minimal-set enumeration;
- cost-weighted selection;
- forbidden variables;
- measurement-cost metadata;
- temporal history restrictions;
- positivity-aware ranking after a data check.

Graph search and data-based ranking are separate stages. Enumeration APIs support streaming callbacks or iterators so all adjustment sets need not be retained.

### 10.5 Identification performance requirements

Required:

- ancestral subgraphs, descendants, districts, and mutilated-graph overlays are cached within a prepared graph;
- node sets use bitsets or sorted dense IDs;
- recursive ID/IDC is implemented with explicit memoization over canonical subproblems;
- expression construction writes into one arena and reuses interned sets;
- graph-ensemble identification groups graphs by relevant subgraph or estimand when possible;
- adjustment enumeration exposes size and result-count limits.

Do not:

- clone the graph at each recursive identification step;
- construct materialized power sets of candidate adjusters;
- enumerate every compatible graph when equivalence-class reasoning can answer the query directly;
- retain all adjustment sets by default for graphs with combinatorial output;
- recompute ancestry, descendants, or districts for every candidate set;
- select a faster heuristic while returning an exact-identification result type.

## 11. Statistical kernel layer [partial]

### 11.1 Linear algebra backend and boundary

Use `faer` as the default dense linear-algebra backend. Backend types are not exposed in public causal APIs. The abstraction is operation-level:

```rust
pub trait DenseLinearAlgebra: Send + Sync {
    fn least_squares(
        &self,
        x: MatrixRef<'_>,
        y: MatrixRef<'_>,
        options: LeastSquaresOptions,
        workspace: &mut LinalgWorkspace,
    ) -> Result<LeastSquaresResult, LinalgError>;

    fn cholesky(
        &self,
        matrix: SymmetricMatrixRef<'_>,
        workspace: &mut LinalgWorkspace,
    ) -> Result<CholeskyFactor, LinalgError>;

    fn symmetric_eigen(
        &self,
        matrix: SymmetricMatrixRef<'_>,
        workspace: &mut LinalgWorkspace,
    ) -> Result<EigenDecomposition, LinalgError>;

    fn svd(
        &self,
        matrix: MatrixRef<'_>,
        options: SvdOptions,
        workspace: &mut LinalgWorkspace,
    ) -> Result<SvdDecomposition, LinalgError>;
}
```

`MatrixRef` and related borrowed views are library-owned and strided. Backend selection occurs once when compiling an analysis plan or constructing an estimator, not inside numerical loops.

Default feature set:

```text
faer dense backend
portable scalar kernels
runtime-selected SIMD kernels where supported
```

Optional builds may provide BLAS/LAPACK, but default wheels must not require a system BLAS. A small reference backend exists for correctness tests on tiny matrices.

Required operations:

- QR with pivoting;
- SVD or rank-revealing decomposition;
- Cholesky and LDLT;
- symmetric eigendecomposition;
- triangular solves;
- weighted least squares;
- covariance and cross-products;
- batched small-matrix operations;
- stable log determinants and condition estimates.

### 11.2 Regression models

Implement:

- OLS;
- weighted least squares;
- logistic regression;
- multinomial logistic regression;
- Poisson and negative-binomial GLMs;
- robust M-estimation;
- 2SLS;
- ridge and lasso utilities where required by algorithms;
  Lasso point estimates ship without analytic standard errors — pair/resampling
  bootstrap only (post-selection classical sandwich is invalid; debiased Lasso is a
  separate estimator, not an SE bolt-on). Ridge retains residual sandwich SEs;
- generalized additive interfaces (cubic B-spline expansion, backfitting
  `fit_gam` / `predict_gam`, `DesignColumnMap` smooth provenance — Gaussian
  identity only; no analytic SEs).

Model fitting is split into design compilation and iterative fitting:

```rust
pub struct CompiledDesign {
    pub matrix: Arc<[f64]>, // column-major dense storage
    pub columns: DesignColumnMap,
    pub contrasts: Vec<RecordedContrast>,
    pub row_selection: RowSelection,
    pub standardization: StandardizationRecord,
}
```

Dense storage stays on `CompiledDesign.matrix` as `Arc<[f64]>` — there is no separate matrix wrapper type. `DesignColumnMap` carries per-column roles plus contrast / standardization provenance links. Repeated fits against the same design reuse compiled contrasts, row selection, standardization, and scratch buffers. Every fit returns rank, condition diagnostics, convergence state, row-selection provenance, iterations, backend, and allocation/copy diagnostics.

### 11.3 Covariance estimators

Implement:

- homoskedastic;
- HC0-HC3;
- cluster-robust;
- multiway cluster-robust;
- HAC/Newey-West;
- panel cluster plus temporal HAC where supported.

Covariance implementations consume retained residuals, scores, and sufficient statistics when available. They do not refit a model unless the method mathematically requires it.

### 11.4 Resampling

Provide a shared resampling engine:

```rust
pub enum ResamplingPlan {
    IidBootstrap,
    BayesianBootstrap,
    ClusterBootstrap { cluster: VariableId },
    MovingBlock { length: usize },
    StationaryBlock { expected_length: f64 },
    CircularBlock { length: usize },
    Permutation(PermutationScheme),
}
```

The engine produces index or weight plans in batches and reuses estimator workspaces. Nested parallelism is controlled by one `ExecutionContext`; algorithms request tasks from it rather than constructing independent pools.

### 11.5 Multiple testing

Implement Bonferroni, Holm, Benjamini-Hochberg, and Benjamini-Yekutieli. Preserve raw and adjusted p-values. Large correction batches use stable indexed sorting with reusable buffers.

### 11.6 Numerical and performance requirements

Required:

- QR or rank-revealing decompositions by default for least squares;
- explicit small-matrix paths for common CI conditioning sizes;
- aligned reusable buffers for residuals, weights, gradients, and Hessians;
- batched reductions and covariance accumulation through `causal-kernels`;
- convergence criteria stated in scale-aware terms;
- condition diagnostics before selecting a faster but less stable path;
- benchmark coverage for tall-skinny, small repeated, rank-deficient, and weighted problems.

Do not:

- form `XᵀX` and invert it as the default OLS implementation;
- allocate residual, gradient, or Hessian vectors in every optimizer iteration;
- dispatch through a trait object for each row or coefficient;
- recreate category contrasts or standardization on every bootstrap fit;
- call a large external BLAS kernel for tiny matrices without benchmarking the overhead;
- use unchecked lower precision in a path that changes conformance behavior;
- hide rank deficiency by returning unstable coefficients without diagnostics;
- treat numerical fallback as invisible: fallback reason and backend are recorded.

## 12. Conditional-independence framework [built]

```rust
pub trait ConditionalIndependenceTest<D> {
    fn prepare(
        &self,
        data: &D,
        plan: &CiPreparationPlan,
        ctx: &ExecutionContext,
    ) -> Result<PreparedCiTest, CiError>;

    fn test(
        &self,
        prepared: &mut PreparedCiTest,
        query: &CiQuery,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiTestResult, CiError>;

    fn test_batch(
        &self,
        prepared: &mut PreparedCiTest,
        queries: &[CiQueryRef<'_>],
        workspace: &mut CiWorkspace,
        output: &mut [CiTestResult],
        ctx: &ExecutionContext,
    ) -> Result<(), CiError>;
}
```

```rust
pub struct CiQuery {
    pub x: Vec<NodeRef>,
    pub y: Vec<NodeRef>,
    pub z: Vec<NodeRef>,
    pub significance: SignificanceMethod,
    pub confidence: Option<ConfidenceMethod>,
}

pub struct CiTestResult {
    pub statistic: f64,
    pub p_value: Option<f64>,
    pub interval: Option<Interval>,
    pub effective_n: usize,
    pub status: CiStatus,
    pub diagnostics: Vec<CiDiagnostic>,
}
```

The framework owns sample planning, sample-index caching, shuffle/bootstrap wrappers, constant-variable handling, and reusable workspaces. Individual tests implement a dependence statistic and, where available, an analytic reference distribution.

Required pinned discovery baseline-parity tests:

- partial correlation;
- robust/nonparanormal partial correlation;
- weighted partial correlation;
- multivariate partial correlation;
- Gaussian-process regression plus distance correlation;
- k-nearest-neighbor conditional mutual information;
- mixed-data kNN conditional mutual information;
- symbolic conditional mutual information;
- G-squared;
- mixed regression CI;
- pairwise multivariate wrapper;
- graph oracle.

Bayesian additions:

- Bayes-factor conditional independence for supported conjugate models;
- posterior dependence probability;
- posterior predictive conditional-independence diagnostics;
- CI tests as screening/proposal mechanisms for posterior graph search.

### 12.1 CI performance requirements

Required:

- partial-correlation batches reuse a common correlation/cross-product workspace;
- residuals are cached only when their full semantic key is known;
- permutation and block-shuffle indexes are generated in batches;
- kNN indexes are reused across compatible queries;
- contingency tables use compact integer accumulation and clear only touched cells;
- constant and duplicate columns are detected during preparation;
- batch APIs preserve deterministic query order in outputs.

Do not:

- allocate `Vec` instances for `X`, `Y`, `Z`, residuals, and row indexes for every test in PCMCI;
- rebuild lag alignment for every candidate edge;
- sort the same conditioning set repeatedly;
- call Python once per CI query;
- use a global result cache without memory bounds;
- parallelize both CI batches and their internal permutation replicates without an assigned nested budget.

## 13. Discovery subsystem [partial]

### 13.1 Common output

```rust
pub struct DiscoveryResult<G> {
    pub evidence: GraphEvidence<G>,
    pub algorithm: AlgorithmRecord,
    pub assumptions: AssumptionSet,
    pub iterations: Vec<DiscoveryIteration>,
    pub diagnostics: Vec<DiscoveryDiagnostic>,
    pub performance: DiscoveryPerformanceRecord,
}
```

### 13.2 Constraint model

```rust
pub struct DiscoveryConstraints<N> {
    pub required: BTreeSet<EdgeConstraint<N>>,
    pub forbidden: BTreeSet<EdgeConstraint<N>>,
    pub tiers: Vec<Vec<N>>,
    pub max_parents: Option<usize>,
    pub temporal: Option<TemporalConstraints>,
}
```

Constraints are validated and compiled to dense indexed masks before execution. Conflicting required and forbidden edges are an error.

### 13.3 Static discovery

Parity and extension target:

- PC;
- FCI and RFCI;
- GES;
- LiNGAM variants where assumptions apply;
- score-based DAG search;
- NOTEARS-style continuous optimization as an optional extension;
- Bayesian DAG posterior search.

Implement PC first because it shares CI machinery and orientation code with temporal discovery.

### 13.4 PCMCI family

Implement a shared `PcmciEngine` containing:

- candidate generation;
- compiled candidate and forbidden-edge masks;
- PC-style parent selection;
- MCI testing;
- lag range handling;
- link assumptions;
- conditioning-set size limits;
- alpha selection;
- statistic and p-value matrices;
- FDR correction;
- iteration diagnostics;
- deterministic tie handling;
- reusable target-local workspaces.

Public algorithms:

- `Pcmci`;
- `PcmciPlus`;
- `Lpcmci`;
- `JpcmciPlus`;
- `Rpcmci` (caller-supplied `RegimeAssignment` per regime; unsupervised regime
  search is out of scope).

Do not implement them as option flags on one giant function. Each owns its assumptions, graph output type, and orientation rules.

### 13.5 PCMCI acceptance behavior

`Pcmci` returns directed lagged edges and supported contemporaneous representation under its configured lag minimum. `PcmciPlus` returns a temporal CPDAG. `Lpcmci` returns a temporal PAG. `JpcmciPlus` includes context nodes and multi-dataset constraints. `Rpcmci` returns regime assignments plus one graph per regime.

### 13.6 Orientation engine

Implement orientation rules as named, individually testable transformations:

```rust
pub trait OrientationRule<G> {
    fn apply(
        &self,
        graph: &mut G,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError>;
}
```

Track:

- edges changed;
- premises used;
- separating sets used;
- conflicts;
- affected local structures to enqueue;
- whether the rule reached a fixed point.

For LPCMCI, discriminating paths and rule scheduling are explicit modules; they are not embedded in a single procedural loop.

### 13.7 Bayesian graph discovery

Bayesian discovery is additive to parity, not a replacement.

```rust
pub trait GraphPosteriorEngine<D, G> {
    fn infer_graphs(
        &self,
        data: &D,
        prior: &GraphPrior<G>,
        mechanisms: &MechanismFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior<G>, DiscoveryError>;
}
```

Initial supported methods:

- exact enumeration for very small DAGs;
- order MCMC or structure MCMC for discrete/small continuous models;
- candidate-edge posterior updates after CI screening;
- dynamic Bayesian network posterior search for bounded lag;
- model averaging over externally supplied graph sets.

The graph posterior stores normalized weights, edge marginals, orientation marginals, effective sample size, chain diagnostics, and rejected invalid graphs in columnar or indexed arrays rather than one boxed object per sample.

### 13.8 Discovery performance requirements

Required:

- target variables are primary parallel work units for PCMCI where semantics permit;
- candidate and adjacency membership use bitsets or compact indexed sets;
- conditioning-set generation writes combinations into reusable buffers;
- MCI and PC queries are submitted through CI batch APIs;
- orientation uses delta queues rather than global rescans;
- iteration logs may be disabled or summarized without affecting algorithm state;
- graph posterior proposals use incremental score updates when mathematically valid;
- memory estimates account for p-value/statistic matrices, candidate sets, caches, and bootstrap replicas.

Do not:

- allocate or hash high-level node tuples in the innermost candidate loop;
- copy the dataset for each target variable;
- materialize every possible conditioning set in advance;
- schedule one parallel task per trivial CI test;
- keep every intermediate graph when only accepted samples and summaries are requested;
- change conditioning order, tie handling, or edge orientation solely to improve speed without recording a semantic deviation;
- claim parity based only on final graph equality when statistics or p-values differ outside tolerance.

## 14. Estimation subsystem [partial]

### 14.1 Estimator contract

```rust
pub trait Estimator<D> {
    type Fit;

    fn prepare(
        &self,
        data: &D,
        estimand: &IdentifiedEstimand,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEstimationProblem, EstimationError>;

    fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, EstimationError>;
}

pub trait FittedEstimator<Q> {
    fn estimate_batch(
        &self,
        queries: &[Q],
        output: &mut [EstimateArtifact],
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError>;
}
```

A fitted estimator is bound to one estimand, data schema, category encoding, row-selection policy, and model configuration. Applying it to an incompatible query is an error.

### 14.2 Frequentist parity estimators

Implement:

- linear regression adjustment;
- generalized linear adjustment;
- distance matching;
- propensity-score matching;
- propensity stratification;
- propensity weighting with ATT/ATE/ATC variants;
- doubly robust/AIPW;
- IV/Wald and 2SLS;
- regression discontinuity;
- two-stage regression for front-door and mediation;
- conditional effects and effect modifiers;
- bootstrap and analytic uncertainty where valid.

EconML adapters and estimator classes are excluded. Native heterogeneous-effect methods may be added independently.

### 14.3 Positivity and overlap

Every treatment-effect estimator must either run an overlap check or require an explicit override. Results include:

- propensity range;
- effective sample size;
- extreme-weight summary;
- target population support;
- excluded regions;
- clipping/trimming policy;
- sensitivity to clipping thresholds.

Overlap calculations reuse fitted propensity values and sorted/indexed buffers rather than repeatedly recomputing them.

### 14.4 Bayesian estimation

Bayesian estimation evaluates identified functionals over posterior draws.

```rust
pub trait PosteriorFunctionalEvaluator {
    fn compile(
        &self,
        functional: &ProbExpr,
        posterior_schema: &PosteriorSchema,
    ) -> Result<CompiledPosteriorFunctional, EstimationError>;

    fn evaluate_batch(
        &self,
        compiled: &CompiledPosteriorFunctional,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError>;
}
```

Supported initial mechanism models:

- conjugate Gaussian linear;
- Bayesian logistic and Poisson regression;
- hierarchical linear/GLM mechanisms after the base backend is stable;
- Bayesian vector autoregression;
- linear Gaussian state-space models;
- Gaussian-process mechanisms as an optional feature.

Output:

```rust
pub struct CausalPosterior {
    pub draws: PosteriorDraws<EffectValue>,
    pub summaries: PosteriorSummary,
    pub identification: IdentificationStatus,
    pub prior_sensitivity: Option<PriorSensitivitySummary>,
    pub diagnostics: InferenceDiagnostics,
}
```

If some posterior graph mass is unidentified, the result preserves that mass. It does not renormalize identified graphs unless explicitly requested.

### 14.5 Initial Bayesian GLM backend

The first Bayesian GLM backend is a native Laplace approximation.

Supported initial likelihood/link combinations:

- Gaussian/identity;
- Bernoulli/logit;
- Bernoulli/probit;
- Poisson/log;
- Gaussian coefficient priors;
- offsets and observation weights;
- dense fixed effects.

Implementation:

1. compile the design and prior precision;
2. find the MAP using damped Newton or trust-region Newton;
3. reuse gradient, Hessian, and factorization workspaces;
4. factor the negative Hessian with Cholesky when positive definite;
5. use a structured LDLT fallback or return a diagnostic failure;
6. expose a multivariate-normal approximation and batched draws;
7. return convergence, separation, conditioning, and approximation diagnostics.

External Stan/PyMC-style adapters may be added after the model IR stabilizes as
optional validation / advanced-inference routes. They are **not** required for
completion or parity; the native Laplace backend is the canonical Bayesian GLM
implementation and the only Bayesian GLM surface tracked in `parity/bayesian.toml`.

### 14.6 Estimation performance requirements

Required:

- design compilation is separate from fitting;
- matching and nearest-neighbor indexes are reused across compatible estimands;
- propensity values and weights are retained for diagnostics and resampling;
- bootstrap fits reuse design and workspace storage;
- posterior functional evaluation is draw-batched and columnar;
- temporal estimators reuse lag plans and sufficient statistics;
- fitted objects expose estimated retained-memory cost.

Do not:

- rebuild the design matrix for each target population query;
- create one Rust object per posterior draw;
- allocate a model object for every bootstrap replicate;
- call a Python model once per row or posterior draw in a fast-path estimator;
- compute diagnostics by silently refitting the estimator;
- report a narrow Laplace posterior without convergence and curvature diagnostics;
- use automatic estimator selection without recording the selected physical and statistical plan.

## 15. Structural causal models [partial]

### 15.1 Model types

Implement:

- probabilistic causal models;
- structural causal models;
- invertible structural causal models;
- static and temporal models;
- fixed-graph and graph-posterior model collections.

A model has a semantic graph and a compiled execution plan:

```rust
pub struct CompiledCausalModel {
    pub node_order: Arc<[DenseNodeId]>,
    pub parent_gathers: Arc<[ParentGatherPlan]>,
    pub mechanisms: CompiledMechanismStore,
    pub output_layout: ModelOutputLayout,
}
```

### 15.2 Mechanism traits

```rust
pub trait ProbabilisticMechanism {
    fn log_prob_batch(
        &self,
        values: ValueBatch<'_>,
        parents: ParentBatch<'_>,
        output: &mut [f64],
        workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError>;

    fn sample_batch(
        &self,
        parents: ParentBatch<'_>,
        rng: &mut dyn RngCore,
        output: &mut ValueBatchMut<'_>,
        workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError>;
}

pub trait StructuralMechanism {
    fn sample_noise_batch(
        &self,
        count: usize,
        rng: &mut dyn RngCore,
        output: &mut NoiseBatchMut<'_>,
    ) -> Result<(), ModelError>;

    fn evaluate_batch(
        &self,
        parents: ParentBatch<'_>,
        noise: NoiseBatch<'_>,
        output: &mut ValueBatchMut<'_>,
        workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError>;
}

pub trait InvertibleMechanism: StructuralMechanism {
    fn infer_noise_batch(
        &self,
        value: ValueBatch<'_>,
        parents: ParentBatch<'_>,
        output: &mut NoiseBatchMut<'_>,
        workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError>;
}
```

Built-in mechanisms compile to concrete enum variants or monomorphized kernels. Dynamic user mechanisms remain available as an explicit slow path.

### 15.3 Automatic mechanism assignment

Provide a registry of candidate mechanism families and an explicit scoring/selection process. Assignment returns candidates, validation scores, selected model, fit cost, and evaluation cost. No silent fallback to a default family.

### 15.4 Sampling

Implement:

- observational ancestral sampling;
- hard interventions;
- soft mechanism interventions;
- simultaneous interventions;
- stochastic interventions;
- temporal intervention sequences;
- posterior predictive interventional sampling;
- conditional interventional sampling where supported.

Sampling compiles interventions to overlays on the immutable execution plan. The model graph and mechanism registry are not cloned for each intervention or draw.

### 15.5 Model evaluation and falsification

Implement:

- mechanism predictive checks;
- residual independence tests;
- graph-based local Markov tests;
- permutation baselines;
- held-out likelihood or predictive scores;
- posterior predictive checks;
- model comparison reports.

### 15.6 Model performance requirements

Required:

- topological order and parent-gather plans are compiled once;
- simulation is batch-oriented and columnar;
- parent values are gathered into reusable aligned buffers or zero-copy strided views;
- posterior parameter draws are processed in blocks sized to memory budget;
- temporal simulation uses ring buffers when full trajectory retention is unnecessary;
- built-in mechanisms provide vectorized batch kernels.

Do not:

- traverse the semantic graph recursively for every generated observation;
- call a trait object once per scalar value;
- clone the whole SCM for every intervention;
- retain all posterior trajectories when only summaries are requested;
- make the dynamic Python mechanism path appear equivalent in performance to compiled Rust mechanisms.

## 16. Counterfactual subsystem [partial]

Counterfactual evaluation follows abduction-action-prediction.

```rust
pub struct CounterfactualEngine<M> {
    pub model: M,
    pub compiled: CompiledCounterfactualPlan,
}

pub struct CounterfactualWorld {
    pub factual_observation: Observation,
    pub inferred_exogenous_state: ExogenousPosterior,
    pub interventions: InterventionSet,
}
```

Required capabilities:

- point and distributional counterfactuals;
- individual treatment effects;
- counterfactual trajectories;
- shared-noise semantics across worlds;
- posterior uncertainty over exogenous states;
- missing factual variables;
- nested counterfactual expressions where identifiable under model assumptions.

The output records whether counterfactuals rely on invertibility, posterior noise inference, or assumed noise distributions.

### 16.1 Counterfactual execution requirements

Required:

- factual abduction is performed once and shared across requested worlds;
- intervention worlds use immutable model structure plus compact overlays;
- shared-noise coupling is represented explicitly and evaluated in batches;
- trajectory queries may stream summaries without retaining every world/time/draw value;
- repeated queries compile their causal expression and gather plan once.

Do not:

- clone the model or graph per world;
- independently resample exogenous noise for worlds that require shared-noise semantics;
- allocate nested `Vec<Vec<Vec<_>>>` structures for draw-by-world-by-time results;
- cross the Python boundary per world or per posterior draw.

## 17. Attribution and inverse explanation [partial]

### 17.1 Query types

- anomaly attribution for one or more samples;
- distribution-change attribution between populations or periods;
- mechanism-change attribution;
- unit-change attribution;
- feature relevance under interventions;
- direct arrow strength and intrinsic influence;
- path-specific contribution;
- root-cause ranking.

### 17.2 Change decomposition

```rust
pub struct ChangeAttributionQuery {
    pub outcome: VariableId,
    pub baseline: PopulationSelector,
    pub comparison: PopulationSelector,
    pub components: AttributionComponents,
    pub allocation: AllocationMethod,
}

pub enum AttributionComponents {
    Inputs,
    Mechanisms,
    Structure,
    InputsAndMechanisms,
    All,
}

pub enum AllocationMethod {
    Sequential { order: Vec<ComponentId> },
    Shapley { approximation: ShapleyConfig },
    PathBased,
}
```

Outputs include posterior or sampling distributions over contributions, interaction terms, path breakdowns, unidentified components, graph sensitivity, compute budget, approximation error estimate, and cache statistics.

### 17.3 Mechanism-change detection

Compare conditional mechanisms between environments/regimes. **Shipped detectors:**

- likelihood-ratio or mean-difference proxies;
- classifier two-sample tests;
- posterior parameter or predictive differences where a posterior is available;
- kernel two-sample tests (MMD² with RBF on residuals);
- change-point models for temporal mechanisms (known-split Gaussian LR; max-CUSUM scan).

Detection and causal attribution are separate. A changed mechanism is not necessarily responsible for the target outcome change.

### 17.4 Attribution performance requirements

Required:

- coalition and path evaluations use a semantic cache keyed by intervention/substitution state;
- Shapley sampling is batched and parallelized at coalition or permutation granularity;
- exact Shapley is rejected above a configured component limit unless explicitly overridden;
- path algorithms use dynamic programming or graph decomposition where valid;
- posterior and graph-ensemble attribution processes draws in bounded blocks;
- contribution summaries may be streamed.

Do not:

- promise exact combinatorial attribution for unconstrained component counts;
- recompute unchanged upstream mechanisms for every coalition when compiled dependency information can prune them;
- store every coalition output when only aggregate contribution summaries are requested;
- hide approximation budget or Monte Carlo error.

## 18. Validation and sensitivity [partial]

### 18.1 Common interface

```rust
pub trait Validator<A> {
    type Prepared;
    type Report;

    fn prepare(
        &self,
        artifact: &A,
        ctx: &ExecutionContext,
    ) -> Result<Self::Prepared, ValidationError>;

    fn validate(
        &self,
        prepared: &mut Self::Prepared,
        ctx: &ExecutionContext,
    ) -> Result<Self::Report, ValidationError>;
}
```

### 18.2 Effect refuters

Implement pinned estimate baseline-parity checks:

- placebo treatment;
- random common cause;
- data-subset refutation;
- bootstrap refutation;
- dummy outcome;
- add-unobserved-common-cause;
- graph refutation;
- overlap assessment;
- overlap-rule diagnostics;
- E-value analysis;
- linear sensitivity;
- partial-linear sensitivity;
- nonparametric sensitivity;
- Reisz-representer diagnostics where applicable.

A refuter result contains the transformed problem, repeated estimates, comparison statistic, failure conditions, and whether the check is informative for the estimator used.

### 18.3 Discovery validation

Implement:

- stability selection over resamples;
- lag-window sensitivity;
- alpha-threshold sensitivity;
- CI-test sensitivity;
- orientation stability;
- regime stability;
- environment holdout;
- synthetic-null calibration;
- false-positive checks using permuted or phase-randomized data.

### 18.4 Bayesian workflow diagnostics

Implement:

- prior predictive simulation;
- posterior predictive simulation;
- chain convergence diagnostics;
- effective sample size;
- divergence counts;
- simulation-based calibration;
- prior sensitivity grids;
- likelihood-family comparison;
- posterior calibration on synthetic SCMs.

These diagnostics do not replace causal identification or refutation.

### 18.5 Validation suite

```rust
let report = ValidationSuite::new()
    .with(PlaceboTreatment::default())
    .with(GraphStability::block_bootstrap(200))
    .with(PriorSensitivity::standard_grid())
    .run(&analysis_result, &ctx)?;
```

The suite executes only compatible validators and returns explicit `NotApplicable` entries for requested incompatible checks.

### 18.6 Validation performance requirements

Required:

- resampling plans, row selections, compiled designs, and estimator workspaces are reused;
- validation suites share one execution budget and one result cache;
- simulation diagnostics support streaming summaries;
- long calibration suites are separated from unit CI but remain automated scheduled gates;
- every validator reports replicate count, retained memory, elapsed work units, and early-stopping behavior.

Do not:

- create a new executor or thread pool per validator;
- refit unchanged nuisance models unless the refuter definition requires it;
- keep all replicate artifacts by default;
- parallelize validator-level and replicate-level work simultaneously without a compiled schedule;
- reduce replicate counts to satisfy a benchmark without revisiting statistical calibration.

## 19. Experiment, measurement, and decision primitives [partial]

This crate provides computation only.

### 19.1 Design objectives

```rust
pub enum DesignObjective {
    ReduceGraphEntropy,
    IncreaseIdentificationProbability { query: QueryId },
    ReduceEffectPosteriorWidth { query: QueryId },
    ReduceDecisionRegret { decision: DecisionProblemId },
    DistinguishModels { models: Vec<ModelId> },
}
```

### 19.2 Candidate plans

```rust
pub enum CandidateDesign {
    Measure(MeasurementPlan),
    Intervene(ExperimentPlan),
    ObserveEnvironment(EnvironmentPlan),
    IncreaseSamplingRate(SamplingPlan),
}
```

Ranking evaluates expected utility, information gain, cost, constraints, and model uncertainty. Returned plans include Monte Carlo error, assumptions, compute budget, and any approximation method.

**Shipped production models (not stubs):** EIG uses a discrete graph-posterior
soft-evidence model; effect-width reduction uses linear-Gaussian / OLS Gram
scaling. These are the algorithms for the current design surface.

### 19.3 Decision analysis

```rust
pub struct DecisionProblem<A, O> {
    pub actions: Vec<A>,
    pub utility: Arc<dyn Utility<A, O>>,
    pub constraints: Vec<Arc<dyn DecisionConstraint<A, O>>>,
}
```

The library returns expected utility, posterior regret, chance-constraint probabilities, and sensitivity to priors/graphs. It does not dispatch the selected action or own approval workflows.

### 19.4 Design computation requirements

Required:

- candidate designs compile to batched simulation/evaluation plans;
- common posterior and graph draws are reused across candidates where unbiased common-random-number comparisons are valid;
- utility and constraint evaluations support batch APIs;
- adaptive Monte Carlo may stop when ranking uncertainty is below a declared threshold;
- expected information calculations stream sufficient summaries when possible;
- result ordering includes uncertainty in the candidate rank.

Do not:

- call a dynamic utility function once per scalar outcome when a batch utility is available;
- allocate a separate posterior copy per candidate design;
- report exact rankings when Monte Carlo uncertainty overlaps materially;
- silently drop candidates because their compute cost is high;
- execute external actions or own organization-specific approval state.

## 20. Incremental causal state [partial]

`causal-state` supports applications that repeatedly update analyses.

```rust
pub struct CausalState {
    pub version: StateVersion,
    pub data_catalog: DataCatalog,
    pub graph_evidence: GraphEvidenceStore,
    pub assumptions: AssumptionSet,
    pub models: ModelStore,
    pub queries: QueryStore,
    pub cached_results: ResultStore,
    pub invalidations: InvalidationLog,
    pub cache_budget: CacheBudget,
}
```

Supported events:

```rust
pub enum StateEvent {
    AppendData(DataBatchRef),
    ReplaceData(DataVersion),
    AddGraphEvidence(GraphEvidenceRecord),
    AddConstraint(GraphConstraintRecord),
    RemoveConstraint(ConstraintId),
    UpdateAssumption(AssumptionRecord),
    RegisterQuery(CausalQuery),
    RecordIntervention(InterventionRecord),
}
```

Applying an event computes invalidation dependencies. It does not automatically rerun expensive analyses unless the caller requests recomputation.

Incremental algorithms may maintain:

- sufficient statistics for linear models (**shipped**);
- streaming covariance matrices (**shipped**);
- cached lagged sample indexes (**shipped**);
- particle-filter state (**shipped** — bootstrap/SIR LGSSM in `causal-state`);
- graph-score caches (**shipped** — Gaussian BIC local scores with parent-set deltas);
- rolling mechanism diagnostics.

Caches are versioned, bounded, and reconstructible. Eviction affects performance only, never semantics. Serialized state contains no process handles, thread pools, callbacks, borrowed buffers, or Python objects.

Do not retain the full historical dataset merely because an incremental statistic can be maintained. Each state component declares whether it requires raw history, a bounded window, or sufficient statistics.

## 21. Common planner and facade [partial]

### 21.1 User workflow

```rust
use causal::prelude::*;

let result = CausalAnalysis::builder()
    .data(tabular)
    .graph(dag) // or .discover_pcmci(...) for temporal discovery
    .query(AverageEffectQuery::binary_ate(t, y))
    .identifier(IdentifierId::BackdoorAdjustment)
    .estimator(EstimatorId::LinearAdjustmentAte)
    .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
    .build()?
    .run(&ctx)?;
```

The same builder applies to tabular and temporal data (`.series(...)` + temporal query / discovery).

### 21.2 Logical and physical planning

Compilation produces two inspectable plans:

```rust
let logical = analysis.compile_logical()?;
logical.validate()?;
let physical = logical.compile_physical(&ctx.capabilities())?;
let result = physical.execute(&ctx)?;
```

`LogicalAnalysisPlan` contains statistical and causal semantics:

- data classification;
- preprocessing semantics;
- discovery algorithm and constraints;
- graph review requirement;
- identifier selection;
- estimator/inference method;
- validation suite;
- expected artifacts.

`PhysicalExecutionPlan` contains execution choices:

- borrowed versus materialized columns;
- dense index maps;
- matrix orientation and layout;
- selected scalar/SIMD/backend kernels;
- batch sizes;
- cache plan;
- workspace sizes;
- parallel task graph;
- deterministic reduction plan;
- estimated peak memory;
- expected Python boundary crossings.

The physical plan is derived without changing logical semantics. Both plans are serializable and recordable.

Compilation fails when the requested combination is invalid, for example:

- PCMCI on tabular data without temporal metadata;
- DAG-only identification on a PAG without a completion or class-aware identifier;
- an estimator incompatible with the identified functional;
- dynamic intervention without a time axis;
- Bayesian mechanism family unsupported by the selected inference backend;
- estimated required memory exceeds budget and no streaming/chunked path exists.

### 21.3 Graph review boundary

Automatic `run()` is allowed when:

- a graph is supplied and validated;
- discovery returns a fully specified graph and the caller permits automatic acceptance;
- the requested query can be evaluated over the returned graph class directly.

Otherwise compilation returns a `ReviewRequired` artifact containing unresolved graph features and their relevance to the query.

### 21.4 Result type

Shipped shape (`crates/causal/src/result.rs`):

```rust
pub struct CausalAnalysisResult {
    pub logical_plan: LogicalAnalysisPlanRecord,
    pub physical_plan: PhysicalExecutionPlanRecord,
    pub identification: IdentificationResult,
    pub estimand: IdentifiedEstimand,
    pub estimate: EffectEstimate,
    pub posterior: Option<CausalPosterior>,
    pub refutations: Vec<RefutationReport>,
    pub diagnostics: Vec<Diagnostic>,
    pub provenance: ProvenanceGraph,
    pub performance: ExecutionPerformanceRecord,
    pub treatment: VariableId,
    pub outcome: VariableId,
}
```

There is no separate `graph` / `validation` field on the Rust result: graph review is a compile-time gate, and refutation reports live in `refutations`. Python nests the same information under `result.validation`.

### 21.5 Planner anti-patterns

Do not:

- hide a large materialization or transpose behind `run()` without recording it;
- select an algorithm solely because it is implemented when its memory or asymptotic behavior is unsuitable for the input;
- let `Auto` choose statistically different semantics based on CPU features;
- allow a Python callback in a compiled parallel hot path without marking the plan as a slow path;
- compile separate physical plans independently in nested algorithms when one parent plan can coordinate resources.

### 21.6 Facade surface that grew in-tree

Beyond the builder/`run` sketch above, the `causal` crate (and Python bindings) expose:

- `prelude` — day-1 re-exports (`CausalAnalysis`, `CausalError`, query/data/graph types, strategy IDs);
- `RefuteSuite` (`None` / `PlaceboAndRcc` / `Full`) on `CausalAnalysisBuilder`;
- `gcm` module — fit GCM, interventional sample, ITE, anomaly / distribution / mechanism-change attribution;
- `strategy_table` — typed `IdentifierId` / `EstimatorId` pairing and validation (wire strings via `From<&str>`);
- `discovery_defaults` — named CI resolution (`parcorr`, `weighted_parcorr`, …) including optional weights;
- Python OO facade: `causal.analyze(..., query=AverageEffect|PulseEffect|SustainedEffect, …)` plus `discover_pcmci*` / `counterfactual_ite` / `sample_do` / `dag_from_*` / `dag_to_*`.

## 22. Error and diagnostic model [built]

Use structured errors for execution failure and diagnostics for scientifically or operationally important conditions that may not be fatal.

The facade error type is `causal::AnalysisError` (structured sum over domain errors).
`CausalError` is a documented type alias of `AnalysisError` for DESIGN naming parity — it is not
a separate core type.

```rust
// crates/causal/src/error.rs (shape; variant set may grow)
pub enum AnalysisError {
    Identify(IdentificationError),
    Estimate(EstimationError),
    Validate(ValidationError),
    Discovery(DiscoveryError),
    Model(ModelError),
    Counterfactual(CounterfactualError),
    Attribution(AttributionError),
    Serialization(IoError),
    Data(DataError),
    Graph(GraphError),
    Design(DesignError),
    State(StateError),
    Schema(SchemaError),
    Compile { message: String },
    Resource { message: String },
    ReviewRequired { message: String },
    Unsupported { message: &'static str },
    Missing { field: &'static str },
}

pub type CausalError = AnalysisError;
```

Python maps each domain variant to a typed `Causal*Error` subclass of `CausalError`
(including `CausalDesignError` / `CausalStateError`).

Examples of scientific diagnostics:

- weak positivity;
- rank deficiency handled by pivoting;
- unidentified posterior graph mass;
- high MCMC autocorrelation;
- unstable orientation;
- assumption untestable;
- temporal sampling gap larger than modeled lag;
- mechanism extrapolation outside training support;
- sensitivity result dominated by one prior choice.

Examples of execution diagnostics:

- Arrow input copied because buffers were incompatible;
- scalar fallback used because SIMD alignment or CPU support was unavailable;
- BLAS backend bypassed for a small-matrix kernel;
- cache disabled by memory budget;
- graph unfolding truncated to requested horizon;
- batch size reduced to satisfy memory limit;
- Python callback forced serial execution;
- deterministic reduction selected over faster nondeterministic reduction.

Diagnostics have stable codes, severity, affected artifact IDs, and machine-readable fields. Performance diagnostics are descriptive; benchmark regressions are test failures, not runtime warnings.

## 23. Execution and performance [partial]

### 23.1 Execution context

```rust
pub struct ExecutionContext {
    pub parallelism: Parallelism,
    pub determinism: Determinism,
    pub rng: RngFactory,
    pub memory: MemoryBudget,
    pub cancellation: CancellationToken,
    pub progress: Option<Arc<dyn ProgressSink>>,
    pub kernel_policy: KernelPolicy,
    pub cache_policy: CachePolicy,
}
```

No core algorithm creates a global thread pool, uses an implicit global RNG, or selects architecture-specific behavior outside `KernelPolicy`.

### 23.2 Kernel dispatch and SIMD

`causal-kernels` provides:

- a scalar reference implementation (correctness gold standard);
- a portable optimized implementation where stable Rust permits it (always compiled; selected via `KernelPolicy`);
- architecture-specific implementations behind runtime feature detection **only when justified by benchmarks**;
- one public semantic entry point per kernel.

Dispatch occurs once per batch or compiled plan. It does not occur per element. Users do not choose SIMD; default `KernelPolicy` allows portable and arch paths. `force_scalar` is the debugging / differential-test escape hatch.

**Sequencing decision:** implement missing kernels as scalar + portable first. Add arch-specific SIMD only for kernels where portable leaves measurable performance on the table (typical candidates: masked reductions, contingency accumulation, bootstrap weighted accumulation). Contiguous `copy` / `gather` often will not justify `unsafe`.

**Semantic contract:** scalar, portable, and arch SIMD share one statistical contract. Optimized paths must pass the same property, conformance, and tolerance-class tests (ADR 0010 / 0011). SIMD and optional BLAS must not change statistical defaults. Silent numerical drift (for example discovery edges flipping from reduction-order or NaN-handling differences) is a release blocker.

**Reduction policy:** prefer deterministic reductions on CI-test and discovery hot paths even when slightly slower. Non-associative / nondeterministic reductions are allowed only when explicitly selected, recorded in execution provenance, and refused under `Determinism::Strict` when no deterministic path exists.

Candidate kernels (scalar + portable required; arch SIMD optional after benches):

- masked sums, means, variances, and covariance accumulation (**shipped**);
- standardization (**shipped** as public `standardize_inplace`); residualization stays inside
  `partial_correlation` / ParCorr workspaces rather than a separate public kernel;
- weighted dot products and bootstrap weighted sum/mean (**shipped**); unweighted dots are
  ordinary slice folds and are not a dedicated `causal-kernels` entry;
- pairwise L1 distance fill (**shipped**);
- contingency-table accumulation helpers (**shipped**);
- lagged gather/copy kernels (**shipped**);
- posterior draw reductions (**deferred** — add when Bayesian / posterior batch paths need a
  shared reduction surface; see TODO).

SIMD kernels must define:

- supported alignment and stride conditions;
- scalar tail behavior;
- NaN and validity semantics;
- deterministic versus non-deterministic reduction behavior;
- tolerance class;
- minimum batch size at which dispatch is beneficial.

### 23.3 Workspaces and allocation policy

Every repeated high-frequency operation with nontrivial scratch space has a workspace API. Workspaces may grow but do not shrink during an operation. They are reused within one execution plan and are not shared concurrently unless explicitly synchronized.

Designated hot paths must have allocation tests covering steady-state execution after preparation. A hot path may allocate for output growth, but repeated scratch allocation is a regression.

### 23.4 Parallel work units

**Decision:** parallelism is owned by `ExecutionContext` and implemented with hand-rolled `std::thread::scope` (as in discovery `engine.rs`). Do **not** introduce `rayon` or any global thread pool. Thread count is a runtime budget (`Parallelism::max_threads`), not a Cargo feature. Nested or irregular work (for example graph MCMC, attribution coalitions) may use a thin library-owned helper; it must still honor `ExecutionContext` budgets and must not expose a third-party pool API.

Parallelize at coarse, deterministic boundaries:

- candidate target variables;
- candidate-link batches;
- bootstrap/permutation replicate batches;
- graph samples;
- posterior predictive draw blocks;
- attribution coalitions or permutation blocks;
- candidate experiments.

The physical planner chooses one primary parallel dimension and assigns nested budgets explicitly. Small tasks remain local to avoid scheduler overhead.

### 23.5 Caching

Cache only pure computations with complete keys:

- lag maps and materialized sample indexes;
- CI-test results and residuals;
- sufficient statistics;
- graph scores;
- separation queries;
- identification results;
- compiled functional evaluators;
- mechanism execution plans.

Caches are bounded, optional, observable, and versioned. Serialized results never rely on an external cache for correctness. Cache keys do not include pointer identity or Python object identity.

### 23.6 Memory planning

The physical planner estimates:

- input views and required copies;
- graph and evidence storage;
- sample/design matrices;
- workspaces per concurrent task;
- posterior or bootstrap batches;
- result retention;
- cache allowance.

If expected peak memory exceeds the budget, the planner selects a smaller batch, streaming summary, sparse representation, or refuses with a structured resource error. It does not proceed until allocator failure.

### 23.7 Determinism

`Determinism::Strict` guarantees:

- stable iteration order;
- deterministic task partitioning where supported;
- seeded independent RNG streams derived from operation IDs;
- stable tie breaking;
- deterministic reductions where a supported path exists (nondeterministic / non-associative kernel reductions are refused — see §23.2);
- recorded floating-point and kernel backend.

It does not guarantee bitwise equality across CPU architectures or alternative BLAS backends unless a backend explicitly provides that guarantee.

### 23.8 Performance contracts

Each designated workload has:

- a canonical fixture and fixture generator;
- correctness assertions;
- peak-memory measurement;
- allocation count or allocated-byte measurement where stable;
- single-thread latency/throughput baseline;
- multi-thread scaling baseline where relevant;
- Python-boundary baseline where relevant;
- accepted regression budget;
- hardware and compiler metadata.

Default merge budget for stable benchmark noise is 5% on median latency and 10% on peak memory. Workloads with higher variance define their own budget. A regression beyond budget requires an approved benchmark-baseline change explaining the cause and why the design remains acceptable.

### 23.9 Prohibited optimization shortcuts

Do not:

- add an optimized path without a scalar reference and differential tests;
- cache a result whose key omits masks, weights, node order, assumptions, or backend-relevant semantics;
- use unsafe aliasing or alignment assumptions outside isolated reviewed kernel modules;
- trade exact graph semantics for approximate traversal without an explicitly different algorithm name;
- change RNG consumption order under strict determinism without a versioned behavior decision;
- optimize benchmark-only fixtures while leaving representative mixed workloads slow;
- use wall-clock benchmarks as the only evidence when allocation or memory growth is the real risk;
- defer all performance validation to a later release.

## 24. Serialization and artifact format [partial]

Shipped: versioned sectioned container with per-section Zstandard compression
(Auto/Always/Never), BLAKE3 checksums, format `0.2` (`STABLE_FORMAT`) with
`0.1→0.2` schema migration, CBOR wire types for graphs/queries/analysis/
mechanisms/plans/provenance, DOT/GML/JSON and NetworkX-compatible DAG exchange,
and `model_bundle` encode/decode. See `docs/artifacts.md`.

Remaining: §24.5 memory-map / stream large Arrow sections without full buffer
load; skip unread sections without materializing payloads; zero-copy write paths
when Arrow buffers are already shared.

### 24.1 Container format

Use a versioned sectioned container:

```text
magic bytes
container version
canonical CBOR manifest
section table
CBOR semantic sections
Arrow IPC numerical/tabular sections
optional compressed posterior sections
checksums
```

Large arrays are not embedded as generic CBOR arrays. They use Arrow IPC or another explicitly versioned binary section.

```rust
pub struct ArtifactManifest {
    pub format_version: FormatVersion,
    pub minimum_reader_version: FormatVersion,
    pub artifact_kind: ArtifactKind,
    pub library_version: SemanticVersion,
    pub artifact_id: ArtifactId,
    pub sections: Vec<SectionDescriptor>,
    pub provenance: ProvenanceWire,
}
```

Every section records:

- stable section identifier;
- content type and encoding version;
- required or optional status;
- compression;
- compressed and uncompressed sizes;
- BLAKE3 checksum;
- logical schema identifier.

Metadata and graph/query/model records use canonical CBOR. JSON is a debugging and interchange representation, not the canonical durable encoding. Section compression defaults to Zstandard where beneficial.

### 24.2 Wire types

Internal Rust structs are not serialized directly as the artifact specification. Each stable artifact has explicit versioned wire types and conversion code:

```text
internal semantic type
    <-> versioned wire type
    <-> CBOR/Arrow section
```

Ordinary internal refactoring must not change the wire format.

### 24.3 Migration

Every breaking schema change requires migration from at least the previous two stable format versions. Unknown optional fields are preserved where practical. Unknown semantic variants fail explicitly rather than being ignored.

### 24.4 Python objects

Artifacts must not serialize arbitrary Python callables. Python-defined mechanisms require one of:

- a registered portable expression/model representation;
- an explicit nonportable marker and Python-only serializer;
- rejection for cross-language export.

### 24.5 Serialization performance requirements

Required:

- readers can skip unneeded sections;
- large arrays can be memory-mapped or streamed where supported;
- checksums and compression operate per section;
- graph and expression wire forms use compact IDs and shared tables;
- artifact writes do not duplicate large Arrow buffers when zero-copy transfer is available.

Do not:

- use `serde` derives on internal structs as the durable contract;
- encode millions of numerical values as nested CBOR objects;
- require loading posterior draws to inspect artifact metadata;
- serialize transient dense graph indexes as stable identities;
- make compression mandatory for already compressed or random-like sections.

## 25. Python package [partial]

### 25.1 Package structure

Canonical layout (matches §3 and `python/pyproject.toml` `python-source = "."`):

```text
python/
  Cargo.toml                 # maturin / PyO3 cdylib
  pyproject.toml
  src/lib.rs                 # Rust bindings → causal._native
  causal/                    # installed Python package
    __init__.py
    py.typed
    data.py
    graph.py
    query.py                 # AverageEffect, PulseEffect, SustainedEffect
    inference.py             # Frequentist, Bayesian
    discovery.py             # PCMCI* configs + discover_* helpers
    estimation.py            # analyze() + nested AnalysisResult
    model.py
    counterfactual.py
    attribution.py
    design.py
    state.py                 # reserved / experimental
    _native.*                # built extension (not checked in)
    *.pyi
  tests/
  examples/
```

Identification and validation are not separate public modules: they surface through
`analyze(...).identification` / `.validation`. Advanced FFI remains on `causal._native`
(including `analyze_ate` / temporal `analyze` used by the OO layer).

### 25.2 Binding rules

- PyO3 bindings contain conversion and ergonomic logic only.
- Statistical algorithms remain in Rust crates.
- Long-running calls release the GIL.
- Rust panics never cross FFI; convert to Python exceptions.
- NumPy arrays are borrowed when contiguous and compatible.
- pandas is converted through Arrow where practical.
- Polars and PyArrow use Arrow C Data Interface/Stream Interface.
- Returned large arrays use NumPy or Arrow buffers, not Python lists.
- A binding method invokes one coarse Rust operation or one explicitly documented slow-path callback loop.

### 25.3 Python API

```python
result = causal.analyze(
    data,  # dict[str, array] or pandas DataFrame
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=causal.AverageEffect(treatment="t", outcome="y"),
    inference=causal.Bayesian(n_draws=1000),
)

# Temporal pulse with a supplied lagged graph:
result = causal.analyze(
    series,
    graph=[("pressure", 1, "defect", 0)],
    query=causal.PulseEffect(
        treatment="pressure",
        outcome="defect",
        active_level=-0.03,
        treatment_lag=1,
        horizon_steps=1,
    ),
)

# Or discover → estimate in one call:
result = causal.analyze(
    series,
    discovery=causal.PCMCI(max_lag=1, alpha=0.05),
    query=causal.PulseEffect(
        treatment="pressure",
        outcome="defect",
        treatment_lag=1,
        horizon_steps=1,
    ),
)
```

Nested result sections (Python):

```python
result.identification
result.estimate
result.posterior
result.validation
result.diagnostics
result.provenance
result.performance
```

`discovery=` accepts `PCMCI` / `PCMCIPlus` / `LPCMCI` for temporal queries (auto-accepts
oriented edges by default). `JPCMCIPlus` / `RPCMCI` still go through `discover_*` helpers.
Attribution, NetworkX adjacency IO, and causal-state append helpers are exported from the
package root and stage modules.

### 25.4 Python extensibility

Allow Python callbacks only at explicit slow-path extension points:

- custom utility functions;
- custom mechanism wrappers;
- custom CI tests;
- custom validators.

Callbacks reacquire the GIL and prevent full Rust parallelism. The physical plan marks callback regions and does not imply native performance. Performance-critical extensions use Rust traits or a future stable plugin ABI.

### 25.5 Wheels and supported versions

Build with maturin.

Minimum supported Rust version:

```text
Rust 1.85
edition 2024
```

Supported Python versions for the first public release:

```text
CPython 3.11, 3.12, 3.13, 3.14
```

Initial wheel matrix (**status:** shipped in CI):

- Linux x86-64 manylinux;
- Linux aarch64 manylinux;
- macOS x86-64 and arm64;
- Windows x86-64.

The default wheel includes the pure-Rust `faer` path and must not require system BLAS. `abi3`, free-threaded Python, PyPy, and optional BLAS wheel variants are experimental until NumPy/Arrow compatibility and performance are measured. Package and crate versions remain `0.1.0` until an explicit 1.0.0 release decision (ADR 0017); artifact format `0.1` is frozen.

### 25.6 Python performance requirements

Required:

- conversion tests measure borrowed bytes, copied bytes, and conversion latency;
- representative end-to-end calls cross Python/Rust once per analysis stage, not once per row/test/draw;
- GIL-release tests verify another Python thread can run during native computation;
- large result objects expose zero-copy NumPy/Arrow views when lifetime safety permits;
- Python wrappers do not reconstruct large Rust result structures as nested dictionaries.

Do not:

- accept object-dtype arrays in a hot path without explicit conversion diagnostics;
- return millions of graph statistics as Python tuples;
- hold the GIL while Rust performs discovery, bootstrap, posterior evaluation, or simulation;
- duplicate algorithms in Python for convenience;
- hide callback-induced serialization of execution.

## 26. Parity management [built]

Parity is tracked in machine-readable manifests pinned to exact upstream references. "Latest upstream" is never an acceptance target.

Pinned baselines:

```text
pinned estimate baseline v0.14
commit 178ecc9c690a02f2801c1f70da2695f5744186cc

pinned discovery baseline stable 5.2.1.25
commit 5a8768754e6103755b006e9357e21c1a58534927

pinned discovery baseline extended snapshot
commit ff3ff13e1481073b8c5833a6fde1c304627a208e
```

```toml
[[capabilities]]
id = "pinned estimate baseline.estimator.linear_regression"
upstream = "py-why/pinned estimate baseline"
upstream_ref = "178ecc9c690a02f2801c1f70da2695f5744186cc"
category = "estimation"
status = "pending"
parity = ["algorithm", "statistical", "documented_edge_cases"]
reference_tests = ["linear_continuous_ate", "effect_modifiers"]
performance_workloads = ["ols_tall_skinny", "bootstrap_ols"]
notes = "Python API parity not required"
```

Statuses (manifest vocabulary — use these in `parity/*.toml` and prose):

- `pending`: not started (must appear on `TODO.md` when it is a DESIGN chapter);
- `in_progress`: implementation underway;
- `done`: implemented with conformance / gate evidence.

Do not use the older draft vocabulary (`not_planned` / `planned` / `implemented` /
`conformant` / `deviates` / `blocked` / `intentional_deviation`) in manifests.
Out-of-scope items are omitted from inventories (see excluded / optional notes in
the relevant `parity/*.toml`) or documented as permanent product contracts in this
file and marked `done` when the shipped surface matches that contract.

Parity dimensions:

- algorithmic semantics;
- statistical behavior;
- documented edge cases and errors;
- artifact/result completeness;
- calibration where applicable;
- performance workload coverage.

Performance is not required to match an upstream implementation exactly, but every parity capability expected to be computationally material must name at least one representative benchmark. Comparative pinned estimate baseline/pinned discovery baseline timings are informative, not the sole target: an upstream implementation may itself be inefficient or may use a different numerical backend.

### 26.1 Estimate / identify parity inventory

Required capability groups:

#### Model and graph workflow

- causal model construction from data and graph;
- explicit treatment/outcome/effect modifiers;
- graph parsing and validation;
- inspectable assumptions;
- model, identify, estimate, refute workflow.

#### Identification

- adjustment sets;
- automatic identification;
- backdoor;
- efficient backdoor;
- general ID;
- identified-estimand representation.

#### Estimation

- distance matching;
- doubly robust;
- generalized linear model;
- instrumental variables;
- linear regression;
- propensity base, matching, stratification, weighting;
- regression discontinuity;
- two-stage regression;
- conditional effects/effect modifiers.

Excluded from required parity:

- EconML adapter;
- arbitrary EconML model composition.

Optional adapter parity:

- CausalML;
- TabPFN or successor external estimators.

#### Refutation and sensitivity

- unobserved common cause;
- overlap and overlap-rule assessment;
- bootstrap;
- data subset;
- dummy outcome;
- E-value;
- graph refutation;
- linear, partial-linear, and nonparametric sensitivity;
- placebo treatment;
- random common cause;
- Reisz-related diagnostics.

#### Do-sampling

- weighting;
- multivariate weighting;
- kernel-density;
- MCMC.

#### GCM

- causal mechanisms and model types;
- automatic mechanism assignment;
- fitting and sampling;
- anomaly scoring and attribution;
- distribution-change attribution and robust variants;
- density estimation and divergence;
- graph/model falsification;
- feature relevance;
- causal influence and arrow strength;
- model evaluation;
- Shapley utilities;
- stochastic models;
- uncertainty utilities;
- unit-change attribution;
- what-if/interventional analysis;
- confidence intervals.

#### Secondary surfaces

pinned estimate baseline’s secondary package (graph learners, transformers, interpreters, time-series
helpers as one grab-bag) is **not mirrored**. Equivalent capabilities are tracked
as dedicated inventory IDs under discovery, transforms, effects, and facade
workflows. `estimate.secondary_surfaces` is `done` under that contract.

### 26.2 Discovery / CI parity inventory

#### Data processing

- dataframe-equivalent temporal data abstraction;
- masks and missing flags;
- multiple datasets;
- time offsets/reference points;
- vector variables;
- transformations, smoothing, binning, ordinal patterns;
- bootstrap generation.

#### Conditional-independence tests

- CMI kNN;
- mixed CMI kNN;
- symbolic CMI;
- GPDC (native RBF-GP + distance correlation; torch backend is not in scope);
- G-squared;
- oracle CI;
- pairwise multivariate wrapper;
- partial correlation;
- multivariate partial correlation;
- weighted partial correlation;
- regression CI;
- robust partial correlation.

#### Discovery

- PCMCI;
- PCMCI+;
- LPCMCI;
- J-PCMCI+;
- RPCMCI;
- shared PCMCI base behavior;
- link assumptions;
- optimization and iteration diagnostics;
- FDR and confidence intervals;
- masked-data behavior.

#### Graphs

- time-series graph representations;
- stationarity handling;
- path and separation functions;
- endpoint/orientation semantics;
- hidden-variable graph handling.

#### Effects and models

- causal effects;
- causal mediation;
- linear mediation;
- model base;
- prediction;
- direct, total, mediated, and conditional effects.

#### Simulation and presentation data

- structural process generators;
- context/regime toy models;
- graph validation for generated processes;
- data structures needed by plotting and notebook adapters.

Plot rendering itself may be implemented in Python or exported to plotting libraries; all underlying plot data and graph layouts required for parity must be produced.

### 26.3 Inventory honesty and release gates

Parity work is inventory + evidence, not a waiver registry:

- `parity/*.toml` — capability inventories (`pending` / `in_progress` / `done`);
- `TODO.md` — unfinished DESIGN chapters corresponding to `pending` rows;
- `parity/release.toml` — release-closure capabilities (artifact schema, wheel matrix,
  conformance docs);
- `scripts/gate_*.sh` — executable gates that fail CI when a claimed `done` row
  lacks evidence.

Marking `done` without a gate path or conformance fixture is incomplete.
`pending` rows must cite `TODO.md` (or an inline DESIGN note) so roadmap work is
not lost. There is no `intentional_deviation` status.

## 27. Licensing and clean implementation process [built]

The project is licensed under:

```text
MIT OR Apache-2.0
```

Source files use:

```text
SPDX-License-Identifier: MIT OR Apache-2.0
```

Contributions require Developer Certificate of Origin sign-off. A CLA is not required initially. This decision must be reconsidered before any proprietary dual-licensing or centralized relicensing plan.

The project is independently implemented from published papers, specifications, and public behavior.

Repository rules:

- implementation PRs cite papers, standards, or independent design notes;
- every substantive algorithm has a machine-readable provenance record under `provenance/`;
- provenance records truthfully disclose prior exposure to upstream implementations;
- no copied or translated source, comments, docstrings, tests, or notebooks from pinned discovery baseline enter the repository;
- pinned estimate baseline code is also not translated line by line despite its permissive license;
- reference libraries may be executed as black-box comparators in isolated conformance tooling;
- reference outputs are generated from public APIs and stored with version, command, environment, and fixture metadata;
- unusual parity quirks require a written decision: match upstream, or accept a permanent product contract in DESIGN and mark the inventory row `done`;
- contributor documentation states that "port" is project shorthand, not a source translation.

A provenance record includes:

```toml
feature_id = "discovery.pcmci"
implementation_crate = "causal-discovery"
source_translation = false
copied_code = false
copied_comments = false
copied_tests = false

papers = [
  { title = "...", doi = "...", sections = ["Algorithm 1", "Appendix B"] }
]

upstream_implementations_observed = [
  { project = "pinned discovery baseline", exposure = "previous familiarity" }
]

test_sources = [
  "independently generated synthetic SCMs",
  "paper example",
  "black-box comparison against pinned baseline"
]
```

Do not commit upstream GPL source, translated GPL tests, or fixtures whose redistribution status is unclear. The clean-implementation and licensing policy must receive legal review before the first public release if permissive commercial use is a project requirement.

## 28. Testing, conformance, and performance gates [built]

### 28.1 Unit tests

Focus on mathematical, structural, and resource invariants:

- graph insertion preserves graph-type invariants;
- separation witnesses are valid;
- latent projection preserves relevant m-separation relations;
- symbolic simplification preserves evaluation;
- resampling respects partitions and masks;
- estimators reject incompatible estimands;
- serialization round trips preserve semantic equality;
- prepared hot paths perform no steady-state scratch allocation beyond declared output growth;
- scalar and optimized kernels return equivalent results within their tolerance class.

### 28.2 Property tests

Generate small random graphs and verify:

- topological ordering;
- d-separation against a reference implementation;
- graph mutilation;
- equivalence-class invariants;
- temporal unfolding consistency;
- adjustment sets block all noncausal paths and contain no forbidden descendants;
- ID expressions evaluate consistently on generated SCMs;
- SIMD and scalar kernels agree under random strides, masks, NaNs, tails, and alignments;
- chunked/streaming execution agrees with full materialization.

### 28.3 Statistical calibration tests

On repeated synthetic data:

- CI tests maintain nominal type-I error within tolerance;
- confidence and credible intervals achieve target coverage under declared models;
- permutation p-values are approximately uniform under null;
- discovery false-positive rates and power meet defined ranges;
- posterior SBC ranks are calibrated;
- bootstrap procedures preserve temporal dependence assumptions.

These tests use fixed simulation budgets and tolerance bands. They run in scheduled CI, not every unit-test invocation.

### 28.4 Conformance tests

Conformance fixtures live under `conformance/<domain>/` (e.g. `pinned estimate baseline/`, `pinned discovery baseline/`, `gcm/`,
`estimate/`, `pag/`, `bayesian/`, `attribution/`, `context/`, `design_state/`, `interchange/`,
`manufacturing/`) plus shared `conformance/gates/`. Each case typically has a README, inputs, and
`expected.json` (or equivalent) rather than a single global `paper_examples|generated|reference_outputs`
tree.

For each parity capability:

1. define a versioned input fixture under the relevant domain directory;
2. run the pinned reference implementation when applicable;
3. capture structured outputs;
4. run the Rust implementation;
5. compare using capability-specific tolerance classes;
6. leave unfinished DESIGN chapters as `pending` in inventories and on `TODO.md`.

Compare:

- selected variables and graph marks;
- statistics and p-values;
- identified estimands;
- numerical estimates;
- validation outcomes;
- documented error behavior.

Randomized procedures compare distributions or summary statistics unless a shared random stream can be reproduced.

### 28.5 Numeric tolerance policy

Every assertion declares one tolerance class:

```rust
pub enum ToleranceClass {
    Exact,
    StableFloat,
    BackendSensitive,
    ResidualBased,
    MonteCarlo,
    PosteriorDistribution,
}
```

Defaults:

```text
StableFloat:      atol 1e-10, rtol 1e-8
BackendSensitive: atol 1e-8,  rtol 1e-6
```

Exact equality is used for graph structure, normalized estimands, masks, category mappings, sample indexes, temporal unfolding, error classes, and artifact metadata.

Ill-conditioned linear algebra uses normalized residual, fitted-value, rank, objective, or subspace comparisons rather than unstable coefficient-by-coefficient equality.

P-value fixtures used for exact graph decisions must remain outside a threshold guard band:

```text
abs(p - alpha) >= max(1e-6, 0.1 * alpha)
```

Monte Carlo comparisons use combined Monte Carlo standard error plus a documented numerical floor. Posterior comparisons use moments, selected quantiles, predictive quantities, calibration, or distribution distances with Monte Carlo error accounted for.

Each fixture stores its chosen tolerances and reason. Widening a tolerance requires review and a written numerical explanation.

### 28.6 Cross-language tests

Every public Python operation has a Rust-equivalent fixture. Verify:

- identical logical plan construction;
- equivalent physical plan where platform capabilities match;
- equivalent data conversion;
- same artifact schema;
- exception mapping;
- expected copy behavior for supported inputs;
- GIL release during long calls;
- no Python-call count proportional to rows, CI tests, posterior draws, or simulation samples on native paths.

### 28.7 Fuzzing

Fuzz:

- graph parsers;
- artifact deserialization;
- expression parser and simplifier;
- temporal sample requests;
- Python conversion boundaries;
- SIMD kernels against scalar references;
- graph workspaces under repeated reuse;
- malformed Arrow metadata and category dictionaries.

### 28.8 Benchmark suites

Benchmark workloads:

- Arrow/NumPy conversion and zero-copy acceptance;
- masked column reductions;
- lag-map construction and lagged sample gathering;
- partial-correlation CI batches across conditioning sizes;
- kNN and contingency CI tests;
- PC and PCMCI parent search;
- PCMCI+ and LPCMCI orientation;
- d/m-separation query batches;
- adjustment search and ID on representative graph sizes;
- OLS/GLM/Laplace fits;
- bootstrap estimation;
- interventional and counterfactual sampling;
- Shapley attribution;
- posterior functional evaluation;
- serialization read/write and partial section loading;
- Python end-to-end calls.

Each workload has small, representative, and stress tiers. Benchmarks record latency, throughput, peak resident memory where available, allocation metrics, thread scaling, and output correctness hashes.

### 28.9 Merge and release gates

A feature PR is not complete when it only passes correctness tests. It must include or update:

- a representative benchmark;
- a memory/allocation assertion for designated hot paths;
- scalar-versus-optimized differential tests when optimized kernels are involved;
- the physical-plan behavior expected for its main workloads.

Release gates include:

- no unexplained benchmark regression beyond workload budget;
- no unbounded memory growth in stress tests;
- no fallback from zero-copy to copy for supported Python inputs without an explicit decision;
- no loss of statistical calibration from an optimization;
- no hot path missing a benchmark owner and baseline.

## 29. Unsafe code and dependency policy [built]

- `#![forbid(unsafe_code)]` in semantic, graph, identification, discovery, estimation, and model crates where practical.
- Unsafe code is allowed only in isolated FFI or kernel modules with a written safety contract, differential tests, Miri where applicable, fuzzing, and architecture-specific CI.
- SIMD modules may use `unsafe` only for feature-detected instructions, alignment-aware loads/stores, or proven aliasing conditions that cannot be expressed safely.
- Every unsafe function documents preconditions and has a safe wrapper that validates them.
- No dependency with unmaintained status or incompatible license enters default features without an explicit decision record.
- Optional GPL dependencies are not permitted in default or distributable permissive builds.
- Numerical dependencies must support wheel distribution without user-installed native libraries in the default configuration.
- Dependency upgrades are benchmarked on designated workloads when they affect Arrow, PyO3, `faer`, random-number generation, serialization, or parallel execution.
- A dependency is not added solely to obtain one small hot-loop primitive that can be implemented and tested locally with less maintenance risk.

## 30. Feature flags [partial]

**Policy:** Cargo features mean “bring a heavy or optional *implementation / adapter*.” They never change which statistical numbers the default path produces. Performance selection lives in `KernelPolicy` and `ExecutionContext`, default-optimized, with minimal user configuration.

**Actual flags today:**

```toml
# causal-data
[features]
default = ["arrow"]
arrow = []                # Arrow ingest adapters (library-owned tables remain core)
```

**Required / always-on (not features):**

- `faer` — required dependency of `causal-stats` (ADR 0001).
- Portable-optimized kernels — always compiled into `causal-kernels`; scalar is selected only via `KernelPolicy` (`force_scalar` / disallow-portable).
- Parallelism — `std::thread::scope` coordinated by `ExecutionContext` (**`rayon` rejected**; do not add a `rayon` feature).

**Allowed optional features** (may land later; adapters / heavy backends only):

```toml
[features]
# not wired as a workspace feature set today
polars = []               # ingest adapter
serde-json = []           # exchange adapter
blas = []                 # additive LA backend; never removes faer from conformance
simd-runtime = []         # compile-in arch SIMD code size / unsafe surface only;
                          # default KernelPolicy still selects it when present (§23.2)
gaussian-process = []     # heavy mechanism backend
hmc = []                  # heavy sampling backend
smc = []                  # heavy sampling backend
networkx-io = []          # graph exchange adapter
plot-data = []            # visualization data export
```

`python` bindings live in the `python/` workspace member, not a crate feature.

Core semantic types do not change shape based on feature flags. A future `simd-runtime` must not select different statistical behavior. Optional `blas` adds a backend and never removes the required `faer` path from conformance testing.

## 31. API and performance stability [built]

Before `1.0`:

- artifact schema versions are explicit and may stabilize earlier than Rust APIs;
- the high-level facade is experimental until integrated static and temporal workflows are complete;
- low-level graph, expression, data-view, matrix-view, and temporal-index types receive stricter compatibility guarantees once used by artifacts or external extensions;
- parity manifests identify behavior changes caused by upstream reference changes;
- physical execution plans may evolve, but logical semantics and performance diagnostics remain stable enough for callers to detect copies, fallbacks, and resource use.

Public trait object safety is required only where runtime extensibility is intended. Generic traits remain generic when performance or associated types are important.

For designated hot APIs, the project documents performance-shape guarantees rather than exact timings, for example:

- batch API exists;
- preparation is separable from repeated execution;
- steady-state scratch allocation is bounded or zero;
- execution can consume borrowed views;
- memory complexity is stated;
- parallelism is externally controlled.

Removing one of these guarantees is an API change even if function signatures remain compatible.

## 32. Initial public API examples [partial]

Examples below match the shipped `0.1.0` surface. Unshipped algorithms (PC, AutoIdentifier,
full ID/IDC) are documented in §10 / §13, not here.

### 32.1 Static, Bayesian

```rust
use causal::prelude::*;

let t = schema.id_of("treatment")?;
let y = schema.id_of("outcome")?;
let result = CausalAnalysis::builder()
    .data(tabular)
    .graph(dag)
    .query(AverageEffectQuery::binary_ate(t, y))
    .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(1000)))
    .build()?
    .run(&ctx)?;

if let Some(posterior) = &result.posterior {
    println!("P(effect < 0) = {}", posterior.probability_below(0.0)?);
}
```

### 32.2 Temporal discovery and effect

```rust
use causal::prelude::*;
use causal::{DiscoveryAccept, FdrControl};

let pressure = schema.id_of("pressure")?;
let defect = schema.id_of("defect")?;
let request = CausalAnalysis::builder()
    .series(series)
    .discover_pcmci(12, 0.05, FdrControl::bh(), DiscoveryAccept::Review)
    .temporal_query(TemporalEffectQuery::pulse(pressure, defect, -0.03))
    .build()?;

match request.compile(&ctx)? {
    CompiledAnalysis::Ready(plan) => {
        let _ = request.execute(&plan, &ctx)?;
    }
    CompiledAnalysis::ReviewRequired(review) => {
        let _ = request.finish_review_and_run(review, &ctx)?;
    }
    other => {
        let _ = other;
    }
}
```

### 32.3 Change attribution

```rust
let attribution = ChangeAttribution::new()
    .outcome(defect)
    .baseline(PopulationSelector::TimeRange { start: 0, end: 100 })
    .comparison(PopulationSelector::TimeRange { start: 100, end: 200 })
    .components(AttributionComponents::All)
    .allocation(AllocationMethod::Shapley {
        approximation: ShapleyConfig::monte_carlo(2_000),
    })
    .run(&model, &data, &ctx)?;
```

### 32.4 Python

```python
result = causal.analyze(
    data,
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=causal.AverageEffect(treatment="t", outcome="y"),
    inference=causal.Bayesian(n_draws=1000),
)
print(result.estimate.ate, result.validation.count)
```

## 33. Adopted architecture decisions [built]

The following decisions are accepted and are no longer open.

1. **Linear algebra:** `faer` is the default dense backend behind an operation-level abstraction. Public APIs use library-owned matrix views. Optional BLAS is additive. (ADR 0001)
2. **Artifact encoding:** canonical CBOR for semantic metadata and Arrow IPC for large arrays, inside a sectioned versioned container with BLAKE3 checksums and optional Zstandard compression. (ADR 0002)
3. **Categoricals:** dictionary-encoded `u32` category IDs with immutable domains. Missingness is separate. Contrasts are explicit model configuration and stored in fitted artifacts. (ADR 0003)
4. **Data API:** stable library-owned data views with Arrow-backed implementations and adapters; Arrow crate types are not the public causal API. (ADR 0004)
5. **Temporal indexing:** stable `(VariableId, offset)` identities and time-major dense indexes for finite unfolding. Dense indexes are not serialized. (ADR 0005)
6. **Initial Bayesian GLM:** native Laplace approximation with a backend-neutral inference interface; external probabilistic-programming adapters are optional forever and not required for completion. (ADR 0006)
7. **Supported versions:** Rust 1.85, edition 2024; CPython 3.11 through 3.14 for the first public release. (ADR 0007)
8. **License and provenance:** `MIT OR Apache-2.0`, DCO sign-off, machine-readable algorithm provenance, and clean-implementation rules. (ADR 0008)
9. **Parity baselines:** pinned estimate baseline v0.14 at commit `178ecc9c690a02f2801c1f70da2695f5744186cc`; pinned discovery baseline stable tag `5.2.1.25` at commit `5a8768754e6103755b006e9357e21c1a58534927`, plus extended snapshot commit `ff3ff13e1481073b8c5833a6fde1c304627a208e` for post-release features. (ADR 0009)
10. **Tolerance policy:** fixture-specific `Exact`, `StableFloat`, `BackendSensitive`, `ResidualBased`, `MonteCarlo`, and `PosteriorDistribution` classes. There is no project-wide epsilon. (ADR 0010)
11. **Performance posture:** performance and correctness are co-equal requirements. Hot paths require prepared/batch APIs, reusable workspaces, memory plans, scalar references, optimized differential tests, and benchmark gates from initial implementation. (ADR 0011)
12. **GCM model:** compiled SCM with intervention overlays for sampling; no per-draw semantic-graph walk or SCM clone. (ADR 0012)
13. **PAG / LPCMCI:** PAG endpoint semantics and LPCMCI orientation inventory (`parity/pag.toml`). (ADR 0013)
14. **Context / regime effects:** multi-environment and regime handling; RPCMCI uses caller-supplied regime labels (`parity/context.toml`). (ADR 0014)
15. **Attribution:** Shapley/change/anomaly attribution surface (`parity/attribution.toml`). (ADR 0015)
16. **Design / state:** experiment-design and incremental-state surface; OLS / streaming cov / particle-filter / graph-score caches shipped (`parity/design_state.toml`). (ADR 0016)
17. **Release prep:** release-closure inventory, wheel matrix, and gate scripts (`parity/release.toml`, ADR 0017). Package version remains 0.1.0 until an explicit 1.0.0 bump.

Record these decisions as ADRs before dependent code is merged. Changing one requires an explicit superseding ADR and migration or compatibility analysis where applicable.

## 34. Definition of completion [planned]

The library reaches the described full scope when:

- all required pinned estimate baseline capabilities except EconML integration are `done` in inventories (or omitted as permanently out of scope in DESIGN);
- all required pinned discovery baseline capabilities are `done` in inventories (or omitted as permanently out of scope in DESIGN);
- every remaining DESIGN chapter is either `done` or explicitly `pending` on `TODO.md`;
- static, temporal, panel, and multi-environment data use the same facade workflow;
- every causal query passes through explicit identification;
- frequentist and Bayesian evaluators consume the same identified-functional IR where mathematically applicable;
- graph uncertainty propagates through identification, estimation, counterfactuals, and attribution;
- Python wheels expose the same Rust implementation without algorithm duplication;
- serialized artifacts contain enough information to reproduce or audit an analysis;
- incremental state and decision/design primitives remain embeddable library components rather than hosted-product features;
- conformance manifests contain no unresolved required items beyond documented `pending` roadmap rows;
- every designated hot path has a documented data layout, preparation path, reusable workspace, batch API, memory complexity, benchmark fixture, regression budget, and responsible owner;
- scalar reference and optimized implementations pass the same semantic, property, conformance, and tolerance tests;
- no supported workflow depends on per-row, per-test, per-draw, or per-sample Python callbacks in its native fast path;
- physical plans expose materialization, copy, backend, SIMD, parallel, cache, and peak-memory decisions;
- no subsystem relies on an unspecified future rewrite to obtain acceptable vectorization, parallelism, or memory behavior.

