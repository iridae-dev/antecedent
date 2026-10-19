# ADR 0026: Internal contract versions

- Status: Accepted
- Date: 2026-10-19
- Extends: [0002](0002-artifact-encoding.md), [0004](0004-data-api.md), [0005](0005-temporal-indexing.md), [0010](0010-tolerance-policy.md), [0011](0011-performance-posture.md), [0018](0018-release-version-history.md), [0019](0019-response-artifact-format.md), [0021](0021-temporal-response.md), [0022](0022-causal-compiler-contract.md)

## Decision

Internal contracts below are versioned separately from the package release and
from the platform floor. ADR 0007 remains the platform contract: Rust 1.85,
edition 2024, and CPython 3.11 through 3.14. This decision does not change it.

### Identity domains

Extends ADR 0022. The encoding those domains hash is the CBOR wire of ADR 0002.

- Format tag `antecedent.identity.v2` (`IDENTITY_FORMAT = 2`). A version-2
  digest is not comparable with a version-1 digest. A canonical-payload change
  bumps this format; it does not reuse a v2 derive-key.
- Each layer keeps its own BLAKE3 derive-key. The closed set is:

  | Domain | Derive-key |
  | --- | --- |
  | Target | `antecedent.identity.target.v1` |
  | Identification | `antecedent.identity.identification.v1` |
  | Identification product | `antecedent.identity.identification_product.v1` |
  | Program | `antecedent.identity.program.v1` |
  | Inference binding | `antecedent.identity.inference_binding.v1` |
  | Observation | `antecedent.identity.observation.v1` |
  | Data snapshot | `antecedent.identity.data_snapshot.v1` |
  | Execution | `antecedent.identity.execution.v1` |
  | Claim | `antecedent.identity.claim.v1` |
  | Score reuse | `antecedent.identity.score_reuse.v1` |
  | Target weights | `antecedent.identity.target_weights.v1` |
  | Learned trial | `antecedent.identity.learned_trial.v1` |
  | Transport certificate | `antecedent.identity.transport_certificate.v1` |

- Owned columnar content identity stays `antecedent.data.storage.v1`, the
  encoding ADR 0022 already binds into the data-snapshot layer. A new layer
  gets a new derive-key. It does not overload an existing one.

### Uncertainty types

Extends ADR 0010 and the uncertainty-kind axis of ADR 0019 and ADR 0021.

- ADR 0010 tolerance classes (`Exact`, `StableFloat`, `BackendSensitive`,
  `ResidualBased`, `MonteCarlo`, `PosteriorDistribution`) stay fixture
  comparison classes. They are not result uncertainty types.
- The result uncertainty vocabulary is `ResponseUncertainty`: `None`,
  `Scalar`, `PointwiseBand`, `SimultaneousBand`, `IdentifiedEnvelopeBand`,
  and `Posterior`. Pointwise and simultaneous bands are distinct types.
  `IdentifiedEnvelopeBand` is a band around an identified envelope, and the
  envelope itself is a separate value. Every interval carries
  `IntervalInterpretation`: `confidence` or `credible`.
- Decision requirements use `UncertaintyKind`: `Point`,
  `StructuralEnvelope`, `Credible`. A requirement matches only its own kind.
  A structural envelope is a set, and a credible claim is a probability.
- Credible draw vectors are in-process retention. The response artifact
  decodes them as absent, so this vocabulary does not advance the artifact
  format.

### Cyclic solution semantics

Extends ADR 0022 and ADR 0005. Contract `cyclic.solution.v1`.

- The initial family is the specified linear equilibrium `X = KX + c + ε`
  with an exact zero diagonal. `K[i, j]` is j's direct mechanism into i.
  A hard clamp deletes clamped equations and leaves the free block
  `(I − K_FF) X_F = K_FH a + c_F + ε_F`, which is factored, not inverted.
  The free disturbance stays the marginal law of the free variables.
- A sufficient `ρ(|K|) < 1` bound (admission gap `1e-12`) certifies that
  family-wide contraction. It is not a simple-SCM certificate: unique
  solvability of every subset is a different claim, and this bound does not
  issue it. `ρ(K) < 1` on the signed matrix, invertibility of `(I − K)`,
  physical stability, and iterative-solver convergence stay separate
  statuses. This contract leaves physical stability unclaimed and does not
  treat the bound as a solver result.
- A failed bound is not an identification failure. Both the admitted and
  the insufficient certificate report that the model was not marked
  unidentified. The insufficient certificate names a query-specific free-block
  check. Specified-model algebra does not identify `K` from observations.
- σ-separation is the conditional-independence query on a directed mixed
  graph that may contain cycles. The answer carries its assumptions: the
  observational Markov reading needs unique solvability per strongly
  connected component, which the query does not check; a simple SCM and
  σ-faithfulness are not assumed; failure to separate is not a proof of
  dependence. Ordinary d-separation on cyclic graphs stays unsettled.
  Acyclification edges used by the procedure are not original direct
  mechanisms.
- Condensation is the feedback schedule. `CyclicDigraph::condensation`
  orders directed strongly connected components, sources before sinks, with
  minimum dense-id tie breaks. `CyclicMixedGraph::condensation` reads the
  same directed edges. A bidirected edge is latent confounding and does not
  join a component. An acyclic condensation does not license ordinary DAG
  adjustment inside a feedback block. The schedule is a construction-time
  dense index. ADR 0005 still holds: serialized artifacts carry variable
  schemas and lag metadata, and `(VariableId, Lag)` identities are unchanged.
  `Lag(0)` remains contemporaneous.

### Retention

Extends ADR 0011. Retention is an execution fact. It rides the execution
identity of ADR 0022 and does not add an identity domain.

- Semantic caches obey `CachePolicy` and `CacheBudget` on
  `ExecutionContext`. A cache hit preserves the scientific identity of the
  retained product and does not grant a new license.
- Score tables, folds, posterior draws, and credible-draw vectors stay on
  the layer that produced them. They are not identification products.
- Published endpoints are the values already computed. Dropping an
  in-process draw vector does not rewrite those endpoints, and the response
  wire continues to omit the vector.
- An optimization that retains fewer indexes or copies must preserve
  replicate rules, seeds, cancellation, and stopping. ADR 0011 forbids a
  silent change to statistical semantics.

### Thread ownership

Extends ADR 0011.

- `ExecutionContext` is the only thread-budget owner. Core algorithms do
  not create a global pool, an implicit global RNG, or a process-wide
  parallelism setting.
- Parallel maps run on `std::thread::scope` through `map_indexed`. Nested
  work takes `serial_inner`, so an outer map that already fills the budget
  does not start another pool. Results return in index order. A failure
  returns the lowest-index error and skips later work.
- The default user count is `available_parallelism` clamped to 16
  (`DEFAULT_USER_THREAD_CAP`). An explicit thread count may exceed that cap.
  `threads = 1` is a serial pin.
- An Antecedent-owned Rayon pool is not part of this contract.

### Data backends

Extends ADR 0004. Durable array bytes stay Arrow IPC under ADR 0002.

- Public causal APIs expose library-owned views (`TableView`, `ColumnView`,
  matrix views). Arrow crate types and any relational engine stay behind
  adapters.
- The Arrow C Data Interface is an in-process import with an explicit
  lifetime. It is not a persistence format and it is not cross-process
  sharing.
- Owned `f64` columns keep the invariant that a row is invalid exactly when
  its value is `NaN`. A backend that distinguishes null from a valid `NaN`
  compiles that policy before a kernel sees the column.
- Content identity of owned storage remains `antecedent.data.storage.v1`.
  A projected or lazy plan does not replace that digest.
- Polars, DataFusion, and DuckDB are not backends of this contract. Adopting
  one requires a later ADR, keeps the public view types, and preserves the
  platform floor in ADR 0007.

### Artifact and API migration

Extends ADR 0002, ADR 0018, ADR 0019, and ADR 0021.

- The stable durable format remains `FormatVersion { major: 0, minor: 5 }`.
  Readers migrate sources 0.1 through 0.5 toward that format. An unknown
  version fails. This decision does not advance the format.
- ADR 0019's move to 0.3 and ADR 0021's move to 0.4 remain historical steps
  inside that registry. Response and temporal-response wire types stay the
  versioned types those ADRs introduced.
- Package version and artifact format stay independent. A release note
  records a package bump. A format bump is a migration plus a superseding
  ADR, as ADR 0018 requires when the format freeze itself changes.
- Public API breaks are explicit. Export returns container bytes, scientific
  readings stay immutable, and a removed name is not kept as a silent alias.
  Adding a public companion record follows ADR 0022's additive rule and does
  not by itself bump the artifact format.

## Consequences

- A change to a derive-key, the uncertainty vocabularies, `cyclic.solution.v1`,
  thread ownership, the public data view, the `f64` missingness rule, or
  `STABLE_FORMAT` requires a superseding ADR.
- Rust 1.85, edition 2024, and CPython 3.11–3.14 stay as ADR 0007 decided
  them. CI and wheel builds keep those floors.
- Existing accepted ADRs stay the record of the decisions they made. This
  ADR names the internal versions those decisions now pin.
