# Integrated Rust facade review

Owner: facade reviewer; integration owns formatting, Cargo validation, gates, API
records and commits. This review traces complete consuming workflows in
`crates/antecedent`, with coordinated fixes in graph and identification owners
where the facade previously dropped caller context. It does not claim an
independent scientific rederivation or line-by-line review of every family.
[Validation](validation.md) is the authority for executed checks and their source
snapshots. Source implementation and a prior passing run are not substitutes for
verification of the final source.

## Ownership and workflow coverage

| Boundary | Actual ownership and disposition |
| --- | --- |
| `lib.rs`, `prelude.rs`, `analysis/mod.rs`, `analysis/execute/mod.rs` | Preserve existing public study/query/data/result exports. Private prepared state remains separate from the public hidden configuration dereference target. Additive inspection and contextual graph APIs have concrete callers and require fresh public API comparison. |
| `planner/{binding,lowered,plan_view}.rs`, `analysis/execute/compile.rs`, `analysis/prepared/{prepare,prepare_identification}.rs` | Strategy fallback remains with `PlanBinding`/`RouteDefaults`. Compile retains the lowered program and identification; prepare installs those products. Statistical lowering, physical admission and actual execution remain distinct owners. |
| `analysis/prepared/{execution,execution_accessors,info}.rs`, `sealed/`, `sealers/` | The closed execution enum and small private operation trait retain family-specific work. Sealer selection and admission remain explicit; no generic wrapper is counted as a concrete family operation. |
| `analysis/{contract,contract_identity}.rs`, `contract/{posterior_programs,program_wire,result_wire}.rs` | Cheap inspection reads no prepared identification. Prepared and executed contracts reuse scientific programs separately from snapshot-bound payloads; independent artifact consumption still verifies the existing digest chain and executable sections. |
| `analysis/prepared/{estimate,lifecycle,operations}.rs`, `analysis/recalc_{dr,receipt,static}.rs` | Estimate, compatible replacement, numerical reconfiguration and recalculation have explicit owners. Sample replacement shares one transaction boundary; summary/period refresh retain their different modality contracts. |
| `analysis/execute/checked_*.rs`, `analysis/proof_support.rs`, `strategy_table/dispatch.rs` | Checked operations verify retained proofs before replaying them. Necessary unsealed replay receives the construction context. Query, graph, scientific claim and numerical procedure checks retain their refusal order. |
| `analysis/execute/{temporal_path,temporal_class_point,panel_path,temporal_class_posterior_response,temporal_pag_mediation}.rs`, `analysis/identification_cache/temporal_class.rs` | The actual caller or posterior worker context reaches class completion searches and checked proof validation, including compile-time window/probe retention. Resource/cancellation failures cannot become scientific unidentified mass. |
| `analysis/helpers.rs`, `result.rs`, `error.rs`, `route_guards.rs` | Keep method-specific effect/response/posterior uncertainty and typed refusal distinctions. Shared rules consolidate concrete repeated responsibilities without replacing scientific methods with one generic procedure. |

## Prepared inspection and producing laws

`contract/posterior_programs.rs` previously prepared full sampled atom handles
under a fresh default context while constructing a posterior contract. It built
empirical providers and discarded them. Per-atom scientific programs now belong
to the original preparation attempt, its context and preparation budget.
`SampledPreparedState` retains immutable program payloads and frozen canonical
query bytes. Contract publication borrows them and does not compile, identify,
fit, integrate or bind atom providers. Identification refusals retain their
existing per-atom scientific disposition.

Frozen-query guards use the existing artifact query codec
(`causal_query_to_wire` followed by canonical CBOR), preserving floating-point
bit distinctions. Frozen graph guards use the existing graph identity wire,
including posterior atoms, weights and temporal metadata. There is no parallel
query or graph encoder.

Functional effect/path clicks now call checked rebinding rather than preparing
another causal program. Distribution/effect/path operations retain the digest
of the storage that produced their empirical laws. That digest covers typed
values, validity, analysis masks and weights; schema compatibility is checked
independently. Successful refresh installs the exact rebound operation through
the existing click notes, so publication reads the producing laws rather than
scanning again or publishing stale providers. Distribution binds its provider
once and executes that binding.

The mutation audit traced every prepared-state constructor/replacement. Sample
replacement changes rows and row-bound evidence. Numerical AIPW/DML/DR changes
admit only their existing average-effect families. Bayesian inference rebinding
admits its checked operations. Score retargeting stamps a temporary result study
without changing the retained query; posterior path/nested/distribution routes
cannot enter score retargeting. Canonical frozen-target checks remain an
additional defense against future internal mutations.

## Transactional refresh and configured plans

`prepared/lifecycle.rs::commit_refresh` checks cancellation before work and
before replacing the handle, after execution and final contract publication.
Tabular, series, panel, multi-environment and configured estimator refresh use
this shared boundary. A failed candidate preserves the old study, operation,
program/contract caches, scores, physical plan and temporal invalidation log.
A cancelled mutating refresh cannot publish a usable partial bootstrap; ordinary
standalone estimates retain their existing partial-result semantics.

Configured refresh also previously changed the executing DML/DR or AIPW options
without replacing the lowered fold/resampling plan or resource quote. It now
stages the new data, re-lowers numerical work through
`lowering::reconfigure_retained`, requotes admission and reseals the operation,
including its physical-plan reference, before executing the transaction. Reuse
requires a nonempty matching scientific retention key and exact retained static
proof seal; shared identification retention also checks canonical compiled
product identity. It never enters a public builder or repeats identification
for compatible settings.

`install_numerical_spec` follows existing estimator ownership: configured
overlap stays in the estimator specification, and the resolved study replicate
count comes from `EstimatorSpec::bootstrap_replicates()`, as at initial build.
This removes duplicate resolved-setting drift in inference/artifact bindings.
Causal targets and identification remain unchanged. DML fold changes preserve
the existing artifact program identity while changing inference binding and
lowered statistical identity. AIPW bootstrap changes can also change existing
artifact program content; full comparison with fresh configuration verifies the
correct identity domains rather than forcing one digest to remain stable.

The candidate clone shares immutable data/program arcs and leases; it does not
copy source rows. Re-lowering numerical configuration is intentional. It is not
an additional scientific compile or fit.

## Concrete retained family operations

The strengthened retained-route gate exposed real wrapper gaps. They are fixed
by concrete operations, rather than exempted from the gate:

- `CheckedStaticCrossWorldOperation` in `execute/checked_static_cross_world.rs`
  serves `CounterfactualClass`, `PathSpecificCpdag`, `PathSpecificMixed` and
  `PathSpecificPag`. Explicit sealers call each actual class/cross-world/path or
  mediation family. Mixed functional programs/providers are prepared once,
  rebound through checked reuse and retained by refresh. Public inspection gains
  `RetainedOperation::StaticCrossWorld` (`static_cross_world`).
- `CheckedClassAttributionOperation` in `execute/checked_family_routes.rs`
  replaces the generic route wrapper for static and temporal graph-class
  attribution and directly executes those families on replacement data.
- `CheckedTemporalAdmgOperation` directly executes temporal ADMG contrasts,
  derivatives and mediation, temporal PAG mediation and their supported lagged
  posterior counterparts. Their distinct scientific/refusal/uncertainty
  semantics remain separate.

`CheckedScientificRoute` shares only the frozen-target/program guard. Changed
canonical query/graph or swapped lowered program refuses before computation.
Its entry guards remain in the consuming entries. Later private phase helpers
separate proof selection, native component estimation/refutation and result
publication without removing the actual family computation from retained-route
traversal. A zero retained-route baseline was observed before the final context
and responsibility changes; the strengthened gate must be rerun on final source.

## Caller context and checked proof construction

The complete temporal class chain now supplies the caller context to configured
identification. Posterior atoms use their actual map worker context. Contextual
calls bypass the process envelope memo, so a warm entry cannot bypass another
caller's cancellation, memory pool or disabled caching. Temporal PAG mediation
passes context through retained window compilation and each component proof;
cross-world probe retention is fallible for the same reason.

Static and temporal completion owners share
`generalized::check_completion_stop`: memory exhaustion and cancellation raise
typed non-scientific errors. Declared operation/depth/completion caps retain
partial-envelope semantics. Scientific non-identification retains its mass;
interruption cannot be converted into unidentified mass or a deferred proof.
Completion enumeration is deterministic and consumes no RNG, so supplying a
context does not change member order, weights or successful scientific bytes.
Static class and MAG-candidate context ownership is coordinated with the methods
reviewer; [the context record](context_ownership.md) describes the full chain.

Unsealed Bayesian basis, robust, DAG ATE/CATE, specialist and cell-AIPW proof
replay now receives the constructing context. Existing premise records carry it;
`CheckedCompileContext` groups the existing physical plan with the borrowed
context for constructors already at their argument boundary. Valid retained
proofs are verified without replay. Bayesian static mediation also reuses its
exact physical-graph/static-proof seal before deriving an unsealed proof.
Static mediation and static response constructors that only consume existing
proofs do not gain unnecessary context parameters. Tiered cell proof validation
is a fixed closure/separation check, not completion/subset enumeration; its
fallback observes cancellation explicitly.

`TemporalPagCompletionSampler::with_budget` is an intentional additive graph API
used by temporal attribution. It synthesizes exactly the same default stationary
window as `new`; `new` delegates with the established default search context.
The new entry supplies caller search limits and context without changing the
existing API's default behavior.

## Verification status and explicit limits

Earlier integration runs exercised the inspection/provider mutation regressions,
retained family lifecycle/artifact cases, configured-plan fresh equivalence and
rollback, population retargeting, and late refresh cancellation. Their results
are recorded in [validation](validation.md). Those passing results precede the
latest context/proof and private responsibility changes and do not certify the
final source snapshot.

The following newly added context regressions await integration execution of the
current source unless a later result is recorded in validation:

| Regression source | Actual boundary exercised |
| --- | --- |
| `antecedent-identify/src/class_envelope_cache.rs` | Warm real memo, contextual cancellation/tiny admission, disabled-cache recomputation and no contextual process retention. |
| `antecedent-identify/src/temporal_pag_path_specific.rs` | Native mediation-window cancellation/admission and unchanged enumeration under another caller seed. |
| `antecedent-graph/src/temporal_pag_completion.rs` | Default-window constructor member equality, caller cancellation and memory stop. |
| `tests/checked_temporal_class_effect_route_evidence.rs` | CPDAG/PAG pulse and sustained preparation after a warm process envelope; enabled/disabled caching, exact contracts/numerics, budget and progress cancellation. |
| `tests/temporal_pag_mediation.rs` | Actual retained mediation preparation/estimate, budget/cancellation and fresh contract/numerical equivalence. |
| `execute/checked_cell_aipw_response.rs`, `prepared/tests/static_operation.rs` | Unsealed cell-response and Bayesian basis construction use the actual cancellation token and memory pool. |

No numerical/reference measurement, artifact schema/version or scientific
identity encoder is repinned here. Four physical-plan report updates belong to
the separately documented aggregate resource correction; they are not new
measured memory evidence. Final lint, API, lifecycle/resource/artifact gates and
native/Python execution remain integration-owned.

Resource guarantees remain bounded by the actual owner: compile quotes are
estimates, live leases are reservations and observations are measured facts.
Neither context propagation nor a quote proves byte-accurate RSS for arbitrary
application providers. Provider `SignalLimits` and runtime admission obligations
remain. Context-free standalone APIs keep their documented defaults; their
bounded process memo is not described as caller-admitted retained program
memory. See [resources](resources.md) and [context ownership](context_ownership.md).

Writable declaration/DTO fields and public hidden configuration targets remain
for compatibility; checked construction and consuming fallible operations own
execution invariants. Breaking encapsulation changes belong to the API pass.
Method-specific inference, stream identities and uncertainty remain separate.
Empirical weighting semantics are unchanged; mutation tests compare against
cold reconstruction rather than granting broader weighted-route support.
