# Checked single-source transport and exact execution

The scientific reference is Bareinboim and Pearl, *A General Algorithm for
Deciding Transportability of Experimental Results*,
[arXiv:1312.7485v1](https://arxiv.org/abs/1312.7485v1), Figure 5 and its s-hedge
obstruction theorem. The evidence setting is one selected source's complete
experimental family and the target observational law. Completeness applies to
that theoretical setting, conditional on completing the algorithm within the
configured resources. It is not completeness for an arbitrary finite catalog,
restricted experiments, or combined sources.

## Identification and checking

`antecedent-identify/src/sid.rs` first attempts ordinary target-only ID, including
ADMGs. Source evidence is needed only if that attempt fails. Both attempts use
existing prepared graph operations and carry kernels in original coordinates;
variables outside an induced district remain external kernel parameters.

| Figure 5 branch | Typed rule | Transformation and consuming evidence |
| --- | --- | --- |
| 1 | `Marginal` | Sum non-outcome variables from the current kernel. Exhaustive branch conformance. |
| 2 | `Ancestors` | Restrict graph and kernel to outcome ancestors. Three-node SCM enumeration and branch conformance. |
| 3 | `Enlarge` | Add irrelevant interventions in the mutilated graph. Average the resulting irrelevant parameter against the normalized carried-kernel marginal. `original_coordinates_and_intervention_enlargement_are_preserved` and branch conformance. |
| 4 | `Districts` | Identify each postintervention district, multiply, and sum nuisance coordinates. Six-node independent front-door districts. |
| 7 | `Factor` | Extract the topologically ordered district kernel from the current kernel. Branch conformance and front-door oracle. |
| 8 | `Recurse` | Recurse inside the containing district with its extracted kernel. Six-node recursive SCM oracle. |
| 10 | `Source` | Check selection separation in the mutilated graph before substituting the source experimental law. Branch conformance and source-snapshot walkthrough. |
| 11 | checked s-hedge | Verify nested selected C-forests, common roots, treatment intersection and ancestral conditions. Exhaustive four-node conformance and witness mutation tests. |

All referenced Rust tests live in
`crates/antecedent-identify/tests/transport_exact_scm.rs`. The four-node test
covers all 65,536 combinations of ordered directed edges, bidirected edges and
selection targets, requires a checked positive or negative outcome, and asserts
that every positive recursive rule was consumed. Numerical tests independently
enumerate finite SCMs; they do not use the identifier to generate oracle truth.

For the obstruction construction, the failed subquery's outcome coordinates
are roots in both forests: all outgoing edges from those roots are omitted.
Each other node retains one outgoing edge toward a child in its forest; smaller
forest nodes choose within the smaller forest. Ancestry restriction ensures
these paths terminate at the roots. Bidirected connectivity is preserved. The
independent checker validates the resulting witness against the original graph,
query, selections, source population and theoretical evidence scope before a
negative result can be published. An unchecked obstruction, cancellation or
exhausted budget never becomes an impossibility claim.

Positive checking validates each recorded local premise and expression
transformation; it does not invoke the recursive solver. `SidDerivationRecord`
and `TransportProofWire` preserve these premises and the original expression
DAG. `SHedgeRecord` preserves the negative witness and its scope. Historical
negative-named records remain `NotCertified`, not proven negatives.

Catalog binding is a separate bounded search: target-first sID, pretreatment
S-admissible standardization, then catalog-available source-first sID. The
standardizer is checked as a supplementary do-calculus rule and requires
observed nondescendants of treatment. It considers at most 20 candidate
coordinates. Diagnostics report searched alternatives, missing factors,
exhaustion, and proposed future experiments separately from missing existing
evidence. Successful binding does not license arbitrary finite-catalog
completeness. Memo keys include graph, selections, query, current kernel,
population and evidence scope; step, depth, memory and cancellation controls
apply to solving and verification.

## Exact laws and physical execution

`ExactDiscreteLaw` is an immutable dense Cartesian table with explicit ordered
axes, levels, population, regime, intervention world and snapshot identity.
Every cell is supplied, including structural zeros. Duplicate axes/levels,
incomplete coverage, nonfinite or negative entries, and normalization failures
are errors. Default absolute/relative tolerances are `1e-12`/`1e-10`; tolerances
are declarations, not permission to repair the law. Python uses numeric-coded
finite domains; native providers also support the core discrete value types.

Preparation verifies the certificate and catalog, resolves every leaf and
intervention value, and preflights checked work and conservative memory bounds.
The physical plan retains the original functional separately. Dense world and
level indexes avoid repeated search; fully assigned masses use direct lookup.
Scoped elimination restores bindings, shares support buffers, and memoizes
compiled nodes by projected free-variable assignments. A cache belongs to one
immutable provider-bound execution and is shared across target atoms, never
across replacement snapshots. Limits can conservatively reject an execution
that a more aggressive optimizer could fit.

Support is a factor-and-assignment obligation. An undefined conditional on a
supplied-law null event can be deferred only as a bounded stochastic extension.
A product with an explicit zero multiplier is then zero for every such
extension. The checked rule records that fact. A positive-weight undefined
conditional fails; arbitrary `0/0` is never assigned zero. Ratio denominators
and conditional support findings remain attached to their original expression
nodes and concrete assignments.

Execution returns the complete target law and checks its probability bounds
and normalization without clipping or renormalizing. Numeric means use
compensated summation; contrasts compare compatible output coordinates. There
are no sampling standard errors or intervals. Cancellation, support failures,
numerical failures and exceeded budgets publish no partial result.

## One prepared lifecycle

The common native handle is `PreparedStudy<S>`: its default state preserves the
existing sampled workflow, while `ExactPreparedState` carries immutable exact
laws and checked transport authority. `StudyBuilder::exact_transport` prepares
that specialization. This keeps sampled `EffectEstimate` uncertainty out of
exact results without inventing a second lifecycle. Python's common `prepare`,
`PreparedAnalysis`, inspection/transformation interfaces and `load` dispatch to
the corresponding native state. The legacy trial-IPW route remains unchanged.

Eight identity layers distinguish target, observation, inference binding,
identification, identification product, program, snapshot and execution.
Preparation retains the native query, graph, selections, evidence contract,
proof and certified functional. Catalog entry order is canonical; Python
regime labels are canonically remapped together with dependent references.
Historical artifact identities are preserved on decode.

Inspection is metadata-only: theorem scope, assumptions, factors, bindings,
formula, supported operations, identities and cached support findings. It does
not fetch data or invoke callbacks. Transformation previews use the shared
invalidation contract. Compatible snapshot replacement retains identification
and invalidates execution claims. Structural/evidence changes require another
preparation. Explicit refresh constructs and executes a candidate before
publishing it, so failure preserves the previous valid state. Stale request or
result identities are rejected. Python display objects cannot substitute for
native certificates or authorize export.

Exact artifacts use a distinct versioned envelope and contain checked proof,
graph, catalog, embedded laws, settings, concrete assignments, identities,
factor support and all four reasoning slots. Independent consumption validates
local proof premises and bindings, then recomputes the embedded exact-law claim
and compares its identities, support and reasoning. It neither reruns
identification nor fits or fetches data. These are self-consistency and
scientific-validity checks, not authentication of who supplied a law.

`examples/python/transport_exact.py` is a tested source-snapshot walkthrough:
identify → inspect → prepare → estimate → replace → refresh → export → load.

## Statistical tables (T6)

Exact supplied tables remain T4: no sampling uncertainty. Empirical tables are
T6.1: one complete-observation frequency joint per `(population, regime, intervention
world)` on catalog-declared finite domains. Several certified leaves from the
same regime project that joint. Empty empirical conditioners are
`sampling_zero`, never a hidden `0/0 = 0`. Dirichlet smoothing and learners
are named later choices; they are not the default.

T6.2 is a joint IID outer bootstrap on the same prepared handle
(`StatisticalPreparedState`). Independent studies share one dataset replicate
across every leaf and treatment-grid point. The grid is evaluated within one
outer resampling transaction; original replicate IDs align contrasts. Unknown, linked-unit, and clustered
dependence keep identification and withhold the interval. Failed replicates
are counted; intervals are pointwise percentile intervals of the plug-in
functional. Trial IPW stays the separately licensed Dahabreh cell and does not
evaluate recursive sID.

`examples/python/transport_statistical.py` is the mixed-sample walkthrough:
inspect → prepare → estimate → replace one source sample → refresh → export →
consume. Consume recomputes the point from embedded fitted joints and does not
re-bootstrap. Version-2 statistical artifacts verify all identity layers,
replicate accounting, atom and mean percentile intervals, support and reasoning.
Raw sample rows are not embedded; re-estimation requires an explicit refresh
with samples. Earlier T6 artifacts lack these checks and must be regenerated.

Missing observations require an explicit missingness model, which the current
empirical provider does not supply. There is no implicit complete-case deletion.
Nominal pointwise uncertainty is separate from an execution-specific calibration
binding; unbound executions report that status explicitly. See the
[T6 review](../audits/transport-t6-review.md) for fixes and calibration scope.

## Complementary sources and retained grids

The single-source theorem scope above remains unchanged. The shared engine now
also has an explicitly scoped [multi-source and retained-grid path](transport-meta-grid.md).
Historical payloads retain their original identities and scientific meanings.
