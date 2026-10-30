# Retained Gaussian compilation and physical selection in 3.0

The production checked linear operation retains its scientific target,
identification product, world obligations and scientific/physical operators.
`PreparedStudy::compiled_linear_program()` inspects that binding without fitting.
Public compilation retains the checked identification, numerical preparation and
scientific program as one product. Preparation consumes that exact product rather
than identifying or preparing the design again. Both the physical plan and the
prepared handle expose the same scientific program. Physical plans now contain a
private compiled product and must be obtained through the compiler rather than
constructed with a struct literal. All four Gaussian examples
execute public prepared handles; the separate demonstration
executor, reconstructed builder and lifecycle cache have been removed.

A deterministic linear point fit is shared through synchronized immutable
ownership. Changed data replaces the fit, and a changed kernel policy requires
another fit. Stochastic uncertainty still executes under caller policy.

Physical planning invokes the capability-filtered cost chooser and retains its
selection and rejection reasons. Quotes use the selected design's row count,
actual contrast width, requested resampling work, worker scratch and retained
output. Raw numeric adjustment uses the direct preparation path; categorical
adjustment requires materialized contrast columns. The selected preparation is
what the checked estimator consumes. A full-row adjustment fit does not advertise
a streamed summary as interchangeable with its influence and uncertainty outputs.
The existing weighted Gaussian capacity reduction retains its streamed choice.
Lazy and accelerator alternatives remain explicitly unavailable.

Preparation reserves its design before allocation, and the compiled owner keeps
that reservation. Execution reserves workspace under the caller's current worker
policy. Frozen fits and returned results share the fitted buffer reservation; each
returned result also reserves its own metadata. Dropping the builder or cancelling
does not release memory still owned by a live result.
`StudyResult::retained_admitted_bytes()` exposes that output reservation. Refresh
rebinds the source digest (inspectable through `source_digest()`), design dimensions and resource plan without repeating
identification. Failed refresh leaves the old handle intact.

These are conservative allocation quotes and shared reservations, not process RSS.
Already resident caller-supplied table buffers are borrowed; their allocation
predates this execution account. Unknown measurements remain absent.

Public Gaussian and four-way behavioral tests replace demonstration-function-name
checks in the checked-execution registry. The accompanying dose-grid/cyclic
integration and verification evidence are recorded with the repair acceptance
tests. Library-wide work
unrelated to this repair remains open; this repair grants no new identification or
calibration guarantees.

The costed physical choice is made before preparing the checked design and is
passed to the estimator's actual assembly implementation. Numeric sources admit
the direct final-buffer gather and materialization alternatives; direct assembly
avoids intermediate column copies. Expanded categorical sources select the
materialized column implementation, whose temporary columns are included in the
admission quote. The checked preparation records the backend actually used.
Full-row adjustment refuses streamed and lazy assembly because its retained-row
uncertainty program requires those rows. Streaming remains available for the
existing sufficient-statistics capacity component, which has a different contract.

The retained recompilation recipe stores only the resource policy needed to bind
new rows. It does not retain caller progress callbacks in physical plans, so the
existing plan unwind-safety auto traits remain intact.

## Recalculation route ownership

Custom recalculation route names are now owned by their capabilities and receipts.
`RequestSupport` and `RefusalReason` carry `Option<Arc<str>>`; these types and
`RecalcCapabilities`, `StageStatus` and `StageEntry` implement `Clone` rather than
`Copy`. Convert a route literal with `.into()` and clone a value when it must be
retained after passing it to a planner. Python custom route strings continue to
work. Wire names, status text and canonical identity inputs are unchanged;
consuming distinct custom receipts no longer consumes a global intern-table slot.

## Sampled and exact counterfactual evidence retention

`SampledCounterfactualTransportEstimate.tables` and `.datasets`, and
`CounterfactualTransportPoint.evidence`, now use an immutable `RetainedSlice` rather than
`Vec`. They dereference to slices and support iteration, indexing, value comparison and cheap
cloning. A clone shares its immutable data and its admission reservation, including when a
collection is retained separately from its parent result. Code that requires an owned `Vec`
can explicitly copy the slice with `to_vec()`. These public field type changes are a Rust source
compatibility break; they do not change the numerical or identity contracts.

## Predictive assessment key provenance

New predictive assessment artifacts use version 2, which stores each observation key's
caller-minted provenance beside its digest. Native assessment identities also bind that
provenance; identities for exclusively ledger-backed keys retain their previous encoding.
Duplicate fitted keys merge provenance conservatively, so a caller-minted copy cannot be
lost during deduplication.

Version 1 artifacts remain readable when their overlap result can be checked without
ledger provenance. A stored version 1 held-out verification is refused because its key
section did not preserve that premise. Re-export keeps the legacy version; reseal from
the original typed observation identities to write version 2.

## Explicit imputation domains

An incomplete column whose observed values are only zero or only one now requires an
explicit binary or continuous domain. Python callers can pass
`MissingPolicy.multiple_imputation(domains=(("x", "binary"),))`; Rust callers can use
`MissingDeclaration::with_imputation_domains` or `Imputer::new_with_domains`.
Declared binary columns retain zero/one support. With only one observed class,
the empirical Bayesian bootstrap stays at that class; it does not assign probability
to an unseen class. Reports disclose this boundary and bind the domain and class
into the imputation assumptions identity. Joint-normal imputation still refuses
incomplete binary columns.

Rust struct literals for `MissingDeclaration`, `ImputationSpecWire`,
`ImputationAssumptions` and `ImputationReport` must supply their new domain fields;
prefer constructors where available. Empty domain declarations preserve earlier
wire bytes and identities. The separate graph-selection and joint-panel preparation
APIs also accept explicit domains.

## Python numeric report boundaries

Posterior constructors now reject invalid summaries and stored-draw shapes immediately,
using the same validation as artifact encoding. Reserved coefficient names must use
canonical `coef_<u32>` notation; malformed names no longer encode as coefficient zero.

Preflight JSON reports disclose nonfinite values through JSON-pointer paths in
`nonfinite_numeric_fields`, distinguishing them from absent optional values. Selective
discovery links carry a `tested` flag. Observation reports expose numeric status fields;
when a required Bayes factor is outside float range, `required_factor` raises
`CausalValueError` and `required_log_factor` retains the finite logarithm. These
reporting changes do not license parameter intervals.

## Cyclic solver diagnostics and numerical boundaries

The Rust `CyclicWorldOutcome::GaussianLaw` and
`CyclicWorldOutcomeAnswer::GaussianLaw` variants add optional `block_diagnostics`;
field patterns and constructors must account for it. `SolveEvidence` adds optional
`contraction_bound_provenance`. Python world answers expose block members, residuals,
infinity-norm condition numbers and whether those numbers are exact or estimated.
A uniqueness certificate does not assert good numerical conditioning. External
contraction evidence records when its proof relies on declared Lipschitz bounds.

Existing cyclic v1/v2 and external v1 artifact payloads retain their earlier schema.
Export omits these added diagnostics; a current consumer recomputes and returns them.
Request identities remain unchanged. Persisting diagnostics in the artifact requires
a future explicit format version.

Direct held-value queries now activate actions at the exact declared time, inclusive,
and return no value for nonfinite times. Mixed-scale Krawczyk and OU solves apply the
existing pivot criterion to an equilibrated dimensionless matrix, refusing scaling
that overflows or erases nonzero entries. Exhausted fitting steps no longer establish
convergence. Confounded IPW results disclose how many raw weights hit the existing cap.

## Sensitivity grids and predictive diagnostics

Sensitivity refuters reject empty partial-R² grids, values outside finite `[0, 1)`,
and pass thresholds outside finite `[0, 1]`. Valid strengths above 0.999 are used
as declared. Grid counts must fit the report's existing u32 count.

Target posterior-predictive checks report the inclusive empirical exceedance
fraction. The alpha verdict is an uncalibrated diagnostic convention; zero means
no exceedances were observed. Finite inputs that overflow a discrepancy now
refuse rather than returning an infinite statistic.

## Sampling work and worker admission

Rust execution contexts accept explicit exposure-probability and HMC work limits.
Exposure admission includes initial estimation and Full-suite redraws together.
HMC admission includes all chains and diagnostics, correlated-prior validation,
and absolute-scale OLS preprocessing. Context-free calls use finite defaults;
requested draw counts and sampling order are preserved.

HMC workspace growth is admitted for the call. Caller-owned capacity retained
after return is outside that temporary lease. Dense Cholesky and solve kernels
check cancellation at call boundaries. `moment_matched_normal_table_with_work_limit`
allows an explicit joint-table ceiling; workflow contexts supply cancellation.
These operational limits add no calibration claim.

### Indexed maps return operational failures

`ExecutionContext::map_indexed`, `map_indexed_with_scratch`, and
`map_indexed_with_scratch_if_free` now return `MapExecutionError<E>`. Match
`Work(error)` to handle the original task error and `Admission(error)` to handle
cancellation, admission refusal, or a worker-start failure. Serial maps acquire a
worker reservation too; nested maps reuse their enclosing worker reservation. The
if-free form can reduce its worker count, but refuses when no worker is available.
No work runs after a refused reservation.

`antecedent_prob::mcmc_summary_parallel` now returns `Result<McmcSummary,
AdmissionError>` so cancellation and worker-start failures are reported rather than
silently replaced by serial computation. The non-parallel `mcmc_summary` is unchanged.

Synchronous numerical helpers reuse an enclosing map worker’s CPU reservation from
the same admission pool. Their scratch bytes remain separately admitted. Direct
serial calls still reserve one CPU slot. This does not authorize additional threads.

## Execution contract access

Rust callers read a `StudyResult` execution stamp with `result.executed_contract()`.
The returned reference is read-only; execution stamps are attached by the engine
and can no longer be replaced or mutated by callers. Match `None` when no prepared
execution produced the result. To run another validation suite, request that suite
through the study or prepared execution rather than changing the result's stamp.

The execution stamp retains both the full snapshot identity and the engine-computed
base snapshot digest used by cheap inspection. Replay evidence remains part of the
full identity. External `ExecutedContract` literals must supply the new
`inspection_data_snapshot` field.

## Python sampling controls

`Execution` accepts `exposure_probability_work_limit` and `hmc_work_limit`. Direct
`interference.estimate` also accepts `work_limit` and cancellation. Exposure limits
cover the initial estimate and Full-suite redraws together; independent calls use
independent ceilings. Native input-copy admission precedes strided-array conversion,
after PyO3 argument extraction.

Python family-table construction runs under its execution scope and reports
cancellation and resource refusals. These controls preserve requested draw counts
and add no calibration claim.

## OU Lyapunov residuals

OU covariance and covariance-direction solves check a dimensionless, max-entry
relative residual against the absolute terms of `M X + X Mᵀ + Q`. The residual
reported for a failed consistency check is now dimensionless. Binary scaling
avoids overflowing intermediate products and removes the previous absolute unit
floor. Symmetrization preserves finite large and subnormal entries. The Schur
separation threshold remains unchanged; a small residual is a consistency check
and does not establish forward accuracy for an ill-conditioned equation.

## GES search schedule

GES now runs the forward Insert phase followed by backward Delete, matching
Chickering (2002) and the pinned causal-learn oracle. The previous additional
Turn phase is no longer executed. Search diagnostics and the algorithm record
state this schedule. GES configuration identities, including bootstrap wrappers,
carry `ges.schedule.insert-delete.v2`; historical GES declarations require a new
search. Other discovery configuration identities retain their existing encoding.

Reference: [Chickering, Section 1](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf).

## Checked discovery declarations

`DiscoverySpec::query(variables)` now returns `Result<DiscoverQuery, QueryError>`.
Propagate the result with `?` before passing the query to another API. Invalid
CI significance levels and temporal CI lags whose doubled window exceeds `u32`
are refused before configuration identity generation. `StudyBuilder::discovery`
retains this typed error and returns it from `build`; a later valid discovery
declaration replaces the pending error. Static CI searches require finite
`0 < alpha < 1`; temporal CI searches also require `2 * max_lag` to fit `u32`.
Generic structural constraint validation and non-CI search options retain their
existing semantics.

## Interface calibration and evidence forwarding

An `Inherited` interface calibration label now requires the output to preserve
its connected upstream quantity, semantic identity, support, uncertainty,
measurement assumptions, and evidence sources. The output may offer a subset
of the upstream operations. Input requirements can be less specific than the
upstream declaration; calibration follows the actual supplied output. A changed
quantity requires its own measured calibration record.

`ProducedPort::with_evidence([])` now declares evidence as unknown. Directly
constructing `EvidenceBasis::Declared` with an empty set also fails a connection
that requires declared or disjoint evidence. Declare the actual sources before
requesting these dependence guarantees.

Inherited measured outputs must retain an interval or posterior operation, as
directly measured outputs already require. Native `check_composition` now refuses
to infer disjoint observations from distinct source IDs alone. Use
`check_composition_with_ledger` to establish disjointness from observation sets.

Interventional distribution validation now checks every target in an intervention
sequence, including nested sequences. An outcome or observational conditioning
variable that appears anywhere among those targets is refused with the existing
typed overlap error, just as for a single-target intervention.
