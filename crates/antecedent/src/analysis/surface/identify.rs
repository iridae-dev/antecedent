//! Identification results, estimands and the transport and recovery identifiers, for reading
//! why a query is or is not identified under a declared structure.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_identify::{
    BinaryIvLaw, GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier,
    GraphIdentificationCase, IdentificationEnvelope, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, NotCertifiedCertificate, PopulationFactor, ProbabilityMass,
    TemporalMediationIdentifier, TransportCertificate, TransportFormula, TransportIdentification,
    TransportIdentifier,
};

pub use antecedent_identify::sid::{
    SHedgeCertificate, SHedgeRecord, SelectionForest, SelectionForestRecord, SidDerivationRecord,
    SidStepRecord, verify_s_hedge,
};
pub use antecedent_identify::{
    BoundTransportFunctional, ClassicalTransportDerivation, ClassicalTransportQuery,
    ClassicalTransportResult, SidLimits, verify_classical_transport,
};

pub use antecedent_identify::CatalogTransportResult;

/// Identify under the explicitly declared classical experimental information family.
/// Ordinary target ID is tried first. No finite-catalog completeness is implied.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::identify_classical_transport`].
///
/// # Errors
/// Invalid coordinates, resource exhaustion, cancellation, or failed verification.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn identify_classical_transport(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ClassicalTransportQuery,
    limits: antecedent_identify::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::ClassicalTransportResult, antecedent_identify::IdentificationError>
{
    antecedent_identify::identify_classical_transport(diagram, query, limits, ctx)
}

/// Try target-first sID, then catalog-aware source/target recursive choices.
/// This bounded two-strategy search does not inherit classical completeness.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::identify_catalog_transport`].
///
/// # Errors
/// Invalid input, exhausted shared search budget, or cancellation. Never a negative theorem claim.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn identify_catalog_transport(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ClassicalTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_identify::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::CatalogTransportResult, antecedent_identify::IdentificationError> {
    antecedent_identify::identify_catalog_transport(diagram, query, catalog, limits, ctx)
}

/// Compute sharp Balke–Pearl-style bounds on `E[Y(1)-Y(0)]` without monotonicity.
///
/// The implementation solves the exact 16-type latent response-program. It assumes randomized
/// instrument assignment/exogeneity and exclusion through the response-type model; those claims
/// must be recorded by the caller in the enclosing identification artifact.
///
/// Returns an error naming the Balke–Pearl instrumental inequality specifically when the
/// response-type program is infeasible, which is distinct from (and reported separately from)
/// malformed input — the latter is rejected earlier by `BinaryIvLaw::validate`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::binary_iv_ate_bounds`].
///
/// # Errors
///
/// The error of [`antecedent_identify::binary_iv_ate_bounds`], unchanged.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled IvBounds route, which calls this kernel function with the same inputs"
)]
pub fn binary_iv_ate_bounds(
    law: antecedent_identify::BinaryIvLaw,
) -> Result<antecedent_core::response::IdentifiedSet<f64>, antecedent_identify::IdentificationError>
{
    antecedent_identify::binary_iv_ate_bounds(law)
}

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Classical meta-transportability (Bareinboim & Pearl 2013, Figure 5).
pub use antecedent_identify::MetaTransportQuery;
/// Conservative structural transportability identification.
pub use antecedent_identify::MissingEvidenceCertificate;
/// Mixed-source rule engine and the decision that runs it.
pub use antecedent_identify::MixedSourceDecision;
/// `TemporalCpdag` / `TemporalPag` identification envelopes.
pub use antecedent_identify::TemporalCompletionGraph;
/// Self-censoring witness and stored-recovery verification.
pub use antecedent_identify::verify_recovery_witness;
/// Contract validation for the bounded single-source z-transport setting.
pub use antecedent_identify::{
    BoundZTransportFunctional, ComponentFactorization, TwoSourceZTransportDecision,
    TwoSourceZTransportQuery, ZTransportDecision, ZTransportDerivation, ZTransportLimitsReceipt,
    ZTransportMissingEvidence, ZTransportObstruction, ZTransportOutcome, ZTransportQuery,
    ZTransportResult, ZTransportSourceSpec,
};
/// General finite counterfactual transport on a static semi-Markovian graph.
pub use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, CounterfactualTransportMissing,
    CounterfactualTransportProblem, CounterfactualTransportQuery, CounterfactualTransportSource,
    counterfactual_transport_refusal,
};
/// ADMG conditional transport on the classical complete-source family.
pub use antecedent_identify::{
    ConditionalObstructionCandidate, ConditionalTransportDecision, ConditionalTransportQuery,
    admg_conditional_refusal,
};
/// Self-certifying two-model witnesses for ADMG conditional transport.
pub use antecedent_identify::{ConditionalWitnessRecord, WitnessKernelRecord, WitnessModelRecord};
/// Identification result types.
pub use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};
/// Identification errors.
pub use antecedent_identify::{IdentificationBudget, IdentificationError};
/// Bounded mixed-source proof search.
pub use antecedent_identify::{
    MixedQuantity, MixedSourceDerivation, MixedSourceQuery, MixedStageRecord, render_quantity,
};
/// The bounded multi-source limited-experiment (mz-transportability) contract.
pub use antecedent_identify::{
    MzSearchRecord, MzStageRecord, MzTransportDecision, MzTransportQuery, MzTransportRoute,
    mz_transport_refusal,
};
/// Exact binary causal observation recovery.
pub use antecedent_identify::{
    ObservationRecoveryQuery, PartiallyObserved, RECOVERY_DEFAULT_LIMITS, RecoveredEffectQuery,
    RecoveryDecision, RecoveryDerivation, RecoveryDetail, RecoveryError, RecoveryLimits,
    RecoveryWitness, WitnessMechanism,
};
/// Study planning over the mz-transport and mixed-source catalogs.
pub use antecedent_identify::{
    STUDY_PLAN_MEMORY_BYTES, StudyPlan, StudyPlanLimits, StudyPlanRoute, StudyPlanStop,
    StudySubsetOutcome,
};
/// Lower `TransportFormula` views into the shared expression arena.
pub use antecedent_identify::{bind_transport_derivation, lower_transport_formula};

/// Identify a bounded single-source restricted-experiment query under the
/// caller's search limits and execution context.
///
/// The registered surrogate formula and direct joint exchange have dedicated
/// local checkers. Other in-bound graphs, including admissible selection
/// diagrams, follow the recursive `TRz` reduction and are rederived on replay.
/// Use `decide_z_transport_with_catalog` when a negative result must be
/// distinguished from incomplete catalog evidence: a line-11 failure without a
/// catalog is `ZTransportResult::NotCertified`, not an obstruction.
///
/// The registered factorization is the four-variable graph `W→Z→X→Y`, `W→Y`,
/// bidirected `W↔Y`, `Z↔Y`, `Z↔X`, controllable `Z`, query `P(Y | do(X))`, and
/// no selection targets. Other graphs up to 12 observed and 4 controllable
/// variables publish the recursive derivation when search succeeds.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::identify_z_transport`].
///
/// # Errors
/// Invalid query coordinates, cancellation, or exhausted computation.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn identify_z_transport(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ZTransportQuery,
    limits: antecedent_identify::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::ZTransportResult, antecedent_identify::IdentificationError> {
    antecedent_identify::identify_z_transport(diagram, query, limits, ctx)
}

/// Identify with `μsID` Figure 5. Target-only ID has priority; scientific negatives
/// require the same independently checked forest obstruction in every source.
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::identify_meta_transport`].
///
/// # Errors
/// Invalid coordinates, cancellation, resource exhaustion, or invalid proof.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn identify_meta_transport(
    graph: &antecedent_graph::admg::Admg,
    query: &antecedent_identify::MetaTransportQuery,
    limits: antecedent_identify::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::ClassicalTransportResult, antecedent_identify::IdentificationError>
{
    antecedent_identify::identify_meta_transport(graph, query, limits, ctx)
}

/// Decide a counterfactual transport query on `problem` from the labelled
/// evidence of `catalog`.
///
/// Stages run in a fixed order, each only when the earlier ones do not decide:
/// the derivation over the supplied regimes; the derivation over the declared
/// family (supplied regimes plus every population's observational law and every
/// source's experiments on subsets of its controllable set), whose success is
/// `CounterfactualTransportDecision::MissingEvidence`; and the bounded witness
/// search over the declared family, whose verified success is
/// `CounterfactualTransportDecision::ProvenNonTransportable`. One
/// `SearchBudget` is charged by every stage, so the limits bound the whole
/// decision; a stop anywhere is `CounterfactualTransportDecision::Exhausted`,
/// never a verdict.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::identify_counterfactual_transport`].
///
/// # Errors
///
/// A malformed query or catalog
/// (`counterfactual_transport.invalid_query`,
/// `counterfactual_transport.invalid_catalog`), or a derivation that fails its own
/// check. Search limits are the caller's; memory is the execution pool or the
/// route default when the pool is unset.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn identify_counterfactual_transport(
    problem: &antecedent_identify::CounterfactualTransportProblem,
    query: &antecedent_identify::CounterfactualTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_core::search::SearchLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    antecedent_identify::CounterfactualTransportDecision,
    antecedent_identify::IdentificationError,
> {
    antecedent_identify::identify_counterfactual_transport(problem, query, catalog, limits, ctx)
}

/// Decide a bounded single-source z-transport query, and on a budget or
/// cancellation return the limits receipt rather than an opaque exhaustion
/// error, so callers can inspect the limits in force and what was consumed.
///
/// Every non-exhaustion outcome is a `ZTransportOutcome::Decided`; only a
/// budget or cooperative cancellation becomes `ZTransportOutcome::Exhausted`.
/// Malformed inputs and invalid catalogs are still returned as errors.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::decide_z_transport_inspecting`].
///
/// # Errors
/// Invalid or unsupported query, or an invalid catalog.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn decide_z_transport_inspecting(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ZTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_identify::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::ZTransportOutcome, antecedent_identify::IdentificationError> {
    antecedent_identify::decide_z_transport_inspecting(diagram, query, catalog, limits, ctx)
}

/// Search two sources separately on a shared causal graph.
///
/// One identifying source is enough, and the other source's catalog is not
/// required: both sources are searched, and a failure of one source does not
/// hide an identification by the other. Two line-11 terminals become one
/// obstruction only for one experiment family. A result that would mix a
/// factor from each source is refused by name. This does not call classical
/// meta-transport, which assumes every source can experiment on every variable.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::decide_two_source_z_transport`].
///
/// # Errors
/// A source population collides with the other source or the target, a
/// selection diagram is invalid, or neither source identifies and one search
/// failed (the first failure is reported).
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn decide_two_source_z_transport(
    graph: &antecedent_graph::admg::Admg,
    query: &antecedent_identify::TwoSourceZTransportQuery,
    catalogs: [&antecedent_core::EvidenceCatalog; 2],
    limits: antecedent_identify::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    antecedent_identify::TwoSourceZTransportDecision,
    antecedent_identify::IdentificationError,
> {
    antecedent_identify::decide_two_source_z_transport(graph, query, catalogs, limits, ctx)
}

/// Decide whether `P(X(1), O)` is recoverable from the named observed pattern law,
/// and, when `effect` is given, identify it downstream, all under one budget.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::decide_observation_recovery`].
///
/// # Errors
///
/// A typed `RecoveryError`: bounds, invalid query, unsupported mechanism,
/// missing margin, witness unavailable, handoff mismatch, effect not identified,
/// or a budget stop with its receipt.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn decide_observation_recovery(
    graph: &antecedent_graph::admg::Admg,
    query: &antecedent_identify::ObservationRecoveryQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    effect: Option<&antecedent_identify::RecoveredEffectQuery>,
    limits: antecedent_identify::RecoveryLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::RecoveryDecision, antecedent_identify::RecoveryError> {
    antecedent_identify::decide_observation_recovery(graph, query, catalog, effect, limits, ctx)
}

/// Decide a bounded multi-source limited-experiment query.
///
/// Stages run in a fixed order, each only when the earlier ones do not
/// identify: the target alone; each source's own z route, in canonical source
/// order; then the combined `TR^mz` search and, when it reaches a forced
/// line-11 state, the replay that certifies it as an obstruction. One
/// `SearchBudget` under `limits` is charged by every engine step of every
/// stage, with the engine's live-state bytes, so the limits bound the whole
/// decision. A formula that does not bind is kept as missing evidence and the
/// next stage still runs; anything else is not certified. A stop anywhere is
/// `MzTransportDecision::Exhausted`, never a verdict.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::decide_mz_transport`].
///
/// # Errors
///
/// Limits above `MZ_TRANSPORT_DEFAULT_LIMITS` or a query outside its bounds
/// (`mz_transport.bounds_exceeded`), invalid input
/// (`validate_mz_transport_query`), an invalid catalog, or a derivation that
/// fails its own check.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn decide_mz_transport(
    graph: &antecedent_graph::admg::Admg,
    query: &antecedent_identify::MzTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_core::search::SearchLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::MzTransportDecision, antecedent_identify::IdentificationError> {
    antecedent_identify::decide_mz_transport(graph, query, catalog, limits, ctx)
}

/// Decide a bounded mixed-source query.
///
/// Stages run in a fixed order, each only when the earlier ones do not identify:
/// target-first sID over the catalog's target evidence, the declared z (one
/// source) or mz (two or more) route, classical meta-transport when two or more
/// declared sources are unrestricted, then the generic rule search over the
/// target population's distributions and, when it fails, the relaxed pass that
/// names the exact missing joint. One `SearchBudget` under `limits` is charged
/// by every step of every stage, so the limits bound the whole decision. A stop
/// anywhere is `MixedSourceDecision::Exhausted`, never a verdict.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::decide_mixed_source`].
///
/// # Errors
///
/// Limits above `MIXED_SOURCE_DEFAULT_LIMITS` or a query outside its bounds
/// (`mixed_search.bounds_exceeded`), a malformed query or catalog
/// (`mixed_search.invalid_query`), a model artifact offered as an experimental
/// law (`mixed_search.posterior_as_law`), or a derivation that fails its own check.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn decide_mixed_source(
    graph: &antecedent_graph::admg::Admg,
    query: &antecedent_identify::MixedSourceQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_core::search::SearchLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_identify::MixedSourceDecision, antecedent_identify::IdentificationError> {
    antecedent_identify::decide_mixed_source(graph, query, catalog, limits, ctx)
}

/// Decide a bounded ADMG conditional transport query against a catalog.
///
/// Stages, charged to one `SearchBudget` under `limits`: the rule-2 reduction
/// and its independent re-check (one charge per separation test of either),
/// the classical sID decision of the reduced
/// joint (every engine step and the witness check), and the classical catalog
/// search binding it. When sID does not certify the reduced joint, the two-model
/// witness search runs instead of the binding (one charge per parameter block
/// and one per null-space direction tried, each with the attempt's live-state
/// estimate); a verified witness is
/// `ConditionalTransportDecision::ProvenNonTransportable`, and no witness is
/// `ConditionalTransportDecision::NotCertified`. A stop anywhere, the witness
/// search included, is `ConditionalTransportDecision::Exhausted` with a
/// receipt of explored and unevaluated stages, never a verdict.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::decide_admg_conditional_transport`].
///
/// # Errors
/// Limits above `ADMG_CONDITIONAL_DEFAULT_LIMITS`
/// (`admg_transport.bounds_exceeded`), an invalid query or catalog, or a
/// derivation that fails its own check.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn decide_admg_conditional_transport(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ConditionalTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: antecedent_core::search::SearchLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<
    antecedent_identify::ConditionalTransportDecision,
    antecedent_identify::IdentificationError,
> {
    antecedent_identify::decide_admg_conditional_transport(diagram, query, catalog, limits, ctx)
}

/// Bind the checked surrogate formula to a sufficient source joint margin.
///
/// When several available regimes supply the same factor (or the same world of
/// a factor whose exchanged coordinate is bound at evaluation), every formula
/// kind binds by one deterministic rule: prefer a family regime, one declaring
/// no intervention levels, otherwise the lowest regime id. Duplicate evidence
/// is never refused, and the choice does not depend on catalog order.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::bind_z_transport_catalog`].
///
/// # Errors
/// Invalid catalog or a missing joint margin covering the formula variables.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn bind_z_transport_catalog(
    diagram: &antecedent_graph::selection::SelectionDiagram,
    query: &antecedent_identify::ZTransportQuery,
    derivation: &antecedent_identify::ZTransportDerivation,
    catalog: &antecedent_core::EvidenceCatalog,
) -> Result<antecedent_identify::BoundZTransportFunctional, antecedent_identify::IdentificationError>
{
    antecedent_identify::bind_z_transport_catalog(diagram, query, derivation, catalog)
}

/// Bind a checked derivation to a catalog. Each leaf binds only to a regime of
/// its own population, so a factor is always evaluated from the source whose
/// premises certified it.
///
/// Duplicate evidence binds by the single-source z-transport rule on every
/// route: a family regime (no declared levels) before a concrete one, otherwise
/// the lowest regime id, independent of catalog order; supplying a factor more
/// than once is never refused. Exact evaluation selects world-bound laws among
/// the cited regimes only.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::bind_mz_transport_catalog`].
///
/// # Errors
///
/// `IdentificationError::MissingEvidence` when a cited regime is absent; an
/// invalid catalog; or a derivation checked against another graph.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn bind_mz_transport_catalog(
    graph: &antecedent_graph::admg::Admg,
    derivation: &antecedent_identify::MzTransportDerivation,
    catalog: &antecedent_core::EvidenceCatalog,
) -> Result<antecedent_identify::BoundMzTransportFunctional, antecedent_identify::IdentificationError>
{
    antecedent_identify::bind_mz_transport_catalog(graph, derivation, catalog)
}

/// Bind a checked derivation to a catalog: the derivation is re-checked against
/// the catalog, so a leaf always reads the regime whose study supplied it.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_identify::bind_mixed_source_catalog`].
///
/// # Errors
/// `mixed_search.invalid_derivation` when the derivation does not check against
/// `graph` and `catalog`; `mixed_search.invalid_query` for a malformed catalog.
#[deprecated(
    since = "3.1.0",
    note = "compile_transport_identification(TransportIdentificationRequest::..) then identify"
)]
pub fn bind_mixed_source_catalog(
    graph: &antecedent_graph::admg::Admg,
    derivation: &antecedent_identify::MixedSourceDerivation,
    catalog: &antecedent_core::EvidenceCatalog,
) -> Result<antecedent_identify::BoundMixedSourceFunctional, antecedent_identify::IdentificationError>
{
    antecedent_identify::bind_mixed_source_catalog(graph, derivation, catalog)
}

/// Transport under graph uncertainty.
pub mod discovered_transport {
    pub use antecedent_identify::discovered_transport::DiscoveredTransportOperationalKind;
}
/// General ID of a whole-program joint intervention on an unrolled panel graph.
pub mod panel_unrolled {
    pub use antecedent_identify::panel_unrolled::{
        PanelNode, UnrolledPanelGraph, UnrolledPanelSpec,
    };
}
/// Exact binary causal observation recovery.
pub mod recovery {
    pub use antecedent_identify::recovery::MISSING_LEVEL;
}
/// Classical single-source sID, scoped to access to all source experiments.
pub mod sid {
    /// Bounded DAG completions of a supplied CPDAG, decided as a finite scenario set.
    pub mod cpdag_completion {
        pub use antecedent_identify::sid::cpdag_completion::{
            CPDAG_EVIDENCE_MISMATCH_DETAIL, CompletionEvidenceBinding, CpdagCompletionInput,
            CpdagEvidence, enumerate_cpdag_completions,
        };
    }
    /// Per-scenario selection differences and invariances.
    pub mod scenario_invariance {
        pub use antecedent_identify::sid::scenario_invariance::{
            InvarianceBody, InvarianceReport, ObstructionWitness,
        };
    }
    /// Finite, explicitly supplied graph/selection scenarios for one transport question.
    pub mod scenarios {
        pub use antecedent_identify::sid::scenarios::{
            ScenarioCoordinate, TransportScenario, TransportScenarioSet,
        };
    }
    /// One finite temporal transport sequence: exact and point-only.
    pub mod temporal_sequence {
        pub use antecedent_identify::sid::temporal_sequence::{
            TemplateRoles, TemporalOutcome, TemporalSequenceSpec, TemporalSlots, unroll_two_slice,
        };
    }
}
