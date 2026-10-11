//! Core types shared across the Antecedent workspace.
//!
//! `antecedent-core` owns identifiers, schemas, assumptions, provenance,
//! contract identities, diagnostics, errors, and execution policy. It must not depend on numerical,
//! graph-algorithm, Arrow, or Python crates.
//!
//! # Names at the boundary, IDs on the hot path
//!
//! Human-readable names live in [`CausalSchema`]. Hot-path APIs take
//! [`VariableId`] values resolved from that schema — never raw strings.
//!
//! ```
//! use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, VariableId};
//!
//! let schema = CausalSchemaBuilder::new()
//!     .continuous("treatment")
//!     .treatment()
//!     .continuous("outcome")
//!     .outcome()
//!     .build()
//!     .unwrap();
//! let t = schema.id_of("treatment").unwrap();
//! let y = schema.id_of("outcome").unwrap();
//! let query = AverageEffectQuery::binary_ate(t, y);
//! assert_eq!(query.treatment, VariableId::from_raw(0));
//! ```
//!
//! Parallelism, budgets, and RNG seeding are configured via [`ExecutionContext`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod assumption;
pub mod capability;
pub mod claim;
pub mod composition_provenance;
pub mod diagnostic;
pub mod error;
pub mod evidence_obligation;
pub mod execution;
pub mod execution_attempt;
pub mod external;
pub mod external_binding;
pub mod external_refusal;
pub mod identification;
pub mod identity;
pub mod ids;
pub mod interval;
pub mod intervention;
pub mod node;
pub mod obligation;
pub mod plan;
pub mod program_binding;
pub mod provenance;
pub mod quantity;
pub mod query;
pub mod reason_code;
pub mod reasoning;
pub mod recalc;
pub mod transport_result;
pub use transport_result::TransportGridFailure;
pub mod response;
pub mod response_coordinates;
pub use response_coordinates::{
    label_coordinates, response_coordinate_skeleton, response_coordinates,
    ResponseCoordinateLabels, ResponseCoordinateSkeleton,
};
pub mod schema;
pub mod search;
pub use search::{
    SearchBudget, SearchLimits, SearchReceipt, SearchStop, DEFAULT_SEARCH_MEMORY_BYTES,
};
pub mod temporal;
pub mod tolerance;
pub mod transform;
pub mod value;

pub use assumption::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ParametricAssumption, PriorAssumption,
};
pub use capability::{
    BlockedOperation, LicensedNeighbor, NextAction, OperationKind, OperationReadiness,
    OperationReport, PremiseChange, SemanticApplicability,
};
pub use claim::{
    claim_compatibility, compose_claims, AcceptanceReport, AttestedEvidence, CalibrationBasis,
    CalibrationView, ClaimCompatibility, ClaimDomainAxis, ClaimDomains, ClaimEnvelope,
    ClaimFreshness, ClaimKind, ClaimOperation, ConsumerProfile, DerivedClaim, DerivedClaimOutcome,
    DomainStatus, EvidenceDependence, HandoffReceipt, HostOperation, SharedEvidenceRef,
};
pub use composition_provenance::{
    CompositionLink, CompositionStage, ProvenanceChain, ProvenanceChainError, MAX_CHAIN_LINKS,
};
pub use diagnostic::{Diagnostic, DiagnosticKind, DiagnosticSet, DiagnosticSeverity};
pub use error::SchemaError;
pub use evidence_obligation::{
    unresolved_assumption_obligations, EvidenceObligation, EvidenceObligationError,
    EvidenceObligationKind, EvidenceObligationSpec, EvidenceOffer, ObligationProvenance,
    ObligationRegime, MAX_OBLIGATION_COORDINATES,
};
pub use execution::{
    default_user_threads, AdaptiveBootstrapBudget, AdaptiveDrawBudget, CacheBudget, CachePolicy,
    CancellationToken, CausalRng, Determinism, ExecutionContext, ExecutionReceipt,
    ExecutionRequestState, KernelPolicy, MemoryBudget, MonteCarloBudget, MonteCarloError,
    NonZeroThreadCount, Parallelism, ProgressSink, RequestIdentity, RngFactory, StreamDomain,
    ARCH_SIMD_COMPILED, DEFAULT_USER_THREAD_CAP,
};
pub use execution::{derive_stream, mix64, splitmix64, GOLDEN_GAMMA};
pub use external::{
    verify_external_object, EvidenceProviderContract, ExternalCapability,
    ExternalCapabilityRequest, ExternalContractError, ExternalEvidenceKind, ExternalPosteriorKind,
    ExternalScientificObject, ExternalTrustState, ExternalVerificationError, LawProviderContract,
    NegotiatedExternalOperation, ObjectVerificationReceipt, PosteriorProviderContract,
    ProviderObjectIdentity, SignalProviderContract, UtilityMonotonicity, UtilityProviderContract,
    VerificationProbe, VerificationProbeKind,
};
pub use external_binding::{
    bind_external_result, BoundExternalClaim, BoundTrustLevel, CheckedCausalContract,
    CheckedEquivalence, EquivalenceScope, ExternalBindingError, ExternalDistribution,
    ExternalPosterior, ExternalResponse, ExternalResult, ExternalResultHeader,
    ExternalUncertaintyMeaning, OBSERVATIONAL_REGIME,
};
pub use external_refusal::{
    capability_from_name, capability_name, probe_from_name, probe_name, ExternalRefusal,
    StructuredRefusal,
};
pub use identification::{IdentificationStatus, POSTERIOR_MASS_TOLERANCE};
pub use identity::{
    to_hex, CanonicalWriter, ContractIdentities, IdentityDomain, IdentityRef, SemanticDigest,
    IDENTITY_FORMAT, IDENTITY_FORMAT_TAG,
};
pub use ids::{
    CategoryDomainId, ComponentId, DistributionRef, DynamicRuleId, EnvironmentId, Lag, ModelId,
    QueryId, RegimeId, StateVersion, VariableId,
};
pub use interval::IntervalMethod;
pub use intervention::{
    Intervention, InterventionError, InterventionSequence, MechanismOverride,
    SequencedIntervention, StochasticPolicy, TemporalPolicy,
};
pub use node::NodeRef;
pub use obligation::{ObligationKind, ObligationRecord, ObligationScope};
pub use plan::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
pub use program_binding::{
    bind_external_result_to_program, check_external_against_program, dose_label,
    ExternalProgramClaim, ProgramBinding, ProgramBoundClaim, ProgramCheck,
};
pub use provenance::{ArtifactId, ProvenanceError, ProvenanceGraph, ProvenanceNode};
pub use quantity::{
    DistributionMeaning, QuantityCondition, QuantityMismatch, QuantityRole, ScientificQuantity,
};
pub use query::{
    policy_graphless_coordinate, randomized_graphless_coordinate, AllocationMethod,
    AnomalyAttributionQuery, AnomalyReference, AssignmentDesign, AttributionComponents,
    AverageEffectQuery, CatalogDistribution, CausalQuery, ChangeAttributionQuery,
    ConditionalEffectQuery, ContinuousDomain, ContinuousDoseResponseQuery, CounterfactualQuery,
    DependenceGroup, DerivativeScale, DerivativeWeighting, DidSamplingDesign,
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceCatalogDelta, EvidenceKind,
    EvidenceProjection, EvidenceRegime, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, FactorNeed, FixedCandidateRegretInputs, FixedGroupDosePolicy,
    GridSpec, InterferenceFunctional, InterferenceQuery, InterventionAssignment,
    InterventionalDistributionQuery, KnownCensoringSurvival, LawOrigin, LicensedWeights,
    LocalPolynomialRatioQuery, LongitudinalRegimeMethod, LongitudinalRegimeQuery,
    MechanismChangeQuery, MediationContrast, MediationQuery, MultiActionPolicyInputs,
    NestedCounterfactualQuery, ObservationAssumption, ObservationSpec, OrderedFloatBits,
    OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, PolicyValueQuery,
    PopulationRegistry, PopulationSelection, PopulationSelector, PredicateExpr, QueryError,
    RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand, RegimeBinding, RegimeKind,
    ResponseFunctional, ResponseQuery, SamplingDesign, SamplingSelection, ShapleyConfig,
    ShapleyMode, SharedData, SurvivalFunctional, SurvivalQuery, SyntheticControlQuery,
    SyntheticPanelMethod, TargetPopulation, TargetSampling, TemporalEffectQuery,
    TemporalResponseLicense, TemporalResponseSpec, TheoremFamily, TheoremScope,
    TransportEvaluateSupport, TransportIdentifySupport, TransportLocation, TransportOutcome,
    TransportOutcomeKind, TransportQuery, TransportSupportCoordinate, TransportUncertaintySupport,
    UnitChangeQuery, UnmetDependency, VariableCoordinate, VariableDomain, EXPOSURE_LEVEL_TOLERANCE,
    MAX_NONPARAMETRIC_RESPONSE_DIM, MAX_TEMPORAL_RESPONSE_CELLS, MAX_TEMPORAL_RESPONSE_HORIZONS,
    TEMPORAL_OBSERVATION_UNLICENSED,
};
pub use query::{CounterfactualEvent, CounterfactualEventQuery};
pub use query::{
    CrossWorldQuery, EdgeRoute, ExogenousCoupling, WorldId, WorldObservation, WorldSpec,
    MAX_CROSS_WORLDS,
};
pub use query::{SmoothedDoseTransportQuery, SmoothingKernel};
pub use reasoning::{
    AssumptionSlot, IdentificationSlot, ReasoningView, SlotAvailability, SupportSlot,
    UncertaintyComponent, UncertaintySlot, UncertaintySource,
};
pub use response::{
    check_static_point_labels, CausalResponse, CredibleDraws, DiagnosticScope,
    HorizonIdentification, IdentifiedSet, IntervalInterpretation, ResponseEnvelope,
    ResponseIdentification, ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion,
    SupportReport, SupportStatus,
};
pub use schema::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, RoleHint, ScalarType, SmallRoleSet,
    ValueType, VariableInProgress, VariableSchema,
};
pub use temporal::{TemporalIndexError, TemporalIndexer, TemporalNodeKey};
pub use tolerance::ToleranceClass;
pub use transform::{
    intent_effects, LayerEffect, SemanticLayer, TransformEffect, TransformIntent,
    TransformationReport,
};
pub use value::{same_intervention_level, Value};

/// Library crate version string from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_semver_like() {
        assert!(super::VERSION.contains('.'));
    }
}
