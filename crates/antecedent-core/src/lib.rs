//! Core types shared across the Antecedent workspace.
//!
//! `antecedent-core` owns identifiers, schemas, assumptions, provenance,
//! contract identities, diagnostics, errors, and execution policy. It must not depend on numerical,
//! graph-algorithm, Arrow, or Python crates. Inference-independent scientific targets live in
//! [`scientific_target`] and do not replace [`CausalQuery`].
//!
//! Operator contracts in [`operator`] type an effect on a target: which graph the
//! edge belongs to, the data roles it reads and writes, and whether frozen-fit
//! reuse is allowed.
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
pub mod calibration_basis;
pub mod callback;
pub mod diagnostic;
pub mod error;
pub mod execution;
pub mod execution_attempt;
pub mod identification;
pub mod identity;
pub mod ids;
pub mod inference_realization;
pub mod inspect;
pub mod interval;
pub mod intervention;
pub mod measurement_model;
pub(crate) mod measurement_record;
pub mod node;
pub mod operator;
pub mod parallel_axis;
pub mod physical_operator;
pub mod provider_capability;
pub mod query;
pub mod reason_code;
pub mod recalc;
pub mod response;
pub mod rng_key;
pub mod route_memory;
pub mod sampling_design;
pub mod schema;
pub mod scientific_target;
pub mod search;
pub use search::{
    DEFAULT_SEARCH_MEMORY_BYTES, SearchBudget, SearchLimits, SearchReceipt, SearchStop,
};
pub mod search_budget;
pub mod temporal;
pub mod text;
pub mod tolerance;
pub mod value;
pub mod value_expr;

pub use assumption::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ParametricAssumption, PriorAssumption,
};
pub use calibration_basis::CalibrationBasis;
pub use callback::{
    BatchCallback, CallbackCall, CallbackError, CallbackImplementation, CallbackOutput,
    CallbackRefusal, CallbackSpec, PinnedDependency, Portability, RecoveryMismatch, RecoveryRecord,
};
pub use diagnostic::{Diagnostic, DiagnosticKind, DiagnosticSet, DiagnosticSeverity};
pub use error::SchemaError;
pub use execution::{
    ARCH_SIMD_COMPILED, AdaptiveBootstrapBudget, AdaptiveDrawBudget, CacheBudget, CachePolicy,
    CancellationToken, CausalRng, DEFAULT_USER_THREAD_CAP, Determinism, ExecutionContext,
    ExecutionReceipt, ExecutionRequestState, KernelPolicy, MemoryBudget, MonteCarloBudget,
    MonteCarloError, NonZeroThreadCount, Parallelism, ProgressSink, RequestIdentity, RngFactory,
    StreamDomain, WeakCancellationToken, default_user_threads,
};
pub use execution::{GOLDEN_GAMMA, derive_stream, mix64, splitmix64};
pub use identification::{IdentificationStatus, POSTERIOR_MASS_TOLERANCE};
pub use identity::{
    CanonicalWriter, ContractIdentities, FNV1A_64_OFFSET, FNV1A_64_PRIME, Fnv1a64, IDENTITY_FORMAT,
    IDENTITY_FORMAT_TAG, IdentityDomain, IdentityRef, SemanticDigest, blake3_hex, fnv1a_u64,
    to_hex,
};
pub use ids::{
    CategoryDomainId, ComponentId, DistributionRef, DynamicRuleId, EnvironmentId, Lag, ModelId,
    QueryId, RegimeId, StateVersion, VariableId,
};
pub use inference_realization::{
    ASSUMPTION_BRANCH_TAG, AssumptionBranch, DeclarationError, GENERALIZED_BAYES_TAG,
    GeneralizedBayes, IdentifyingRestriction, LearningRate, LikelihoodModel, LossModel,
    MissingLearningRate, NextIdentificationAttempt, ORDINARY_BAYESIAN_LIKELIHOOD_TAG,
    OrdinaryBayesianLikelihood, PARTIAL_IDENTIFICATION_ENVELOPE_TAG, PartialIdentificationEnvelope,
    PosteriorSample, PosteriorSampleKind, PosteriorSampleRefusal, Prior,
    PriorIdentificationRefusal, RealizationKind, RefusedSampleKind, RestrictedPosterior,
    SCORE_FREQUENTIST_TAG, SampleNextAttempt, ScoreFrequentist, ScoreModel,
};
pub use interval::IntervalMethod;
pub use intervention::{
    Intervention, InterventionError, InterventionSequence, MechanismOverride,
    SequencedIntervention, StochasticPolicy, TemporalPolicy,
};
pub use node::NodeRef;
pub use operator::{
    EvidenceClaims, FrozenFitReuse, Operator, OperatorDeclaration, OperatorEffect, OperatorError,
    OperatorGraph, OperatorPorts, OperatorRole,
};
pub use physical_operator::DeclaredTolerance as PhysicalTolerance;
pub use physical_operator::{
    AdmittedBytes, ByteEstimateKind, ByteEstimates, CancellationPoint, CancellationPoints,
    ComputeCapability, DesignDependencyId, DeviceName, NumericalPolicy, PhysicalDependencies,
    PhysicalEffect, PhysicalOperator, PhysicalOperatorBuilder, PhysicalOperatorError,
    SourceDependencyId,
};
pub use provider_capability::{
    CapabilityRequest, CapabilitySet, CompiledCapabilityRequest, FailedCapabilityAttempt,
    ProviderCapability, ProviderDeclaration, ProviderDeclarationError, ProviderId,
};
pub use query::{
    AllocationMethod, AnomalyAttributionQuery, AnomalyReference, AssignmentDesign,
    AttributionComponents, AverageEffectQuery, CatalogDistribution, CausalQuery, CausalQueryKind,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, ContinuousDoseResponseQuery,
    CounterfactualQuery, DependenceGroup, DerivativeScale, DerivativeWeighting, DidSamplingDesign,
    DistributionAvailability, EXPOSURE_LEVEL_TOLERANCE, Environment, EvidenceCatalog,
    EvidenceCatalogDelta, EvidenceKind, EvidenceProjection, EvidenceRegime, ExposureLevel,
    ExposureMapping, ExposurePropensityProvenance, FactorNeed, FixedCandidateRegretInputs,
    FixedGroupDosePolicy, GridSpec, InterferenceFunctional, InterferenceQuery,
    InterventionAssignment, InterventionalDistributionQuery, KnownCensoringSurvival, LawOrigin,
    LicensedWeights, LocalPolynomialRatioQuery, LongitudinalRegimeMethod, LongitudinalRegimeQuery,
    MAX_NONPARAMETRIC_RESPONSE_DIM, MAX_TEMPORAL_RESPONSE_CELLS, MAX_TEMPORAL_RESPONSE_HORIZONS,
    MechanismChangeQuery, MediationContrast, MediationQuery, MultiActionPolicyInputs,
    NestedCounterfactualQuery, ObservationAssumption, ObservationSpec, OrderedFloatBits,
    OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, PolicyValueQuery,
    PopulationRegistry, PopulationSelection, PopulationSelector, PredicateExpr, QueryError,
    RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand, RegimeBinding, RegimeKind,
    ResponseFunctional, ResponseQuery, SamplingDesign, SamplingSelection, ShapleyConfig,
    ShapleyMode, SharedData, SurvivalFunctional, SurvivalQuery, SyntheticControlQuery,
    SyntheticPanelMethod, TEMPORAL_OBSERVATION_UNLICENSED, TargetPopulation, TargetSampling,
    TemporalEffectQuery, TemporalResponseLicense, TemporalResponseSpec, TheoremFamily,
    TheoremScope, TransportEvaluateSupport, TransportIdentifySupport, TransportLocation,
    TransportOutcome, TransportOutcomeKind, TransportQuery, TransportSupportCoordinate,
    TransportUncertaintySupport, UnitChangeQuery, UnmetDependency, VariableCoordinate,
    VariableDomain, policy_graphless_coordinate, randomized_graphless_coordinate,
};
pub use query::{CounterfactualEvent, CounterfactualEventQuery};
pub use query::{
    CrossWorldQuery, EdgeRoute, ExogenousCoupling, MAX_CROSS_WORLDS, WorldId, WorldObservation,
    WorldSpec,
};
pub use query::{SmoothedDoseTransportQuery, SmoothingKernel};
pub use response::{
    CausalResponse, CredibleDraws, DiagnosticScope, HorizonIdentification, IdentifiedSet,
    IntervalInterpretation, ResponseEnvelope, ResponseIdentification, ResponseUncertainty,
    ResponseValue, SupportDiagnostic, SupportRegion, SupportReport, SupportStatus,
};
pub use rng_key::{
    BatchSize, ChainId, DeclaredTolerance, FailedRngAttempt, FoldId, ReductionPolicy, ReplicateId,
    RngKey, SemanticOperation, StreamRequest, ToleranceError, WorkerId,
};
pub use route_memory::{
    AdmittedRouteMemory, EncodedWidth, RouteBudget, RouteMemoryComponent, RouteMemoryDeclaration,
    RouteMemoryError, RouteMemoryEstimate,
};
pub use sampling_design::SamplingDesign as DeclaredSamplingDesign;
pub use sampling_design::{
    AllowedRule, AttemptedRule, BoundEffectiveSampleSize, BoundResampling, BoundSplitting,
    BoundVariance, DesignRules, EffectiveSampleSizeRule, ResamplingRule, SamplingDesignDeclaration,
    SamplingDesignError, SplittingRule, VarianceRule,
};
pub use schema::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, RoleHint, ScalarType, SmallRoleSet,
    ValueType, VariableInProgress, VariableSchema,
};
pub use scientific_target::{
    Conditioning, ContrastForm, ContrastTerm, CrossWorldPremise, DoseTimeSurface, GraphFamily,
    HistoryWindow, InferenceSelection, InitialLaw, InterventionContrast, InterventionWorld,
    MediationArms, MediationComponent, MediationTarget, ModelAssumptions, ObservationUnit,
    OutcomeQuantity, QuantityUnits, SCIENTIFIC_TARGET_FORMAT, SCIENTIFIC_TARGET_IDENTITY_TAG,
    ScientificTarget, ScientificTargetError, ScientificWorldId, TargetForm, TimeFrame, WorldAction,
    WorldBinding, WorldCollection,
};
pub use temporal::{TemporalIndexError, TemporalIndexer, TemporalNodeKey};
pub use text::is_blank;
pub use tolerance::ToleranceClass;
pub use value::{Value, same_intervention_level};
pub use value_expr::{
    ArithOp, CompareOp, ExprNumber, ExprType, ExpressionFault, LogicOp, MAX_EXPR_DEPTH,
    MAX_EXPR_NODES, PolicyReads, ValueExpr, clip_value,
};

/// Label of a structural (Fréchet-Hoeffding) envelope over every coupling of
/// the marginals; the single definition the contract and stats crates re-export.
pub const STRUCTURAL_ENVELOPE_LABEL: &str = "structural_envelope";

/// Library crate version string from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_semver_like() {
        assert!(super::VERSION.contains('.'));
    }
}
