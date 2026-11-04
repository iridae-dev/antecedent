//! Typed causal queries (re-exported from `antecedent-core`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_core::IvBoundsQuery;
pub use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExposureLevel, ExposureMapping, GridSpec,
    HistoryMatchQuery, InterferenceFunctional, InterferenceQuery, InterventionalDistributionQuery,
    LocalPolynomialRatioQuery, LongitudinalRegimeQuery, MechanismChangeQuery, MediationContrast,
    MediationQuery, OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, RandomizationDesign,
    RandomizedEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SequentialContrastQuery, SyntheticControlQuery,
    SyntheticPanelMethod, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec,
    TransportQuery, UnitChangeQuery,
};
pub use antecedent_core::{
    AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery,
};
pub use antecedent_core::{
    AttributeQuery, AttributeTarget, ChangeEstimator, ChangeMeasure, DiscoverQuery,
    DiscoveredClass, DiscoveryInference, DiscoveryMethod, InfluenceAllocation,
    IntrinsicInfluenceTarget, MechanismChangeTest, NoiseDeclaration, ScreenWeight, SearchBounds,
    SearchSchedule, SimulateQuery, SimulationTarget,
};
pub use antecedent_core::{
    OutcomeMisclassification, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget, TemporalRegimeMethod,
};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Hypothetical evidence changes for transport planning.
pub use antecedent_core::EvidenceCatalogDelta;
/// Randomized interference queries.
pub use antecedent_core::ExposurePropensityProvenance;
/// Longitudinal regime value contract for one outcome row per subject.
pub use antecedent_core::LongitudinalRegimeMethod;
/// A narrowly supported natural direct effect query.
pub use antecedent_core::NestedCounterfactualQuery;
/// Stable node identity before dense graph indexing.
pub use antecedent_core::NodeRef;
/// Target-population resolution against tabular rows.
pub use antecedent_core::PopulationRegistry;
/// Query submodule.
pub use antecedent_core::QueryError;
/// Kernel-smoothed dose-response transport query.
pub use antecedent_core::SmoothedDoseTransportQuery;
/// Transport theorem scopes, typed outcomes, and stage-specific support.
pub use antecedent_core::TransportOutcomeKind;
/// Typed value expressions for programmable interventions.
pub use antecedent_core::ValueExpr;
/// Query submodule.
pub use antecedent_core::{
    AllocationMethod, AnomalyReference, AttributionComponents, PopulationSelector, ShapleyConfig,
};
/// Two-period and selected staggered group-time difference-in-differences queries.
pub use antecedent_core::{
    AttritionShiftBound, DidAggregation, DidBasePeriod, DidComparisonGroup, DidNuisanceMethod,
    DidResamplingScheme, HeterogeneityRobustDid, PanelObservationProcess, RolloutDidEstimator,
};
/// Causal-response result and support vocabulary.
pub use antecedent_core::{CausalResponse, IdentifiedSet, IntervalInterpretation, SupportStatus};
/// Caller-supplied conditional continuous-dose response design.
pub use antecedent_core::{ContinuousDoseResponseQuery, FixedGroupDosePolicy};
/// A graph-independent counterfactual event query.
pub use antecedent_core::{CounterfactualEvent, CounterfactualEventQuery};
/// A graph-independent cross-world query: worlds, per-edge inputs, coupling.
pub use antecedent_core::{CrossWorldQuery, EdgeRoute, ExogenousCoupling, WorldId, WorldSpec};
/// Environments, evidence regimes, and supplied transport catalogs.
pub use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceProjection, EvidenceRegime, InterventionAssignment, LawOrigin, RegimeBinding,
    RegimeKind, SamplingDesign, SamplingSelection, SupportedWeights, TargetSampling,
    VariableCoordinate, VariableDomain,
};
/// Continuous causal-response queries.
pub use antecedent_core::{FeedbackDeclaration, ObservationAssumption, ObservationSpec};
/// Query contract for randomized binary and multi-action policy value.
pub use antecedent_core::{FixedCandidateRegretInputs, MultiActionPolicyInputs, PolicyValueQuery};
/// Interventions on causal variables.
pub use antecedent_core::{
    InterventionSequence, MechanismOverride, SequencedIntervention, StochasticPolicy,
    TemporalBudget, TemporalMaterializationError, TemporalMaterializationLimits,
};
/// Randomized survival and competing-risk query contract.
pub use antecedent_core::{KnownCensoringSurvival, SurvivalFunctional, SurvivalQuery};
/// Declared local-polynomial fuzzy discontinuity and regression kink design.
pub use antecedent_core::{LocalBandwidthRule, LocalRatioOptions};
/// Query submodule.
pub use antecedent_core::{PredicateExpr, TargetPopulation};
/// Temporal node identity and dense unfolding indexes (ADR 0005 / ).
pub use antecedent_core::{TemporalIndexer, TemporalNodeKey};
