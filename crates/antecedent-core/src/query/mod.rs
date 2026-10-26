//! Typed causal queries.
//!
//! Hot paths bind [`VariableId`](crate::ids::VariableId)s; names are resolved only at API boundaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod association;
mod attribute;
mod attribution;
mod average;
mod continuous_dose;
mod counterfactual;
mod counterfactual_event;
mod cross_world;
mod did;
mod discover;
mod distribution;
mod error;
mod fault;
mod functional;
mod history_match;
mod interference;
mod iv_bounds;
mod local_polynomial_ratio;
mod longitudinal_regime;
mod mediation;
mod nested_counterfactual;
mod policy_value;
mod population;
mod probability;
mod probability_rows;
mod randomized;
mod response;
mod sequential_contrast;
mod simulate;
mod smoothed_dose;
mod survival;
mod synthetic_control;
mod target;
mod temporal;
mod temporal_design;
mod transport;
mod transport_catalog;
mod transport_contract;
mod transport_delta;
mod transport_distribution;
mod variants;

pub use crate::intervention::TemporalPolicy;

pub use association::{
    AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery,
    MAX_REFUSED_AFTER_BYTES, MIN_PREDICTIVE_FOLDS,
};
pub use attribute::{
    AttributeQuery, AttributeTarget, ChangeEstimator, ChangeMeasure, InfluenceAllocation,
    IntrinsicInfluenceTarget, MechanismChangeTest, NoiseDeclaration,
};
pub use attribution::{
    AllocationMethod, AnomalyAttributionQuery, AnomalyReference, AttributionComponents,
    ChangeAttributionQuery, MechanismChangeQuery, OrderedFloatBits, PopulationSelector,
    ShapleyConfig, ShapleyMode, UnitChangeQuery,
};
pub use average::AverageEffectQuery;
pub use continuous_dose::{ContinuousDoseResponseQuery, FixedGroupDosePolicy};
pub use counterfactual::CounterfactualQuery;
pub use counterfactual_event::{CounterfactualEvent, CounterfactualEventQuery};
pub use cross_world::{
    CrossWorldQuery, EdgeRoute, ExogenousCoupling, WorldId, WorldObservation,
    WorldSpec,
};
pub use did::{
    DidAggregation, DidBasePeriod, DidComparisonGroup, DidNuisanceMethod, DidResampling,
    DidResamplingScheme, DidSamplingDesign, HeterogeneityRobustDid, PanelDidQuery,
    RolloutDidEstimator,
};
pub use discover::{
    DiscoverQuery, DiscoveredClass, DiscoveryInference, DiscoveryMethod, ScreenWeight,
    SearchBounds, SearchSchedule,
};
pub use distribution::{InterventionalDistributionQuery, PathSpecificEffectQuery};
pub use error::{QueryError, RandomizedEffectFault};
pub use functional::OutcomeFunctional;
pub use history_match::HistoryMatchQuery;
pub use interference::{
    AssignmentDesign, EXPOSURE_LEVEL_TOLERANCE, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, InterferenceFunctional, InterferenceQuery,
};
pub use iv_bounds::IvBoundsQuery;
pub use local_polynomial_ratio::{
    LocalBandwidthRule, LocalPolynomialRatioQuery, LocalRatioOptions,
};
pub use longitudinal_regime::{LongitudinalRegimeMethod, LongitudinalRegimeQuery};
pub use mediation::{ConditionalEffectQuery, MediationContrast, MediationQuery};
pub use nested_counterfactual::NestedCounterfactualQuery;
pub use policy_value::{
    FixedCandidateRegretInputs, MultiActionPolicyInputs, PolicyValueQuery,
    policy_graphless_coordinate,
};
pub use population::{PopulationRegistry, PopulationSelection};
pub use fault::{
    AssociationFault,
    AttributeFault,
    DiscoverFault,
    HistoryMatchFault,
    InterferenceFault,
    InterventionFault,
    IvBoundsFault,
    LocalPolynomialRatioFault,
    LongitudinalRegimeFault,
    PanelDesignFault,
    PolicyValueFault,
    ResponseFault,
    SequentialContrastFault,
    SimulateFault,
    SurvivalFault,
    TemporalDesignFault,
    TransportFault,
};
pub use probability_rows::{ProbabilityRows, ProbabilityTable};
pub use randomized::{
    RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand, randomized_graphless_coordinate,
};
pub use response::{
    ContinuousDomain, DerivativeScale, DerivativeWeighting, FeedbackDeclaration, GridSpec,
    MAX_NONPARAMETRIC_RESPONSE_DIM, MAX_TEMPORAL_RESPONSE_CELLS,
    ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseQuery,
    TEMPORAL_OBSERVATION_UNLICENSED, TemporalResponseLicense, TemporalResponseSpec,
};
pub use sequential_contrast::SequentialContrastQuery;
pub use simulate::{SimulateQuery, SimulationTarget};
pub use smoothed_dose::{SmoothedDoseTransportQuery, SmoothingKernel};
pub use survival::{KnownCensoringSurvival, SurvivalFunctional, SurvivalQuery};
pub use synthetic_control::{
    SyntheticControlQuery, SyntheticInterval, SyntheticPanelMethod, SyntheticRollout,
};
pub use target::{PredicateExpr, TargetPopulation};
pub use temporal::TemporalEffectQuery;
pub use temporal_design::{
    OutcomeMisclassification, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget, TemporalRegimeMethod,
};
pub use transport::TransportQuery;
pub use transport_catalog::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceProjection, EvidenceRegime, FactorNeed, InterventionAssignment, LawOrigin,
    LicensedWeights, RegimeBinding, RegimeKind, SamplingDesign, SamplingSelection, TargetSampling,
    UnmetDependency, VariableCoordinate, VariableDomain,
};
pub use transport_contract::{
    ComputationLimits, ExperimentFamily, GraphAssumptionSet, OutcomeGuarantee, TheoremFamily,
    TheoremReference, TheoremScope, TransportDistributionFamily, TransportEvaluateSupport,
    TransportIdentifySupport, TransportLocation, TransportOutcome, TransportOutcomeKind,
    TransportQueryScope, TransportSupportCoordinate, TransportUncertaintySupport,
};
pub use transport_delta::EvidenceCatalogDelta;
pub use transport_distribution::{CatalogDistribution, SharedData};
pub use variants::{CausalQuery, CausalQueryKind, QueryVariant};

impl CausalQuery {
    /// Target population of a population-scoped query.
    ///
    /// `None` for kinds that carry no [`TargetPopulation`] (counterfactual,
    /// attribution, change, transport, and interference queries). The single
    /// owner of which query kinds are population-scoped.
    #[must_use]
    pub const fn target_population(&self) -> Option<&TargetPopulation> {
        match self {
            Self::AverageEffect(inner) => Some(&inner.target_population),
            Self::TemporalEffect(inner) => Some(&inner.target_population),
            Self::Mediation(inner) => Some(&inner.target_population),
            Self::Distribution(inner) => Some(&inner.target_population),
            Self::PathSpecific(inner) => Some(&inner.target_population),
            Self::Response(inner) => Some(&inner.target_population),
            Self::ConditionalEffect(inner) => Some(&inner.inner.target_population),
            Self::Counterfactual(_)
            | Self::NestedCounterfactual(_)
            | Self::AnomalyAttribution(_)
            | Self::ChangeAttribution(_)
            | Self::MechanismChange(_)
            | Self::UnitChange(_)
            | Self::Transport(_)
            | Self::Interference(_)
            | Self::RandomizedEffect(_)
            | Self::PolicyValue(_)
            | Self::ContinuousDoseResponse(_)
            | Self::PanelDid(_)
            | Self::SyntheticControl(_)
            | Self::LocalPolynomialRatio(_)
            | Self::Survival(_)
            | Self::LongitudinalRegime(_)
            | Self::SequentialContrast(_)
            | Self::HistoryMatch(_)
            | Self::Discover(_)
            | Self::Attribute(_)
            | Self::Simulate(_)
            | Self::TemporalDesign(_)
            | Self::IvBounds(_)
            | Self::Association(_) => None,
        }
    }

    /// Mutable target population; `None` exactly when [`Self::target_population`] is.
    pub fn target_population_mut(&mut self) -> Option<&mut TargetPopulation> {
        match self {
            Self::AverageEffect(inner) => Some(&mut inner.target_population),
            Self::TemporalEffect(inner) => Some(&mut inner.target_population),
            Self::Mediation(inner) => Some(&mut inner.target_population),
            Self::Distribution(inner) => Some(&mut inner.target_population),
            Self::PathSpecific(inner) => Some(&mut inner.target_population),
            Self::Response(inner) => Some(&mut inner.target_population),
            Self::ConditionalEffect(inner) => Some(&mut inner.inner.target_population),
            Self::Counterfactual(_)
            | Self::NestedCounterfactual(_)
            | Self::AnomalyAttribution(_)
            | Self::ChangeAttribution(_)
            | Self::MechanismChange(_)
            | Self::UnitChange(_)
            | Self::Transport(_)
            | Self::Interference(_)
            | Self::RandomizedEffect(_)
            | Self::PolicyValue(_)
            | Self::ContinuousDoseResponse(_)
            | Self::PanelDid(_)
            | Self::SyntheticControl(_)
            | Self::LocalPolynomialRatio(_)
            | Self::Survival(_)
            | Self::LongitudinalRegime(_)
            | Self::SequentialContrast(_)
            | Self::HistoryMatch(_)
            | Self::Discover(_)
            | Self::Attribute(_)
            | Self::Simulate(_)
            | Self::TemporalDesign(_)
            | Self::IvBounds(_)
            | Self::Association(_) => None,
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "encode_tests.rs"]
mod encode_tests;
