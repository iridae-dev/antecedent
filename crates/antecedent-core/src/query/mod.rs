//! Typed causal queries.
//!
//! Hot paths bind [`VariableId`](crate::ids::VariableId)s; names are resolved only at API boundaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod attribution;
mod average;
mod continuous_dose;
mod counterfactual;
mod counterfactual_event;
mod cross_world;
mod did;
mod distribution;
mod error;
mod functional;
mod interference;
mod local_polynomial_ratio;
mod longitudinal_regime;
mod mediation;
mod nested_counterfactual;
mod policy_value;
mod population;
mod randomized;
mod response;
mod smoothed_dose;
mod survival;
mod synthetic_control;
mod target;
mod temporal;
mod transport;
mod transport_catalog;
mod transport_contract;
mod transport_delta;
mod transport_distribution;

pub use crate::intervention::TemporalPolicy;

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
    CrossWorldQuery, EdgeRoute, ExogenousCoupling, MAX_CROSS_WORLDS, WorldId, WorldObservation,
    WorldSpec,
};
pub use did::{DidSamplingDesign, PanelDidQuery};
pub use distribution::{InterventionalDistributionQuery, PathSpecificEffectQuery};
pub use error::QueryError;
pub use functional::OutcomeFunctional;
pub use interference::{
    AssignmentDesign, EXPOSURE_LEVEL_TOLERANCE, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, InterferenceFunctional, InterferenceQuery,
};
pub use local_polynomial_ratio::LocalPolynomialRatioQuery;
pub use longitudinal_regime::{LongitudinalRegimeMethod, LongitudinalRegimeQuery};
pub use mediation::{ConditionalEffectQuery, MediationContrast, MediationQuery};
pub use nested_counterfactual::NestedCounterfactualQuery;
pub use policy_value::{
    FixedCandidateRegretInputs, MultiActionPolicyInputs, PolicyValueQuery,
    policy_graphless_coordinate,
};
pub use population::{PopulationRegistry, PopulationSelection};
pub use randomized::{
    RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand, randomized_graphless_coordinate,
};
pub use response::{
    ContinuousDomain, DerivativeScale, DerivativeWeighting, GridSpec,
    MAX_NONPARAMETRIC_RESPONSE_DIM, MAX_TEMPORAL_RESPONSE_CELLS, MAX_TEMPORAL_RESPONSE_HORIZONS,
    ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseQuery,
    TEMPORAL_OBSERVATION_UNLICENSED, TemporalResponseLicense, TemporalResponseSpec,
};
pub use smoothed_dose::{SmoothedDoseTransportQuery, SmoothingKernel};
pub use survival::{KnownCensoringSurvival, SurvivalFunctional, SurvivalQuery};
pub use synthetic_control::{SyntheticControlQuery, SyntheticPanelMethod};
pub use target::{PredicateExpr, TargetPopulation};
pub use temporal::TemporalEffectQuery;
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

/// Top-level causal query enum.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CausalQuery {
    /// Average / population effect (static).
    AverageEffect(AverageEffectQuery),
    /// Temporal effect over a discrete horizon.
    TemporalEffect(TemporalEffectQuery),
    /// Counterfactual / unit-level what-if query .
    Counterfactual(CounterfactualQuery),
    /// Fixed-contract nested natural direct effect.
    NestedCounterfactual(NestedCounterfactualQuery),
    /// Anomaly attribution for one or more units .
    AnomalyAttribution(AnomalyAttributionQuery),
    /// Distribution / population change attribution .
    ChangeAttribution(ChangeAttributionQuery),
    /// Mechanism-change detection — not attribution .
    MechanismChange(MechanismChangeQuery),
    /// Per-unit change attribution .
    UnitChange(UnitChangeQuery),
    /// Mediation (direct / mediated / natural effects).
    Mediation(MediationQuery),
    /// Conditional average effect given modifiers.
    ConditionalEffect(ConditionalEffectQuery),
    /// Interventional distribution P(Y | do(...)).
    Distribution(InterventionalDistributionQuery),
    /// Path-specific effect / contribution.
    PathSpecific(PathSpecificEffectQuery),
    /// Continuous response, derivative, policy response, or Jacobian.
    Response(ResponseQuery),
    /// Structurally transported response between populations.
    Transport(TransportQuery),
    /// Randomization-based causal effect under interference.
    Interference(InterferenceQuery),
    /// Bernoulli randomized two-arm intention-to-treat effect.
    RandomizedEffect(RandomizedEffectQuery),
    /// Fixed randomized binary or multi-action policy value.
    PolicyValue(PolicyValueQuery),
    /// Local conditional continuous-dose response with supplied dose density.
    ContinuousDoseResponse(ContinuousDoseResponseQuery),
    /// Balanced two-period panel difference in differences.
    PanelDid(PanelDidQuery),
    /// Balanced-panel synthetic control with one treated unit and donor pool.
    SyntheticControl(SyntheticControlQuery),
    /// Fixed-bandwidth local fuzzy discontinuity or regression kink ratio.
    LocalPolynomialRatio(LocalPolynomialRatioQuery),
    /// Randomized right-censored survival or competing-risk functional.
    Survival(SurvivalQuery),
    /// Prespecified longitudinal regime value over subject histories.
    LongitudinalRegime(LongitudinalRegimeQuery),
}

impl CausalQuery {
    /// Construct an average-effect query.
    #[must_use]
    pub fn average_effect(query: AverageEffectQuery) -> Self {
        Self::AverageEffect(query)
    }

    /// Construct a temporal-effect query.
    #[must_use]
    pub fn temporal_effect(query: TemporalEffectQuery) -> Self {
        Self::TemporalEffect(query)
    }

    /// Construct a counterfactual query.
    #[must_use]
    pub fn counterfactual(query: CounterfactualQuery) -> Self {
        Self::Counterfactual(query)
    }

    /// Construct a nested counterfactual query.
    #[must_use]
    pub fn nested_counterfactual(query: NestedCounterfactualQuery) -> Self {
        Self::NestedCounterfactual(query)
    }

    /// Construct an anomaly attribution query.
    #[must_use]
    pub fn anomaly_attribution(query: AnomalyAttributionQuery) -> Self {
        Self::AnomalyAttribution(query)
    }

    /// Construct a change attribution query.
    #[must_use]
    pub fn change_attribution(query: ChangeAttributionQuery) -> Self {
        Self::ChangeAttribution(query)
    }

    /// Construct a mechanism-change detection query.
    #[must_use]
    pub fn mechanism_change(query: MechanismChangeQuery) -> Self {
        Self::MechanismChange(query)
    }

    /// Construct a unit-change attribution query.
    #[must_use]
    pub fn unit_change(query: UnitChangeQuery) -> Self {
        Self::UnitChange(query)
    }

    /// Construct a mediation query.
    #[must_use]
    pub fn mediation(query: MediationQuery) -> Self {
        Self::Mediation(query)
    }

    /// Construct a conditional-effect query.
    #[must_use]
    pub fn conditional_effect(query: ConditionalEffectQuery) -> Self {
        Self::ConditionalEffect(query)
    }

    /// Construct an interventional-distribution query.
    #[must_use]
    pub fn distribution(query: InterventionalDistributionQuery) -> Self {
        Self::Distribution(query)
    }

    /// Construct a path-specific effect query.
    #[must_use]
    pub fn path_specific(query: PathSpecificEffectQuery) -> Self {
        Self::PathSpecific(query)
    }

    /// Construct a continuous-response query.
    #[must_use]
    pub fn response(query: ResponseQuery) -> Self {
        Self::Response(query)
    }

    /// Construct a transportability query.
    #[must_use]
    pub fn transport(query: TransportQuery) -> Self {
        Self::Transport(query)
    }

    /// Construct an interference query.
    #[must_use]
    pub fn interference(query: InterferenceQuery) -> Self {
        Self::Interference(query)
    }

    /// Construct a randomized ITT query.
    #[must_use]
    pub fn randomized_effect(query: RandomizedEffectQuery) -> Self {
        Self::RandomizedEffect(query)
    }

    /// Construct a held-out policy value query.
    #[must_use]
    pub fn policy_value(query: PolicyValueQuery) -> Self {
        Self::PolicyValue(query)
    }

    /// Construct a balanced-panel `DiD` query.
    #[must_use]
    pub fn panel_did(query: PanelDidQuery) -> Self {
        Self::PanelDid(query)
    }

    /// Construct a synthetic-control query.
    #[must_use]
    pub fn synthetic_control(query: SyntheticControlQuery) -> Self {
        Self::SyntheticControl(query)
    }

    /// Construct a local fuzzy-discontinuity or regression-kink query.
    #[must_use]
    pub fn local_polynomial_ratio(query: LocalPolynomialRatioQuery) -> Self {
        Self::LocalPolynomialRatio(query)
    }

    /// Construct a randomized survival or competing-risk query.
    #[must_use]
    pub fn survival(query: SurvivalQuery) -> Self {
        Self::Survival(query)
    }

    /// Construct a prespecified longitudinal regime value query.
    #[must_use]
    pub fn longitudinal_regime(query: LongitudinalRegimeQuery) -> Self {
        Self::LongitudinalRegime(query)
    }
}

impl From<AverageEffectQuery> for CausalQuery {
    fn from(query: AverageEffectQuery) -> Self {
        Self::AverageEffect(query)
    }
}

impl From<TemporalEffectQuery> for CausalQuery {
    fn from(query: TemporalEffectQuery) -> Self {
        Self::TemporalEffect(query)
    }
}

impl From<CounterfactualQuery> for CausalQuery {
    fn from(query: CounterfactualQuery) -> Self {
        Self::Counterfactual(query)
    }
}

impl From<NestedCounterfactualQuery> for CausalQuery {
    fn from(query: NestedCounterfactualQuery) -> Self {
        Self::NestedCounterfactual(query)
    }
}

impl From<MediationQuery> for CausalQuery {
    fn from(query: MediationQuery) -> Self {
        Self::Mediation(query)
    }
}

impl From<ConditionalEffectQuery> for CausalQuery {
    fn from(query: ConditionalEffectQuery) -> Self {
        Self::ConditionalEffect(query)
    }
}

impl From<InterventionalDistributionQuery> for CausalQuery {
    fn from(query: InterventionalDistributionQuery) -> Self {
        Self::Distribution(query)
    }
}

impl From<PathSpecificEffectQuery> for CausalQuery {
    fn from(query: PathSpecificEffectQuery) -> Self {
        Self::PathSpecific(query)
    }
}

impl From<AnomalyAttributionQuery> for CausalQuery {
    fn from(query: AnomalyAttributionQuery) -> Self {
        Self::AnomalyAttribution(query)
    }
}

impl From<ChangeAttributionQuery> for CausalQuery {
    fn from(query: ChangeAttributionQuery) -> Self {
        Self::ChangeAttribution(query)
    }
}

impl From<MechanismChangeQuery> for CausalQuery {
    fn from(query: MechanismChangeQuery) -> Self {
        Self::MechanismChange(query)
    }
}

impl From<UnitChangeQuery> for CausalQuery {
    fn from(query: UnitChangeQuery) -> Self {
        Self::UnitChange(query)
    }
}

impl From<ResponseQuery> for CausalQuery {
    fn from(query: ResponseQuery) -> Self {
        Self::Response(query)
    }
}

impl From<TransportQuery> for CausalQuery {
    fn from(query: TransportQuery) -> Self {
        Self::Transport(query)
    }
}

impl From<InterferenceQuery> for CausalQuery {
    fn from(query: InterferenceQuery) -> Self {
        Self::Interference(query)
    }
}

impl From<RandomizedEffectQuery> for CausalQuery {
    fn from(query: RandomizedEffectQuery) -> Self {
        Self::RandomizedEffect(query)
    }
}

impl From<PanelDidQuery> for CausalQuery {
    fn from(query: PanelDidQuery) -> Self {
        Self::PanelDid(query)
    }
}

impl From<SyntheticControlQuery> for CausalQuery {
    fn from(query: SyntheticControlQuery) -> Self {
        Self::SyntheticControl(query)
    }
}

impl From<LocalPolynomialRatioQuery> for CausalQuery {
    fn from(query: LocalPolynomialRatioQuery) -> Self {
        Self::LocalPolynomialRatio(query)
    }
}

impl From<SurvivalQuery> for CausalQuery {
    fn from(query: SurvivalQuery) -> Self {
        Self::Survival(query)
    }
}

impl From<LongitudinalRegimeQuery> for CausalQuery {
    fn from(query: LongitudinalRegimeQuery) -> Self {
        Self::LongitudinalRegime(query)
    }
}

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
            | Self::LongitudinalRegime(_) => None,
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
            | Self::LongitudinalRegime(_) => None,
        }
    }

    /// Validate the inner query.
    ///
    /// # Errors
    ///
    /// Propagates inner [`QueryError`].
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::AverageEffect(q) => q.validate(),
            Self::TemporalEffect(q) => q.validate(),
            Self::Counterfactual(q) => q.validate(),
            Self::NestedCounterfactual(q) => q.validate(),
            Self::AnomalyAttribution(q) => q.validate(),
            Self::ChangeAttribution(q) => q.validate(),
            Self::MechanismChange(q) => q.validate(),
            Self::UnitChange(q) => q.validate(),
            Self::Mediation(q) => q.validate(),
            Self::ConditionalEffect(q) => q.validate(),
            Self::Distribution(q) => q.validate(),
            Self::PathSpecific(q) => q.validate(),
            Self::Response(q) => q.validate(),
            Self::Transport(q) => q.validate(),
            Self::Interference(q) => q.validate(),
            Self::RandomizedEffect(q) => q.validate(),
            Self::PolicyValue(q) => q.validate(),
            Self::ContinuousDoseResponse(q) => q.validate(),
            Self::PanelDid(q) => q.validate(),
            Self::SyntheticControl(q) => q.validate(),
            Self::LocalPolynomialRatio(q) => q.validate(),
            Self::Survival(q) => q.validate(),
            Self::LongitudinalRegime(q) => q.validate(),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
