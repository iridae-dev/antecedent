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
    AttributeQuery, AttributeTarget, ChangeEstimator, ChangeMeasure, DiscoverQuery,
    DiscoveredClass, DiscoveryInference, DiscoveryMethod, InfluenceAllocation,
    IntrinsicInfluenceTarget, MechanismChangeTest, NoiseDeclaration, ScreenWeight, SearchBounds,
    SearchSchedule, SimulateQuery, SimulationTarget,
};
pub use antecedent_core::{
    OutcomeMisclassification, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget,
};
