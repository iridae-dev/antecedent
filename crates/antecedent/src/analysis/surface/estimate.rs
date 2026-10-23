//! Estimation stage types and strategy identifiers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_data::{NetworkData, NetworkEdge};

pub use antecedent_estimate::{
    CausalPosterior, ConditionalLinearAdjustment, ContinuousResponseOptions, EffectEstimate,
    EstimationError, InterferenceEstimate, OverlapPolicy, ResponseGridInputs,
    TemporalEffectSurface, TemporalLinearPredictor, TemporalMediationEstimator,
    TemporalMediationGrid, TemporalMediationIdentifiedSet, TemporalMediationSlice,
    TemporalMediationUncertainty, TransportEffectEstimate, TransportOverlapDiagnostic,
    TransportOverlapReport, TransportResponseGridEstimate, own_treatment_level,
    transport_augmented_response_grid,
};

pub use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, EstimatorId, IdentifierId, StaticEstimateWorkspaces,
    estimand_compatible_with_estimator, estimate_provenance_step, identification_status_acceptable,
    identify_provenance_step, identify_static_query, identify_static_query_with_rd,
    require_identified, select_estimand, validate_distribution_pair, validate_path_specific_pair,
    validate_static_pair,
};

pub use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    ExactSupportRecord, ExactTransportData, LawTolerance,
};

pub use antecedent_estimate::white_innovation_posterior::{
    InnovationWhitenessGate, InnovationWhitenessReport,
};
