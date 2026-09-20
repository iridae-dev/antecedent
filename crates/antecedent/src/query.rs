//! Typed causal queries (re-exported from `antecedent-core`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExposureLevel, ExposureMapping, GridSpec,
    InterferenceFunctional, InterferenceQuery, InterventionalDistributionQuery,
    MechanismChangeQuery, MediationContrast, MediationQuery, OutcomeFunctional,
    PathSpecificEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec,
    TransportQuery, UnitChangeQuery,
};
