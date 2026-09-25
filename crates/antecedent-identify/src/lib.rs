//! Causal identification algorithms.
//!
//! Identify first; estimate second. Primary entry points include
//! [`BackdoorIdentifier`], [`FrontDoorIdentifier`], [`IdIdentifier`],
//! [`IdcIdentifier`], and [`AutoIdentifier`].
//!
//! ```
//! use antecedent_identify::BackdoorIdentifier;
//!
//! let id = BackdoorIdentifier::new();
//! let _ = id;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod assumptions;
pub mod auto;
pub mod backdoor;
pub mod bounds;
pub mod efficient;
pub(crate) mod enum_masks;
pub mod envelope;
pub mod error;
pub mod frontdoor;
pub mod generalized;
pub mod hedge;
pub mod id;
pub mod idc;
pub mod identifier;
pub(crate) mod intervention_support;
pub mod iv;
pub mod path_specific;
pub mod prepared;
pub mod rd;
pub mod response;
pub(crate) mod response_id;
pub mod result;
pub(crate) mod selection_separation;
pub mod sid;
pub mod temporal_backdoor;
pub mod temporal_generalized;
mod temporal_mag;
pub mod temporal_mediation;
pub mod tiered;
pub mod transport;
pub use sid::{
    bind_z_transport_catalog, decide_z_transport_with_catalog, identify_catalog_transport,
    identify_classical_transport, identify_meta_catalog, identify_meta_transport,
    identify_z_transport, identify_z_transport_surrogate, identify_z_transport_with_limits,
    validate_z_experiment_family, validate_z_transport_query, verify_classical_transport,
    verify_meta_s_hedge, verify_meta_transport, verify_z_transport_derivation,
    verify_z_transport_obstruction, BoundTransportFunctional, BoundZTransportFunctional,
    CatalogTransportResult, CheckedTransportDerivation, ClassicalTransportDerivation,
    ClassicalTransportQuery, ClassicalTransportResult, MetaSource, MetaTransportQuery, SidLimits,
    ZExperimentFamilyError, ZFactorObligation, ZProofOperation, ZTransportDecision,
    ZTransportDerivation, ZTransportDerivationRecord, ZTransportMissingEvidence,
    ZTransportObstruction, ZTransportObstructionRecord, ZTransportProofInspection, ZTransportQuery,
    ZTransportResult, ZTransportTerminalRecord, Z_TRANSPORT_MAX_CONTROLLABLE,
    Z_TRANSPORT_MAX_FAMILY_REGIMES, Z_TRANSPORT_MAX_OBSERVED,
};
mod transport_lower;

#[cfg(test)]
mod id_scm_property;
#[cfg(test)]
mod mag_id_bruteforce;
/// Hidden parser for the frozen external `graph_dot` oracles used by tests.
///
/// Compiled only for this crate's tests and under the `test-util` feature: it panics on
/// malformed input, which is fine for frozen fixtures and wrong for a library API.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub mod oracle_dot;

pub use auto::{AutoIdentifier, PreparedAutoGraph};
pub use backdoor::{
    AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph, RankedAdjustmentSet,
    BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
};
pub use bounds::{binary_iv_ate_bounds, BinaryIvLaw};
pub use efficient::EfficientBackdoorIdentifier;
pub use envelope::{
    carries_identified_mass, search_truncated, GraphFeature, GraphIdentificationCase,
    IdentificationEnvelope, ProbabilityMass,
};
pub use error::IdentificationError;
pub use frontdoor::{
    FrontDoorIdentifier, FrontDoorSearchConfig, FRONTDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
};
pub use generalized::{
    GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier,
    CAPPED_COMPLETION_DIAGNOSTIC_CODE, CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
    MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
};
pub use hedge::{HedgeCertificate, HedgeProblem};
pub use id::IdIdentifier;
pub use idc::IdcIdentifier;
pub use identifier::{IdentificationWorkspace, Identifier};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use joint_response::JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE;
pub use path_specific::PathSpecificIdentifier;
pub use prepared::{dag_to_admg, PreparedAdmg};
pub use rd::{SharpRdConfig, SharpRdIdentifier};
pub use response::ResponseIdentifier;
pub use response_id::{identify_cpdag_response_general, identify_pag_response_general};
pub use result::{
    DerivationStep, DerivationTrace, EstimandClaim, IdentificationPerformanceRecord,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{
    TemporalBackdoorIdentifier, TemporalIdentificationResult, PARENT_ADJUSTMENT_RULE,
};
pub use temporal_generalized::{TemporalClassEnvelope, TemporalCompletionGraph};
pub use temporal_mediation::TemporalMediationIdentifier;
pub use tiered::{
    identify_tiered, identify_tiered_joint, identify_tiered_joint_on, identify_tiered_on,
    NO_LATENT_TO_OUTCOME, TIERED_ADJUSTMENT_REFUSE, TIERED_JOINT_ADJUSTMENT_REFUSE,
    TIERED_JOINT_UNKNOWN_REFUSE,
};
pub use transport::{
    MissingEvidenceCertificate, NotCertifiedCertificate, PopulationFactor, TransportCertificate,
    TransportFormula, TransportIdentification, TransportIdentifier,
};
pub use transport_lower::{
    bind_transport_derivation, lower_transport_formula, lower_transport_mean,
};

mod joint_response;
