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
pub mod cross_world;
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
    BoundMixedSourceFunctional, MIXED_SOURCE_DEFAULT_LIMITS, MIXED_SOURCE_MAX_DISTRIBUTIONS,
    MIXED_SOURCE_MAX_DO, MIXED_SOURCE_MAX_MOVE, MIXED_SOURCE_MAX_OBSERVED, MIXED_SOURCE_RULE_SET,
    MixedExclusion, MixedInput, MixedMissingEvidence, MixedMissingLeaf, MixedQuantity, MixedRule,
    MixedSearchInspection, MixedSearchSummary, MixedSourceDecision, MixedSourceDerivation,
    MixedSourceDerivationRecord, MixedSourceLeaf, MixedSourceQuery, MixedStageRecord, MixedStep,
    MixedStepRecord, ValidatedMixedSourceQuery, bind_mixed_source_catalog, decide_mixed_source,
    mixed_source_rule_names, render_frontier, render_quantity, validate_mixed_source_query,
    verify_mixed_source_derivation,
};
pub use sid::{
    BoundMzTransportFunctional, BoundTransportFunctional, BoundZTransportFunctional,
    CatalogTransportResult, CheckedTransportDerivation, ClassicalTransportDerivation,
    ClassicalTransportQuery, ClassicalTransportResult, ComponentFactorization,
    MZ_TRANSPORT_DEFAULT_LIMITS, MZ_TRANSPORT_MAX_CANDIDATE_REGIMES,
    MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE, MZ_TRANSPORT_MAX_OBSERVED, MZ_TRANSPORT_MAX_SOURCES,
    MZ_TRANSPORT_MEMORY_BYTES, mz_transport_refusal, MetaSource, MetaTransportQuery, MzSearchInspection,
    MzSearchRecord, MzStageRecord, MzTransportDecision,
    MzTransportDerivation, MzTransportDerivationRecord, MzTransportObstruction, MzTransportQuery,
    MzTransportRoute, SidLimits, TwoSourceZTransportComponent, TwoSourceZTransportDecision,
    TwoSourceZTransportQuery, ValidatedMzTransportQuery, Z_TRANSPORT_MAX_CONTROLLABLE,
    Z_TRANSPORT_MAX_FAMILY_REGIMES, Z_TRANSPORT_MAX_OBSERVED, ZExperimentFamilyError,
    ZFactorObligation, ZProofOperation, ZTransportBudgetKind, ZTransportDecision,
    ZTransportDerivation, ZTransportDerivationRecord, ZTransportLimitsReceipt,
    ZTransportMissingEvidence, ZTransportNotCertifiedInspection, ZTransportNotCertifiedKind,
    ZTransportObstruction, ZTransportObstructionRecord, ZTransportOutcome,
    ZTransportProofInspection, ZTransportQuery, ZTransportResult, ZTransportSourceSpec,
    ZTransportTerminalRecord, bind_mz_transport_catalog, bind_z_transport_catalog,
    decide_mz_transport, decide_two_source_z_transport, decide_z_transport_inspecting,
    decide_z_transport_with_catalog, identify_catalog_transport, identify_classical_transport,
    identify_meta_catalog, identify_meta_transport, identify_z_transport,
    identify_z_transport_reporting, validate_mz_transport_query, validate_z_experiment_family,
    validate_z_transport_query, verify_classical_transport, verify_meta_s_hedge,
    verify_meta_transport, verify_mz_transport_obstruction, verify_z_transport_derivation,
    verify_z_transport_obstruction,
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
    AdjustmentSearchConfig, BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE, BackdoorIdentifier,
    PreparedIdentificationGraph, RankedAdjustmentSet,
};
pub use bounds::{BinaryIvLaw, binary_iv_ate_bounds};
pub use efficient::EfficientBackdoorIdentifier;
pub use envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
    carries_identified_mass, search_truncated,
};
pub use error::{IdentificationBudget, IdentificationError};
pub use frontdoor::{
    FRONTDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE, FrontDoorIdentifier, FrontDoorSearchConfig,
};
pub use generalized::{
    CAPPED_COMPLETION_DIAGNOSTIC_CODE, CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
    GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier,
    MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
};
pub use hedge::{HedgeCertificate, HedgeProblem};
pub use id::IdIdentifier;
pub use idc::IdcIdentifier;
pub use identifier::{IdentificationWorkspace, Identifier};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use joint_response::JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE;
pub use path_specific::PathSpecificIdentifier;
pub use prepared::{PreparedAdmg, dag_to_admg};
pub use rd::{SharpRdConfig, SharpRdIdentifier};
pub use response::ResponseIdentifier;
pub use response_id::{identify_cpdag_response_general, identify_pag_response_general};
pub use result::{
    DerivationStep, DerivationTrace, EstimandClaim, IdentificationPerformanceRecord,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{
    PARENT_ADJUSTMENT_RULE, TemporalBackdoorIdentifier, TemporalIdentificationResult,
};
pub use temporal_generalized::{TemporalClassEnvelope, TemporalCompletionGraph};
pub use temporal_mediation::TemporalMediationIdentifier;
pub use tiered::{
    NO_LATENT_TO_OUTCOME, TIERED_ADJUSTMENT_REFUSE, TIERED_JOINT_ADJUSTMENT_REFUSE,
    TIERED_JOINT_UNKNOWN_REFUSE, identify_tiered, identify_tiered_joint, identify_tiered_joint_on,
    identify_tiered_on,
};
pub use transport::{
    MissingEvidenceCertificate, NotCertifiedCertificate, PopulationFactor, TransportCertificate,
    TransportFormula, TransportIdentification, TransportIdentifier,
};
pub use transport_lower::{
    bind_transport_derivation, lower_transport_formula, lower_transport_mean,
};

mod joint_response;
