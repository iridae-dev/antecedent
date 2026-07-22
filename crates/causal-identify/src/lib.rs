//! Causal identification algorithms.
//!
//! Identify first; estimate second. Primary entry points include
//! [`BackdoorIdentifier`], [`FrontDoorIdentifier`], [`IdIdentifier`],
//! [`IdcIdentifier`], and [`AutoIdentifier`].
//!
//! ```
//! use causal_identify::BackdoorIdentifier;
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
pub mod result;
pub mod temporal_backdoor;
pub mod temporal_mediation;

#[cfg(test)]
mod id_scm_property;

pub use auto::{AutoIdentifier, PreparedAutoGraph};
pub use backdoor::{
    AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph, RankedAdjustmentSet,
};
pub use efficient::EfficientBackdoorIdentifier;
pub use envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
pub use error::IdentificationError;
pub use frontdoor::{FrontDoorIdentifier, FrontDoorSearchConfig};
pub use generalized::{GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier};
pub use hedge::HedgeCertificate;
pub use id::IdIdentifier;
pub use idc::IdcIdentifier;
pub use identifier::{IdentificationWorkspace, Identifier};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use path_specific::PathSpecificIdentifier;
pub use prepared::{PreparedAdmg, dag_to_admg};
pub use rd::{SharpRdConfig, SharpRdIdentifier};
pub use result::{
    DerivationStep, DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{TemporalBackdoorIdentifier, TemporalIdentificationResult};
pub use temporal_mediation::TemporalMediationIdentifier;
