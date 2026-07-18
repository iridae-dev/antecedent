//! Causal identification algorithms.
//!
//! Identify first; estimate second. Primary entry points include
//! [`BackdoorIdentifier`], [`FrontDoorIdentifier`], and
//! [`InstrumentalVariableIdentifier`]. Full ID/IDC (`AutoIdentifier`) is not
//! shipped yet (DESIGN.md §10).
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
pub mod backdoor;
pub(crate) mod enum_masks;
pub mod efficient;
pub mod envelope;
pub mod error;
pub mod frontdoor;
pub mod generalized;
pub mod identifier;
pub mod iv;
pub mod rd;
pub mod result;
pub mod temporal_backdoor;
pub mod temporal_mediation;

#[cfg(test)]
mod id_scm_property;

pub use backdoor::{AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph};
pub use efficient::EfficientBackdoorIdentifier;
pub use envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
pub use error::IdentificationError;
pub use frontdoor::{FrontDoorIdentifier, FrontDoorSearchConfig};
pub use generalized::{GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier};
pub use identifier::{IdentificationWorkspace, Identifier};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use rd::{SharpRdConfig, SharpRdIdentifier};
pub use result::{
    DerivationStep, DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{TemporalBackdoorIdentifier, TemporalIdentificationResult};
pub use temporal_mediation::TemporalMediationIdentifier;
