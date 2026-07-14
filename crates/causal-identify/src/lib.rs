//! Causal identification algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod assumptions;
pub mod backdoor;
pub mod efficient;
pub mod envelope;
pub mod error;
pub mod frontdoor;
pub mod generalized;
pub mod iv;
pub mod rd;
pub mod result;
pub mod temporal_backdoor;

pub use backdoor::{AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph};
pub use efficient::EfficientBackdoorIdentifier;
pub use envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
pub use error::IdentificationError;
pub use frontdoor::{FrontDoorIdentifier, FrontDoorSearchConfig};
pub use generalized::{GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use rd::{SharpRdConfig, SharpRdIdentifier};
pub use result::{
    DerivationStep, DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{TemporalBackdoorIdentifier, TemporalIdentificationResult};
