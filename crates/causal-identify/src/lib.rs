//! Causal identification algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod backdoor;
pub mod efficient;
pub mod error;
pub mod frontdoor;
pub mod iv;
pub mod result;
pub mod temporal_backdoor;

pub use backdoor::{AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph};
pub use efficient::EfficientBackdoorIdentifier;
pub use error::IdentificationError;
pub use frontdoor::{FrontDoorIdentifier, FrontDoorSearchConfig};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use result::{
    DerivationStep, DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{TemporalBackdoorIdentifier, TemporalIdentificationResult};
