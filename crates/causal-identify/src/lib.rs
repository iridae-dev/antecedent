//! Causal identification algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod backdoor;
pub mod error;
pub mod result;
pub mod temporal_backdoor;

pub use backdoor::{AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph};
pub use error::IdentificationError;
pub use result::{
    DerivationStep, DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::TemporalBackdoorIdentifier;
