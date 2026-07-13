//! Causal discovery algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod constraints;
pub mod engine;
pub mod error;
pub mod result;

pub use constraints::{DiscoveryConstraints, TemporalConstraints};
pub use engine::{DiscoveryWorkspace, PcmciEngine};
pub use error::DiscoveryError;
pub use result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, GraphEvidence, LaggedLink, ScoredLink,
};
