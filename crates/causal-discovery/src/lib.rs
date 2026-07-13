//! Causal discovery algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod combinations;
pub mod constraints;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod orientation;
pub mod pcmci;
pub mod result;

pub use constraints::{
    CandidateCatalog, CompiledConstraints, DiscoveryConstraints, TemporalConstraints,
};
pub use engine::{DiscoveryWorkspace, PcmciEngine};
pub use error::DiscoveryError;
pub use evidence::graph_evidence_from_scored;
pub use orientation::{
    MeekR1, MeekR2, OrientCollider, OrientationError, OrientationQueue, OrientationRule,
    OrientationState, RuleDelta, run_orientation_to_fixed_point,
};
pub use pcmci::Pcmci;
pub use result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, GraphEvidence, LaggedLink, ScoredLink,
};
