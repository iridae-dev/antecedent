//! Causal discovery algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ci;
pub mod combinations;
pub mod constraints;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod orientation;
pub mod pcmci;
pub mod pcmci_plus;
pub mod result;

pub use ci::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

pub use constraints::{
    CandidateCatalog, CompiledConstraints, DiscoveryConstraints, TemporalConstraints,
};
pub use engine::{DiscoveryWorkspace, PcmciEngine};
pub use error::DiscoveryError;
pub use evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, graph_evidence_from_scored,
    graph_evidence_from_scored_with_sepsets, threshold_scored_links,
};
pub use orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationError, OrientationQueue,
    OrientationRule, OrientationState, RuleDelta, run_orientation_to_fixed_point,
};
pub use pcmci::Pcmci;
pub use pcmci_plus::PcmciPlus;
pub use result::{
    AlgorithmRecord, CpdagDiscoveryResult, CpdagGraphEvidence, DagDiscoveryResult, DagGraphEvidence,
    DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult,
    EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, LaggedParent, PcSepsets, ScoredLink,
    SepsetKey,
};
