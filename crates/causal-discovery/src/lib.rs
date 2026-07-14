//! Causal discovery algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ci;
pub mod combinations;
pub mod constraints;
pub mod discriminating_paths;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod lpcmci;
pub mod orientation;
pub mod pcmci;
pub mod pcmci_plus;
pub mod result;
pub mod rule_scheduling;

pub use ci::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

pub use constraints::{
    CandidateCatalog, CompiledConstraints, CrossEnvLinkAssumption, DiscoveryConstraints,
    MultiDatasetConstraints, TemporalConstraints,
};
pub use discriminating_paths::{DiscriminatingPath, find_discriminating_paths};
pub use engine::{DiscoveryWorkspace, PcmciEngine};
pub use error::DiscoveryError;
pub use evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, graph_evidence_from_scored,
    graph_evidence_from_scored_with_sepsets, pag_evidence_from_oriented, pag_from_scored_links,
    threshold_scored_links,
};
pub use lpcmci::Lpcmci;
pub use orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationError, OrientationQueue,
    OrientationRule, OrientationState, RuleDelta, run_orientation_to_fixed_point,
};
pub use pcmci::Pcmci;
pub use pcmci_plus::PcmciPlus;
pub use result::{
    AlgorithmRecord, CpdagDiscoveryResult, CpdagGraphEvidence, DagDiscoveryResult,
    DagGraphEvidence, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, LaggedParent,
    PagDiscoveryResult, PagGraphEvidence, PcSepsets, ScoredLink, SepsetKey,
};
pub use rule_scheduling::{
    LpcmciDiscriminatingPathRule, LpcmciOrientCollider, LpcmciOrientationRule, LpcmciR1,
    run_lpcmci_orientation,
};
