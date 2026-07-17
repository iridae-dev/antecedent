//! Causal discovery algorithms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod algorithm;
pub mod ci;
pub mod combinations;
pub mod constraints;
pub mod discriminating_paths;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod jpcmci_plus;
pub mod lpcmci;
pub mod lpcmci_phases;
pub mod orientation;
pub mod pcmci;
pub mod pcmci_plus;
pub mod pipeline;
pub mod result;
pub mod rpcmci;
pub mod rule_scheduling;
pub mod uncovered_paths;
pub mod weakly_minimal;

pub use algorithm::DiscoveryAlgorithm;
pub use ci::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

pub use constraints::{
    CandidateCatalog, CompiledConstraints, ContextKind, CrossEnvLinkAssumption,
    DiscoveryConstraints, JpcmciNodeRole, MultiDatasetConstraints, SpaceDummyCiMode,
    TemporalConstraints,
};
pub use discriminating_paths::{DiscriminatingPath, find_discriminating_paths};
pub use engine::{DiscoveryWorkspace, PcmciEngine};
pub use error::DiscoveryError;
pub use evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, graph_evidence_from_scored,
    graph_evidence_from_scored_with_sepsets, pag_evidence_from_oriented, pag_from_scored_links,
    symmetrize_contemporaneous_links, threshold_scored_links, threshold_scored_links_bh,
};
pub use causal_stats::{FdrAdjustment, MultipleTestingMethod};
pub use jpcmci_plus::{JpcmciPlus, JpcmciPlusDiscoveryResult};
pub use lpcmci::Lpcmci;
pub use orientation::{
    ContempMeekR1, ContempMeekR2, ContempMeekR3, MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider,
    OrientationError, OrientationQueue, OrientationRule, OrientationState, RuleDelta,
    run_orientation_to_fixed_point,
};
pub use pcmci::Pcmci;
pub use pcmci_plus::PcmciPlus;
pub use pipeline::{
    algorithm_record, lagged_node_index, orientation_state_from_sepsets, push_diagnostic,
    with_links_retained,
};
pub use result::{
    AlgorithmRecord, CpdagDiscoveryResult, CpdagGraphEvidence, DagDiscoveryResult,
    DagGraphEvidence, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, LaggedParent,
    PagDiscoveryResult, PagGraphEvidence, PcSepsets, ScoredLink, SepsetKey,
};
pub use rpcmci::{
    RegimeAssignment, RegimeGraphCollection, Rpcmci, RpcmciDiscoveryResult, regime_edge_counts,
    two_regime_half_split,
};
pub use rule_scheduling::{
    LpcmciApr, LpcmciDiscriminatingPathRule, LpcmciMmr, LpcmciOrientCollider, LpcmciOrientationRule,
    LpcmciR1, LpcmciR10, LpcmciR2, LpcmciR3, LpcmciR8, LpcmciR9, default_lpcmci_rules,
    run_lpcmci_orientation,
};
