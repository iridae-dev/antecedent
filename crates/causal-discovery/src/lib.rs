//! Causal discovery algorithms.
//!
//! Shipped surface: PCMCI family ([`Pcmci`], [`PcmciPlus`], [`Lpcmci`],
//! [`JpcmciPlus`], [`Rpcmci`]), static [`Pc`], classic static [`Fci`] / [`Rfci`] →
//! [`causal_graph::Pag`], score-based [`Ges`] → [`causal_graph::Cpdag`],
//! [`DirectLingam`] → [`causal_graph::Dag`], and Zhang FCI orientation plumbing
//! via [`PagOps`] / [`FciOrientationRule`] (DESIGN.md §13.3 / §13.6).
//! NOTEARS remains optional / unshipped.
//!
//! ```
//! use causal_discovery::Pcmci;
//!
//! let alg = Pcmci::new();
//! let _ = alg;
//! ```
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
pub mod fci;
pub mod ges;
pub mod jpcmci_plus;
pub mod lpcmci;
pub mod lpcmci_phases;
pub mod lingam;
pub mod orientation;
pub mod pc;
pub mod pcmci;
pub mod pcmci_family;
pub mod pcmci_plus;
pub mod pipeline;
pub mod possible_d_sep;
pub mod result;
pub mod rfci;
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
pub use fci::{Fci, StaticPagDiscoveryResult};
pub use ges::Ges;
pub use jpcmci_plus::{JpcmciPlus, JpcmciPlusDiscoveryResult};
pub use lingam::{DirectLingam, StaticDagDiscoveryResult};
pub use lpcmci::Lpcmci;
pub use orientation::{
    OrientCollider, OrientationError, OrientationRule, OrientationState, PagOps,
    StaticOrientationRule, run_static_orientation_to_fixed_point,
};
pub use pc::{Pc, StaticCpdagDiscoveryResult};
pub use pcmci::Pcmci;
pub use pcmci_plus::PcmciPlus;
pub use possible_d_sep::{possible_d_sep, pds_triple_ok, PossibleDSepBudget};
pub use result::{
    AlgorithmRecord, CpdagDiscoveryResult, CpdagGraphEvidence, DagDiscoveryResult,
    DagGraphEvidence, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, LaggedParent,
    PagDiscoveryResult, PagGraphEvidence, PcSepsets, ScoredLink, SepsetKey,
};
pub use rfci::Rfci;
pub use rpcmci::{
    RegimeAssignment, RegimeGraphCollection, Rpcmci, RpcmciDiscoveryResult, regime_edge_counts,
    two_regime_half_split,
};
pub use rule_scheduling::{
    FciOrientationRule, LpcmciApr, LpcmciDiscriminatingPathRule, LpcmciMmr, LpcmciOrientCollider,
    LpcmciOrientationRule, LpcmciR1, LpcmciR10, LpcmciR2, LpcmciR3, LpcmciR8, LpcmciR9,
    default_fci_rules, default_lpcmci_rules, run_fci_orientation_to_fixed_point,
    run_lpcmci_orientation,
};

// Additional orientation helpers stay under their modules for advanced callers:
// `causal_discovery::orientation::*` and `causal_discovery::rule_scheduling::*`.
