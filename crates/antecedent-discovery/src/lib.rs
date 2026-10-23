//! Causal discovery algorithms.
//!
//! Shipped surface: PCMCI family ([`Pcmci`], [`PcmciPlus`], [`Lpcmci`],
//! [`JpcmciPlus`], [`Rpcmci`]), static [`Pc`], classic static [`Fci`] / [`Rfci`] →
//! [`antecedent_graph::Pag`], score-based [`Ges`] → [`antecedent_graph::Cpdag`],
//! [`DirectLingam`] / [`Notears`] → [`antecedent_graph::Dag`], Bayesian graph discovery
//! ([`ExactDagPosterior`], [`StructureMcmc`], [`OrderMcmc`], [`CiScreenedPosterior`],
//! [`DbnPosterior`]; ), and Zhang FCI orientation plumbing via
//! [`PagOps`] / [`FciOrientationRule`].
//!
//! ```
//! use antecedent_discovery::Pcmci;
//!
//! let alg = Pcmci::new();
//! let _ = alg;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod debug_ci {
    macro_rules! debug_with_ci {
        ($ty:ident $(, $field:ident)*) => {
            impl std::fmt::Debug for $ty {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.debug_struct(stringify!($ty))
                        .field("constraints", &self.constraints)
                        .field("ci", &"<dyn ConditionalIndependence>")
                        .field("fdr", &self.fdr)
                        $(.field(stringify!($field), &self.$field))*
                        .finish()
                }
            }
        };
    }
    pub(crate) use debug_with_ci;
}

pub(crate) use debug_ci::debug_with_ci;

pub mod adjacency_mask;
pub mod ci;
pub mod ci_screened_posterior;
pub mod combinations;
pub mod conflict_resolution;
pub mod constraints;
pub mod cyclic_fci;
pub mod dbn_posterior;
pub mod discriminating_paths;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod exact_enumeration;
pub mod exact_marginals;
pub mod exact_pag_posterior;
pub mod fci;
pub mod ges;
pub mod graph_mcmc;
pub mod graph_posterior;
#[cfg(test)]
mod graph_posterior_calibration;
pub mod graph_score;
mod indexing;
pub mod jpcmci_plus;
pub mod lingam;
pub mod lpcmci;
pub mod lpcmci_phases;
pub mod notears;
pub mod order_mcmc;
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
pub mod selective;
pub mod stability;
mod static_skeleton;
pub mod structure_mcmc;
#[cfg(test)]
mod test_tables;
pub mod uncovered_paths;
pub mod weakly_minimal;

pub use adjacency_mask::{AdjacencyMask, MAX_POSTERIOR_NODES};
pub use ci::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

pub use antecedent_stats::{FdrAdjustment, MultipleTestingMethod};
pub use ci_screened_posterior::{CiScreenedPosterior, CiSoftWeight};
pub use constraints::{
    BackgroundKnowledge, CandidateCatalog, CiMaskType, CompiledConstraints, ContextKind,
    CrossEnvLinkAssumption, DiscoveryConstraints, JpcmciNodeRole, KnowledgeFault, KnowledgeScope,
    MultiDatasetConstraints, SpaceDummyCiMode, TemporalConstraints, TimeDummyCiMode,
};
pub use dbn_posterior::{
    DbnPosterior, temporal_cpdag_from_dbn_masks, temporal_dag_from_dbn_masks,
    temporal_pag_from_dbn_masks,
};
pub use discriminating_paths::DiscriminatingPathBudget;
pub use engine::{DiscoveryWorkspace, PcmciEngine};
pub use error::DiscoveryError;
pub use evidence::threshold_scored_links;
pub use exact_enumeration::ExactDagPosterior;
pub use exact_marginals::{EXACT_MARGINAL_MAX_NODES, ExactDagMarginals, exact_dag_marginals};
pub use exact_pag_posterior::{
    PAG_ENUM_MAX_NODES, PAG_MCMC_MAX_NODES, PagMcmcSchedule, exact_pag_posterior,
    pag_posterior_mcmc,
};
pub use fci::{Fci, StaticPagDiscoveryResult};
pub use ges::Ges;
pub use graph_posterior::{
    EXACT_ENUM_MAX_NODES, GraphPosterior, GraphPosteriorAtomKind, GraphPosteriorEngine, GraphPrior,
    SamplingSummary, adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask, edge_bit,
    has_edge, mask_is_dag, n_directed_edges, pag_from_adjacency_mask, parents_of, set_edge,
};
pub use jpcmci_plus::JpcmciPlus;
pub use lingam::{DirectLingam, StaticDagDiscoveryResult};
pub use lpcmci::Lpcmci;
pub use notears::{Notears, NotearsDiscoveryResult};
pub use order_mcmc::OrderMcmc;
pub use orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationError, OrientationQueue,
    OrientationRule, OrientationState, PagOps, RuleDelta, run_orientation_to_fixed_point,
};
pub use pc::{Pc, StaticCpdagDiscoveryResult};
pub use pcmci::Pcmci;
pub use pcmci_plus::PcmciPlus;
pub use possible_d_sep::{PossibleDSepBudget, possible_d_sep};
pub use result::{
    AlgorithmRecord, CpdagDiscoveryResult, DagDiscoveryResult, DiscoveryDiagnostic,
    DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult, EdgeEvidence, EvidenceSource,
    GraphEvidence, LaggedLink, LaggedParent, PagDiscoveryResult, PcSepsets, ScoredLink,
};
pub use rfci::Rfci;
pub use rpcmci::{
    RegimeAssignment, RegimeGraphCollection, Rpcmci, RpcmciDiscoveryResult, two_regime_half_split,
};
pub use rule_scheduling::{
    LpcmciDiscriminatingPathRule, LpcmciOrientCollider, LpcmciOrientationRule, LpcmciR1,
    run_lpcmci_orientation,
};
pub use stability::{EdgeSelection, MarkSelection, StructureSelection, StructureStability};
pub use structure_mcmc::StructureMcmc;

// Additional orientation helpers stay under their modules for advanced callers:
// `antecedent_discovery::orientation::*` and `antecedent_discovery::rule_scheduling::*`.
