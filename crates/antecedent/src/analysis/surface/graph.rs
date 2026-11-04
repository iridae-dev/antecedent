//! Graph types and construction helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_core::Lag;
pub use antecedent_graph::{
    Admg, CompletionSampler, Cpdag, CpdagCompletion, CpdagCompletionSampler, CpdagReview, Dag,
    DagReview, DenseNodeId, Pag, PagCompletion, PagReview, SelectionDiagram, TemporalCpdag,
    TemporalDag, TemporalPag, TemporalPagReview, TieredBackground, WithinTier, ensure_lagged,
    is_mec_member, latent_project,
};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// d-separation for DAGs.
pub use antecedent_graph::DSeparationWorkspace;
/// Graph construction and validation errors.
pub use antecedent_graph::GraphError;
/// `antecedent-graph` items at the crate root.
pub use antecedent_graph::NodeRef;
/// m-separation for ADMGs, and class-wide m-separation for PAGs.
pub use antecedent_graph::PagSeparation;
/// Temporal ADMG: lagged nodes, directed edges, and bidirected latent edges.
pub use antecedent_graph::TemporalAdmg;
/// Dense ids and edge endpoints.
pub use antecedent_graph::{Endpoint, MarkedEdge, MiddleMark};
/// Lazy finite unfolding of temporal DAGs and graph-review artifacts.
pub use antecedent_graph::{TemporalCpdagReview, TemporalGraphReview};
