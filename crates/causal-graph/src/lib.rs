//! Causal graph types, dense indexes, and traversal workspaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod admg;
pub mod algo;
pub mod ancestry;
pub mod completion;
pub mod cpdag;
pub mod dag;
pub mod dsep;
pub mod error;
pub(crate) mod marked_storage;
pub mod msep;
pub mod overlay;
pub mod pag;
pub mod projection;
pub mod temporal;
pub mod temporal_pag;
pub mod types;
pub mod unfold;
pub mod workspace;

pub use admg::Admg;
pub use completion::{CompletionSampler, PagCompletion, is_mag_completion};
pub use cpdag::{Cpdag, TemporalCpdag};
pub use dag::Dag;
pub use dsep::{DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult};
pub use error::GraphError;
pub use overlay::{DagView, GraphOverlay};
pub use pag::{DefiniteStatusPath, DefiniteStatusPathSearch, Pag};
pub use projection::{latent_project, projection_preserves_msep_sample};
pub use temporal::TemporalDag;
pub use temporal_pag::{TemporalPag, TemporalPagReview};
pub use causal_core::NodeRef;
pub use types::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark};
pub use unfold::{
    LazyUnfoldedTemporalGraph, TemporalCpdagReview, TemporalGraphReview, UnfoldedTemporalGraph,
    ensure_lagged,
};
pub use workspace::{BitSet, GraphWorkspace};
