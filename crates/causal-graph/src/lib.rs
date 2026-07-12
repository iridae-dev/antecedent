//! Causal graph types, dense indexes, and traversal workspaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod dag;
pub mod error;
pub mod temporal;
pub mod types;
pub mod workspace;

pub use dag::Dag;
pub use error::GraphError;
pub use temporal::TemporalDag;
pub use types::{DenseNodeId, Endpoint, MarkedEdge, NodeRef};
pub use workspace::{BitSet, GraphWorkspace};
