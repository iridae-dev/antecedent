//! Analysis provenance graph (DESIGN.md §7, §21.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::assumption::AssumptionSet;

/// Stable artifact identifier string.
pub type ArtifactId = Arc<str>;

/// Node in the provenance graph describing how an artifact was produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceNode {
    /// Artifact this node describes.
    pub artifact_id: ArtifactId,
    /// Operation that produced the artifact.
    pub operation: Arc<str>,
    /// Upstream artifact IDs.
    pub parents: Arc<[ArtifactId]>,
    /// Assumptions in force for this operation.
    pub assumptions: AssumptionSet,
    /// Library version string at production time.
    pub library_version: Arc<str>,
    /// Optional configuration digest or label.
    pub config_digest: Option<Arc<str>>,
}

/// Directed provenance graph (nodes only; edges implied by `parents`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProvenanceGraph {
    /// Provenance nodes in insertion order.
    pub nodes: Vec<ProvenanceNode>,
}

impl ProvenanceGraph {
    /// Empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node.
    pub fn push(&mut self, node: ProvenanceNode) {
        self.nodes.push(node);
    }

    /// Look up a node by artifact id.
    #[must_use]
    pub fn get(&self, artifact_id: &str) -> Option<&ProvenanceNode> {
        self.nodes.iter().find(|n| &*n.artifact_id == artifact_id)
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
