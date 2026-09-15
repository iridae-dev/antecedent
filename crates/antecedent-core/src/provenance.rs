//! Analysis provenance graph.
//!
//! [`ProvenanceGraph::push`] stays append-only. Claim ancestry uses
//! [`ProvenanceGraph::try_push`] / [`ProvenanceGraph::validate`]: unique
//! identities, no cycles, and parent binding. A nonempty id or a
//! syntactically valid parent link is not a complete lineage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
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

/// Provenance ancestry validation error.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ProvenanceError {
    /// Empty artifact id.
    #[error("provenance node has an empty artifact id")]
    EmptyArtifactId,
    /// Duplicate artifact id in this graph.
    #[error("duplicate provenance artifact id `{id}`")]
    DuplicateIdentity {
        /// Conflicting id.
        id: Arc<str>,
    },
    /// Parent list contains a cycle through `id`.
    #[error("provenance cycle through `{id}`")]
    Cycle {
        /// Artifact id on the cycle.
        id: Arc<str>,
    },
    /// Parent id is empty.
    #[error("provenance parent of `{child}` is empty")]
    EmptyParent {
        /// Child artifact id.
        child: Arc<str>,
    },
}

impl ProvenanceGraph {
    /// Empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node without ancestry validation.
    ///
    /// Prefer [`Self::try_push`] at claim acceptance.
    pub fn push(&mut self, node: ProvenanceNode) {
        self.nodes.push(node);
    }

    /// Insert a node after checking unique identity, nonempty ids, and that
    /// adding it would not create a cycle among nodes already in the graph
    /// plus this node. Unresolved external parent ids are allowed; they are
    /// recorded as missing, not invented.
    ///
    /// # Errors
    ///
    /// Empty ids, duplicate identity, or a cycle.
    pub fn try_push(&mut self, node: ProvenanceNode) -> Result<(), ProvenanceError> {
        Self::validate_node(&self.nodes, &node)?;
        self.nodes.push(node);
        Ok(())
    }

    /// Validate unique identities, nonempty ids, and acyclicity of the
    /// current graph. External parent references (ids not in this graph)
    /// are returned so a receiver can record unresolved lineage.
    ///
    /// # Errors
    ///
    /// Empty ids, duplicate identity, or a cycle.
    pub fn validate(&self) -> Result<Arc<[ArtifactId]>, ProvenanceError> {
        let mut seen = BTreeSet::new();
        for node in &self.nodes {
            Self::validate_node_shape(node)?;
            if !seen.insert(node.artifact_id.clone()) {
                return Err(ProvenanceError::DuplicateIdentity { id: node.artifact_id.clone() });
            }
        }
        self.assert_acyclic()?;
        Ok(self.unresolved_parents())
    }

    /// Parent artifact ids that are not nodes in this graph.
    #[must_use]
    pub fn unresolved_parents(&self) -> Arc<[ArtifactId]> {
        let present: BTreeSet<&str> = self.nodes.iter().map(|n| n.artifact_id.as_ref()).collect();
        let mut missing = BTreeSet::new();
        for node in &self.nodes {
            for parent in node.parents.iter() {
                if !present.contains(parent.as_ref()) {
                    missing.insert(parent.clone());
                }
            }
        }
        Arc::from(missing.into_iter().collect::<Vec<_>>())
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

    fn validate_node(
        existing: &[ProvenanceNode],
        node: &ProvenanceNode,
    ) -> Result<(), ProvenanceError> {
        Self::validate_node_shape(node)?;
        if existing.iter().any(|n| n.artifact_id == node.artifact_id) {
            return Err(ProvenanceError::DuplicateIdentity { id: node.artifact_id.clone() });
        }
        let mut nodes: Vec<ProvenanceNode> = existing.to_vec();
        nodes.push(node.clone());
        Self { nodes }.assert_acyclic()
    }

    fn validate_node_shape(node: &ProvenanceNode) -> Result<(), ProvenanceError> {
        if node.artifact_id.is_empty() {
            return Err(ProvenanceError::EmptyArtifactId);
        }
        for parent in node.parents.iter() {
            if parent.is_empty() {
                return Err(ProvenanceError::EmptyParent { child: node.artifact_id.clone() });
            }
        }
        Ok(())
    }

    fn assert_acyclic(&self) -> Result<(), ProvenanceError> {
        let index: BTreeMap<&str, usize> =
            self.nodes.iter().enumerate().map(|(i, n)| (n.artifact_id.as_ref(), i)).collect();
        let mut state = vec![0u8; self.nodes.len()];
        for i in 0..self.nodes.len() {
            self.dfs_acyclic(i, &index, &mut state)?;
        }
        Ok(())
    }

    fn dfs_acyclic(
        &self,
        i: usize,
        index: &BTreeMap<&str, usize>,
        state: &mut [u8],
    ) -> Result<(), ProvenanceError> {
        match state[i] {
            1 => {
                return Err(ProvenanceError::Cycle { id: self.nodes[i].artifact_id.clone() });
            }
            2 => return Ok(()),
            _ => {}
        }
        state[i] = 1;
        for parent in self.nodes[i].parents.iter() {
            if let Some(&j) = index.get(parent.as_ref()) {
                self.dfs_acyclic(j, index, state)?;
            }
        }
        state[i] = 2;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assumption::AssumptionSet;

    fn node(id: &str, parents: &[&str]) -> ProvenanceNode {
        ProvenanceNode {
            artifact_id: Arc::from(id),
            operation: Arc::from("test"),
            parents: parents.iter().map(|p| Arc::<str>::from(*p)).collect(),
            assumptions: AssumptionSet::new(),
            library_version: Arc::from("1.9.0"),
            config_digest: None,
        }
    }

    #[test]
    fn try_push_rejects_duplicates_and_cycles() {
        let mut graph = ProvenanceGraph::new();
        graph.try_push(node("a", &[])).unwrap();
        assert!(matches!(
            graph.try_push(node("a", &[])),
            Err(ProvenanceError::DuplicateIdentity { .. })
        ));
        graph.try_push(node("b", &["a"])).unwrap();
        assert!(matches!(graph.try_push(node("c", &["c"])), Err(ProvenanceError::Cycle { .. })));
        let missing = graph.validate().unwrap();
        assert!(missing.is_empty());
    }

    #[test]
    fn external_parents_are_unresolved_not_cycles() {
        let mut graph = ProvenanceGraph::new();
        graph.try_push(node("child", &["external-source"])).unwrap();
        let missing = graph.validate().unwrap();
        assert_eq!(missing.as_ref(), [Arc::<str>::from("external-source")]);
    }

    #[test]
    fn push_still_accepts_invalid_nodes() {
        let mut graph = ProvenanceGraph::new();
        graph.push(node("", &[]));
        assert_eq!(graph.len(), 1);
        assert!(graph.validate().is_err());
    }
}
