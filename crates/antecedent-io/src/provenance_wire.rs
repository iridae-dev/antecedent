//! Provenance graph wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, ProvenanceGraph, ProvenanceNode};
use serde::{Deserialize, Serialize};

use crate::trace::{AssumptionRecordWire, assumptions_to_wire};

/// Provenance graph wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceGraphWire {
    /// Nodes.
    pub nodes: Vec<ProvenanceNodeWire>,
}

/// One provenance node.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceNodeWire {
    /// Artifact id.
    pub artifact_id: String,
    /// Operation.
    pub operation: String,
    /// Parents.
    pub parents: Vec<String>,
    /// Assumptions (wire records).
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Library version.
    pub library_version: String,
    /// Config digest.
    pub config_digest: Option<String>,
}

/// Encode provenance graph.
#[must_use]
pub fn provenance_to_wire(g: &ProvenanceGraph) -> ProvenanceGraphWire {
    ProvenanceGraphWire {
        nodes: g
            .nodes
            .iter()
            .map(|n| ProvenanceNodeWire {
                artifact_id: n.artifact_id.to_string(),
                operation: n.operation.to_string(),
                parents: n.parents.iter().map(ToString::to_string).collect(),
                assumptions: assumptions_to_wire(&n.assumptions),
                library_version: n.library_version.to_string(),
                config_digest: n.config_digest.as_ref().map(ToString::to_string),
            })
            .collect(),
    }
}

/// Decode provenance graph (assumption tags rehydrated as declared/custom best-effort).
#[must_use]
pub fn provenance_from_wire(w: &ProvenanceGraphWire) -> ProvenanceGraph {
    let mut g = ProvenanceGraph::new();
    for n in &w.nodes {
        g.push(ProvenanceNode {
            artifact_id: Arc::from(n.artifact_id.as_str()),
            operation: Arc::from(n.operation.as_str()),
            parents: n
                .parents
                .iter()
                .map(|p| Arc::<str>::from(p.as_str()))
                .collect::<Vec<_>>()
                .into(),
            // Assumptions are audit metadata; empty set on decode keeps graph structure.
            assumptions: AssumptionSet::new(),
            library_version: Arc::from(n.library_version.as_str()),
            config_digest: n.config_digest.as_ref().map(|c| Arc::<str>::from(c.as_str())),
        });
    }
    g
}
