//! Temporal discovery graph wire types (stable `TemporalNodeKey` / Lag edges).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, VariableId};
use antecedent_graph::{NodeRef, TemporalDag};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Stable lagged node key on the wire (lag magnitude, contemporaneous = 0).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TemporalNodeKeyWire {
    /// Variable.
    pub variable: u32,
    /// Lag magnitude (`Lag::raw`).
    pub lag: u32,
}

/// Temporal DAG edge list.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveryTemporalGraphWire {
    /// Graph class.
    pub kind: String,
    /// Nodes.
    pub nodes: Vec<TemporalNodeKeyWire>,
    /// Directed edges by node index into `nodes`.
    pub directed: Vec<(u32, u32)>,
}

/// Historical Rust name for [`DiscoveryTemporalGraphWire`].
///
/// CBOR and JSON field names are unchanged. The alias keeps existing imports compiling.
pub type TemporalGraphWire = DiscoveryTemporalGraphWire;

/// Discovery header.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DiscoveryHeaderWire {
    /// Algorithm id.
    pub algorithm_id: String,
    /// Config.
    pub algorithm_config: String,
    /// Performance.
    pub ci_tests: u64,
    /// Links retained.
    pub links_retained: u64,
    /// Targets.
    pub targets: u64,
    /// Lagged frame bytes.
    pub lagged_frame_bytes: u64,
    /// Workers.
    pub worker_threads: u32,
    /// Iterations.
    pub iterations: Vec<(String, u64)>,
    /// Diagnostics.
    pub diagnostics: Vec<(String, String)>,
}

/// Edge evidence wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EdgeEvidenceWire {
    /// Link.
    pub link: LaggedLinkWire,
    /// Statistic.
    pub statistic: Option<f64>,
    /// P-value.
    pub p_value: Option<f64>,
    /// Adjusted p.
    pub adjusted_p_value: Option<f64>,
    /// Interval.
    pub interval: Option<(f64, f64)>,
    /// Provenance tags.
    pub provenance: Vec<String>,
}

/// Lagged link wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LaggedLinkWire {
    /// Source variable.
    pub source: u32,
    /// Source lag magnitude.
    pub source_lag: u32,
    /// Target.
    pub target: u32,
    /// Target lag magnitude.
    pub target_lag: u32,
}

/// Encode temporal DAG.
///
/// # Errors
///
/// Non-lagged nodes.
pub fn temporal_dag_to_wire(g: &TemporalDag) -> Result<TemporalGraphWire, IoError> {
    let mut nodes = Vec::with_capacity(g.node_count());
    for n in g.nodes() {
        match n {
            NodeRef::Lagged { variable, lag } => {
                nodes.push(TemporalNodeKeyWire { variable: variable.raw(), lag: lag.raw() });
            }
            other => {
                return Err(IoError::Convert(format!(
                    "temporal wire requires Lagged nodes, got {other:?}"
                )));
            }
        }
    }
    let mut directed = Vec::new();
    for e in g.edges() {
        if let Some((a, b)) = e.parent_child() {
            directed.push((a.raw(), b.raw()));
        }
    }
    Ok(DiscoveryTemporalGraphWire { kind: "temporal_dag".into(), nodes, directed })
}

/// Decode temporal DAG.
///
/// # Errors
///
/// Invalid edges.
pub fn temporal_dag_from_wire(w: &TemporalGraphWire) -> Result<TemporalDag, IoError> {
    let mut g = TemporalDag::empty();
    let mut ids = Vec::new();
    for n in &w.nodes {
        let id = g
            .add_lagged(VariableId::from_raw(n.variable), Lag::from_raw(n.lag))
            .map_err(|e| IoError::Convert(e.to_string()))?;
        ids.push(id);
    }
    for &(a, b) in &w.directed {
        let from = *ids.get(a as usize).ok_or_else(|| IoError::Convert("bad edge".into()))?;
        let to = *ids.get(b as usize).ok_or_else(|| IoError::Convert("bad edge".into()))?;
        g.insert_directed(from, to).map_err(|e| IoError::Convert(e.to_string()))?;
    }
    Ok(g)
}
