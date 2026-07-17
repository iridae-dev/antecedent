//! Temporal discovery graph wire types (stable TemporalNodeKey / Lag edges).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{AssumptionSet, Lag, VariableId};
use causal_discovery::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
};
use causal_graph::{NodeRef, TemporalDag, TemporalGraphReview};
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
pub struct TemporalGraphWire {
    /// Graph class.
    pub kind: String,
    /// Nodes.
    pub nodes: Vec<TemporalNodeKeyWire>,
    /// Directed edges by node index into `nodes`.
    pub directed: Vec<(u32, u32)>,
}

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
    Ok(TemporalGraphWire { kind: "temporal_dag".into(), nodes, directed })
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

/// Encode DAG discovery result header + graph + evidence.
///
/// # Errors
///
/// Graph encode failures.
pub fn discovery_dag_sections(
    result: &DiscoveryResult<TemporalDag, TemporalGraphReview>,
) -> Result<(DiscoveryHeaderWire, TemporalGraphWire, Vec<EdgeEvidenceWire>), IoError> {
    let header = DiscoveryHeaderWire {
        algorithm_id: result.algorithm.id.to_string(),
        algorithm_config: result.algorithm.config.to_string(),
        ci_tests: result.performance.ci_tests,
        links_retained: result.performance.links_retained,
        targets: result.performance.targets,
        lagged_frame_bytes: result.performance.lagged_frame_bytes,
        worker_threads: result.performance.worker_threads,
        iterations: result
            .iterations
            .iter()
            .map(|i| (i.label.to_string(), i.ci_tests))
            .collect(),
        diagnostics: result
            .diagnostics
            .iter()
            .map(|d| (d.code.to_string(), d.message.to_string()))
            .collect(),
    };
    let graph = temporal_dag_to_wire(&result.evidence.graph)?;
    let evidence = result.evidence.edge_evidence.iter().map(edge_evidence_to_wire).collect();
    Ok((header, graph, evidence))
}

fn edge_evidence_to_wire(e: &EdgeEvidence) -> EdgeEvidenceWire {
    EdgeEvidenceWire {
        link: LaggedLinkWire {
            source: e.link.source.raw(),
            source_lag: e.link.source_lag.raw(),
            target: e.link.target.raw(),
            target_lag: e.link.target_lag.raw(),
        },
        statistic: e.statistic,
        p_value: e.p_value,
        adjusted_p_value: e.adjusted_p_value,
        interval: e.interval,
        provenance: e.provenance.iter().map(|p| p.to_string()).collect(),
    }
}

/// Rebuild a minimal DAG discovery result from sections.
///
/// # Errors
///
/// Graph decode failures.
pub fn discovery_dag_from_sections(
    header: &DiscoveryHeaderWire,
    graph: &TemporalGraphWire,
    evidence: &[EdgeEvidenceWire],
) -> Result<DiscoveryResult<TemporalDag, TemporalGraphReview>, IoError> {
    let dag = temporal_dag_from_wire(graph)?;
    let edge_evidence: Vec<EdgeEvidence> = evidence
        .iter()
        .map(|e| EdgeEvidence {
            link: LaggedLink {
                source: VariableId::from_raw(e.link.source),
                source_lag: Lag::from_raw(e.link.source_lag),
                target: VariableId::from_raw(e.link.target),
                target_lag: Lag::from_raw(e.link.target_lag),
            },
            statistic: e.statistic,
            p_value: e.p_value,
            adjusted_p_value: e.adjusted_p_value,
            interval: e.interval,
            separating_sets: Arc::from([]),
            provenance: e.provenance.iter().map(|p| Arc::<str>::from(p.as_str())).collect::<Vec<_>>().into(),
        })
        .collect();
    let links: Vec<ScoredLink> = edge_evidence
        .iter()
        .filter_map(|e| {
            Some(ScoredLink {
                link: e.link,
                statistic: e.statistic?,
                p_value: e.p_value?,
                adjusted_p_value: e.adjusted_p_value,
            })
        })
        .collect();
    let review = TemporalGraphReview::from_graph(dag.clone(), header.algorithm_id.as_str());
    Ok(DiscoveryResult {
        evidence: GraphEvidence {
            graph: dag,
            edge_evidence: edge_evidence.into(),
            links: links.into(),
            source: EvidenceSource::Discovery {
                algorithm: Arc::from(header.algorithm_id.as_str()),
            },
        },
        algorithm: AlgorithmRecord {
            id: Arc::from(header.algorithm_id.as_str()),
            config: Arc::from(header.algorithm_config.as_str()),
        },
        assumptions: AssumptionSet::new(),
        iterations: header
            .iterations
            .iter()
            .map(|(l, c)| DiscoveryIteration {
                label: Arc::from(l.as_str()),
                ci_tests: *c,
            })
            .collect(),
        diagnostics: header
            .diagnostics
            .iter()
            .map(|(c, m)| DiscoveryDiagnostic {
                code: Arc::from(c.as_str()),
                message: Arc::from(m.as_str()),
            })
            .collect(),
        performance: DiscoveryPerformanceRecord {
            ci_tests: header.ci_tests,
            links_retained: header.links_retained,
            targets: header.targets,
            lagged_frame_bytes: header.lagged_frame_bytes,
            worker_threads: header.worker_threads,
        },
        review,
        sepsets: HashMap::new(),
    })
}
