//! Lazy finite unfolding of temporal DAGs and graph-review artifacts.
//!
//! [`LazyUnfoldedTemporalGraph`] holds the template and window indexer; [`materialize`] replicates
//! the in-window edges into a static [`Dag`]. Per-edge queries exist only as the test oracle for
//! that replication.
//!
//! [`materialize`]: LazyUnfoldedTemporalGraph::materialize
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_core::{TemporalIndexer, TemporalNodeKey};

use crate::dag::Dag;
use crate::error::GraphError;
use crate::temporal::TemporalDag;
use crate::types::{DenseNodeId, NodeRef};

/// Eager materialisation of a [`TemporalDag`] over a finite indexer window.
#[derive(Clone, Debug)]
pub struct UnfoldedTemporalGraph {
    /// Static DAG whose dense ids match [`TemporalIndexer::dense_id`].
    pub dag: Dag,
    /// Indexer used for the unfolding.
    pub indexer: TemporalIndexer,
}

/// An admitted temporal DAG window; clones share the graph and its lease.
#[derive(Clone, Debug)]
pub struct AdmittedUnfoldedTemporalGraph(std::sync::Arc<AdmittedUnfoldedTemporalGraphInner>);
#[derive(Debug)]
struct AdmittedUnfoldedTemporalGraphInner {
    window: UnfoldedTemporalGraph,
    topo: Vec<DenseNodeId>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl AdmittedUnfoldedTemporalGraph {
    /// Topological order, retained under this window's lease.
    #[must_use]
    pub fn topological_order(&self) -> &[DenseNodeId] {
        &self.0.topo
    }
}
impl std::ops::Deref for AdmittedUnfoldedTemporalGraph {
    type Target = UnfoldedTemporalGraph;
    fn deref(&self) -> &Self::Target {
        &self.0.window
    }
}

/// Lazy finite unfolding: template edges are replicated on query, not upfront.
#[derive(Clone, Debug)]
pub struct LazyUnfoldedTemporalGraph {
    /// Stationary / lagged summary graph.
    pub template: TemporalDag,
    /// Finite window indexer.
    pub indexer: TemporalIndexer,
}

impl TemporalDag {
    /// Materialize a finite window using a shared logical budget and retained admission.
    /// Clones of this handle share the graph and lease; a graph copied by a caller
    /// from the borrowed `dag` is caller-owned.
    /// # Errors
    /// Cancellation, logical limits, admission, or invalid graph coordinates.
    pub fn unfold_within(
        &self,
        indexer: TemporalIndexer,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<AdmittedUnfoldedTemporalGraph, GraphError> {
        use antecedent_core::TemporalMaterializationError as T;
        let limit = |dimension| GraphError::TemporalMaterialization(T::Limit { dimension });
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        let nodes = u64::try_from(indexer.dense_len()).map_err(|_| limit("graph nodes"))?;
        let limits = budget.limits();
        if nodes > limits.max_graph_nodes {
            return Err(limit("graph nodes"));
        }
        // Each key of the indexer must fit the public i32 offset representation.
        if indexer.history() > (i32::MAX as u32) + 1 || indexer.horizon() > (i32::MAX as u32) + 1 {
            return Err(limit("window coordinates"));
        }
        let template_nodes = self.nodes().len() as u64;
        budget.charge_work(template_nodes)?;
        let mut templates = 0u64;
        for (i, _) in self.nodes().iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let id = DenseNodeId::try_from_usize(i)?;
            if self.temporal_key(id).is_none() {
                return Err(GraphError::InvalidEndpoints {
                    message: "unfold requires lagged nodes",
                });
            }
            let n = self.children(id).len() as u64;
            budget.charge_work(n)?;
            templates = templates.checked_add(n).ok_or_else(|| limit("graph edges"))?;
        }
        let slices = u64::from(indexer.history())
            .checked_add(u64::from(indexer.horizon()))
            .ok_or_else(|| limit("graph nodes"))?;
        let edges = templates.checked_mul(slices).ok_or_else(|| limit("graph edges"))?;
        if edges > limits.max_graph_edges {
            return Err(limit("graph edges"));
        }
        // Each attempted insertion scans one child and one parent row, each bounded by nodes.
        let work = nodes
            .checked_mul(8)
            .and_then(|n| edges.checked_mul(8).and_then(|e| n.checked_add(e)))
            .and_then(|base| {
                edges
                    .checked_mul(nodes)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|scans| base.checked_add(scans))
            })
            .ok_or_else(|| limit("work"))?;
        budget.charge_work(work)?;
        let bytes = nodes
            .checked_mul(512)
            .and_then(|n| edges.checked_mul(256).and_then(|e| n.checked_add(e)))
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| limit("bytes"))?;
        if bytes > limits.max_bytes {
            return Err(limit("bytes"));
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
            other => T::Admission(other),
        })?;
        let mut dag = Dag::empty();
        for dense in 0..indexer.dense_len() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let key = indexer
                .key_of(u32::try_from(dense).map_err(|_| GraphError::TooManyNodes)?)
                .map_err(|_| GraphError::TooManyNodes)?;
            dag.add_node(NodeRef::Unfolded { variable: key.variable, offset: key.offset })?;
        }
        for (i, _) in self.nodes().iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let from = DenseNodeId::try_from_usize(i)?;
            let a = self
                .temporal_key(from)
                .ok_or(GraphError::InvalidEndpoints { message: "unfold requires lagged nodes" })?;
            for &to in self.children(from) {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                let b = self.temporal_key(to).ok_or(GraphError::InvalidEndpoints {
                    message: "unfold requires lagged nodes",
                })?;
                for delta in stationary_shifts(&indexer, i64::from(a.offset), i64::from(b.offset)) {
                    if ctx.cancellation.is_cancelled() {
                        return Err(T::Cancelled.into());
                    }
                    let a_offset = i32::try_from(i64::from(a.offset) + delta)
                        .map_err(|_| limit("window coordinates"))?;
                    let b_offset = i32::try_from(i64::from(b.offset) + delta)
                        .map_err(|_| limit("window coordinates"))?;
                    let ai = indexer
                        .dense_id(TemporalNodeKey { variable: a.variable, offset: a_offset })
                        .map_err(|_| GraphError::InvalidEndpoints {
                            message: "unfold endpoint outside window",
                        })?;
                    let bi = indexer
                        .dense_id(TemporalNodeKey { variable: b.variable, offset: b_offset })
                        .map_err(|_| GraphError::InvalidEndpoints {
                            message: "unfold endpoint outside window",
                        })?;
                    if ai != bi {
                        dag.directed_mut()
                            .push_if_absent(DenseNodeId::from_raw(ai), DenseNodeId::from_raw(bi));
                    }
                }
            }
        }
        let mut indegree = Vec::with_capacity(indexer.dense_len());
        let mut ready = Vec::new();
        for i in 0..indexer.dense_len() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let id = DenseNodeId::try_from_usize(i)?;
            indegree.push(dag.parents(id).len());
            if indegree[i] == 0 {
                ready.push(id);
            }
        }
        let mut topo = Vec::with_capacity(indexer.dense_len());
        while let Some(id) = ready.pop() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            topo.push(id);
            for child in dag.children(id) {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                indegree[child.as_usize()] -= 1;
                if indegree[child.as_usize()] == 0 {
                    ready.push(*child);
                }
            }
        }
        if topo.len() != indexer.dense_len() {
            return Err(GraphError::Cycle { from: 0, to: 0 });
        }
        Ok(AdmittedUnfoldedTemporalGraph(std::sync::Arc::new(AdmittedUnfoldedTemporalGraphInner {
            window: UnfoldedTemporalGraph { dag, indexer },
            topo,
            _lease: lease,
        })))
    }

    /// Lazy unfold over a finite indexer window .
    ///
    /// # Errors
    ///
    /// Unknown/non-lagged template nodes.
    pub fn unfold_lazy(
        &self,
        indexer: TemporalIndexer,
    ) -> Result<LazyUnfoldedTemporalGraph, GraphError> {
        // Validate all nodes are lagged up front.
        for (i, _) in self.nodes().iter().enumerate() {
            let id = DenseNodeId::try_from_usize(i)?;
            let _ = self
                .temporal_key(id)
                .ok_or(GraphError::InvalidEndpoints { message: "unfold requires lagged nodes" })?;
        }
        Ok(LazyUnfoldedTemporalGraph { template: self.clone(), indexer })
    }

    /// Eager unfold into a static [`Dag`] (materialises the full window).
    ///
    /// This materializes a window, not a certificate that its history is sufficient: parents
    /// that fall before the earliest slice are dropped, so d-separation statements about nodes
    /// near the start of a truncated window can be false for the stationary process.
    /// Identification must separately certify the query's boundary closure.
    ///
    /// # Errors
    ///
    /// Unknown/non-lagged nodes, indexer issues, or cycle insertion.
    pub fn unfold(
        &self,
        indexer: TemporalIndexer,
    ) -> Result<AdmittedUnfoldedTemporalGraph, GraphError> {
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.unfold_within(indexer, &mut budget, &ctx)
    }
}

impl LazyUnfoldedTemporalGraph {
    /// Whether a concrete directed edge exists under the template replication. Test oracle
    /// for [`Self::materialize`].
    ///
    /// # Errors
    ///
    /// Endpoints outside the indexer window.
    #[cfg(test)]
    pub fn has_edge(&self, from: TemporalNodeKey, to: TemporalNodeKey) -> Result<bool, GraphError> {
        let _ = self.indexer.dense_id(from).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        let _ = self.indexer.dense_id(to).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        for (from_i, _) in self.template.nodes().iter().enumerate() {
            let from_id = DenseNodeId::try_from_usize(from_i)?;
            let Some(from_key) = self.template.temporal_key(from_id) else {
                continue;
            };
            for &to_id in self.template.children(from_id) {
                let Some(to_key) = self.template.temporal_key(to_id) else {
                    continue;
                };
                if edge_matches(from_key, to_key, from, to) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Materialise all in-window replicated edges into a static DAG.
    ///
    /// # Errors
    ///
    /// Indexer / cycle insertion failures.
    pub fn materialize(&self) -> Result<AdmittedUnfoldedTemporalGraph, GraphError> {
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.template.unfold_within(self.indexer.clone(), &mut budget, &ctx)
    }
}

#[cfg(test)]
fn edge_matches(
    template_from: TemporalNodeKey,
    template_to: TemporalNodeKey,
    concrete_from: TemporalNodeKey,
    concrete_to: TemporalNodeKey,
) -> bool {
    template_from.variable == concrete_from.variable
        && template_to.variable == concrete_to.variable
        && concrete_from.offset.wrapping_sub(template_from.offset)
            == concrete_to.offset.wrapping_sub(template_to.offset)
}

/// Translations `delta` that keep an edge with endpoint offsets `a_offset`, `b_offset` inside the
/// indexer window `[-history, horizon - 1]`. Template offsets are <= 0, so a fully lagged edge
/// (both endpoints negative) is shifted into the newest slices too.
pub(crate) fn stationary_shifts(
    indexer: &TemporalIndexer,
    a_offset: i64,
    b_offset: i64,
) -> std::ops::RangeInclusive<i64> {
    let lower = -i64::from(indexer.history());
    let upper = i64::from(indexer.horizon()) - 1;
    (lower - a_offset.min(b_offset))..=(upper - a_offset.max(b_offset))
}

/// Review-required temporal DAG artifact .
#[derive(Clone, Debug)]
pub struct TemporalGraphReview {
    /// Proposed discovery graph.
    pub graph: TemporalDag,
    /// Edges awaiting explicit acceptance (serializable keys).
    pub pending_edges: Arc<[(TemporalNodeKey, TemporalNodeKey)]>,
    /// Algorithm id that produced the proposal.
    pub algorithm: Arc<str>,
}

impl TemporalGraphReview {
    /// Construct a review artifact listing all current edges as pending.
    #[must_use]
    pub fn from_graph(graph: TemporalDag, algorithm: impl Into<Arc<str>>) -> Self {
        let mut pending = Vec::new();
        for (i, _) in graph.nodes().iter().enumerate() {
            let from = DenseNodeId::try_from_usize(i).expect("node fit");
            let Some(from_key) = graph.temporal_key(from) else {
                continue;
            };
            for &to in graph.children(from) {
                if let Some(to_key) = graph.temporal_key(to) {
                    pending.push((from_key, to_key));
                }
            }
        }
        Self { graph, pending_edges: Arc::from(pending), algorithm: algorithm.into() }
    }

    /// Accept a pending edge by endpoints (no-op if absent).
    #[must_use]
    pub fn accept_edge(mut self, from: TemporalNodeKey, to: TemporalNodeKey) -> Self {
        self.pending_edges = crate::types::without_pending(&self.pending_edges, (from, to));
        self
    }

    /// Whether all pending edges have been accepted.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending_edges.is_empty()
    }
}

/// Review-required temporal CPDAG artifact .
///
/// Directed edges must be accepted; undirected edges must be explicitly oriented
/// before the CPDAG can be completed to a [`TemporalDag`] for identification.
#[derive(Clone, Debug)]
pub struct TemporalCpdagReview {
    /// Proposed discovery CPDAG.
    pub graph: crate::TemporalCpdag,
    /// Directed edges awaiting acceptance.
    pub pending_edges: Arc<[(TemporalNodeKey, TemporalNodeKey)]>,
    /// Undirected edges awaiting explicit orientation `(a, b)` with `a` lexicographically first.
    pub pending_undirected: Arc<[(TemporalNodeKey, TemporalNodeKey)]>,
    /// Algorithm id that produced the proposal.
    pub algorithm: Arc<str>,
}

impl TemporalCpdagReview {
    /// Construct a review listing all directed edges as pending and all undirected as pending orientation.
    #[must_use]
    pub fn from_cpdag(graph: crate::TemporalCpdag, algorithm: impl Into<Arc<str>>) -> Self {
        let mut pending = Vec::new();
        let mut undirected = Vec::new();
        for e in graph.edges() {
            if let Some((from, to)) = e.parent_child() {
                if let (Some(fk), Some(tk)) = (graph.temporal_key(from), graph.temporal_key(to)) {
                    pending.push((fk, tk));
                }
            } else if e.is_undirected() {
                if let (Some(ak), Some(bk)) = (graph.temporal_key(e.a), graph.temporal_key(e.b)) {
                    if (ak.variable, ak.offset) <= (bk.variable, bk.offset) {
                        undirected.push((ak, bk));
                    } else {
                        undirected.push((bk, ak));
                    }
                }
            }
        }
        Self {
            graph,
            pending_edges: Arc::from(pending),
            pending_undirected: Arc::from(undirected),
            algorithm: algorithm.into(),
        }
    }

    /// Accept a pending directed edge (no-op if absent).
    #[must_use]
    pub fn accept_edge(mut self, from: TemporalNodeKey, to: TemporalNodeKey) -> Self {
        self.pending_edges = crate::types::without_pending(&self.pending_edges, (from, to));
        self
    }

    /// Orient an undirected edge as `from -> to` and remove it from pending undirected.
    ///
    /// # Errors
    ///
    /// Missing undirected edge, cycle, or unknown nodes.
    pub fn orient_edge(
        mut self,
        from: TemporalNodeKey,
        to: TemporalNodeKey,
    ) -> Result<Self, GraphError> {
        let from_id = self.resolve_key(from)?;
        let to_id = self.resolve_key(to)?;
        self.graph.orient_undirected(from_id, to_id)?;
        (self.pending_edges, self.pending_undirected) =
            crate::types::orient_pending(&self.pending_edges, &self.pending_undirected, from, to);
        Ok(self)
    }

    /// Whether all directed edges are accepted and no undirected edges remain.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending_edges.is_empty() && self.pending_undirected.is_empty()
    }

    /// Complete to a [`TemporalDag`] after review (errors if undirected edges remain).
    ///
    /// # Errors
    ///
    /// Incomplete review or conversion failure.
    pub fn try_into_temporal_dag(self) -> Result<TemporalDag, GraphError> {
        if !self.is_complete() {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalCpdagReview is incomplete; accept directed and orient undirected edges first",
            });
        }
        self.graph.try_into_temporal_dag()
    }

    fn resolve_key(&self, key: TemporalNodeKey) -> Result<DenseNodeId, GraphError> {
        for i in 0..self.graph.node_count() {
            let id = DenseNodeId::try_from_usize(i)?;
            if self.graph.temporal_key(id) == Some(key) {
                return Ok(id);
            }
        }
        Err(GraphError::UnknownNode { id: key.variable.raw() })
    }
}

/// Helper for tests / discovery: ensure a lagged node exists.
pub fn ensure_lagged(
    graph: &mut TemporalDag,
    variable: VariableId,
    lag: antecedent_core::Lag,
) -> Result<DenseNodeId, GraphError> {
    for (i, n) in graph.nodes().iter().enumerate() {
        if let NodeRef::Lagged { variable: v, lag: l } = n {
            if *v == variable && *l == lag {
                return DenseNodeId::try_from_usize(i);
            }
        }
    }
    graph.add_lagged(variable, lag)
}

#[cfg(test)]
mod tests {
    use antecedent_core::TemporalIndexer;
    use antecedent_core::{Lag, VariableId};

    use super::*;
    use crate::dsep::DSeparationWorkspace;

    #[test]
    fn lazy_has_edge_matches_materialize() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let lazy = g.unfold_lazy(indexer.clone()).unwrap();
        let from = TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 };
        let to = TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 };
        assert!(lazy.has_edge(from, to).unwrap());
        let unfolded = lazy.materialize().unwrap();
        assert_eq!(unfolded.dag.node_count(), 6);
    }

    #[test]
    fn unfolded_nodes_are_window_slots_never_schema_variables() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        let indexer = TemporalIndexer::new(2, 1, 3).unwrap();
        let unfolded = g.unfold(indexer.clone()).unwrap();
        assert_eq!(unfolded.dag.node_count(), 8);
        for (dense, node) in unfolded.dag.nodes().iter().enumerate() {
            let key = indexer.key_of(u32::try_from(dense).unwrap()).unwrap();
            assert_eq!(*node, NodeRef::Unfolded { variable: key.variable, offset: key.offset });
            assert!(!matches!(node, NodeRef::Static(_)));
        }
        // Positive offsets (the future window) are expressible; `Lagged` could not.
        assert!(
            unfolded
                .dag
                .nodes()
                .contains(&NodeRef::Unfolded { variable: VariableId::from_raw(1), offset: 2 })
        );
    }

    #[test]
    fn unfold_replicates_lagged_edge() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let unfolded = g.unfold(indexer).unwrap();
        assert_eq!(unfolded.dag.node_count(), 6);
        let mut edge_count = 0usize;
        for i in 0..unfolded.dag.node_count() {
            edge_count += unfolded.dag.children(DenseNodeId::from_raw(i as u32)).len();
        }
        // The template edge x[-1] -> y[0] repeats at each translation inside the 3-slot
        // window: exactly x[-1] -> y[0] and x[0] -> y[1], and nothing else.
        assert_eq!(edge_count, 2);
        let node = |v: u32, offset: i32| {
            DenseNodeId::from_raw(
                unfolded
                    .indexer
                    .dense_id(TemporalNodeKey { variable: VariableId::from_raw(v), offset })
                    .unwrap(),
            )
        };
        assert_eq!(unfolded.dag.children(node(0, -1)), &[node(1, 0)]);
        assert_eq!(unfolded.dag.children(node(0, 0)), &[node(1, 1)]);
    }

    #[test]
    fn unfold_dsep_on_chain() {
        let mut g = TemporalDag::empty();
        let x1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        g.insert_directed(y0, z0).unwrap();
        let indexer = TemporalIndexer::new(3, 1, 1).unwrap();
        let unfolded = g.unfold(indexer).unwrap();
        let mut ws = DSeparationWorkspace::default();
        let y = DenseNodeId::from_raw(
            unfolded
                .indexer
                .dense_id(TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 })
                .unwrap(),
        );
        let z = DenseNodeId::from_raw(
            unfolded
                .indexer
                .dense_id(TemporalNodeKey { variable: VariableId::from_raw(2), offset: 0 })
                .unwrap(),
        );
        let x = DenseNodeId::from_raw(
            unfolded
                .indexer
                .dense_id(TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 })
                .unwrap(),
        );
        // Chain X→Y→Z: Y and Z are d-connected unconditionally; conditioning on Y
        // d-separates X from Z.
        assert!(!unfolded.dag.is_d_separated(y, z, &[], &mut ws).unwrap());
        assert!(unfolded.dag.is_d_separated(x, z, &[y], &mut ws).unwrap());
        assert!(!unfolded.dag.is_d_separated(x, z, &[], &mut ws).unwrap());
    }

    #[test]
    fn unfold_replicates_edge_with_both_endpoints_lagged() {
        let mut g = TemporalDag::empty();
        let x2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let y1 = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        g.insert_directed(x2, y1).unwrap();
        // Window offsets are {-1, 0}: the template edge only fits shifted by +1.
        let indexer = TemporalIndexer::new(2, 1, 1).unwrap();
        let lazy = g.unfold_lazy(indexer.clone()).unwrap();
        let from = TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 };
        let to = TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 };
        assert!(lazy.has_edge(from, to).unwrap());

        let unfolded = lazy.materialize().unwrap();
        let edges: Vec<_> = unfolded.dag.edges().collect();
        assert_eq!(edges.len(), 1);
        let (f, t) = edges[0].parent_child().unwrap();
        assert_eq!(f.raw(), indexer.dense_id(from).unwrap());
        assert_eq!(t.raw(), indexer.dense_id(to).unwrap());

        // Lazy and eager views agree on every in-window pair.
        for &va in &[0u32, 1] {
            for oa in -1..=0 {
                for &vb in &[0u32, 1] {
                    for ob in -1..=0 {
                        let a = TemporalNodeKey { variable: VariableId::from_raw(va), offset: oa };
                        let b = TemporalNodeKey { variable: VariableId::from_raw(vb), offset: ob };
                        let dense_a = DenseNodeId::from_raw(indexer.dense_id(a).unwrap());
                        let dense_b = DenseNodeId::from_raw(indexer.dense_id(b).unwrap());
                        let eager = unfolded.dag.children(dense_a).contains(&dense_b);
                        assert_eq!(lazy.has_edge(a, b).unwrap(), eager, "{a:?} -> {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn review_accept_clears_pending() {
        let mut g = TemporalDag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        let review = TemporalGraphReview::from_graph(g, "pcmci");
        assert!(!review.is_complete());
        let from = TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 };
        let to = TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 };
        let review = review.accept_edge(from, to);
        assert!(review.is_complete());
    }

    fn random_small_template(rng: &mut antecedent_core::CausalRng) -> (TemporalDag, u32, u32, u32) {
        let n_vars = 2 + (rng.next_u64() % 2) as u32; // 2..=3
        let max_lag = 1 + (rng.next_u64() % 2) as u32; // 1..=2
        let history = max_lag;
        let horizon = 1 + (rng.next_u64() % 2) as u32; // 1..=2

        let mut g = TemporalDag::empty();
        let mut ids = Vec::new();
        for v in 0..n_vars {
            for lag in 0..=max_lag {
                let id = g.add_lagged(VariableId::from_raw(v), Lag::from_raw(lag)).unwrap();
                ids.push((v, lag, id));
            }
        }
        // Time-respecting edges only: source lag ≥ target lag; contemporaneous
        // edges ordered by variable id so the template stays a DAG.
        for &(va, la, a) in &ids {
            for &(vb, lb, b) in &ids {
                if a == b {
                    continue;
                }
                let time_ok = la > lb || (la == lb && va < vb);
                if !time_ok {
                    continue;
                }
                if rng.next_u64() % 3 == 0 {
                    let _ = g.insert_directed(a, b);
                }
            }
        }
        (g, n_vars, history, horizon)
    }

    /// Reconstruct a DAG by querying every lazy `has_edge` pair in-window.
    #[allow(clippy::cast_possible_wrap, reason = "fixture dimensions are small literals")]
    fn dag_from_lazy_scan(lazy: &LazyUnfoldedTemporalGraph) -> Dag {
        let window = Dag::from_unfolding(&lazy.indexer).unwrap();
        let mut dag = window.dag.clone();
        let min_off = -(lazy.indexer.history() as i32);
        let max_off = (lazy.indexer.horizon() as i32) - 1;
        let n_vars = lazy.indexer.variable_count();
        for va in 0..n_vars {
            for oa in min_off..=max_off {
                for vb in 0..n_vars {
                    for ob in min_off..=max_off {
                        let from =
                            TemporalNodeKey { variable: VariableId::from_raw(va), offset: oa };
                        let to = TemporalNodeKey { variable: VariableId::from_raw(vb), offset: ob };
                        if !lazy.has_edge(from, to).unwrap() {
                            continue;
                        }
                        let a = DenseNodeId::from_raw(lazy.indexer.dense_id(from).unwrap());
                        let b = DenseNodeId::from_raw(lazy.indexer.dense_id(to).unwrap());
                        if a != b {
                            let _ = dag.insert_directed(a, b);
                        }
                    }
                }
            }
        }
        dag
    }

    /// Lazy `has_edge` agrees with materialize over random small stationary templates.
    #[test]
    #[allow(clippy::cast_possible_wrap, reason = "fixture dimensions are small literals")]
    fn property_lazy_unfold_matches_materialize_on_small_templates() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(91);
        for _ in 0..50 {
            let (g, n_vars, history, horizon) = random_small_template(&mut rng);
            let indexer = TemporalIndexer::new(n_vars, history, horizon).unwrap();
            let lazy = g.unfold_lazy(indexer.clone()).unwrap();
            let unfolded = lazy.materialize().unwrap();
            let min_off = -(history as i32);
            let max_off = (horizon as i32) - 1;
            for va in 0..n_vars {
                for oa in min_off..=max_off {
                    for vb in 0..n_vars {
                        for ob in min_off..=max_off {
                            let from =
                                TemporalNodeKey { variable: VariableId::from_raw(va), offset: oa };
                            let to =
                                TemporalNodeKey { variable: VariableId::from_raw(vb), offset: ob };
                            let dense_a = DenseNodeId::from_raw(indexer.dense_id(from).unwrap());
                            let dense_b = DenseNodeId::from_raw(indexer.dense_id(to).unwrap());
                            let eager = unfolded.dag.children(dense_a).contains(&dense_b);
                            assert_eq!(
                                lazy.has_edge(from, to).unwrap(),
                                eager,
                                "lazy≠materialize {from:?}->{to:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// d-separation on the lazy-scanned DAG agrees with materialize on random templates.
    #[test]
    fn property_unfolded_dsep_lazy_scan_matches_materialize() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(113);
        let mut ws = DSeparationWorkspace::default();
        for _ in 0..30 {
            let (g, n_vars, history, horizon) = random_small_template(&mut rng);
            let indexer = TemporalIndexer::new(n_vars, history, horizon).unwrap();
            let lazy = g.unfold_lazy(indexer).unwrap();
            let unfolded = lazy.materialize().unwrap();
            let scanned = dag_from_lazy_scan(&lazy);
            let n = unfolded.dag.node_count() as u32;
            assert_eq!(scanned.node_count(), unfolded.dag.node_count());
            for i in 0..n {
                let u = DenseNodeId::from_raw(i);
                let mut a = scanned.children(u).to_vec();
                let mut b = unfolded.dag.children(u).to_vec();
                a.sort_by_key(|x| x.raw());
                b.sort_by_key(|x| x.raw());
                assert_eq!(a, b, "lazy-scan adjacency ≠ materialize at {i}");
            }
            for _ in 0..10 {
                let x = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
                let mut y = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
                while y == x {
                    y = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
                }
                let mut z = Vec::new();
                for i in 0..n {
                    let v = DenseNodeId::from_raw(i);
                    if v == x || v == y {
                        continue;
                    }
                    if rng.next_u64() % 3 == 0 {
                        z.push(v);
                    }
                }
                let lazy_sep = scanned.is_d_separated(x, y, &z, &mut ws).unwrap();
                let mat_sep = unfolded.dag.is_d_separated(x, y, &z, &mut ws).unwrap();
                assert_eq!(
                    lazy_sep,
                    mat_sep,
                    "unfolded d-sep mismatch x={} y={}",
                    x.raw(),
                    y.raw()
                );
            }
        }
    }
}

#[cfg(test)]
mod admitted_dag_window_tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{Lag, TemporalBudget, TemporalMaterializationError, VariableId};

    #[test]
    fn admitted_window_matches_small_unfolding_and_clones_retain_lease() {
        let mut graph = TemporalDag::empty();
        let old = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(old, now).unwrap();
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let expected = graph.unfold(indexer.clone()).unwrap();
        let pool = AdmissionPool::fixed(1, 65_536);
        let ctx = antecedent_core::ExecutionContext::for_tests(3).with_admission(pool.clone());
        let before = pool.available().bytes();
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        let window = graph.unfold_within(indexer, &mut budget, &ctx).unwrap();
        assert_eq!(window.dag, expected.dag);
        let held = pool.available().bytes();
        assert!(held < before);
        let clone = window.clone();
        drop(window);
        assert_eq!(pool.available().bytes(), held);
        drop(clone);
        assert_eq!(pool.available().bytes(), before);
        ctx.cancellation.cancel();
        assert!(matches!(
            graph.unfold_within(TemporalIndexer::new(2, 1, 1).unwrap(), &mut budget, &ctx),
            Err(GraphError::TemporalMaterialization(TemporalMaterializationError::Cancelled))
        ));
    }
}
