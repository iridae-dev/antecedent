//! Lazy finite unfolding of temporal DAGs and graph-review artifacts.
//!
//! Stationary algorithms query edges on demand via [`LazyUnfoldedTemporalGraph`].
//! Full materialisation is available when a static [`Dag`] is required .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::sync::Arc;

use causal_core::VariableId;
use causal_core::{TemporalIndexer, TemporalNodeKey};

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

/// Lazy finite unfolding: template edges are replicated on query, not upfront.
#[derive(Clone, Debug)]
pub struct LazyUnfoldedTemporalGraph {
    /// Stationary / lagged summary graph.
    pub template: TemporalDag,
    /// Finite window indexer.
    pub indexer: TemporalIndexer,
}

impl TemporalDag {
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
    /// # Errors
    ///
    /// Unknown/non-lagged nodes, indexer issues, or cycle insertion.
    pub fn unfold(&self, indexer: TemporalIndexer) -> Result<UnfoldedTemporalGraph, GraphError> {
        self.unfold_lazy(indexer)?.materialize()
    }
}

impl LazyUnfoldedTemporalGraph {
    /// Whether a concrete directed edge exists under the template replication.
    ///
    /// # Errors
    ///
    /// Endpoints outside the indexer window.
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
    pub fn materialize(&self) -> Result<UnfoldedTemporalGraph, GraphError> {
        let n = self.indexer.dense_len();
        let n_u32 = u32::try_from(n).map_err(|_| GraphError::TooManyNodes)?;
        let mut dag = Dag::with_variables(n_u32);

        for (from_i, _) in self.template.nodes().iter().enumerate() {
            let from = DenseNodeId::try_from_usize(from_i)?;
            let from_key = self
                .template
                .temporal_key(from)
                .ok_or(GraphError::InvalidEndpoints { message: "unfold requires lagged nodes" })?;
            for &to in self.template.children(from) {
                let to_key =
                    self.template.temporal_key(to).ok_or(GraphError::InvalidEndpoints {
                        message: "unfold requires lagged nodes",
                    })?;
                insert_replicated_edges(&mut dag, &self.indexer, from_key, to_key)?;
            }
        }

        Ok(UnfoldedTemporalGraph { dag, indexer: self.indexer.clone() })
    }
}

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

fn insert_replicated_edges(
    dag: &mut Dag,
    indexer: &TemporalIndexer,
    from_key: TemporalNodeKey,
    to_key: TemporalNodeKey,
) -> Result<(), GraphError> {
    let min_off = -(indexer.history() as i32);
    let max_off = (indexer.horizon() as i32) - 1;
    // Template offsets are <= 0; shift far enough that even a fully lagged edge
    // (both endpoints negative) lands in the newest window slices.
    let lo = min_off - from_key.offset.min(to_key.offset);
    let hi = max_off - from_key.offset.max(to_key.offset);
    for delta in lo..=hi {
        let a = TemporalNodeKey {
            variable: from_key.variable,
            offset: from_key.offset.saturating_add(delta),
        };
        let b = TemporalNodeKey {
            variable: to_key.variable,
            offset: to_key.offset.saturating_add(delta),
        };
        if a.offset < min_off || a.offset > max_off || b.offset < min_off || b.offset > max_off {
            continue;
        }
        let from_dense = indexer.dense_id(a).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        let to_dense = indexer.dense_id(b).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        let from = DenseNodeId::from_raw(from_dense);
        let to = DenseNodeId::from_raw(to_dense);
        if from == to {
            continue;
        }
        match dag.insert_directed(from, to) {
            Ok(()) | Err(GraphError::DuplicateEdge { .. }) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
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
        let pending: Vec<_> =
            self.pending_edges.iter().copied().filter(|e| *e != (from, to)).collect();
        self.pending_edges = Arc::from(pending);
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
        let pending: Vec<_> =
            self.pending_edges.iter().copied().filter(|e| *e != (from, to)).collect();
        self.pending_edges = Arc::from(pending);
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
        let undirected: Vec<_> = self
            .pending_undirected
            .iter()
            .copied()
            .filter(|&(a, b)| (a, b) != (from, to) && (a, b) != (to, from))
            .collect();
        self.pending_undirected = Arc::from(undirected);
        // Newly directed edge still needs acceptance unless already accepted.
        if !self.pending_edges.iter().any(|e| *e == (from, to)) {
            let mut pending = self.pending_edges.to_vec();
            pending.push((from, to));
            self.pending_edges = Arc::from(pending);
        }
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
    lag: causal_core::Lag,
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
#[allow(clippy::many_single_char_names)]
mod tests {
    use causal_core::TemporalIndexer;
    use causal_core::{Lag, VariableId};

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
        assert!(edge_count >= 1);
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

    fn random_small_template(rng: &mut causal_core::CausalRng) -> (TemporalDag, u32, u32, u32) {
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
    fn dag_from_lazy_scan(lazy: &LazyUnfoldedTemporalGraph) -> Dag {
        let n = lazy.indexer.dense_len();
        let mut dag = Dag::with_variables(u32::try_from(n).unwrap());
        let min_off = -(lazy.indexer.history() as i32);
        let max_off = (lazy.indexer.horizon() as i32) - 1;
        let n_vars = lazy.indexer.variable_count();
        for va in 0..n_vars {
            for oa in min_off..=max_off {
                for vb in 0..n_vars {
                    for ob in min_off..=max_off {
                        let from =
                            TemporalNodeKey { variable: VariableId::from_raw(va), offset: oa };
                        let to =
                            TemporalNodeKey { variable: VariableId::from_raw(vb), offset: ob };
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
    fn property_lazy_unfold_matches_materialize_on_small_templates() {
        use causal_core::CausalRng;

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
                            let from = TemporalNodeKey {
                                variable: VariableId::from_raw(va),
                                offset: oa,
                            };
                            let to = TemporalNodeKey {
                                variable: VariableId::from_raw(vb),
                                offset: ob,
                            };
                            let dense_a =
                                DenseNodeId::from_raw(indexer.dense_id(from).unwrap());
                            let dense_b =
                                DenseNodeId::from_raw(indexer.dense_id(to).unwrap());
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
        use causal_core::CausalRng;

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
                assert_eq!(a, b, "lazy-scan adjacency ≠ materialize at {}", i);
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
                    lazy_sep, mat_sep,
                    "unfolded d-sep mismatch x={} y={}",
                    x.raw(),
                    y.raw()
                );
            }
        }
    }
}
