//! Prepared ADMG with §10.5 identification caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{AssumptionSet, VariableId};
use antecedent_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace, NodeRef};

use crate::error::IdentificationError;
use crate::node_map::NodeMap;

#[derive(Clone, Debug)]
enum PreparedGraph {
    Ordinary(Arc<Admg>),
    TemporalWindow(antecedent_graph::UnrolledTemporalAdmg),
}
impl std::ops::Deref for PreparedGraph {
    type Target = Admg;
    fn deref(&self) -> &Admg {
        match self {
            Self::Ordinary(graph) => graph,
            Self::TemporalWindow(window) => window.admg(),
        }
    }
}

/// Prepared semi-Markovian graph for ID/IDC.
///
/// Owns one [`Admg`] and caches the topological order and the variable-to-node index.
/// Ancestral closures are memoized on the identification run ([`AncestorMemo`]), not here,
/// so cloning a prepared graph does not copy that table and the graph stays immutable
/// during the recursion.
#[derive(Clone, Debug)]
pub struct PreparedAdmg {
    admg: PreparedGraph,
    // The temporal graph's lease is retained in PreparedGraph; this second
    // lease owns the topological order and node index built from it.
    _temporal_index_lease: Option<Arc<antecedent_core::execution::AdmissionLease>>,
    declared_assumptions: AssumptionSet,
    /// Dense node of each static variable.
    var_index: Arc<NodeMap>,
    /// Topological order of all nodes (directed edges).
    topo: Arc<[DenseNodeId]>,
}

/// Ancestral closures for one identification run.
///
/// The key is the seed set intersected with the active set, then the active set.
/// The value is the closure [`Admg::ancestors_within`] returns for that pair.
#[derive(Clone, Debug, Default)]
pub(crate) struct AncestorMemo {
    map: HashMap<(BitSet, BitSet), BitSet>,
}

impl PreparedAdmg {
    /// Prepare from an ADMG with no extra declared assumptions.
    ///
    /// # Errors
    ///
    /// Directed-cycle validation failure.
    pub fn new(admg: Admg) -> Result<Self, IdentificationError> {
        Self::with_assumptions(admg, AssumptionSet::new())
    }

    /// Prepare from an ADMG, retaining caller-declared assumptions.
    ///
    /// # Errors
    ///
    /// Directed-cycle validation failure.
    pub fn with_assumptions(
        admg: Admg,
        declared_assumptions: AssumptionSet,
    ) -> Result<Self, IdentificationError> {
        Self::with_graph_owner(PreparedGraph::Ordinary(Arc::new(admg)), declared_assumptions, None)
    }

    /// Prepare an admitted temporal window without cloning its graph. The
    /// index has separate admission and follows this handle across clones.
    /// # Errors
    /// Graph validation, cancellation, logical limits or admission.
    pub fn from_unrolled_within(
        window: antecedent_graph::UnrolledTemporalAdmg,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        use crate::error::IdentificationBudget;
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        let limits = budget.limits();
        if !window.is_admitted() {
            return Err(IdentificationError::unsupported("temporal graph is not admitted"));
        }
        let graph = window.admg();
        let nodes = u64::try_from(graph.node_count())
            .map_err(|_| IdentificationError::budget(IdentificationBudget::Memory))?;
        if nodes > limits.max_graph_nodes {
            return Err(IdentificationError::budget(IdentificationBudget::Memory));
        }
        budget
            .charge_work(nodes)
            .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
        let mut directed = 0u64;
        let mut bidirected_adjacency = 0u64;
        for i in 0..graph.node_count() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let id = DenseNodeId::from_index(i);
            directed = directed
                .checked_add(graph.children(id).len() as u64)
                .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
            bidirected_adjacency = bidirected_adjacency
                .checked_add(graph.bidirected_neighbors(id).len() as u64)
                .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
        }
        // Each undirected edge occurs in two adjacency rows.
        let edges = directed
            .checked_add(bidirected_adjacency / 2)
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
        if edges > limits.max_graph_edges {
            return Err(IdentificationError::budget(IdentificationBudget::Memory));
        }
        // Counting, Kahn traversal and NodeMap construction each scan nodes;
        // Kahn scans each directed edge once. The extra passes are conservative.
        let work = nodes
            .checked_mul(7)
            .and_then(|n| directed.checked_mul(2).and_then(|d| n.checked_add(d)))
            .and_then(|n| n.checked_add(bidirected_adjacency))
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
        budget
            .charge_work(work)
            .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
        let bytes = nodes
            .checked_mul(512)
            .and_then(|n| edges.checked_mul(64).and_then(|e| n.checked_add(e)))
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
        if bytes > limits.max_bytes {
            return Err(IdentificationError::budget(IdentificationBudget::Memory));
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
            _ => IdentificationError::budget(IdentificationBudget::Memory),
        })?;
        // Reproduce the graph's Kahn order, polling cancellation inside both
        // node and edge loops. The window was built by checked insertions.
        let graph = window.admg();
        let mut indegree = Vec::with_capacity(graph.node_count());
        let mut ready = Vec::new();
        for i in 0..graph.node_count() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let id = DenseNodeId::from_index(i);
            let count = u32::try_from(graph.parents(id).len())
                .map_err(|_| IdentificationError::budget(IdentificationBudget::Memory))?;
            indegree.push(count);
            if count == 0 {
                ready.push(id);
            }
        }
        let mut topo = Vec::with_capacity(graph.node_count());
        while let Some(node) = ready.pop() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            topo.push(node);
            for &child in graph.children(node) {
                if ctx.cancellation.is_cancelled() {
                    return Err(IdentificationError::Cancelled);
                }
                let degree = &mut indegree[child.as_usize()];
                *degree = degree.checked_sub(1).ok_or(IdentificationError::InvariantViolated {
                    message: "temporal graph indegree underflow",
                })?;
                if *degree == 0 {
                    ready.push(child);
                }
            }
        }
        if topo.len() != graph.node_count() {
            return Err(IdentificationError::Graph(antecedent_graph::GraphError::Cycle {
                from: 0,
                to: 0,
            }));
        }
        let var_index = NodeMap::new_with_cancel(graph.nodes(), ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        Ok(Self {
            admg: PreparedGraph::TemporalWindow(window),
            declared_assumptions: AssumptionSet::new(),
            var_index: Arc::new(var_index),
            topo: Arc::from(topo),
            _temporal_index_lease: Some(Arc::new(lease)),
        })
    }

    fn with_graph_owner(
        admg: PreparedGraph,
        declared_assumptions: AssumptionSet,
        lease: Option<Arc<antecedent_core::execution::AdmissionLease>>,
    ) -> Result<Self, IdentificationError> {
        admg.validate().map_err(IdentificationError::from)?;
        let topo: Arc<[DenseNodeId]> = Arc::from(admg.topological_order().ok_or_else(|| {
            IdentificationError::construction(
                antecedent_graph::error::CyclicDagRefusal::unlicensed(
                    antecedent_graph::error::DagOperation::Identification,
                    "prepared_admg",
                    "a directed cycle belongs in a CyclicMixedGraph; condensation names the \
                     feedback and sigma_separates answers a conditional-independence question. \
                     This preparation has no cyclic identification license",
                )
                .to_string(),
            )
        })?);
        let var_index = NodeMap::new(admg.nodes());
        Ok(Self {
            admg,
            declared_assumptions,
            var_index: Arc::new(var_index),
            topo,
            _temporal_index_lease: lease,
        })
    }

    /// Embed a DAG as an ADMG (directed edges only; no bidirected).
    ///
    /// # Errors
    ///
    /// Node/edge construction failures.
    pub fn from_dag(dag: &Dag) -> Result<Self, IdentificationError> {
        Self::from_dag_with_assumptions(dag, AssumptionSet::new())
    }

    /// Embed a DAG as an ADMG with declared assumptions.
    ///
    /// # Errors
    ///
    /// Node/edge construction failures.
    pub fn from_dag_with_assumptions(
        dag: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<Self, IdentificationError> {
        Self::with_assumptions(dag_to_admg(dag)?, assumptions)
    }

    /// Borrow the ADMG.
    #[must_use]
    pub fn admg(&self) -> &Admg {
        &self.admg
    }

    /// Assumptions declared at prepare time.
    #[must_use]
    pub fn declared_assumptions(&self) -> &AssumptionSet {
        &self.declared_assumptions
    }

    /// Topological order over all nodes.
    #[must_use]
    pub fn topo(&self) -> &[DenseNodeId] {
        &self.topo
    }

    /// Map a variable id to its dense node.
    ///
    /// # Errors
    ///
    /// Unknown variable.
    pub fn var_to_dense(&self, id: VariableId) -> Result<DenseNodeId, IdentificationError> {
        self.var_index.dense(id)
    }

    /// Dense node → variable id.
    ///
    /// # Errors
    ///
    /// Non-static or unknown node.
    pub fn dense_to_var(&self, id: DenseNodeId) -> Result<VariableId, IdentificationError> {
        self.var_index.variable(id)
    }

    /// Ancestral closure of `seeds` within `active` (including seeds).
    ///
    /// Seeds outside `active` are dropped before the walk, which is what
    /// [`Admg::ancestors_within`] does by skipping inactive seeds. The signature
    /// stays `&mut self` so existing callers keep compiling; the closure is not
    /// stored on `self`.
    #[allow(
        clippy::needless_pass_by_ref_mut,
        reason = "public signature stays &mut self; the closure is memoized on the identification run"
    )]
    pub fn ancestors_within(
        &mut self,
        seeds: &BitSet,
        active: &BitSet,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        ancestral_closure(&self.admg, seeds, active, ws)
    }

    /// [`Self::ancestors_within`], reusing `memo` for this run.
    ///
    /// A hit is the set the uncached walk returns for the same seeds and active set.
    pub(crate) fn ancestors_cached(
        &self,
        seeds: &BitSet,
        active: &BitSet,
        ws: &mut GraphWorkspace,
        memo: &mut AncestorMemo,
    ) -> BitSet {
        debug_assert_eq!(seeds.bit_len(), active.bit_len());
        let mut seed_key = seeds.clone();
        seed_key.intersect_with(active);
        let key = (seed_key, active.clone());
        if let Some(cached) = memo.map.get(&key) {
            return cached.clone();
        }
        let computed = self.admg.ancestors_within(&key.0, active, None, ws);
        memo.map.insert(key, computed.clone());
        computed
    }

    /// Ancestral closure of `seeds` in `G_{\overline{intervene}}` restricted to `active`.
    ///
    /// Incoming directed edges to nodes in `intervene` are ignored.
    pub fn ancestors_bar_x(
        &self,
        seeds: &BitSet,
        active: &BitSet,
        intervene: &BitSet,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        self.admg.ancestors_within(seeds, active, Some(intervene), ws)
    }

    /// C-components (districts) of the subgraph induced by `nodes` under bidirected edges.
    #[must_use]
    pub fn c_components(&self, nodes: &BitSet) -> Vec<BitSet> {
        self.admg.district_components_within(nodes)
    }

    /// Whether the induced subgraph on `nodes` is a single C-component covering all of `nodes`.
    #[must_use]
    pub fn is_single_c_component(&self, nodes: &BitSet) -> bool {
        if !nodes.any() {
            return true;
        }
        let comps = self.c_components(nodes);
        comps.len() == 1 && comps[0].equal_set(nodes)
    }
}

fn ancestral_closure(
    admg: &Admg,
    seeds: &BitSet,
    active: &BitSet,
    ws: &mut GraphWorkspace,
) -> BitSet {
    debug_assert_eq!(seeds.bit_len(), active.bit_len());
    let mut seed_key = seeds.clone();
    seed_key.intersect_with(active);
    admg.ancestors_within(&seed_key, active, None, ws)
}

/// The identification-facing variable id carried by node `dense` in `nodes`, if any.
///
/// `Static(v)` carries `v`. An `Unfolded` node (a finite-window slot) has no variable
/// label of its own — the same template variable repeats across every slot it occupies —
/// so identification code addresses it by its own dense position instead, treating each
/// slot as a distinct synthetic variable `VariableId::from_raw(dense)`. This matches how
/// unfolded temporal DAGs are identified: `temporal_backdoor::identify_temporal` retargets
/// treatment/outcome onto `VariableId::from_raw(dense)` before calling into this crate's
/// static identifiers. Any other node kind (`Lagged`, `Context`) carries no identification
/// variable and returns `None`.
pub(crate) fn node_variable_id(nodes: &[NodeRef], dense: usize) -> Option<VariableId> {
    match nodes.get(dense)? {
        NodeRef::Static(v) => Some(*v),
        NodeRef::Unfolded { .. } => Some(VariableId::from_raw(u32::try_from(dense).ok()?)),
        NodeRef::Lagged { .. } | NodeRef::Context { .. } => None,
    }
}

/// Dense index of the static or unfolded node addressed by `id` in `nodes`.
///
/// The one place that maps a variable to its position in a graph's node list; node order
/// is a property of the graph, never assumed to equal the variable's raw id (except for
/// `Unfolded` nodes, which are addressed by dense position; see [`node_variable_id`]).
///
/// # Errors
///
/// No node carries `id`.
pub(crate) fn dense_of_static(
    nodes: &[NodeRef],
    id: VariableId,
) -> Result<DenseNodeId, IdentificationError> {
    // A single lookup needs no index; callers resolving many variables against one graph use
    // [`crate::node_map::NodeMap`], which is built once and answers in O(1).
    nodes
        .iter()
        .enumerate()
        .find_map(|(index, node)| match node {
            NodeRef::Static(v) => (*v == id).then(|| DenseNodeId::from_index(index)),
            NodeRef::Unfolded { .. } => (u32::try_from(index).ok() == Some(id.raw()))
                .then(|| DenseNodeId::from_index(index)),
            NodeRef::Lagged { .. } | NodeRef::Context { .. } => None,
        })
        .ok_or(IdentificationError::UnknownVariable { id })
}

/// Convert a static DAG into an ADMG (no bidirected edges).
///
/// # Errors
///
/// Graph construction failures.
pub fn dag_to_admg(dag: &Dag) -> Result<Admg, IdentificationError> {
    Admg::from_dag(dag).map_err(IdentificationError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_embed_preserves_edges() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let prep = PreparedAdmg::from_dag(&dag).unwrap();
        assert_eq!(prep.admg().children(DenseNodeId::from_raw(0)), &[DenseNodeId::from_raw(1)]);
        assert!(!prep.is_single_c_component(&{
            let mut b = BitSet::with_len(3);
            b.insert(DenseNodeId::from_raw(0));
            b.insert(DenseNodeId::from_raw(1));
            b
        }));
    }

    #[test]
    fn var_to_dense_follows_node_labels_not_raw_ids() {
        let mut g = Admg::empty();
        g.add_node(NodeRef::Static(VariableId::from_raw(7))).unwrap();
        g.add_node(NodeRef::Static(VariableId::from_raw(3))).unwrap();
        let prep = PreparedAdmg::new(g).unwrap();
        assert_eq!(prep.var_to_dense(VariableId::from_raw(7)).unwrap(), DenseNodeId::from_raw(0));
        assert_eq!(prep.var_to_dense(VariableId::from_raw(3)).unwrap(), DenseNodeId::from_raw(1));
        assert!(matches!(
            prep.var_to_dense(VariableId::from_raw(0)),
            Err(IdentificationError::UnknownVariable { .. })
        ));
        assert_eq!(
            dense_of_static(prep.admg().nodes(), VariableId::from_raw(3)).unwrap(),
            DenseNodeId::from_raw(1)
        );
    }

    #[test]
    fn topo_matches_graph_topological_order() {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(3, 1), (1, 0), (3, 2)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let expected = g.topological_order().unwrap();
        let prep = PreparedAdmg::new(g).unwrap();
        assert_eq!(prep.topo(), expected.as_slice());
    }

    #[test]
    fn c_components_respect_bidirected() {
        let mut g = Admg::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let prep = PreparedAdmg::new(g).unwrap();
        let mut nodes = BitSet::with_len(3);
        for i in 0..3u32 {
            nodes.insert(DenseNodeId::from_raw(i));
        }
        let comps = prep.c_components(&nodes);
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn run_local_ancestor_memo_matches_the_uncached_closure() {
        let mut g = Admg::with_variables(4);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(2)).unwrap();
        let mut prep = PreparedAdmg::new(g).unwrap();
        let mut seeds = BitSet::with_len(4);
        seeds.insert(DenseNodeId::from_raw(2));
        let mut active = BitSet::with_len(4);
        for i in 0..3u32 {
            active.insert(DenseNodeId::from_raw(i));
        }
        let mut ws = GraphWorkspace::default();
        let fresh = prep.ancestors_within(&seeds, &active, &mut ws);
        let mut memo = AncestorMemo::default();
        let once = prep.ancestors_cached(&seeds, &active, &mut ws, &mut memo);
        let twice = prep.ancestors_cached(&seeds, &active, &mut ws, &mut memo);
        assert!(fresh.equal_set(&once));
        assert!(twice.equal_set(&once));
        assert_eq!(memo.map.len(), 1);
        assert!(once.contains(DenseNodeId::from_raw(0)));
        assert!(once.contains(DenseNodeId::from_raw(1)));
        assert!(once.contains(DenseNodeId::from_raw(2)));
        assert!(!once.contains(DenseNodeId::from_raw(3)));
    }
}

#[cfg(test)]
mod temporal_owner_tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{Lag, TemporalMaterializationLimits};
    #[test]
    fn prepared_temporal_window_keeps_both_leases_across_clones() {
        let mut graph = antecedent_graph::TemporalAdmg::empty();
        graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let limits = TemporalMaterializationLimits {
            max_offsets: 8,
            max_work: 1024,
            max_bytes: 50_000,
            max_graph_nodes: 8,
            max_graph_edges: 8,
        };
        let pool = AdmissionPool::fixed(1, 50_000);
        let ctx = antecedent_core::ExecutionContext::for_tests(1).with_admission(pool.clone());
        let window = graph
            .unroll_window_over_within(
                2,
                1,
                &mut antecedent_core::TemporalBudget::new(limits),
                &ctx,
            )
            .unwrap();
        let prepared = PreparedAdmg::from_unrolled_within(
            window,
            &mut antecedent_core::TemporalBudget::new(limits),
            &ctx,
        )
        .unwrap();
        assert_eq!(prepared.admg().node_count(), 2);
        let used = pool.available().bytes();
        assert!(used < 50_000);
        let clone = prepared.clone();
        drop(prepared);
        assert_eq!(pool.available().bytes(), used);
        drop(clone);
        assert_eq!(pool.available().bytes(), 50_000);
    }
}
