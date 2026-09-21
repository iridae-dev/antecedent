//! Prepared ADMG with §10.5 identification caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{AssumptionSet, VariableId};
use antecedent_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace, NodeRef};

use crate::error::IdentificationError;

/// Prepared semi-Markovian graph for ID/IDC.
///
/// Owns one [`Admg`] and caches the topological order, the variable-to-node index, and
/// ancestral closures so recursive ID does not clone the graph per step.
#[derive(Clone, Debug)]
pub struct PreparedAdmg {
    admg: Admg,
    declared_assumptions: AssumptionSet,
    /// Dense node of each static variable.
    var_index: Arc<HashMap<VariableId, DenseNodeId>>,
    /// Topological order of all nodes (directed edges).
    topo: Arc<[DenseNodeId]>,
    /// Memo of ancestral closures keyed by (seed-set, active-set).
    ancestor_memo: HashMap<(BitSet, BitSet), BitSet>,
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
        admg.validate().map_err(IdentificationError::from)?;
        let topo: Arc<[DenseNodeId]> = Arc::from(admg.topological_order().ok_or_else(|| {
            IdentificationError::msg("ADMG topological order incomplete (cycle?)")
        })?);
        let mut var_index = HashMap::with_capacity(admg.node_count());
        for (i, node) in admg.nodes().iter().enumerate() {
            if let NodeRef::Static(v) = node {
                var_index
                    .entry(*v)
                    .or_insert_with(|| DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
        Ok(Self {
            admg,
            declared_assumptions,
            var_index: Arc::new(var_index),
            topo,
            ancestor_memo: HashMap::new(),
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
        self.var_index.get(&id).copied().ok_or(IdentificationError::UnknownVariable { id })
    }

    /// Dense node → variable id.
    ///
    /// # Errors
    ///
    /// Non-static or unknown node.
    pub fn dense_to_var(&self, id: DenseNodeId) -> Result<VariableId, IdentificationError> {
        match self.admg.nodes().get(id.as_usize()) {
            Some(NodeRef::Static(v)) => Ok(*v),
            _ => Err(IdentificationError::msg(format!("unknown dense node {}", id.raw()))),
        }
    }

    /// Ancestral closure of `seeds` within `active` (including seeds), with memoization.
    pub fn ancestors_within(
        &mut self,
        seeds: &BitSet,
        active: &BitSet,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        debug_assert_eq!(seeds.bit_len(), active.bit_len());
        // Key must include `active`: reachability depends on which nodes exist.
        let mut seed_key = seeds.clone();
        seed_key.intersect_with(active);
        let key = (seed_key, active.clone());
        if let Some(cached) = self.ancestor_memo.get(&key) {
            return cached.clone();
        }
        let computed = self.admg.ancestors_within(&key.0, active, None, ws);
        self.ancestor_memo.insert(key, computed.clone());
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
        self.admg.districts_within(nodes)
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

/// Dense index of the static node labelled `id` in `nodes`.
///
/// The one place that maps a variable to its position in a graph's node list; node order
/// is a property of the graph, never assumed to equal the variable's raw id.
///
/// # Errors
///
/// No static node carries `id`.
pub(crate) fn dense_of_static(
    nodes: &[NodeRef],
    id: VariableId,
) -> Result<DenseNodeId, IdentificationError> {
    nodes
        .iter()
        .position(|node| *node == NodeRef::Static(id))
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node index fits")))
        .ok_or(IdentificationError::UnknownVariable { id })
}

/// Convert a static DAG into an ADMG (no bidirected edges).
///
/// # Errors
///
/// Graph construction failures.
pub fn dag_to_admg(dag: &Dag) -> Result<Admg, IdentificationError> {
    let mut admg = Admg::empty();
    for node in dag.nodes() {
        admg.add_node(*node).map_err(IdentificationError::from)?;
    }
    for e in dag.edges() {
        let (from, to) = e.parent_child().expect("dag edge");
        admg.insert_directed(from, to).map_err(IdentificationError::from)?;
    }
    Ok(admg)
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
}
