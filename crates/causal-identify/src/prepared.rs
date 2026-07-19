//! Prepared ADMG with §10.5 identification caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{AssumptionSet, VariableId};
use causal_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace, NodeRef};

use crate::error::IdentificationError;

/// Prepared semi-Markovian graph for ID/IDC.
///
/// Owns one [`Admg`] and caches district labels, topological order, and
/// ancestral closures so recursive ID does not clone the graph per step.
#[derive(Clone, Debug)]
pub struct PreparedAdmg {
    admg: Admg,
    declared_assumptions: AssumptionSet,
    /// District id per dense node over the full graph (`0..n_districts-1`).
    district_of: Arc<[u32]>,
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
        let district_of: Arc<[u32]> = Arc::from(admg.districts());
        let topo = Arc::from(topological_order(&admg)?);
        Ok(Self {
            admg,
            declared_assumptions,
            district_of,
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

    /// District label of `id` in the full graph.
    #[must_use]
    pub fn district_of(&self, id: DenseNodeId) -> u32 {
        self.district_of[id.as_usize()]
    }

    /// Full-graph district labels.
    #[must_use]
    pub fn districts(&self) -> &[u32] {
        &self.district_of
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
        for (i, node) in self.admg.nodes().iter().enumerate() {
            if let NodeRef::Static(v) = node {
                if *v == id {
                    return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
                }
            }
        }
        Err(IdentificationError::UnknownVariable { id })
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
        let computed = ancestors_in_admg(&self.admg, &key.0, active, None, ws);
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
        ancestors_in_admg(&self.admg, seeds, active, Some(intervene), ws)
    }

    /// C-components (districts) of the subgraph induced by `nodes` under bidirected edges.
    #[must_use]
    pub fn c_components(&self, nodes: &BitSet) -> Vec<BitSet> {
        let n = self.admg.node_count();
        let mut seen = BitSet::with_len(n);
        let mut comps = Vec::new();
        for id in nodes.to_dense_ids() {
            if seen.contains(id) {
                continue;
            }
            let mut comp = BitSet::with_len(n);
            let mut stack = vec![id];
            seen.insert(id);
            comp.insert(id);
            while let Some(u) = stack.pop() {
                for &v in self.admg.bidirected_neighbors(u) {
                    if !nodes.contains(v) || seen.contains(v) {
                        continue;
                    }
                    seen.insert(v);
                    comp.insert(v);
                    stack.push(v);
                }
            }
            comps.push(comp);
        }
        comps
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

fn topological_order(admg: &Admg) -> Result<Vec<DenseNodeId>, IdentificationError> {
    let n = admg.node_count();
    let mut indeg = vec![0u32; n];
    for i in 0..n {
        let u = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        for &c in admg.children(u) {
            indeg[c.as_usize()] = indeg[c.as_usize()].saturating_add(1);
        }
    }
    let mut queue: Vec<DenseNodeId> = (0..n)
        .filter(|&i| indeg[i] == 0)
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop() {
        order.push(u);
        for &c in admg.children(u) {
            let i = c.as_usize();
            indeg[i] = indeg[i].saturating_sub(1);
            if indeg[i] == 0 {
                queue.push(c);
            }
        }
    }
    if order.len() != n {
        return Err(IdentificationError::msg("ADMG topological order incomplete (cycle?)"));
    }
    Ok(order)
}

/// Ancestors of `seeds` within `active`. When `bar_x` is set, skip parents of those nodes.
fn ancestors_in_admg(
    admg: &Admg,
    seeds: &BitSet,
    active: &BitSet,
    bar_x: Option<&BitSet>,
    ws: &mut GraphWorkspace,
) -> BitSet {
    let n = admg.node_count();
    let mut out = BitSet::with_len(n);
    ws.prepare(n);
    for id in seeds.to_dense_ids() {
        if !active.contains(id) {
            continue;
        }
        if !out.contains(id) {
            out.insert(id);
            ws.frontier.push(id);
        }
    }
    while let Some(u) = ws.frontier.pop() {
        // In G_{\bar X}, intervened nodes have no incoming edges.
        if bar_x.is_some_and(|bx| bx.contains(u)) {
            continue;
        }
        for &p in admg.parents(u) {
            if !active.contains(p) || out.contains(p) {
                continue;
            }
            out.insert(p);
            ws.frontier.push(p);
        }
    }
    out
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
