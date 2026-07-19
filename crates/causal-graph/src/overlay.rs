//! Intervention / mutilation overlays on an immutable [`Dag`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::dag::Dag;
use crate::dsep::{DSeparationWorkspace, SeparationResult};
use crate::error::GraphError;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

/// Compact edge-visibility mask for graph surgery without cloning adjacency.
///
/// A directed edge `from → to` is visible iff
/// `!hide_outgoing(from) && !hide_incoming(to)`.
#[derive(Clone, Debug)]
pub struct GraphOverlay {
    /// Nodes whose incoming directed edges are hidden (`do(X)` mutilation).
    hide_incoming: BitSet,
    /// Nodes whose outgoing directed edges are hidden (`G_{\bar T}`).
    hide_outgoing: BitSet,
}

impl GraphOverlay {
    /// Observational overlay (all edges visible) for `n` nodes.
    #[must_use]
    pub fn observational(n: usize) -> Self {
        Self { hide_incoming: BitSet::with_len(n), hide_outgoing: BitSet::with_len(n) }
    }

    /// Do-intervention overlay: hide all edges into each intervened node.
    #[must_use]
    pub fn do_intervention(n: usize, intervened: &[DenseNodeId]) -> Self {
        let mut overlay = Self::observational(n);
        for &v in intervened {
            if v.as_usize() < n {
                overlay.hide_incoming.insert(v);
            }
        }
        overlay
    }

    /// Hide all outgoing edges from each node (backdoor / IV mutilation style).
    #[must_use]
    pub fn remove_outgoing(n: usize, nodes: &[DenseNodeId]) -> Self {
        let mut overlay = Self::observational(n);
        for &v in nodes {
            if v.as_usize() < n {
                overlay.hide_outgoing.insert(v);
            }
        }
        overlay
    }

    /// Whether every directed edge of the base graph remains visible.
    #[must_use]
    pub fn is_observational(&self) -> bool {
        !self.hide_incoming.any() && !self.hide_outgoing.any()
    }

    /// Whether directed edge `from → to` is visible under this overlay.
    #[must_use]
    pub fn edge_visible(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        !self.hide_outgoing.contains(from) && !self.hide_incoming.contains(to)
    }
}

/// Borrowed view of a [`Dag`] under a [`GraphOverlay`] (no adjacency clone).
#[derive(Clone, Copy, Debug)]
pub struct DagView<'a> {
    dag: &'a Dag,
    overlay: &'a GraphOverlay,
}

impl<'a> DagView<'a> {
    /// Node count (same as the base DAG).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.dag.node_count()
    }

    /// Base DAG.
    #[must_use]
    pub fn dag(&self) -> &'a Dag {
        self.dag
    }

    /// Overlay mask.
    #[must_use]
    pub fn overlay(&self) -> &'a GraphOverlay {
        self.overlay
    }

    /// Collect visible parents of `id` into `out` (cleared first).
    pub fn parents_into(&self, id: DenseNodeId, out: &mut Vec<DenseNodeId>) {
        out.clear();
        if id.as_usize() >= self.dag.node_count() {
            return;
        }
        for &p in self.dag.parents(id) {
            if self.overlay.edge_visible(p, id) {
                out.push(p);
            }
        }
    }

    /// Collect visible children of `id` into `out` (cleared first).
    pub fn children_into(&self, id: DenseNodeId, out: &mut Vec<DenseNodeId>) {
        out.clear();
        if id.as_usize() >= self.dag.node_count() {
            return;
        }
        for &c in self.dag.children(id) {
            if self.overlay.edge_visible(id, c) {
                out.push(c);
            }
        }
    }

    /// Ancestors under the overlay (including seeds).
    pub fn ancestors_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        self.dag.ancestors_of_with(nodes, out, ws, Some(self.overlay));
    }

    /// Descendants under the overlay (including seeds).
    pub fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        self.dag.descendants_of_with(nodes, out, ws, Some(self.overlay));
    }

    /// d-separation on the overlay view (boolean; no path alloc).
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn is_d_separated(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Result<bool, GraphError> {
        self.dag.is_d_separated_with(x, y, z, ws, Some(self.overlay))
    }

    /// d-separation with witness on the overlay view.
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn d_separation(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Result<SeparationResult, GraphError> {
        self.dag.d_separation_with(x, y, z, ws, Some(self.overlay))
    }

    /// Materialize a concrete [`Dag`] with only visible edges (nodes preserved).
    ///
    /// # Errors
    ///
    /// Node-count overflow (should not occur for a valid base DAG).
    pub fn materialize(&self) -> Result<Dag, GraphError> {
        let n = u32::try_from(self.dag.node_count()).map_err(|_| GraphError::TooManyNodes)?;
        let mut out = Dag::with_variables(n);
        // Base is a valid DAG; removing edges cannot create cycles.
        for i in 0..self.dag.node_count() {
            let from = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            for &to in self.dag.children(from) {
                if self.overlay.edge_visible(from, to) {
                    out.insert_directed_unchecked(from, to);
                }
            }
        }
        Ok(out)
    }
}

impl Dag {
    /// Borrow `self` under an intervention / mutilation overlay.
    #[must_use]
    pub fn view<'a>(&'a self, overlay: &'a GraphOverlay) -> DagView<'a> {
        DagView { dag: self, overlay }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsep::DSeparationWorkspace;

    fn chain3() -> Dag {
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        g
    }

    #[test]
    fn do_intervention_view_matches_materialize() {
        let g = chain3();
        let t = DenseNodeId::from_raw(1);
        let overlay = GraphOverlay::do_intervention(g.node_count(), &[t]);
        let view = g.view(&overlay);
        let mut parents = Vec::new();
        view.parents_into(t, &mut parents);
        assert!(parents.is_empty());
        let mut children = Vec::new();
        view.children_into(DenseNodeId::from_raw(0), &mut children);
        assert!(children.is_empty());
        view.children_into(t, &mut children);
        assert_eq!(children, vec![DenseNodeId::from_raw(2)]);

        let m = view.materialize().unwrap();
        assert!(m.parents(t).is_empty());
        assert!(m.children(DenseNodeId::from_raw(0)).is_empty());
        assert_eq!(m.children(t).len(), 1);
    }

    #[test]
    fn remove_outgoing_hides_treatment_children() {
        let g = chain3();
        let t = DenseNodeId::from_raw(1);
        let overlay = GraphOverlay::remove_outgoing(g.node_count(), &[t]);
        let view = g.view(&overlay);
        let mut children = Vec::new();
        view.children_into(t, &mut children);
        assert!(children.is_empty());
        let mut parents = Vec::new();
        view.parents_into(t, &mut parents);
        assert_eq!(parents, vec![DenseNodeId::from_raw(0)]);
    }

    #[test]
    fn view_dsep_matches_materialized_mutilate() {
        // A → T → Y with confounder A → Y; do(T) blocks A→T.
        let mut g = Dag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(a, y).unwrap();

        let overlay = GraphOverlay::do_intervention(g.node_count(), &[t]);
        let view = g.view(&overlay);
        let materialized = view.materialize().unwrap();
        let mut ws = DSeparationWorkspace::default();
        let view_sep = view.is_d_separated(t, y, &[a], &mut ws).unwrap();
        let mat_sep = materialized.is_d_separated(t, y, &[a], &mut ws).unwrap();
        assert_eq!(view_sep, mat_sep);
        // Under do(T), path T→Y remains; A does not separate T from Y.
        assert!(!view_sep);
    }

    /// Overlay d-sep agrees with `Dag::mutilate` on random multi-treatment graphs.
    #[test]
    fn property_overlay_dsep_matches_mutilate_on_random_dags() {
        use causal_core::CausalRng;

        let mut rng = CausalRng::from_seed(17);
        let mut ws = DSeparationWorkspace::default();
        for _ in 0..40 {
            let n = 4 + (rng.next_u64() % 3) as u32; // 4..=6
            let mut g = Dag::with_variables(n);
            let mut order: Vec<u32> = (0..n).collect();
            for i in (1..n as usize).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for i in 0..n as usize {
                for j in (i + 1)..n as usize {
                    if rng.next_u64() % 3 == 0 {
                        let _ = g.insert_directed(
                            DenseNodeId::from_raw(order[i]),
                            DenseNodeId::from_raw(order[j]),
                        );
                    }
                }
            }
            let k = 1 + (rng.next_u64() as usize % (n as usize).min(3));
            let mut treated = Vec::new();
            while treated.len() < k {
                let t = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
                if !treated.contains(&t) {
                    treated.push(t);
                }
            }
            let overlay = GraphOverlay::do_intervention(g.node_count(), &treated);
            let view = g.view(&overlay);
            let mutilated = g.mutilate(&treated).unwrap();
            // Structure agreement.
            let mat = view.materialize().unwrap();
            for i in 0..n {
                let u = DenseNodeId::from_raw(i);
                assert_eq!(mat.children(u), mutilated.children(u));
            }
            // d-sep agreement on random queries.
            for _ in 0..12 {
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
                    if rng.next_u64() % 2 == 0 {
                        z.push(v);
                    }
                }
                let view_sep = view.is_d_separated(x, y, &z, &mut ws).unwrap();
                let mut_sep = mutilated.is_d_separated(x, y, &z, &mut ws).unwrap();
                assert_eq!(
                    view_sep, mut_sep,
                    "overlay≠mutilate d-sep x={} y={} z={:?} T={:?}",
                    x.raw(),
                    y.raw(),
                    z.iter().map(|v| v.raw()).collect::<Vec<_>>(),
                    treated.iter().map(|v| v.raw()).collect::<Vec<_>>()
                );
            }
        }
    }
}
