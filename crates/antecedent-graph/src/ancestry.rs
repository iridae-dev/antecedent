//! Directed ancestry, descendants, and intervention mutilation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::algo::closure_into;
use crate::dag::Dag;
use crate::error::GraphError;
use crate::overlay::GraphOverlay;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

impl Dag {
    /// Collect all ancestors of `nodes` (including `nodes` themselves) into `out`.
    pub fn ancestors_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        self.ancestors_of_with(nodes, out, ws, None);
    }

    /// Ancestors under an optional edge-visibility overlay.
    pub(crate) fn ancestors_of_with(
        &self,
        nodes: &[DenseNodeId],
        out: &mut BitSet,
        ws: &mut GraphWorkspace,
        overlay: Option<&GraphOverlay>,
    ) {
        closure_into(
            self.node_count(),
            |u| self.parents(u),
            nodes.iter().copied(),
            out,
            ws,
            |u, p| overlay.is_none_or(|ov| ov.edge_visible(p, u)),
        );
    }

    /// Collect all descendants of `nodes` (including `nodes`) into `out`.
    pub fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        self.descendants_of_with(nodes, out, ws, None);
    }

    /// Descendants under an optional edge-visibility overlay.
    pub(crate) fn descendants_of_with(
        &self,
        nodes: &[DenseNodeId],
        out: &mut BitSet,
        ws: &mut GraphWorkspace,
        overlay: Option<&GraphOverlay>,
    ) {
        closure_into(
            self.node_count(),
            |u| self.children(u),
            nodes.iter().copied(),
            out,
            ws,
            |u, c| overlay.is_none_or(|ov| ov.edge_visible(u, c)),
        );
    }

    /// Mutilate the graph under intervention: remove all edges into each
    /// intervened node. Returns a new DAG (nodes preserved).
    ///
    /// Prefer [`Dag::view`] with [`GraphOverlay::do_intervention`] on hot paths
    /// to avoid cloning adjacency.
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn mutilate(&self, intervened: &[DenseNodeId]) -> Result<Dag, GraphError> {
        for &v in intervened {
            self.validate_node(v)?;
        }
        let overlay = GraphOverlay::do_intervention(self.node_count(), intervened);
        self.view(&overlay).materialize()
    }

}
