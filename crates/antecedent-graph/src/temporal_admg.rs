//! Temporal ADMG: lagged nodes, directed edges, and bidirected latent edges.
//!
//! A [`TemporalAdmg`] whose variables each appear once projects onto a static
//! [`Admg`] with the same variable ids and the same mixed-graph edges. General
//! identification of that projection is the same functional. A variable that
//! occupies more than one lag does not have that projection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_core::{Lag, TemporalNodeKey, VariableId};

use crate::admg::Admg;
use crate::algo::bfs_reaches;
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// A finite window of a [`TemporalAdmg`] template as one static [`Admg`].
///
/// Coordinate `period * variable_count + variable` is the variable `variable` in that period
/// (the layout of a `TemporalIndexer` over the same variable count). Template edges are
/// replicated at their delay (the lag of the cause minus the lag of the effect) wherever both
/// endpoints fall in the window. A cause that falls before
/// the window is an observed variable the window does not contain: it is treated as a latent
/// common cause of its in-window effects (and of its in-window bidirected partners), so a
/// confounding path through the truncated history is kept, never dropped. History deeper than
/// the direct lags of the window's first periods is the initial state, as in the unfolding of
/// a [`crate::TemporalDag`].
#[derive(Clone, Debug)]
pub struct UnrolledTemporalAdmg {
    admg: Admg,
    variables: Vec<VariableId>,
    periods: u32,
}

impl UnrolledTemporalAdmg {
    /// The window as a static ADMG whose node `i` has variable id `i`.
    #[must_use]
    pub fn admg(&self) -> &Admg {
        &self.admg
    }

    /// Number of periods in the window.
    #[must_use]
    pub const fn periods(&self) -> u32 {
        self.periods
    }

    /// The variable ids of one period, in coordinate order (`0 .. variable_count`).
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }

    /// Window coordinate of template `variable` in `period`, if both exist.
    #[must_use]
    pub fn coordinate(&self, variable: VariableId, period: u32) -> Option<VariableId> {
        if period >= self.periods {
            return None;
        }
        let index = self.variables.iter().position(|v| *v == variable)?;
        let count = u32::try_from(self.variables.len()).ok()?;
        Some(VariableId::from_raw(period * count + u32::try_from(index).ok()?))
    }

    /// Template variable and period of a window coordinate.
    #[must_use]
    pub fn locate(&self, coordinate: VariableId) -> Option<(VariableId, u32)> {
        let count = u32::try_from(self.variables.len()).ok()?;
        let period = coordinate.raw().checked_div(count)?;
        let index = usize::try_from(coordinate.raw() % count).ok()?;
        (period < self.periods).then(|| (self.variables[index], period))
    }
}

/// Acyclic directed mixed graph over lagged (`VariableId`, `Lag`) nodes.
#[derive(Clone, Debug)]
pub struct TemporalAdmg {
    nodes: Vec<NodeRef>,
    children: Vec<Vec<DenseNodeId>>,
    parents: Vec<Vec<DenseNodeId>>,
    bidirected: Vec<Vec<DenseNodeId>>,
    insert_ws: GraphWorkspace,
}

impl TemporalAdmg {
    /// Empty temporal ADMG.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            bidirected: Vec::new(),
            insert_ws: GraphWorkspace::default(),
        }
    }

    /// Node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Nodes in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        &self.nodes
    }

    /// Add a lagged node.
    ///
    /// # Errors
    ///
    /// Non-lagged node refs or capacity overflow.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        match node {
            NodeRef::Lagged { .. } => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg accepts only Lagged nodes",
                });
            }
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
        self.bidirected.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    /// Convenience: add `variable` at `lag`.
    ///
    /// # Errors
    ///
    /// Capacity overflow.
    pub fn add_lagged(
        &mut self,
        variable: VariableId,
        lag: Lag,
    ) -> Result<DenseNodeId, GraphError> {
        self.add_node(NodeRef::Lagged { variable, lag })
    }

    /// Insert directed edge with the same temporal rules as [`crate::TemporalDag`].
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, cycles, contemporaneous self-edges, or edges
    /// that point from the future into the past.
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        if let (
            NodeRef::Lagged { variable: v1, lag: l1 },
            NodeRef::Lagged { variable: v2, lag: l2 },
        ) = (self.nodes[from.as_usize()], self.nodes[to.as_usize()])
        {
            if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
                return Err(GraphError::ContemporaneousSelfEdge { variable: v1 });
            }
            crate::types::reject_future_to_past(&self.nodes, from, to)?;
        }
        if self.children[from.as_usize()].contains(&to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        let mut ws = core::mem::take(&mut self.insert_ws);
        let cycle = bfs_reaches(&self.children, to, from, &mut ws);
        self.insert_ws = ws;
        if cycle {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.children[from.as_usize()].push(to);
        self.parents[to.as_usize()].push(from);
        Ok(())
    }

    /// Insert bidirected edge `a ↔ b` (latent confounder).
    ///
    /// Bidirected edges have no time direction. Distinct lagged nodes may be
    /// joined, including across lags.
    ///
    /// # Errors
    ///
    /// Unknown nodes, self-loop, or duplicate.
    pub fn insert_bidirected(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if a == b {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalAdmg rejects bidirected self-loops",
            });
        }
        if self.bidirected[a.as_usize()].contains(&b) {
            return Err(GraphError::DuplicateEdge { from: a.raw(), to: b.raw() });
        }
        self.bidirected[a.as_usize()].push(b);
        self.bidirected[b.as_usize()].push(a);
        Ok(())
    }

    /// Children.
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.children.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Parents.
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.parents.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Bidirected neighbors.
    #[must_use]
    pub fn bidirected_neighbors(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.bidirected.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Whether any bidirected edge is present.
    #[must_use]
    pub fn has_bidirected(&self) -> bool {
        self.bidirected.iter().any(|neighbors| !neighbors.is_empty())
    }

    /// Directed and bidirected edges as marked edges.
    #[must_use]
    pub fn edges(&self) -> Vec<MarkedEdge> {
        let mut edges = Vec::new();
        for (i, kids) in self.children.iter().enumerate() {
            let from = DenseNodeId::from_index(i);
            for &to in kids {
                edges.push(MarkedEdge::directed(from, to));
            }
        }
        for (i, neighbors) in self.bidirected.iter().enumerate() {
            let a = DenseNodeId::from_index(i);
            for &b in neighbors {
                if a < b {
                    edges.push(MarkedEdge::bidirected(a, b));
                }
            }
        }
        edges
    }

    /// Map dense id to a serializable [`TemporalNodeKey`].
    #[must_use]
    pub fn temporal_key(&self, id: DenseNodeId) -> Option<TemporalNodeKey> {
        match self.nodes.get(id.as_usize())? {
            NodeRef::Lagged { variable, lag } => {
                let offset = -i32::try_from(lag.raw()).ok()?;
                Some(TemporalNodeKey { variable: *variable, offset })
            }
            _ => None,
        }
    }

    /// Project onto a static [`Admg`] when that projection is the same functional.
    ///
    /// Each variable may appear once. The static node keeps that variable id, and
    /// directed and bidirected edges keep their dense endpoints. General
    /// identification then addresses the same variables on the same mixed graph.
    ///
    /// # Errors
    ///
    /// A variable occupies more than one lag, or a node is not lagged. That graph
    /// does not have this projection.
    pub fn project_preserving_functional(&self) -> Result<Admg, GraphError> {
        let mut seen = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let NodeRef::Lagged { variable, .. } = node else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg projection requires lagged nodes",
                });
            };
            if !seen.insert(*variable) {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg projection preserves the functional only when each \
                              variable appears once",
                });
            }
        }
        let mut admg = Admg::empty();
        for node in &self.nodes {
            let NodeRef::Lagged { variable, .. } = node else { unreachable!("checked above") };
            admg.add_node(NodeRef::Static(*variable))?;
        }
        for (i, kids) in self.children.iter().enumerate() {
            let from = DenseNodeId::from_index(i);
            for &to in kids {
                admg.insert_directed(from, to)?;
            }
        }
        for (i, neighbors) in self.bidirected.iter().enumerate() {
            let a = DenseNodeId::from_index(i);
            for &b in neighbors {
                if a < b {
                    admg.insert_bidirected(a, b)?;
                }
            }
        }
        Ok(admg)
    }

    /// Unroll the template over `periods` consecutive periods into one static [`Admg`], numbering
    /// each period over the variable ids `0 ..= largest id in the template`.
    ///
    /// # Errors
    ///
    /// As [`Self::unroll_window_over`].
    pub fn unroll_window(&self, periods: u32) -> Result<UnrolledTemporalAdmg, GraphError> {
        let largest = self
            .nodes
            .iter()
            .filter_map(|node| match node {
                NodeRef::Lagged { variable, .. } => Some(variable.raw()),
                _ => None,
            })
            .max()
            .ok_or(GraphError::InvalidEndpoints {
                message: "TemporalAdmg unrolling needs at least one period and one variable",
            })?;
        self.unroll_window_over(periods, largest + 1)
    }

    /// Unroll the template over `periods` consecutive periods into one static [`Admg`], with
    /// `variable_count` coordinates per period: the variable ids `0 .. variable_count`, whether
    /// or not the template names them. A variable the template does not name is isolated in the
    /// window, and a window numbered over a schema's variable count has the layout of the
    /// schema's own unfolding (`period * variable_count + variable`).
    ///
    /// # Errors
    ///
    /// No periods, a non-lagged node, a template variable outside `0 .. variable_count`, a window
    /// too large to number, or an edge the window cannot hold (a same-period cycle).
    pub fn unroll_window_over(
        &self,
        periods: u32,
        variable_count: u32,
    ) -> Result<UnrolledTemporalAdmg, GraphError> {
        for node in &self.nodes {
            let NodeRef::Lagged { variable, .. } = node else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg unrolling requires lagged nodes",
                });
            };
            if variable.raw() >= variable_count {
                return Err(GraphError::InvalidEndpoints {
                    message: "a template variable lies outside the window's variable count",
                });
            }
        }
        let variables: Vec<VariableId> = (0..variable_count).map(VariableId::from_raw).collect();
        let count = variable_count;
        let total = periods.checked_mul(count).ok_or(GraphError::TooManyNodes)?;
        if periods == 0 || count == 0 {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalAdmg unrolling needs at least one period and one variable",
            });
        }
        let template = |id: DenseNodeId| -> Result<(usize, i64), GraphError> {
            match self.nodes.get(id.as_usize()) {
                Some(&NodeRef::Lagged { variable, lag }) => {
                    Ok((variable.as_usize(), i64::from(lag.raw())))
                }
                _ => Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg unrolling requires lagged nodes",
                }),
            }
        };
        let coordinate = |index: usize, period: i64| -> DenseNodeId {
            // `period` is in `0..periods` and `index < count` at every call.
            let flat =
                u32::try_from(period).unwrap_or(0) * count + u32::try_from(index).unwrap_or(0);
            DenseNodeId::from_raw(flat)
        };
        let window = i64::from(periods);
        let mut admg = Admg::with_variables(total);
        // In-window effects of each observed cause that precedes the window.
        let mut truncated: std::collections::BTreeMap<(usize, i64), Vec<DenseNodeId>> =
            std::collections::BTreeMap::new();
        let tolerate = |result: Result<(), GraphError>| match result {
            Err(GraphError::DuplicateEdge { .. }) => Ok(()),
            other => other,
        };
        for (parent_id, kids) in self.children.iter().enumerate() {
            let (pi, plag) = template(DenseNodeId::from_index(parent_id))?;
            for &kid in kids {
                let (ci, clag) = template(kid)?;
                let delay = plag - clag;
                for child_period in 0..window {
                    let parent_period = child_period - delay;
                    if parent_period >= window {
                        continue;
                    }
                    let child = coordinate(ci, child_period);
                    if parent_period < 0 {
                        truncated.entry((pi, parent_period)).or_default().push(child);
                    } else {
                        tolerate(admg.insert_directed(coordinate(pi, parent_period), child))?;
                    }
                }
            }
        }
        for (a_id, partners) in self.bidirected.iter().enumerate() {
            let (ai, alag) = template(DenseNodeId::from_index(a_id))?;
            for &partner in partners {
                if partner.as_usize() < a_id {
                    continue;
                }
                let (bi, blag) = template(partner)?;
                let delay = alag - blag;
                for b_period in 0..window {
                    let a_period = b_period - delay;
                    if a_period >= window {
                        continue;
                    }
                    let b = coordinate(bi, b_period);
                    if a_period < 0 {
                        truncated.entry((ai, a_period)).or_default().push(b);
                    } else {
                        let a = coordinate(ai, a_period);
                        if a != b {
                            tolerate(admg.insert_bidirected(a, b))?;
                        }
                    }
                }
            }
        }
        for effects in truncated.values() {
            for (i, &a) in effects.iter().enumerate() {
                for &b in &effects[i + 1..] {
                    if a != b {
                        tolerate(admg.insert_bidirected(a, b))?;
                    }
                }
            }
        }
        Ok(UnrolledTemporalAdmg { admg, variables, periods })
    }

    /// Lift a static [`Admg`] to contemporaneous lagged nodes.
    ///
    /// Variable ids and dense order are unchanged, so
    /// [`Self::project_preserving_functional`] returns an isomorphic graph.
    ///
    /// # Errors
    ///
    /// The static graph contains a non-static node, or edge insertion fails.
    pub fn from_static_contemporaneous(admg: &Admg) -> Result<Self, GraphError> {
        let mut graph = Self::empty();
        for node in admg.nodes() {
            let NodeRef::Static(variable) = node else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg lift accepts only static ADMG nodes",
                });
            };
            graph.add_lagged(*variable, Lag::CONTEMPORANEOUS)?;
        }
        for i in 0..admg.node_count() {
            let from = DenseNodeId::from_index(i);
            for &to in admg.children(from) {
                graph.insert_directed(from, to)?;
            }
            for &to in admg.bidirected_neighbors(from) {
                if from < to {
                    graph.insert_bidirected(from, to)?;
                }
            }
        }
        Ok(graph)
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contemporaneous_lift_projects_to_the_same_mixed_graph() {
        let mut admg = Admg::with_variables(3);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let temporal = TemporalAdmg::from_static_contemporaneous(&admg).unwrap();
        let projected = temporal.project_preserving_functional().unwrap();
        assert!(projected.has_bidirected());
        assert_eq!(projected.children(DenseNodeId::from_raw(0)), &[DenseNodeId::from_raw(1)]);
        assert_eq!(
            projected.bidirected_neighbors(DenseNodeId::from_raw(0)),
            &[DenseNodeId::from_raw(2)]
        );
    }

    #[test]
    fn repeated_variable_does_not_project() {
        let mut graph = TemporalAdmg::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(past, now).unwrap();
        assert!(graph.project_preserving_functional().is_err());
    }

    #[test]
    fn rejects_future_to_past_directed_edge() {
        let mut graph = TemporalAdmg::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        assert!(matches!(graph.insert_directed(now, past), Err(GraphError::FutureToPast { .. })));
    }
}
