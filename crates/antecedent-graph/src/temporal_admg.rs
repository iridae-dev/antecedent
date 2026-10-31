//! Temporal ADMG: lagged nodes, directed edges, and bidirected latent edges.
//!
//! A [`TemporalAdmg`] whose variables each appear once projects onto a static
//! [`Admg`] with the same variable ids and the same mixed-graph edges. General
//! identification of that projection is the same functional. A variable that
//! occupies more than one lag does not have that projection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{Lag, TemporalNodeKey, VariableId};

use crate::admg::Admg;
use crate::directed::DirectedCore;
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};

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
    inner: Arc<UnrolledTemporalAdmgInner>,
}
#[derive(Debug)]
struct UnrolledTemporalAdmgInner {
    admg: Admg,
    variables: Vec<VariableId>,
    position: std::collections::HashMap<VariableId, usize>,
    periods: u32,
    _lease: Option<antecedent_core::execution::AdmissionLease>,
}
impl UnrolledTemporalAdmg {
    /// Whether this window carries an admission lease.
    #[must_use]
    pub fn is_admitted(&self) -> bool {
        self.inner._lease.is_some()
    }
    /// The window graph. A caller explicitly cloning this borrowed graph owns
    /// and must separately account for that independent copy.
    #[must_use]
    pub fn admg(&self) -> &Admg {
        &self.inner.admg
    }
    /// Number of periods in the window.
    #[must_use]
    pub fn periods(&self) -> u32 {
        self.inner.periods
    }
    /// Variable IDs for one period.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.inner.variables
    }
    /// Map a variable and period to its window coordinate.
    #[must_use]
    pub fn coordinate(&self, variable: VariableId, period: u32) -> Option<VariableId> {
        if period >= self.inner.periods {
            return None;
        }
        let index = *self.inner.position.get(&variable)?;
        let count = u32::try_from(self.inner.variables.len()).ok()?;
        Some(VariableId::from_raw(period * count + u32::try_from(index).ok()?))
    }
    /// Invert a window coordinate.
    #[must_use]
    pub fn locate(&self, coordinate: VariableId) -> Option<(VariableId, u32)> {
        let count = u32::try_from(self.inner.variables.len()).ok()?;
        let period = coordinate.raw().checked_div(count)?;
        let index = usize::try_from(coordinate.raw() % count).ok()?;
        (period < self.inner.periods).then(|| (self.inner.variables[index], period))
    }
}

/// Acyclic directed mixed graph over lagged (`VariableId`, `Lag`) nodes.
#[derive(Clone, Debug)]
pub struct TemporalAdmg {
    /// Lagged nodes and the acyclic directed edges: the core a [`crate::TemporalDag`] and an
    /// [`Admg`] also wrap.
    directed: DirectedCore,
    bidirected: Vec<Vec<DenseNodeId>>,
}

impl TemporalAdmg {
    /// Empty temporal ADMG.
    #[must_use]
    pub fn empty() -> Self {
        Self { directed: DirectedCore::new(), bidirected: Vec::new() }
    }

    /// Node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.directed.node_count()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directed.is_empty()
    }

    /// Nodes in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        self.directed.nodes()
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
        let id = self.directed.push_labeled(node)?;
        self.bidirected.push(Vec::new());
        Ok(id)
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
        ) = (self.nodes()[from.as_usize()], self.nodes()[to.as_usize()])
        {
            if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
                return Err(GraphError::ContemporaneousSelfEdge { variable: v1 });
            }
            crate::types::reject_future_to_past(self.nodes(), from, to)?;
        }
        self.directed.try_insert(from, to)
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
        self.directed.children_of(id)
    }

    /// Parents.
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.parents_of(id)
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
        let mut edges: Vec<MarkedEdge> = self.directed.marked_edges().collect();
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
        match self.nodes().get(id.as_usize())? {
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
        let mut seen = HashSet::with_capacity(self.nodes().len());
        for node in self.nodes() {
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
        for node in self.nodes() {
            let NodeRef::Lagged { variable, .. } = node else { unreachable!("checked above") };
            admg.add_node(NodeRef::Static(*variable))?;
        }
        for (i, kids) in self.directed.child_adj().iter().enumerate() {
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
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        budget.charge_work(self.nodes().len() as u64)?;
        let mut largest = None;
        for node in self.nodes() {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            if let NodeRef::Lagged { variable, .. } = node {
                largest = Some(largest.map_or(variable.raw(), |old: u32| old.max(variable.raw())));
            }
        }
        let largest = largest.ok_or(GraphError::InvalidEndpoints {
            message: "TemporalAdmg unrolling needs at least one period and one variable",
        })?;
        self.unroll_window_over_within(
            periods,
            largest.checked_add(1).ok_or(GraphError::TooManyNodes)?,
            &mut budget,
            &ctx,
        )
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
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.unroll_window_over_within(periods, variable_count, &mut budget, &ctx)
    }

    /// Quote and admit a finite temporal graph before any unroll allocation.
    /// The returned handle and its clones share the retained admission.
    /// # Errors
    /// Invalid dimensions, logical/byte limits, cancellation or graph construction.
    pub fn unroll_window_over_within(
        &self,
        periods: u32,
        variable_count: u32,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<UnrolledTemporalAdmg, GraphError> {
        use antecedent_core::TemporalMaterializationError as T;
        let limit = |dimension| GraphError::TemporalMaterialization(T::Limit { dimension });
        let limits = budget.limits();
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        if periods == 0 || variable_count == 0 {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalAdmg unrolling needs at least one period and one variable",
            });
        }
        let nodes = u64::from(periods)
            .checked_mul(u64::from(variable_count))
            .ok_or_else(|| limit("graph nodes"))?;
        if nodes > u64::from(u32::MAX) || nodes > limits.max_graph_nodes {
            return Err(limit("graph nodes"));
        }
        let template_nodes = u64::try_from(self.nodes().len()).map_err(|_| limit("work"))?;
        budget.charge_work(template_nodes.checked_mul(4).ok_or_else(|| limit("work"))?)?;
        // Only edges with positive delay can originate before the window.
        // Count those boundary candidates separately; treating every replicated
        // edge as truncated would overcharge ordinary long windows quadratically.
        let lag = |index: usize| -> Result<i64, GraphError> {
            match self.nodes().get(index) {
                Some(NodeRef::Lagged { lag, .. }) => Ok(i64::from(lag.raw())),
                _ => Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg unrolling requires lagged nodes",
                }),
            }
        };
        let mut directed_templates = 0u64;
        let mut bidirected_templates = 0u64;
        let mut boundary = 0u64;
        let mut count_edge = |from: usize, to: usize| -> Result<(), GraphError> {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let delay = lag(from)? - lag(to)?;
            let truncated = if delay > 0 {
                u64::try_from(delay).unwrap_or(u64::MAX).min(u64::from(periods))
            } else {
                0
            };
            boundary = boundary.checked_add(truncated).ok_or_else(|| limit("graph edges"))?;
            Ok(())
        };
        for (from, children) in self.directed.child_adj().iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            for child in children {
                directed_templates =
                    directed_templates.checked_add(1).ok_or_else(|| limit("graph edges"))?;
                budget.charge_work(2)?;
                count_edge(from, child.as_usize())?;
            }
        }
        for (from, partners) in self.bidirected.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            for partner in partners {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                if partner.as_usize() < from {
                    continue;
                }
                bidirected_templates =
                    bidirected_templates.checked_add(1).ok_or_else(|| limit("graph edges"))?;
                budget.charge_work(4)?;
                count_edge(from, partner.as_usize())?;
            }
        }
        let directed = directed_templates
            .checked_mul(u64::from(periods))
            .ok_or_else(|| limit("graph edges"))?;
        let bidirected = bidirected_templates
            .checked_mul(u64::from(periods))
            .ok_or_else(|| limit("graph edges"))?;
        // Count the actual pre-window source/period groups before quoting
        // cliques. This temporary map is separately admitted; independent
        // sources do not induce quadratic phantom edge charges.
        budget.charge_work(boundary.checked_mul(512).ok_or_else(|| limit("work"))?)?;
        let group_bytes = boundary
            .checked_mul(512)
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| limit("bytes"))?;
        if group_bytes > limits.max_bytes {
            return Err(limit("bytes"));
        }
        let group_lease = ctx.acquire_admission(0, group_bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
            other => T::Admission(other),
        })?;
        let mut groups = std::collections::BTreeMap::<(usize, i64), u64>::new();
        let mut pairs = 0u64;
        let mut count_groups = |from: usize, to: usize| -> Result<(), GraphError> {
            let NodeRef::Lagged { variable, .. } = self
                .nodes()
                .get(from)
                .ok_or(GraphError::InvalidEndpoints { message: "missing temporal node" })?
            else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg unrolling requires lagged nodes",
                });
            };
            let delay = lag(from)? - lag(to)?;
            let end = delay.max(0).min(i64::from(periods));
            for period in 0..end {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                let n = groups.entry((variable.as_usize(), period - delay)).or_default();
                pairs = pairs.checked_add(*n).ok_or_else(|| limit("graph edges"))?;
                *n = n.checked_add(1).ok_or_else(|| limit("graph edges"))?;
                if pairs > limits.max_graph_edges {
                    return Err(limit("graph edges"));
                }
            }
            Ok(())
        };
        for (from, children) in self.directed.child_adj().iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            for child in children {
                count_groups(from, child.as_usize())?;
            }
        }
        for (from, partners) in self.bidirected.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            for partner in partners {
                if partner.as_usize() >= from {
                    count_groups(from, partner.as_usize())?;
                }
            }
        }
        drop(groups);
        drop(group_lease);
        let edges = directed
            .checked_add(bidirected)
            .and_then(|n| n.checked_add(pairs))
            .ok_or_else(|| limit("graph edges"))?;
        if edges > limits.max_graph_edges {
            return Err(limit("graph edges"));
        }
        // A directed insertion performs one BFS (at most V + directed edges);
        // a bidirected insertion scans a row of at most V distinct neighbors.
        let work = nodes
            .checked_add(directed)
            .and_then(|n| n.checked_add(bidirected))
            .and_then(|n| {
                directed
                    .checked_mul(nodes.checked_add(directed)?.checked_add(1)?)
                    .and_then(|d| n.checked_add(d))
            })
            .and_then(|n| {
                bidirected
                    .checked_add(pairs)?
                    .checked_mul(nodes.checked_add(1)?)
                    .and_then(|b| n.checked_add(b))
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
        let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
            other => T::Admission(other),
        })?;
        self.unroll_window_over_inner(periods, variable_count, Some(ctx), Some(lease))
    }

    fn unroll_window_over_inner(
        &self,
        periods: u32,
        variable_count: u32,
        ctx: Option<&antecedent_core::ExecutionContext>,
        lease: Option<antecedent_core::execution::AdmissionLease>,
    ) -> Result<UnrolledTemporalAdmg, GraphError> {
        let poll = || {
            if ctx.is_some_and(|c| c.cancellation.is_cancelled()) {
                Err(GraphError::TemporalMaterialization(
                    antecedent_core::TemporalMaterializationError::Cancelled,
                ))
            } else {
                Ok(())
            }
        };
        poll()?;
        for node in self.nodes() {
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
        let count = variable_count;
        let total = periods.checked_mul(count).ok_or(GraphError::TooManyNodes)?;
        if periods == 0 || count == 0 {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalAdmg unrolling needs at least one period and one variable",
            });
        }
        let mut variables = Vec::with_capacity(variable_count as usize);
        for id in 0..variable_count {
            poll()?;
            variables.push(VariableId::from_raw(id));
        }
        let template = |id: DenseNodeId| -> Result<(usize, i64), GraphError> {
            match self.nodes().get(id.as_usize()) {
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
        let mut admg = Admg::empty();
        for id in 0..total {
            poll()?;
            admg.add_node(NodeRef::Static(VariableId::from_raw(id)))?;
        }
        // In-window effects of each observed cause that precedes the window.
        let mut truncated: std::collections::BTreeMap<(usize, i64), Vec<DenseNodeId>> =
            std::collections::BTreeMap::new();
        let tolerate = |result: Result<(), GraphError>| match result {
            Err(GraphError::DuplicateEdge { .. }) => Ok(()),
            other => other,
        };
        for (parent_id, kids) in self.directed.child_adj().iter().enumerate() {
            poll()?;
            let (pi, plag) = template(DenseNodeId::from_index(parent_id))?;
            for &kid in kids {
                let (ci, clag) = template(kid)?;
                let delay = plag - clag;
                for child_period in 0..window {
                    poll()?;
                    let parent_period = child_period - delay;
                    if parent_period >= window {
                        continue;
                    }
                    let child = coordinate(ci, child_period);
                    if parent_period < 0 {
                        truncated.entry((pi, parent_period)).or_default().push(child);
                    } else {
                        let parent = coordinate(pi, parent_period);
                        tolerate(match ctx {
                            Some(ctx) => admg.insert_directed_temporal(parent, child, ctx),
                            None => admg.insert_directed(parent, child),
                        })?;
                    }
                }
            }
        }
        for (a_id, partners) in self.bidirected.iter().enumerate() {
            poll()?;
            let (ai, alag) = template(DenseNodeId::from_index(a_id))?;
            for &partner in partners {
                if partner.as_usize() < a_id {
                    continue;
                }
                let (bi, blag) = template(partner)?;
                let delay = alag - blag;
                for b_period in 0..window {
                    poll()?;
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
                    poll()?;
                    if a != b {
                        tolerate(admg.insert_bidirected(a, b))?;
                    }
                }
            }
        }
        let mut position = std::collections::HashMap::with_capacity(variables.len());
        for (index, variable) in variables.iter().enumerate() {
            poll()?;
            position.entry(*variable).or_insert(index);
        }
        poll()?;
        Ok(UnrolledTemporalAdmg {
            inner: Arc::new(UnrolledTemporalAdmgInner {
                admg,
                variables,
                position,
                periods,
                _lease: lease,
            }),
        })
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

#[cfg(test)]
mod admitted_unroll_tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{TemporalMaterializationError, TemporalMaterializationLimits};
    fn limits() -> TemporalMaterializationLimits {
        TemporalMaterializationLimits {
            max_offsets: 16,
            max_work: 1000,
            max_bytes: 8192,
            max_graph_nodes: 16,
            max_graph_edges: 64,
        }
    }
    #[test]
    fn unroll_checks_before_allocating_and_clones_retain_lease() {
        let mut graph = TemporalAdmg::empty();
        graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let pool = AdmissionPool::fixed(1, 8192);
        let ctx = antecedent_core::ExecutionContext::for_tests(8).with_admission(pool.clone());
        assert!(matches!(
            graph.unroll_window_over_within(
                u32::MAX,
                u32::MAX,
                &mut antecedent_core::TemporalBudget::new(limits()),
                &ctx
            ),
            Err(GraphError::TemporalMaterialization(TemporalMaterializationError::Limit { .. }))
        ));
        let window = graph
            .unroll_window_over_within(
                3,
                2,
                &mut antecedent_core::TemporalBudget::new(limits()),
                &ctx,
            )
            .unwrap();
        assert_eq!(window.admg().node_count(), 6);
        let remaining = pool.available().bytes();
        assert!(remaining < 8192);
        let clone = window.clone();
        drop(window);
        assert_eq!(pool.available().bytes(), remaining);
        drop(clone);
        assert_eq!(pool.available().bytes(), 8192);
        ctx.cancellation.cancel();
        assert!(matches!(
            graph.unroll_window_over_within(
                1,
                1,
                &mut antecedent_core::TemporalBudget::new(limits()),
                &ctx
            ),
            Err(GraphError::TemporalMaterialization(TemporalMaterializationError::Cancelled))
        ));
    }
    #[test]
    fn long_window_charges_boundary_candidates_not_all_replicated_edges() {
        let mut graph = TemporalAdmg::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let one = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let two = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(past, one).unwrap();
        graph.insert_directed(past, two).unwrap();
        let bounds = TemporalMaterializationLimits {
            max_offsets: 1000,
            max_work: 250_000,
            max_bytes: 512_000,
            max_graph_nodes: 300,
            max_graph_edges: 300,
        };
        let ctx = antecedent_core::ExecutionContext::for_tests(8)
            .with_admission(AdmissionPool::fixed(1, 512_000));
        let window = graph
            .unroll_window_over_within(
                100,
                3,
                &mut antecedent_core::TemporalBudget::new(bounds),
                &ctx,
            )
            .unwrap();
        assert_eq!(window.admg().node_count(), 300);
    }

    #[test]
    fn unrelated_boundary_sources_do_not_create_phantom_cliques() {
        let mut graph = TemporalAdmg::empty();
        for id in 0..100 {
            let source = graph.add_lagged(VariableId::from_raw(id), Lag::from_raw(1)).unwrap();
            let target =
                graph.add_lagged(VariableId::from_raw(100 + id), Lag::CONTEMPORANEOUS).unwrap();
            graph.insert_directed(source, target).unwrap();
        }
        let bounds = TemporalMaterializationLimits {
            max_offsets: 4,
            max_work: 300_000,
            max_bytes: 512_000,
            max_graph_nodes: 400,
            max_graph_edges: 200,
        };
        let ctx = antecedent_core::ExecutionContext::for_tests(8)
            .with_admission(AdmissionPool::fixed(1, 512_000));
        let window = graph
            .unroll_window_over_within(
                2,
                200,
                &mut antecedent_core::TemporalBudget::new(bounds),
                &ctx,
            )
            .unwrap();
        assert_eq!(window.admg().node_count(), 400);
    }
}
