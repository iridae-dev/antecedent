//! One finite unfolding for a temporal query window.
//!
//! History grows by walking template ancestry. The static DAG is materialised
//! once, at the history that closes the truncated boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, TemporalIndexer, TemporalNodeKey};
use antecedent_graph::{AdmittedUnfoldedTemporalGraph, TemporalDag};
use std::collections::BTreeSet;

use crate::error::IdentificationError;

/// The unfolded window and the dense ids of the seeds it was grown for.
pub(crate) struct GrownWindow {
    /// History depth of [`Self::unfolded`].
    pub history: u32,
    /// Static DAG for that history. Dense ids match the unfolding indexer.
    pub unfolded: AdmittedUnfoldedTemporalGraph,
    /// Dense id of each seed, in seed order.
    pub dense: Vec<u32>,
    pub _lease: antecedent_core::execution::AdmissionLease,
}

/// Grow history until ancestral closure of `seeds` no longer sits on the
/// truncated boundary, then unfold that window once.
///
/// `history_cap` stops growth. Reaching it while the boundary is still open
/// is [`IdentificationError::NotCertified`], with the same refusal
/// `identify_temporal` used when each candidate history was unfolded from scratch.
///
/// # Errors
///
/// Indexer or unfold failures, a seed outside the window, or the history-cap refusal.
#[must_use = "the grown window is the only unfolding of this query"]
pub(crate) fn grow_window(
    template: &TemporalDag,
    seeds: &[TemporalNodeKey],
    variable_count: u32,
    horizon: u32,
    base_history: u32,
    history_cap: u32,
    chain_cap: u32,
    ctx: &ExecutionContext,
    budget: &mut antecedent_core::TemporalBudget,
) -> Result<GrownWindow, IdentificationError> {
    use crate::error::IdentificationBudget;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    let node_count = template.nodes().len() as u64;
    budget
        .charge_work(node_count)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let mut edge_count = 0u64;
    for i in 0..template.nodes().len() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        edge_count = edge_count
            .checked_add(
                template.children(antecedent_graph::DenseNodeId::from_index(i)).len() as u64
            )
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    }
    budget
        .charge_work(edge_count)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    if u64::from(variable_count) > budget.limits().max_graph_nodes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let edge_bytes = u64::from(variable_count)
        .checked_mul(128)
        .and_then(|n| edge_count.checked_mul(256).and_then(|e| n.checked_add(e)))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
    if edge_bytes > budget.limits().max_bytes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let _edge_lease = ctx.acquire_admission(0, edge_bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
        _ => IdentificationError::budget(IdentificationBudget::Memory),
    })?;
    let edges = StationaryEdges::from_template(template, variable_count, ctx)?;
    let mut depth = base_history;
    let history = loop {
        check_cancelled(Some(ctx))?;
        let indexer = TemporalIndexer::new(variable_count, depth, horizon)
            .map_err(|e| IdentificationError::construction(e))?;
        budget
            .charge_work(seeds.len() as u64)
            .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
        for seed in seeds {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            indexer
                .dense_id(*seed)
                .map_err(|_| IdentificationError::UnknownVariable { id: seed.variable })?;
        }
        let slices = u64::from(depth)
            .checked_add(u64::from(horizon))
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
        let nodes = u64::from(variable_count)
            .checked_mul(slices)
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
        let visits = edge_count
            .checked_mul(slices)
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
        if nodes > budget.limits().max_graph_nodes || visits > budget.limits().max_graph_edges {
            return Err(IdentificationError::budget(IdentificationBudget::Memory));
        }
        let work = nodes
            .checked_add(visits)
            .and_then(|n| n.checked_mul(128))
            .and_then(|n| n.checked_add(seeds.len() as u64))
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
        budget
            .charge_work(work)
            .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
        let bytes = nodes
            .checked_mul(256)
            .and_then(|n| visits.checked_mul(64).and_then(|v| n.checked_add(v)))
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
        if bytes > budget.limits().max_bytes {
            return Err(IdentificationError::budget(IdentificationBudget::Memory));
        }
        let _walk_lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
            _ => IdentificationError::budget(IdentificationBudget::Memory),
        })?;
        if !edges.ancestry_touches_boundary(seeds, depth, horizon, Some(ctx))? {
            break depth;
        }
        if depth >= history_cap {
            return Err(history_cap_refusal(history_cap, chain_cap));
        }
        depth += 1;
    };

    let indexer = TemporalIndexer::new(variable_count, history, horizon)
        .map_err(|e| IdentificationError::construction(e))?;
    let unfolded = template.unfold_within(indexer, budget, ctx).map_err(|error| match error {
        antecedent_graph::GraphError::TemporalMaterialization(error) => {
            crate::error::temporal_materialization_error(error)
        }
        other => IdentificationError::construction(other),
    })?;
    budget
        .charge_work(seeds.len() as u64)
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let bytes = (seeds.len() as u64)
        .checked_mul(16)
        .and_then(|n| n.checked_add(256))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
    if bytes > budget.limits().max_bytes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
        _ => IdentificationError::budget(IdentificationBudget::Memory),
    })?;
    let mut dense = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        dense.push(
            unfolded
                .indexer
                .dense_id(*seed)
                .map_err(|_| IdentificationError::UnknownVariable { id: seed.variable })?,
        );
    }
    // The growth loop already checked that this depth closes the boundary and,
    // if it advanced, that the immediately preceding depth did not. Replaying
    // those checks after materialization would allocate a second graph under a
    // different context and add no new fixed-point evidence.
    Ok(GrownWindow { history, unfolded, dense, _lease: lease })
}

fn check_cancelled(context: Option<&ExecutionContext>) -> Result<(), IdentificationError> {
    if context.is_some_and(|ctx| ctx.cancellation.is_cancelled())
        || antecedent_core::numerical_admission::is_cancelled()
    {
        return Err(IdentificationError::Cancelled);
    }
    Ok(())
}

/// Whether `from` reaches `to` in the stationary unfolding, including `from == to`.
///
/// `history` and `horizon` are the same window [`TemporalIndexer::new`] would build.
/// Callers still resolve dense ids themselves so an offset outside that window
/// keeps the indexer's error.
pub(crate) fn reaches_in_window(
    template: &TemporalDag,
    from: TemporalNodeKey,
    to: TemporalNodeKey,
    variable_count: u32,
    history: u32,
    horizon: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<bool, IdentificationError> {
    use crate::error::IdentificationBudget;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    if from == to {
        return Ok(true);
    }
    let variable_count = u64::from(variable_count);
    if variable_count > budget.limits().max_graph_nodes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let template_nodes = template.nodes().len() as u64;
    budget.charge_work(template_nodes).map_err(crate::error::temporal_materialization_error)?;
    let mut edge_count = 0u64;
    for i in 0..template.nodes().len() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        edge_count = edge_count
            .checked_add(
                template.children(antecedent_graph::DenseNodeId::from_index(i)).len() as u64
            )
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    }
    let min_off = -i64::from(history);
    let max_off = i64::from(horizon) - 1;
    // TemporalDag edges cannot move backwards in time, so only periods at or
    // after the starting point can be reached. Include an out-of-window seed.
    let start = i64::from(from.offset).max(min_off);
    let slices = u64::try_from((max_off - start + 1).max(0))
        .map_err(|_| IdentificationError::budget(IdentificationBudget::Steps))?;
    let states = variable_count
        .checked_mul(slices)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    let extra_seed_edges = if i64::from(from.offset) < min_off || i64::from(from.offset) > max_off {
        edge_count
    } else {
        0
    };
    let visits = edge_count
        .checked_mul(slices)
        .and_then(|n| n.checked_add(extra_seed_edges))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    if visits > budget.limits().max_graph_edges {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let work = states
        .checked_add(visits)
        .and_then(|n| n.checked_mul(128))
        .and_then(|n| n.checked_add(edge_count))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    budget.charge_work(work).map_err(crate::error::temporal_materialization_error)?;
    let bytes = variable_count
        .checked_mul(128)
        .and_then(|n| edge_count.checked_mul(256).and_then(|e| n.checked_add(e)))
        .and_then(|n| states.checked_mul(256).and_then(|v| n.checked_add(v)))
        .and_then(|n| visits.checked_mul(64).and_then(|v| n.checked_add(v)))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
    if bytes > budget.limits().max_bytes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    let _lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
        _ => IdentificationError::budget(IdentificationBudget::Memory),
    })?;
    let edges = StationaryEdges::from_template(template, variable_count as u32, ctx)?;
    let target = (to.variable.raw(), i64::from(to.offset));
    let mut seen = BTreeSet::new();
    let mut stack = vec![(from.variable.raw(), i64::from(from.offset))];
    while let Some((variable, offset)) = stack.pop() {
        check_cancelled(Some(ctx))?;
        if !seen.insert((variable, offset)) {
            continue;
        }
        if (variable, offset) == target {
            return Ok(true);
        }
        stack.extend(edges.children(variable, offset, min_off, max_off, Some(ctx))?);
    }
    Ok(false)
}

/// The refusal when window growth stops at `history_cap` with ancestry still
/// crossing the truncated boundary.
///
/// A confounder chain that repeats no variable reaches at most `chain_cap`
/// slices back (`variable_count * template_max_lag + |min_offset|`). Reaching
/// that depth therefore means an ancestor lies on a lagged cycle, such as an
/// autoregressive edge `x(t-1) -> x(t)`, whose ancestry never ends: no finite
/// window certifies it, and a larger `max_history_lag` cannot help. Only a
/// caller-set `max_history_lag` below `chain_cap` can stop growth early, and
/// then raising it is the remedy.
fn history_cap_refusal(history_cap: u32, chain_cap: u32) -> IdentificationError {
    let message = if history_cap < chain_cap {
        "temporal unfolding reached the query's max_history_lag while confounder ancestry \
         still crossed the truncated boundary; cannot certify backdoor identification over \
         the finite window (raise max_history_lag)"
    } else {
        "temporal unfolding reached its history cap while confounder ancestry still crossed \
         the truncated boundary: an ancestor of the treatment or outcome lies on a lagged \
         cycle (for example an autoregressive edge x(t-1) -> x(t)), so no finite window \
         certifies backdoor identification; review whether that lagged edge belongs in the \
         graph"
    };
    IdentificationError::NotCertified { message }
}

/// Template edges replicated by the same shift rule as [`TemporalDag::unfold`].
struct StationaryEdges {
    /// `incoming[child_var] = (parent_var, parent_template_offset, child_template_offset)`.
    incoming: Vec<Vec<(u32, i32, i32)>>,
    /// `outgoing[parent_var] = (child_var, parent_template_offset, child_template_offset)`.
    outgoing: Vec<Vec<(u32, i32, i32)>>,
}

impl StationaryEdges {
    fn from_template(
        template: &TemporalDag,
        variable_count: u32,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let n = variable_count as usize;
        let mut incoming = vec![Vec::new(); n];
        let mut outgoing = vec![Vec::new(); n];
        for edge in template.edges() {
            check_cancelled(Some(ctx))?;
            let Some((from, to)) = edge.parent_child() else {
                continue;
            };
            let (Some(from_key), Some(to_key)) =
                (template.temporal_key(from), template.temporal_key(to))
            else {
                continue;
            };
            let parent = from_key.variable.raw() as usize;
            let child = to_key.variable.raw() as usize;
            if parent >= n || child >= n {
                continue;
            }
            incoming[child].push((from_key.variable.raw(), from_key.offset, to_key.offset));
            outgoing[parent].push((to_key.variable.raw(), from_key.offset, to_key.offset));
        }
        Ok(Self { incoming, outgoing })
    }

    fn ancestry_touches_boundary(
        &self,
        seeds: &[TemporalNodeKey],
        history: u32,
        horizon: u32,
        context: Option<&ExecutionContext>,
    ) -> Result<bool, IdentificationError> {
        let min_off = -i64::from(history);
        let max_off = i64::from(horizon) - 1;
        let mut seen = BTreeSet::new();
        let mut stack: Vec<_> =
            seeds.iter().map(|seed| (seed.variable.raw(), i64::from(seed.offset))).collect();
        while let Some((var, offset)) = stack.pop() {
            check_cancelled(context)?;
            if !seen.insert((var, offset)) {
                continue;
            }
            if self.has_cut_parent(var, offset, min_off, context)? {
                return Ok(true);
            }
            stack.extend(self.parents(var, offset, min_off, max_off, context)?);
        }
        Ok(false)
    }

    /// Whether the node `(var, offset)` has a template parent earlier than `min_off`, which
    /// the window therefore omits. Any such node, at any slice, leaves the window's ancestry
    /// open: with template lag `k`, nodes within `k` slices of the boundary lose parents,
    /// not just the nodes on the boundary slice.
    fn has_cut_parent(
        &self,
        var: u32,
        offset: i64,
        min_off: i64,
        context: Option<&ExecutionContext>,
    ) -> Result<bool, IdentificationError> {
        let Some(edges) = self.incoming.get(var as usize) else {
            return Ok(false);
        };
        for &(parent_var, parent_off, child_off) in edges {
            check_cancelled(context)?;
            let Some(delta) = offset.checked_sub(i64::from(child_off)) else {
                continue;
            };
            let parent = i64::from(parent_off) + delta;
            if parent < min_off && !(parent_var == var && parent == offset) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn parents(
        &self,
        var: u32,
        offset: i64,
        min_off: i64,
        max_off: i64,
        context: Option<&ExecutionContext>,
    ) -> Result<Vec<(u32, i64)>, IdentificationError> {
        Self::shift(&self.incoming, var, offset, min_off, max_off, true, context)
    }

    fn children(
        &self,
        var: u32,
        offset: i64,
        min_off: i64,
        max_off: i64,
        context: Option<&ExecutionContext>,
    ) -> Result<Vec<(u32, i64)>, IdentificationError> {
        Self::shift(&self.outgoing, var, offset, min_off, max_off, false, context)
    }

    /// Replicate template edges onto one concrete endpoint.
    ///
    /// `incoming` anchors the shift on the child template offset and emits parents.
    /// `outgoing` anchors it on the parent template offset and emits children.
    /// Coordinates are computed in i64; out-of-window or self edges are dropped,
    /// matching the checked temporal DAG unfolding.
    fn shift(
        table: &[Vec<(u32, i32, i32)>],
        var: u32,
        offset: i64,
        min_off: i64,
        max_off: i64,
        incoming: bool,
        context: Option<&ExecutionContext>,
    ) -> Result<Vec<(u32, i64)>, IdentificationError> {
        let Some(edges) = table.get(var as usize) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for &(other_var, parent_off, child_off) in edges {
            check_cancelled(context)?;
            let anchor = if incoming { child_off } else { parent_off };
            let Some(delta) = offset.checked_sub(i64::from(anchor)) else {
                continue;
            };
            let parent_concrete = i64::from(parent_off) + delta;
            let child_concrete = i64::from(child_off) + delta;
            let (anchored, other) = if incoming {
                (child_concrete, parent_concrete)
            } else {
                (parent_concrete, child_concrete)
            };
            if anchored != offset || other < min_off || other > max_off {
                continue;
            }
            if other_var == var && other == offset {
                continue;
            }
            out.push((other_var, other));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use antecedent_core::{Lag, VariableId};

    #[test]
    fn symbolic_growth_agrees_with_unfolded_ancestor_oracle() {
        use antecedent_graph::{BitSet, DenseNodeId, GraphWorkspace};
        let mut graph = TemporalDag::empty();
        let ancestor = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(2)).unwrap();
        let target = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(ancestor, target).unwrap();
        let seed = TemporalNodeKey { variable: VariableId::from_raw(0), offset: 0 };
        let ctx = ExecutionContext::for_tests(8);
        let edges = StationaryEdges::from_template(&graph, 2, &ctx).unwrap();
        for history in 0..=3 {
            let indexer = TemporalIndexer::new(2, history, 1).unwrap();
            let dense = indexer.dense_id(seed).unwrap();
            let unfolded = graph.unfold(indexer).unwrap();
            let mut ancestors = BitSet::with_len(unfolded.dag.node_count());
            let mut workspace = GraphWorkspace::default();
            unfolded.dag.ancestors_of(
                &[DenseNodeId::from_raw(dense)],
                &mut ancestors,
                &mut workspace,
            );
            let mut static_boundary = false;
            for i in 0..unfolded.dag.node_count() {
                let id = DenseNodeId::from_index(i);
                if ancestors.contains(id) {
                    let key = unfolded.indexer.key_of(id.raw()).unwrap();
                    static_boundary |= edges
                        .has_cut_parent(
                            key.variable.raw(),
                            i64::from(key.offset),
                            -i64::from(history),
                            Some(&ctx),
                        )
                        .unwrap();
                }
            }
            assert_eq!(
                edges.ancestry_touches_boundary(&[seed], history, 1, Some(&ctx)).unwrap(),
                static_boundary
            );
        }
    }

    #[test]
    fn in_window_reachability_fits_the_single_edge_limit() {
        use antecedent_core::{Lag, TemporalBudget, VariableId};
        let mut graph = TemporalDag::empty();
        let x = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x, y).unwrap();
        let ctx = ExecutionContext::for_tests(0);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_graph_edges = 1;
        let mut budget = TemporalBudget::new(limits);
        assert!(
            reaches_in_window(
                &graph,
                TemporalNodeKey::contemporaneous(VariableId::from_raw(0)),
                TemporalNodeKey::contemporaneous(VariableId::from_raw(1)),
                2,
                0,
                1,
                &mut budget,
                &ctx,
            )
            .unwrap()
        );
    }

    #[test]
    fn cyclic_history_respects_one_cumulative_work_budget() {
        let mut graph = TemporalDag::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(past, now).unwrap();
        let seed = TemporalNodeKey { variable: VariableId::from_raw(0), offset: 0 };
        let ctx = ExecutionContext::for_tests(5);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_work = 700;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        assert!(matches!(
            grow_window(&graph, &[seed], 1, 1, 0, u32::MAX, 1, &ctx, &mut budget),
            Err(IdentificationError::Budget { budget: crate::error::IdentificationBudget::Steps })
        ));
    }

    // A caller can deliberately authorize more than the default graph ceiling.
    // The former debug-only replay used the smaller default and panicked here.
    #[cfg(debug_assertions)]
    #[test]
    fn growth_above_default_graph_limit_does_not_replay_with_defaults() {
        let mut graph = TemporalDag::empty();
        let ancestor = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(2)).unwrap();
        let target = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(ancestor, target).unwrap();
        let seed = TemporalNodeKey { variable: VariableId::from_raw(0), offset: -50_000 };
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 128 * 1024 * 1024);
        let ctx = ExecutionContext::for_tests(10).with_admission(pool.clone());
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_graph_nodes = 110_000;
        limits.max_work = 20_000_000_000;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        let window =
            grow_window(&graph, &[seed], 2, 1, 50_000, 50_010, 50_010, &ctx, &mut budget).unwrap();
        assert_eq!(window.history, 50_002);
        assert!(window.unfolded.dag.node_count() > 100_000);
        drop(window);
        assert_eq!(pool.available().bytes(), 128 * 1024 * 1024);
    }

    #[test]
    fn growth_uses_caller_budget_without_debug_graph_replay() {
        let mut graph = TemporalDag::empty();
        let ancestor = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(2)).unwrap();
        let seed_node = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(ancestor, seed_node).unwrap();
        let seed = TemporalNodeKey { variable: VariableId::from_raw(0), offset: 0 };
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 100_000);
        let ctx = ExecutionContext::for_tests(6).with_admission(pool.clone());
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_work = 2_500;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        let window = grow_window(&graph, &[seed], 2, 1, 0, 10, 10, &ctx, &mut budget).unwrap();
        assert_eq!(window.history, 2);
        assert!(pool.available().bytes() < 100_000);
        drop(window);
        assert_eq!(pool.available().bytes(), 100_000);
        let mut small = limits;
        small.max_work = 700;
        let mut budget = antecedent_core::TemporalBudget::new(small);
        assert!(matches!(
            grow_window(&graph, &[seed], 2, 1, 0, 10, 10, &ctx, &mut budget),
            Err(IdentificationError::Budget { .. })
        ));
        assert_eq!(pool.available().bytes(), 100_000);
    }

    #[test]
    fn cancelled_context_stops_growth_before_unfolding() {
        let mut graph = TemporalDag::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(past, now).unwrap();
        let seed = TemporalNodeKey { variable: VariableId::from_raw(0), offset: 0 };
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        assert!(matches!(
            grow_window(
                &graph,
                &[seed],
                1,
                1,
                1,
                100,
                1,
                &ctx,
                &mut antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
            ),
            Err(IdentificationError::Cancelled)
        ));
        let fresh = ExecutionContext::for_tests(2);
        let mut fresh_budget =
            antecedent_core::TemporalBudget::new(fresh.temporal_materialization_limits());
        assert!(matches!(
            antecedent_core::numerical_admission::with_context(&ctx, || grow_window(
                &graph,
                &[seed],
                1,
                1,
                1,
                100,
                1,
                &fresh,
                &mut fresh_budget
            )),
            Err(IdentificationError::Cancelled)
        ));
    }
}
