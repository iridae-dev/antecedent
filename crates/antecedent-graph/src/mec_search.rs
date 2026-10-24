//! Orientation search shared by the static and temporal CPDAG completion samplers.
//!
//! A CPDAG's Markov equivalence class is the set of DAGs with the same skeleton and the same
//! unshielded colliders whose orientations extend the CPDAG's compelled edges. The search
//! orients one undirected edge at a time and prunes a partial assignment as soon as it closes
//! a directed cycle (or, on temporal graphs, a future→past edge — refused by
//! `orient_undirected`) or creates an unshielded collider the CPDAG does not carry.
//!
//! The search is bounded by a [`SearchBudget`] (operations, depth, memory, cancellation), not by
//! a count of undirected edges: a CPDAG of any size completes when pruning keeps the visited
//! states within the budget. Every partial orientation popped is one charged operation. A budget
//! stop is sticky and typed ([`MecSearch::search_stop`]); it is a resource outcome, never a claim
//! that the remainder holds no member.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, SearchBudget, SearchLimits, SearchReceipt, SearchStop};

use crate::cpdag::{Cpdag, TemporalCpdag};
use crate::dag::Dag;
use crate::error::GraphError;
use crate::temporal::TemporalDag;
use crate::types::{DenseNodeId, MarkedEdge};

/// The CPDAG operations the search needs.
pub(crate) trait MecCpdag: Clone {
    fn node_count(&self) -> usize;
    fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId>;
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool;
    fn edges(&self) -> Vec<MarkedEdge>;
    fn conflict_edge_count(&self) -> usize;
    fn orient_undirected(&mut self, from: DenseNodeId, to: DenseNodeId) -> Result<(), GraphError>;
}

/// The DAG operations the membership test needs.
pub(crate) trait MecDag {
    fn node_count(&self) -> usize;
    fn parents(&self, id: DenseNodeId) -> &[DenseNodeId];
    fn children(&self, id: DenseNodeId) -> &[DenseNodeId];
    fn edges(&self) -> Vec<MarkedEdge>;
}

macro_rules! forward_mec_cpdag {
    ($ty:ty) => {
        impl MecCpdag for $ty {
            fn node_count(&self) -> usize {
                <$ty>::node_count(self)
            }
            fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
                <$ty>::parents(self, id)
            }
            fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
                <$ty>::has_edge(self, a, b)
            }
            fn edges(&self) -> Vec<MarkedEdge> {
                <$ty>::edges(self)
            }
            fn conflict_edge_count(&self) -> usize {
                <$ty>::conflict_edge_count(self)
            }
            fn orient_undirected(
                &mut self,
                from: DenseNodeId,
                to: DenseNodeId,
            ) -> Result<(), GraphError> {
                <$ty>::orient_undirected(self, from, to)
            }
        }
    };
}

macro_rules! forward_mec_dag {
    ($ty:ty) => {
        impl MecDag for $ty {
            fn node_count(&self) -> usize {
                <$ty>::node_count(self)
            }
            fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
                <$ty>::parents(self, id)
            }
            fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
                <$ty>::children(self, id)
            }
            fn edges(&self) -> Vec<MarkedEdge> {
                <$ty>::edges(self).collect()
            }
        }
    };
}

forward_mec_cpdag!(Cpdag);
forward_mec_cpdag!(TemporalCpdag);
forward_mec_dag!(Dag);
forward_mec_dag!(TemporalDag);

fn sorted_pair(a: DenseNodeId, b: DenseNodeId) -> (u32, u32) {
    if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) }
}

/// Unshielded colliders `(low parent, centre, high parent)` of a CPDAG, sorted.
pub(crate) fn unshielded_colliders_cpdag<G: MecCpdag>(g: &G) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for center_idx in 0..g.node_count() {
        let Ok(center_raw) = u32::try_from(center_idx) else {
            break;
        };
        let center = DenseNodeId::from_raw(center_raw);
        let parents = g.parents(center);
        for left_i in 0..parents.len() {
            for right_i in (left_i + 1)..parents.len() {
                let (left, right) = (parents[left_i], parents[right_i]);
                if !g.has_edge(left, right) {
                    let (lo, hi) = sorted_pair(left, right);
                    out.push((lo, center.raw(), hi));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Unshielded colliders of a DAG, sorted.
fn unshielded_colliders_dag<D: MecDag>(g: &D) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for center_idx in 0..g.node_count() {
        let Ok(center_raw) = u32::try_from(center_idx) else {
            break;
        };
        let center = DenseNodeId::from_raw(center_raw);
        let parents = g.parents(center);
        for left_i in 0..parents.len() {
            for right_i in (left_i + 1)..parents.len() {
                let (left, right) = (parents[left_i], parents[right_i]);
                let adjacent =
                    g.children(left).contains(&right) || g.children(right).contains(&left);
                if !adjacent {
                    let (lo, hi) = sorted_pair(left, right);
                    out.push((lo, center.raw(), hi));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Whether `dag` is a Markov-equivalence member of `cpdag`: every compelled edge present,
/// every undirected edge oriented exactly one way, no edge outside the skeleton, and the
/// same unshielded colliders. Node identity (`same_nodes`) is the caller's check.
pub(crate) fn is_mec_member<G: MecCpdag, D: MecDag>(cpdag: &G, dag: &D, same_nodes: bool) -> bool {
    if !same_nodes {
        return false;
    }
    for e in cpdag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !dag.children(from).contains(&to) {
                return false;
            }
        } else if e.is_undirected() {
            let ab = dag.children(e.a).contains(&e.b);
            let ba = dag.children(e.b).contains(&e.a);
            if ab == ba {
                // missing or both — not a simple orientation
                return false;
            }
        } else if e.is_conflict() {
            return false;
        }
    }
    for e in dag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !cpdag.has_edge(from, to) {
                return false;
            }
        }
    }
    unshielded_colliders_cpdag(cpdag) == unshielded_colliders_dag(dag)
}

/// True when `center` already has an unshielded collider not present in `allowed` (sorted).
fn has_forbidden_collider_at<G: MecCpdag>(
    allowed: &[(u32, u32, u32)],
    g: &G,
    center: DenseNodeId,
) -> bool {
    let parents = g.parents(center);
    for left_i in 0..parents.len() {
        for right_i in (left_i + 1)..parents.len() {
            let (left, right) = (parents[left_i], parents[right_i]);
            if g.has_edge(left, right) {
                continue;
            }
            let (lo, hi) = sorted_pair(left, right);
            if allowed.binary_search(&(lo, center.raw(), hi)).is_err() {
                return true;
            }
        }
    }
    false
}

/// The execution context a search keeps for its budget checks.
///
/// It is read after construction only (its progress sink is notified, never mutated), so a
/// panic cannot leave it in a state a later observer sees broken. The progress sink is not
/// itself `RefUnwindSafe`, which would otherwise strip that auto trait from every sampler.
#[derive(Clone, Debug)]
struct ReadOnlyContext(Arc<ExecutionContext>);

impl UnwindSafe for ReadOnlyContext {}
impl RefUnwindSafe for ReadOnlyContext {}

/// Depth-first orientation search over the undirected edges of a CPDAG.
#[derive(Clone, Debug)]
pub(crate) struct MecSearch<G: MecCpdag> {
    pub(crate) base: G,
    /// Undirected edges `(a, b)` with `a.raw() <= b.raw()`.
    undirected: Vec<(DenseNodeId, DenseNodeId)>,
    /// Unshielded colliders already present in `base` (sorted).
    allowed_colliders: Vec<(u32, u32, u32)>,
    pub(crate) max_completions: usize,
    pub(crate) next_index: usize,
    /// Partial orientation and index of the next edge to orient.
    stack: Vec<(G, usize)>,
    limits: SearchLimits,
    ctx: ReadOnlyContext,
    memory_limit_bytes: u64,
    /// Estimated live bytes of one stacked partial orientation.
    state_bytes: u64,
    operations: usize,
    depth_reached: usize,
    stop: Option<SearchStop>,
    /// Partial-orientation states whose storage is already leased from the context's pool.
    leased_states: usize,
    /// The leases themselves, held while the search (and any clone of it) lives.
    leases: Arc<std::sync::Mutex<Vec<antecedent_core::execution::AdmissionLease>>>,
}

/// States leased at a time, so the pool is asked once per block of the stack's growth.
const LEASE_BLOCK_STATES: usize = 16;

impl<G: MecCpdag> MecSearch<G> {
    /// # Errors
    ///
    /// `conflict_message` when the CPDAG carries conflict (`x-x`) edges.
    pub(crate) fn new(
        cpdag: G,
        max_completions: usize,
        limits: SearchLimits,
        ctx: &ExecutionContext,
        conflict_message: &'static str,
    ) -> Result<Self, GraphError> {
        if cpdag.conflict_edge_count() > 0 {
            return Err(GraphError::InvalidEndpoints { message: conflict_message });
        }
        let mut undirected = Vec::new();
        let mut edge_count = 0usize;
        for e in cpdag.edges() {
            edge_count += 1;
            if e.is_undirected() {
                let (a, b) = if e.a.raw() <= e.b.raw() { (e.a, e.b) } else { (e.b, e.a) };
                undirected.push((a, b));
            }
        }
        undirected.sort_by_key(|(a, b)| (a.raw(), b.raw()));
        undirected.dedup();
        let allowed_colliders = unshielded_colliders_cpdag(&cpdag);
        let state_bytes = u64::try_from(
            cpdag
                .node_count()
                .saturating_add(edge_count.saturating_mul(2))
                .saturating_mul(std::mem::size_of::<MarkedEdge>()),
        )
        .unwrap_or(u64::MAX);
        let start = cpdag.clone();
        Ok(Self {
            base: cpdag,
            undirected,
            allowed_colliders,
            max_completions,
            next_index: 0,
            stack: vec![(start, 0)],
            limits,
            ctx: ReadOnlyContext(Arc::new(ctx.clone())),
            memory_limit_bytes: ctx
                .memory_limit_bytes()
                .map_or(antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES, |hard| {
                    hard.min(antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES)
                }),
            state_bytes,
            operations: 0,
            depth_reached: 0,
            stop: None,
            leased_states: 0,
            leases: Arc::default(),
        })
    }

    pub(crate) fn n_undirected(&self) -> usize {
        self.undirected.len()
    }

    /// Whether the stream ended before the search finished: the retention cap was reached
    /// with orientations still pending, or the search budget stopped it.
    pub(crate) fn hit_cap(&self) -> bool {
        (self.next_index >= self.max_completions && !self.stack.is_empty()) || self.stop.is_some()
    }

    /// The bound that stopped the search, if a budget (not the retention cap) did.
    pub(crate) fn search_stop(&self) -> Option<SearchStop> {
        self.stop
    }

    /// Receipt of a budget stop: the limits in force, the states visited and the orientations
    /// left pending. `None` while no budget has stopped the search.
    pub(crate) fn search_receipt(&self) -> Option<SearchReceipt> {
        let stop = self.stop?;
        Some(SearchReceipt {
            stop,
            operations_limit: self.limits.operations,
            depth_limit: self.limits.depth,
            memory_limit_bytes: Some(self.memory_limit_bytes),
            operations_consumed: (self.operations > 0).then_some(self.operations),
            depth_reached: (self.operations > 0).then_some(self.depth_reached),
            explored: vec![format!(
                "{} orientation states visited, {} completion(s) yielded",
                self.operations, self.next_index
            )],
            unevaluated: vec![format!(
                "{} partial orientation(s) of {} undirected edge(s) left unexpanded",
                self.stack.len(),
                self.undirected.len()
            )],
        })
    }

    /// The next fully oriented CPDAG (every undirected edge oriented, no forbidden collider).
    pub(crate) fn next_oriented(&mut self) -> Option<G> {
        if self.next_index >= self.max_completions || self.stop.is_some() || self.stack.is_empty() {
            return None;
        }
        let remaining = SearchLimits {
            operations: self.limits.operations.saturating_sub(self.operations),
            depth: self.limits.depth,
        };
        let ctx = Arc::clone(&self.ctx.0);
        let mut budget = match SearchBudget::with_memory(remaining, self.memory_limit_bytes, &ctx) {
            Ok(budget) => budget,
            Err(receipt) => {
                self.stop = Some(receipt.stop);
                return None;
            }
        };
        let found = self.expand(&mut budget);
        self.operations = self.operations.saturating_add(budget.operations());
        self.depth_reached = self.depth_reached.max(budget.depth_reached());
        found
    }

    fn expand(&mut self, budget: &mut SearchBudget<'_>) -> Option<G> {
        while let Some((g, edge_i)) = self.stack.pop() {
            let live = u64::try_from(self.stack.len().saturating_add(1)).unwrap_or(u64::MAX);
            if let Err(stop) = budget.charge(edge_i, live.saturating_mul(self.state_bytes)) {
                self.stop = Some(stop);
                self.stack.push((g, edge_i));
                return None;
            }
            if edge_i == self.undirected.len() {
                return Some(g);
            }
            // The two states this expansion may push are funded by the pool before they exist.
            if let Err(stop) = self.lease_states(budget, self.stack.len().saturating_add(3)) {
                self.stop = Some(stop);
                self.stack.push((g, edge_i));
                return None;
            }
            let (a, b) = self.undirected[edge_i];
            // Push reverse first so a→b is explored first (LIFO).
            for (from, to) in [(b, a), (a, b)] {
                let mut next_g = g.clone();
                if next_g.orient_undirected(from, to).is_err() {
                    continue; // cycle (or temporal future→past)
                }
                // A new arrow into `to` can only create colliders centred at `to`.
                if has_forbidden_collider_at(&self.allowed_colliders, &next_g, to) {
                    continue;
                }
                self.stack.push((next_g, edge_i + 1));
            }
        }
        None
    }

    /// Lease storage for `needed` stacked states in blocks, before they are allocated.
    fn lease_states(&mut self, budget: &SearchBudget<'_>, needed: usize) -> Result<(), SearchStop> {
        if needed <= self.leased_states {
            return Ok(());
        }
        let target = needed.div_ceil(LEASE_BLOCK_STATES).saturating_mul(LEASE_BLOCK_STATES);
        let grow = target - self.leased_states;
        let bytes = u64::try_from(grow).unwrap_or(u64::MAX).saturating_mul(self.state_bytes);
        let lease = budget.lease_storage(bytes)?;
        self.leases.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(lease);
        self.leased_states = target;
        Ok(())
    }

    /// Record one yielded completion and return its stream index.
    pub(crate) fn record_yield(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }
}
