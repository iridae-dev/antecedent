//! Orientation rules and local delta queues (DESIGN.md §13.6).
//!
//! Rules enqueue only neighbors of changed edges — never a full-graph edge scan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names, clippy::redundant_closure_for_method_calls)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use causal_graph::{Cpdag, DenseNodeId, GraphError, MarkedEdge, TemporalCpdag};

use crate::error::DiscoveryError;

/// Shared CPDAG operations for Meek / collider rules on static or temporal graphs.
pub trait CpdagOps {
    /// Node count.
    fn node_count(&self) -> usize;
    /// Directed parents.
    fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId>;
    /// Directed children.
    fn children(&self, id: DenseNodeId) -> Vec<DenseNodeId>;
    /// Undirected neighbors.
    fn undirected_neighbors(&self, id: DenseNodeId) -> Vec<DenseNodeId>;
    /// Whether any edge exists.
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool;
    /// Marked edge if present.
    fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge>;
    /// Orient undirected `from → to`.
    ///
    /// # Errors
    ///
    /// Graph mutation failures.
    fn orient_undirected(&mut self, from: DenseNodeId, to: DenseNodeId) -> Result<(), GraphError>;
    /// Mark conflict on `{a,b}`.
    ///
    /// # Errors
    ///
    /// Graph mutation failures.
    fn mark_conflict(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError>;
}

impl CpdagOps for TemporalCpdag {
    fn node_count(&self) -> usize {
        Self::node_count(self)
    }
    fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        Self::parents(self, id)
    }
    fn children(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        Self::children(self, id)
    }
    fn undirected_neighbors(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        Self::undirected_neighbors(self, id)
    }
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        Self::has_edge(self, a, b)
    }
    fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge> {
        Self::edge_between(self, a, b)
    }
    fn orient_undirected(&mut self, from: DenseNodeId, to: DenseNodeId) -> Result<(), GraphError> {
        Self::orient_undirected(self, from, to)
    }
    fn mark_conflict(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        Self::mark_conflict(self, a, b)
    }
}

impl CpdagOps for Cpdag {
    fn node_count(&self) -> usize {
        Self::node_count(self)
    }
    fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        Self::parents(self, id)
    }
    fn children(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        Self::children(self, id)
    }
    fn undirected_neighbors(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        Self::undirected_neighbors(self, id)
    }
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        Self::has_edge(self, a, b)
    }
    fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge> {
        Self::edge_between(self, a, b)
    }
    fn orient_undirected(&mut self, from: DenseNodeId, to: DenseNodeId) -> Result<(), GraphError> {
        Self::orient_undirected(self, from, to)
    }
    fn mark_conflict(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        Self::mark_conflict(self, a, b)
    }
}

/// Orientation-layer errors.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum OrientationError {
    /// Graph mutation failed.
    #[error(transparent)]
    Graph(#[from] causal_graph::GraphError),
    /// Rule precondition failed.
    #[error("orientation precondition: {message}")]
    Precondition {
        /// Detail.
        message: &'static str,
    },
    /// Path / discriminating-path search hit `max_paths` or `max_len` mid-decision.
    ///
    /// Silent truncation can change orientations; callers must widen budgets or fail closed.
    #[error(
        "orientation path search budget exhausted in {rule} (max_paths={max_paths}, max_len={max_len})"
    )]
    SearchBudgetExhausted {
        /// Rule id that hit the budget.
        rule: &'static str,
        /// Path-count budget.
        max_paths: usize,
        /// Path-length budget.
        max_len: usize,
    },
    /// Non-graph orientation failure message.
    #[error("{0}")]
    Message(String),
}

impl OrientationError {
    /// Ad-hoc message helper.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl From<OrientationError> for DiscoveryError {
    fn from(e: OrientationError) -> Self {
        match e {
            OrientationError::Graph(g) => DiscoveryError::Graph(g),
            OrientationError::Precondition { message } => DiscoveryError::Unsupported { message },
            OrientationError::SearchBudgetExhausted { rule, max_paths, max_len } => {
                DiscoveryError::Orientation(format!(
                    "path search budget exhausted in {rule} (max_paths={max_paths}, max_len={max_len})"
                ))
            }
            OrientationError::Message(m) => DiscoveryError::Orientation(m),
        }
    }
}

/// Local work queue of nodes whose incident undirected edges may need orientation.
#[derive(Clone, Debug, Default)]
pub struct OrientationQueue {
    inner: VecDeque<DenseNodeId>,
    pending: HashSet<u32>,
}

impl OrientationQueue {
    /// Empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a node if not already pending.
    pub fn push(&mut self, id: DenseNodeId) {
        if self.pending.insert(id.raw()) {
            self.inner.push_back(id);
        }
    }

    /// Pop next node.
    pub fn pop(&mut self) -> Option<DenseNodeId> {
        let id = self.inner.pop_front()?;
        self.pending.remove(&id.raw());
        Some(id)
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Pending count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Separating sets and conflict bookkeeping for orientation.
#[derive(Clone, Debug, Default)]
pub struct OrientationState {
    /// Sepset for unordered pair `(min(a,b), max(a,b))`.
    pub sepsets: HashMap<(u32, u32), Arc<[DenseNodeId]>>,
    /// Keys whose stored sepset is weakly minimal (LPCMCI Def. 1).
    pub weakly_minimal: HashSet<(u32, u32)>,
    /// Number of orientation conflicts recorded this run.
    pub conflicts: u32,
    /// Unordered edge keys `(min,max)` that participated in a conflict.
    ///
    /// Matching edges are also marked Conflict–Conflict (`x-x`) in the graph when present.
    pub conflict_edges: HashSet<(u32, u32)>,
}

impl OrientationState {
    /// Record a separating set for undirected edge `{a,b}`.
    pub fn set_sepset(&mut self, a: DenseNodeId, b: DenseNodeId, sep: Arc<[DenseNodeId]>) {
        let key = if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) };
        self.sepsets.insert(key, sep);
    }

    /// Lookup sepset.
    #[must_use]
    pub fn sepset(&self, a: DenseNodeId, b: DenseNodeId) -> Option<&[DenseNodeId]> {
        let key = if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) };
        self.sepsets.get(&key).map(AsRef::as_ref)
    }

    /// Mark `{a,b}`'s sepset as weakly minimal.
    pub fn mark_weakly_minimal(&mut self, a: DenseNodeId, b: DenseNodeId) {
        let key = if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) };
        self.weakly_minimal.insert(key);
    }

    /// Whether the stored sepset for `{a,b}` is weakly minimal.
    #[must_use]
    pub fn is_weakly_minimal(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        let key = if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) };
        self.weakly_minimal.contains(&key)
    }

    /// Record an orientation conflict on `{a,b}` (cycle or opposite direction).
    pub fn record_conflict(&mut self, delta: &mut RuleDelta, a: DenseNodeId, b: DenseNodeId, kind: &str) {
        let key = if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) };
        self.conflict_edges.insert(key);
        self.conflicts = self.conflicts.saturating_add(1);
        delta.conflicts = delta.conflicts.saturating_add(1);
        delta.premises.push(Arc::from(format!(
            "conflict({kind}): {}—{}",
            key.0, key.1
        )));
    }
}

/// Try to orient an undirected edge `from → to`.
///
/// Cycle conflicts are recorded, the edge is marked `x-x`, and the run continues.
/// Other graph errors propagate.
pub(crate) fn try_orient_undirected<G: CpdagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    delta: &mut RuleDelta,
    from: DenseNodeId,
    to: DenseNodeId,
    premise: impl Into<Arc<str>>,
) -> Result<bool, OrientationError> {
    match graph.orient_undirected(from, to) {
        Ok(()) => {
            delta.edges_changed += 1;
            delta.fixed_point = false;
            delta.premises.push(premise.into());
            Ok(true)
        }
        Err(GraphError::Cycle { .. }) => {
            state.record_conflict(delta, from, to, "cycle");
            if graph.mark_conflict(from, to).is_ok() {
                delta.edges_changed += 1;
                delta.fixed_point = false;
            }
            Ok(false)
        }
        Err(e) => Err(OrientationError::from(e)),
    }
}

/// Result of one rule application.
#[derive(Clone, Debug, Default)]
pub struct RuleDelta {
    /// Directed orientations applied this call.
    pub edges_changed: u32,
    /// Human-readable premises (diagnostics).
    pub premises: Vec<Arc<str>>,
    /// Conflicts encountered.
    pub conflicts: u32,
    /// Nodes newly enqueued.
    pub enqueued: u32,
    /// Whether the rule had no further work on the current focus.
    pub fixed_point: bool,
}

/// Named orientation transform on a [`TemporalCpdag`].
pub trait OrientationRule {
    /// Rule id for diagnostics.
    fn name(&self) -> &'static str;

    /// Apply the rule, mutating the graph and enqueueing only local neighbors of changes.
    ///
    /// # Errors
    ///
    /// Graph mutation or precondition failures.
    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError>;
}

/// Named orientation transform on a static [`Cpdag`].
pub trait StaticOrientationRule {
    /// Rule id for diagnostics.
    fn name(&self) -> &'static str;

    /// Apply the rule on a static CPDAG.
    ///
    /// # Errors
    ///
    /// Graph mutation or precondition failures.
    fn apply(
        &self,
        graph: &mut Cpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError>;
}

fn enqueue_neighbors<G: CpdagOps>(graph: &G, id: DenseNodeId, queue: &mut OrientationQueue) -> u32 {
    let before = queue.len();
    queue.push(id);
    for n in graph.children(id) {
        queue.push(n);
    }
    for n in graph.parents(id) {
        queue.push(n);
    }
    for n in graph.undirected_neighbors(id) {
        queue.push(n);
    }
    u32::try_from(queue.len().saturating_sub(before)).unwrap_or(u32::MAX)
}

/// Drain the orientation queue into a focus set, or scan all nodes when empty.
fn focus_nodes<G: CpdagOps>(graph: &G, queue: &mut OrientationQueue) -> Vec<DenseNodeId> {
    if queue.is_empty() {
        (0..graph.node_count())
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
            .collect()
    } else {
        let mut v = Vec::new();
        while let Some(n) = queue.pop() {
            v.push(n);
        }
        v
    }
}

/// Meek R1: if `a → b — c` and `a` not adjacent to `c`, orient `b → c`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR1;

fn apply_meek_r1<G: CpdagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
    let focus = focus_nodes(graph, queue);
    let mut changed_nodes = Vec::new();
    for b in focus {
        for a in graph.parents(b) {
            for c in graph.undirected_neighbors(b) {
                if !graph.has_edge(a, c) {
                    let premise = format!(
                        "meek.r1: {}→{}—{} and {} not adj {}",
                        a.raw(),
                        b.raw(),
                        c.raw(),
                        a.raw(),
                        c.raw()
                    );
                    if try_orient_undirected(graph, state, &mut delta, b, c, premise)? {
                        changed_nodes.push(b);
                        changed_nodes.push(c);
                    }
                }
            }
        }
    }
    for n in changed_nodes {
        delta.enqueued += enqueue_neighbors(graph, n, queue);
    }
    Ok(delta)
}

impl OrientationRule for MeekR1 {
    fn name(&self) -> &'static str {
        "meek.r1"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r1(graph, state, queue)
    }
}

impl StaticOrientationRule for MeekR1 {
    fn name(&self) -> &'static str {
        "meek.r1"
    }

    fn apply(
        &self,
        graph: &mut Cpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r1(graph, state, queue)
    }
}

/// Meek R2: if `a → b → c` and `a — c`, orient `a → c`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR2;

fn apply_meek_r2<G: CpdagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
    let focus = focus_nodes(graph, queue);
    let mut changed = Vec::new();
    for a in &focus {
        for b in graph.children(*a) {
            for c in graph.children(b) {
                if graph.edge_between(*a, c).is_some_and(|e| e.is_undirected()) {
                    let premise = format!(
                        "meek.r2: {}→{}→{} and {}—{}",
                        a.raw(),
                        b.raw(),
                        c.raw(),
                        a.raw(),
                        c.raw()
                    );
                    if try_orient_undirected(graph, state, &mut delta, *a, c, premise)? {
                        changed.push(*a);
                        changed.push(c);
                    }
                }
            }
        }
    }
    for n in changed {
        delta.enqueued += enqueue_neighbors(graph, n, queue);
    }
    Ok(delta)
}

impl OrientationRule for MeekR2 {
    fn name(&self) -> &'static str {
        "meek.r2"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r2(graph, state, queue)
    }
}

impl StaticOrientationRule for MeekR2 {
    fn name(&self) -> &'static str {
        "meek.r2"
    }

    fn apply(
        &self,
        graph: &mut Cpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r2(graph, state, queue)
    }
}

/// Meek R3: if `a — b` and ∃ `c,d` with `a — c → b`, `a — d → b`, `c` not adj `d`, orient `a → b`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR3;

fn apply_meek_r3<G: CpdagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
    let focus = focus_nodes(graph, queue);
    let mut changed = Vec::new();
    for a in &focus {
        let und_a: Vec<DenseNodeId> = graph.undirected_neighbors(*a);
        for &b in &und_a {
            let mut mediators = Vec::new();
            for &c in &und_a {
                if c == b {
                    continue;
                }
                if graph.children(c).contains(&b) {
                    mediators.push(c);
                }
            }
            let mut orient = false;
            'pairs: for i in 0..mediators.len() {
                for j in (i + 1)..mediators.len() {
                    if !graph.has_edge(mediators[i], mediators[j]) {
                        orient = true;
                        break 'pairs;
                    }
                }
            }
            if orient {
                let premise =
                    format!("meek.r3: {}—{} via nonadjacent mediators", a.raw(), b.raw());
                if try_orient_undirected(graph, state, &mut delta, *a, b, premise)? {
                    changed.push(*a);
                    changed.push(b);
                }
            }
        }
    }
    for n in changed {
        delta.enqueued += enqueue_neighbors(graph, n, queue);
    }
    Ok(delta)
}

impl OrientationRule for MeekR3 {
    fn name(&self) -> &'static str {
        "meek.r3"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r3(graph, state, queue)
    }
}

impl StaticOrientationRule for MeekR3 {
    fn name(&self) -> &'static str {
        "meek.r3"
    }

    fn apply(
        &self,
        graph: &mut Cpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r3(graph, state, queue)
    }
}

/// Meek R4: if `a — b` and ∃ `c,d` with `a — c → d → b`, adj(`a`,`d`), not adj(`c`,`b`), orient `a → b`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR4;

fn apply_meek_r4<G: CpdagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
    let focus = focus_nodes(graph, queue);
    let mut changed = Vec::new();
    for a in &focus {
        for b in graph.undirected_neighbors(*a) {
            let mut orient = false;
            for c in graph.undirected_neighbors(*a) {
                if c == b {
                    continue;
                }
                for d in graph.children(c) {
                    if !graph.children(d).contains(&b) {
                        continue;
                    }
                    if !graph.has_edge(*a, d) {
                        continue;
                    }
                    if graph.has_edge(c, b) {
                        continue;
                    }
                    orient = true;
                    break;
                }
                if orient {
                    break;
                }
            }
            if orient {
                let premise =
                    format!("meek.r4: {}—{} via discriminating path", a.raw(), b.raw());
                if try_orient_undirected(graph, state, &mut delta, *a, b, premise)? {
                    changed.push(*a);
                    changed.push(b);
                }
            }
        }
    }
    for n in changed {
        delta.enqueued += enqueue_neighbors(graph, n, queue);
    }
    Ok(delta)
}

impl OrientationRule for MeekR4 {
    fn name(&self) -> &'static str {
        "meek.r4"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r4(graph, state, queue)
    }
}

impl StaticOrientationRule for MeekR4 {
    fn name(&self) -> &'static str {
        "meek.r4"
    }

    fn apply(
        &self,
        graph: &mut Cpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_meek_r4(graph, state, queue)
    }
}

fn is_contemporaneous_node(graph: &TemporalCpdag, id: DenseNodeId) -> bool {
    match graph.nodes().get(id.raw() as usize) {
        Some(causal_graph::NodeRef::Lagged { lag, .. }) => lag.is_contemporaneous(),
        _ => false,
    }
}

/// Meek R1 restricted to contemporaneous undirected edges (PCMCI+ / pinned baseline).
#[derive(Clone, Copy, Debug, Default)]
pub struct ContempMeekR1;

impl OrientationRule for ContempMeekR1 {
    fn name(&self) -> &'static str {
        "meek.r1.contemp"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed_nodes = Vec::new();
        for b in focus {
            for a in graph.parents(b) {
                for c in graph.undirected_neighbors(b) {
                    if !is_contemporaneous_node(graph, b) || !is_contemporaneous_node(graph, c) {
                        continue;
                    }
                    if !graph.has_edge(a, c) {
                        let premise = format!(
                            "meek.r1.contemp: {}→{}—{} and {} not adj {}",
                            a.raw(),
                            b.raw(),
                            c.raw(),
                            a.raw(),
                            c.raw()
                        );
                        if try_orient_undirected(graph, state, &mut delta, b, c, premise)? {
                            changed_nodes.push(b);
                            changed_nodes.push(c);
                        }
                    }
                }
            }
        }
        for n in changed_nodes {
            delta.enqueued += enqueue_neighbors(graph, n, queue);
        }
        Ok(delta)
    }
}

/// Meek R2 restricted to contemporaneous undirected edges (PCMCI+ / pinned baseline).
#[derive(Clone, Copy, Debug, Default)]
pub struct ContempMeekR2;

impl OrientationRule for ContempMeekR2 {
    fn name(&self) -> &'static str {
        "meek.r2.contemp"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed = Vec::new();
        for a in &focus {
            for b in graph.children(*a) {
                for c in graph.children(b) {
                    if !is_contemporaneous_node(graph, *a) || !is_contemporaneous_node(graph, c) {
                        continue;
                    }
                    if graph.edge_between(*a, c).is_some_and(|e| e.is_undirected()) {
                        let premise = format!(
                            "meek.r2.contemp: {}→{}→{} and {}—{}",
                            a.raw(),
                            b.raw(),
                            c.raw(),
                            a.raw(),
                            c.raw()
                        );
                        if try_orient_undirected(graph, state, &mut delta, *a, c, premise)? {
                            changed.push(*a);
                            changed.push(c);
                        }
                    }
                }
            }
        }
        for n in changed {
            delta.enqueued += enqueue_neighbors(graph, n, queue);
        }
        Ok(delta)
    }
}

/// Meek R3 restricted to contemporaneous undirected edges (PCMCI+ / pinned baseline).
#[derive(Clone, Copy, Debug, Default)]
pub struct ContempMeekR3;

impl OrientationRule for ContempMeekR3 {
    fn name(&self) -> &'static str {
        "meek.r3.contemp"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed = Vec::new();
        for a in &focus {
            if !is_contemporaneous_node(graph, *a) {
                continue;
            }
            let und_a: Vec<DenseNodeId> = graph
                .undirected_neighbors(*a)
                .into_iter()
                .filter(|&n| is_contemporaneous_node(graph, n))
                .collect();
            for &b in &und_a {
                let mut mediators = Vec::new();
                for &c in &und_a {
                    if c == b {
                        continue;
                    }
                    if graph.children(c).contains(&b) {
                        mediators.push(c);
                    }
                }
                let mut orient = false;
                'pairs: for i in 0..mediators.len() {
                    for j in (i + 1)..mediators.len() {
                        if !graph.has_edge(mediators[i], mediators[j]) {
                            orient = true;
                            break 'pairs;
                        }
                    }
                }
                if orient {
                    let premise = format!(
                        "meek.r3.contemp: {}—{} via nonadjacent mediators",
                        a.raw(),
                        b.raw()
                    );
                    if try_orient_undirected(graph, state, &mut delta, *a, b, premise)? {
                        changed.push(*a);
                        changed.push(b);
                    }
                }
            }
        }
        for n in changed {
            delta.enqueued += enqueue_neighbors(graph, n, queue);
        }
        Ok(delta)
    }
}

/// Collider orientation when sepset is known: for an unshielded triple `a * c * b` with both
/// legs *into or undirected at* `c` (undirected legs, or legs already directed into `c`, e.g.
/// lagged edges auto-oriented by time), `a` not adj `b`, `c ∉ Sep(a,b)` → orient the
/// undirected leg(s) toward `c`.
///
/// Considering already-directed legs matters: Meek R1's premise is that every collider is
/// oriented first, so skipping `X_{t−τ} → c — b` triples here would let R1 orient `c → b`
/// on triples the data mark as colliders at `c`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OrientCollider;

fn apply_orient_collider<G: CpdagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
    let focus = focus_nodes(graph, queue);
    let mut changed = Vec::new();
    for c in &focus {
        let mut legs: Vec<(DenseNodeId, LegKind)> =
            graph.undirected_neighbors(*c).into_iter().map(|n| (n, LegKind::Undirected)).collect();
        legs.extend(graph.parents(*c).into_iter().map(|n| (n, LegKind::IntoC)));
        legs.extend(graph.children(*c).into_iter().map(|n| (n, LegKind::OutOfC)));
        for i in 0..legs.len() {
            for j in (i + 1)..legs.len() {
                let (a, a_kind) = legs[i];
                let (b, b_kind) = legs[j];
                if matches!(a_kind, LegKind::IntoC) && matches!(b_kind, LegKind::IntoC) {
                    continue;
                }
                if graph.has_edge(a, b) {
                    continue;
                }
                let Some(sep) = state.sepset(a, b) else {
                    continue;
                };
                if sep.iter().any(|x| *x == *c) {
                    continue;
                }
                for &(endpoint, kind) in &[(a, a_kind), (b, b_kind)] {
                    match kind {
                        LegKind::OutOfC => {
                            state.record_conflict(
                                &mut delta,
                                endpoint,
                                *c,
                                "opposite_direction",
                            );
                            if graph.mark_conflict(endpoint, *c).is_ok() {
                                delta.edges_changed += 1;
                                delta.fixed_point = false;
                                changed.push(endpoint);
                                changed.push(*c);
                            }
                        }
                        LegKind::Undirected => {
                            let premise = format!(
                                "collider: {}→{}←{} (c not in sepset)",
                                a.raw(),
                                c.raw(),
                                b.raw()
                            );
                            if try_orient_undirected(
                                graph, state, &mut delta, endpoint, *c, premise,
                            )? {
                                changed.push(endpoint);
                                changed.push(*c);
                            }
                        }
                        LegKind::IntoC => {}
                    }
                }
            }
        }
    }
    for n in changed {
        delta.enqueued += enqueue_neighbors(graph, n, queue);
    }
    Ok(delta)
}

impl OrientationRule for OrientCollider {
    fn name(&self) -> &'static str {
        "orient.collider"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_orient_collider(graph, state, queue)
    }
}

impl StaticOrientationRule for OrientCollider {
    fn name(&self) -> &'static str {
        "orient.collider"
    }

    fn apply(
        &self,
        graph: &mut Cpdag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_orient_collider(graph, state, queue)
    }
}

#[derive(Clone, Copy, Debug)]
enum LegKind {
    Undirected,
    IntoC,
    OutOfC,
}

/// Run rules to a fixed point, seeding the queue with all nodes once.
///
/// # Errors
///
/// Propagates rule failures.
pub fn run_orientation_to_fixed_point(
    graph: &mut TemporalCpdag,
    rules: &[&dyn OrientationRule],
    state: &mut OrientationState,
) -> Result<RuleDelta, OrientationError> {
    let mut queue = OrientationQueue::new();
    for i in 0..graph.node_count() {
        queue.push(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
    }
    let mut total = RuleDelta::default();
    let mut guard = 0u32;
    loop {
        guard += 1;
        if guard > 10_000 {
            return Err(OrientationError::Precondition {
                message: "orientation did not reach fixed point within iteration budget",
            });
        }
        let mut any = false;
        for rule in rules {
            let d = rule.apply(graph, state, &mut queue)?;
            total.edges_changed += d.edges_changed;
            total.conflicts += d.conflicts;
            total.enqueued += d.enqueued;
            total.premises.extend(d.premises);
            if d.edges_changed > 0 {
                any = true;
            }
        }
        if !any && queue.is_empty() {
            total.fixed_point = true;
            break;
        }
        if !any {
            // drain idle queue
            while queue.pop().is_some() {}
            total.fixed_point = true;
            break;
        }
    }
    Ok(total)
}

/// Run static CPDAG orientation rules to a fixed point.
///
/// # Errors
///
/// Propagates rule failures.
pub fn run_static_orientation_to_fixed_point(
    graph: &mut Cpdag,
    rules: &[&dyn StaticOrientationRule],
    state: &mut OrientationState,
) -> Result<RuleDelta, OrientationError> {
    let mut queue = OrientationQueue::new();
    for i in 0..graph.node_count() {
        queue.push(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
    }
    let mut total = RuleDelta::default();
    let mut guard = 0u32;
    loop {
        guard += 1;
        if guard > 10_000 {
            return Err(OrientationError::Precondition {
                message: "orientation did not reach fixed point within iteration budget",
            });
        }
        let mut any = false;
        for rule in rules {
            let d = rule.apply(graph, state, &mut queue)?;
            total.edges_changed += d.edges_changed;
            total.conflicts += d.conflicts;
            total.enqueued += d.enqueued;
            total.premises.extend(d.premises);
            if d.edges_changed > 0 {
                any = true;
            }
        }
        if !any && queue.is_empty() {
            total.fixed_point = true;
            break;
        }
        if !any {
            while queue.pop().is_some() {}
            total.fixed_point = true;
            break;
        }
    }
    Ok(total)
}


#[cfg(test)]
#[path = "orientation_tests.rs"]
mod tests;
