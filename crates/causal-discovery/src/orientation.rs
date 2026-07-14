//! Orientation rules and local delta queues (DESIGN.md §13.6, Phase 5).
//!
//! Rules enqueue only neighbors of changed edges — never a full-graph edge scan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names, clippy::redundant_closure_for_method_calls)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use causal_graph::{DenseNodeId, TemporalCpdag};

use crate::error::DiscoveryError;

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
            OrientationError::Precondition { message } => {
                DiscoveryError::Unsupported { message }
            }
            OrientationError::Message(m) => DiscoveryError::stats_msg(m),
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
    /// Whether a conflict was recorded.
    pub conflicts: u32,
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

/// Named orientation transform.
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

fn enqueue_neighbors(graph: &TemporalCpdag, id: DenseNodeId, queue: &mut OrientationQueue) -> u32 {
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
fn focus_nodes(graph: &TemporalCpdag, queue: &mut OrientationQueue) -> Vec<DenseNodeId> {
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

impl OrientationRule for MeekR1 {
    fn name(&self) -> &'static str {
        "meek.r1"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed_nodes = Vec::new();
        for b in focus {
            for a in graph.parents(b) {
                for c in graph.undirected_neighbors(b) {
                    if !graph.has_edge(a, c) {
                        graph
                            .orient_undirected(b, c)
                            .map_err(OrientationError::from)?;
                        delta.edges_changed += 1;
                        delta.fixed_point = false;
                        delta.premises.push(Arc::from(format!(
                            "meek.r1: {}→{}—{} and {} not adj {}",
                            a.raw(),
                            b.raw(),
                            c.raw(),
                            a.raw(),
                            c.raw()
                        )));
                        changed_nodes.push(b);
                        changed_nodes.push(c);
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

/// Meek R2: if `a → b → c` and `a — c`, orient `a → c`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR2;

impl OrientationRule for MeekR2 {
    fn name(&self) -> &'static str {
        "meek.r2"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed = Vec::new();
        for a in &focus {
            for b in graph.children(*a) {
                for c in graph.children(b) {
                    if graph.edge_between(*a, c).is_some_and(|e| e.is_undirected()) {
                        graph
                            .orient_undirected(*a, c)
                            .map_err(OrientationError::from)?;
                        delta.edges_changed += 1;
                        delta.fixed_point = false;
                        delta.premises.push(Arc::from(format!(
                            "meek.r2: {}→{}→{} and {}—{}",
                            a.raw(),
                            b.raw(),
                            c.raw(),
                            a.raw(),
                            c.raw()
                        )));
                        changed.push(*a);
                        changed.push(c);
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

/// Meek R3: if `a — b` and ∃ `c,d` with `a — c → b`, `a — d → b`, `c` not adj `d`, orient `a → b`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR3;

impl OrientationRule for MeekR3 {
    fn name(&self) -> &'static str {
        "meek.r3"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed = Vec::new();
        for a in &focus {
            let und_a: Vec<DenseNodeId> = graph.undirected_neighbors(*a);
            for &b in &und_a {
                // Common children of paths a—c→b
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
                    graph
                        .orient_undirected(*a, b)
                        .map_err(OrientationError::from)?;
                    delta.edges_changed += 1;
                    delta.fixed_point = false;
                    delta.premises.push(Arc::from(format!(
                        "meek.r3: {}—{} via nonadjacent mediators",
                        a.raw(),
                        b.raw()
                    )));
                    changed.push(*a);
                    changed.push(b);
                }
            }
        }
        for n in changed {
            delta.enqueued += enqueue_neighbors(graph, n, queue);
        }
        Ok(delta)
    }
}

/// Meek R4: if `a — b` and ∃ `c,d` with `a — c → d → b`, adj(`a`,`d`), not adj(`c`,`b`), orient `a → b`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeekR4;

impl OrientationRule for MeekR4 {
    fn name(&self) -> &'static str {
        "meek.r4"
    }

    fn apply(
        &self,
        graph: &mut TemporalCpdag,
        _state: &mut OrientationState,
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
                    graph
                        .orient_undirected(*a, b)
                        .map_err(OrientationError::from)?;
                    delta.edges_changed += 1;
                    delta.fixed_point = false;
                    delta.premises.push(Arc::from(format!(
                        "meek.r4: {}—{} via discriminating path",
                        a.raw(),
                        b.raw()
                    )));
                    changed.push(*a);
                    changed.push(b);
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
        let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
        let focus = focus_nodes(graph, queue);
        let mut changed = Vec::new();
        for c in &focus {
            // Legs eligible to point into c: undirected neighbors and existing parents.
            let mut legs: Vec<(DenseNodeId, bool)> =
                graph.undirected_neighbors(*c).into_iter().map(|n| (n, true)).collect();
            legs.extend(graph.parents(*c).into_iter().map(|n| (n, false)));
            for i in 0..legs.len() {
                for j in (i + 1)..legs.len() {
                    let (a, a_undirected) = legs[i];
                    let (b, b_undirected) = legs[j];
                    if !a_undirected && !b_undirected {
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
                    // Orient a→c←b (only undirected legs need work).
                    let mut oriented = 0u32;
                    if a_undirected && graph.edge_between(a, *c).is_some_and(|e| e.is_undirected())
                    {
                        graph
                            .orient_undirected(a, *c)
                            .map_err(OrientationError::from)?;
                        oriented += 1;
                        changed.push(a);
                        changed.push(*c);
                    }
                    if b_undirected && graph.edge_between(b, *c).is_some_and(|e| e.is_undirected())
                    {
                        graph
                            .orient_undirected(b, *c)
                            .map_err(OrientationError::from)?;
                        oriented += 1;
                        changed.push(b);
                        changed.push(*c);
                    }
                    if oriented > 0 {
                        delta.edges_changed += oriented;
                        delta.fixed_point = false;
                        delta.premises.push(Arc::from(format!(
                            "collider: {}→{}←{} (c not in sepset)",
                            a.raw(),
                            c.raw(),
                            b.raw()
                        )));
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

#[cfg(test)]
mod tests {
    use causal_core::{Lag, VariableId};
    use causal_graph::TemporalCpdag;

    use super::*;

    #[test]
    fn meek_r1_orients_chain() {
        // a → b — c  ⇒  b → c
        let mut g = TemporalCpdag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_undirected(b, c).unwrap();
        let mut state = OrientationState::default();
        let mut queue = OrientationQueue::new();
        queue.push(b);
        let d = MeekR1.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(d.edges_changed >= 1);
        assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
        assert!(d.enqueued > 0);
        assert!(d.enqueued < 20); // local, not full-graph blowup
    }

    #[test]
    fn collider_with_sepset() {
        let mut g = TemporalCpdag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_undirected(a, c).unwrap();
        g.insert_undirected(c, b).unwrap();
        let mut state = OrientationState::default();
        // Sep(a,b) empty ⇒ c not in sepset ⇒ collider
        state.set_sepset(a, b, Arc::from([]));
        let mut queue = OrientationQueue::new();
        let d = OrientCollider.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(d.edges_changed >= 2);
        assert_eq!(g.edge_between(a, c).unwrap().parent_child(), Some((a, c)));
        assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
    }

    #[test]
    fn collider_fires_on_triple_with_directed_lagged_leg() {
        // X@1 → K (auto-oriented by time), K — J undirected, X@1 ⟂ J with sepset ∅
        // excluding K ⇒ collider at K: orient J → K. Meek R1 must not run first and
        // orient K → J.
        let mut g = TemporalCpdag::empty();
        let x1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let k = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let j = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, k).unwrap();
        g.insert_undirected(k, j).unwrap();
        let mut state = OrientationState::default();
        state.set_sepset(x1, j, Arc::from([]));
        let rules: [&dyn OrientationRule; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
        assert_eq!(g.edge_between(j, k).unwrap().parent_child(), Some((j, k)));
    }

    #[test]
    fn meek_r3_orients_diagonal() {
        // a—b with a—c→b and a—d→b, c not adj d ⇒ a→b
        let mut g = TemporalCpdag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let d = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_undirected(a, b).unwrap();
        g.insert_undirected(a, c).unwrap();
        g.insert_undirected(a, d).unwrap();
        g.insert_directed(c, b).unwrap();
        g.insert_directed(d, b).unwrap();
        let mut state = OrientationState::default();
        let mut queue = OrientationQueue::new();
        let delta = MeekR3.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(delta.edges_changed >= 1);
        assert_eq!(g.edge_between(a, b).unwrap().parent_child(), Some((a, b)));
    }

    #[test]
    fn fixed_point_runner() {
        let mut g = TemporalCpdag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_undirected(b, c).unwrap();
        let mut state = OrientationState::default();
        let rules: [&dyn OrientationRule; 4] = [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
        let d = run_orientation_to_fixed_point(&mut g, &rules, &mut state).unwrap();
        assert!(d.fixed_point);
        assert_eq!(g.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
    }
}
