//! LPCMCI orientation rule scheduling (DESIGN.md §13.6).
//!
//! Explicit scheduler module — rules are applied via a delta queue, not a
//! single procedural blob that rescans all edges after every change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names, clippy::similar_names)]

use std::collections::HashSet;

use causal_graph::{DenseNodeId, Endpoint, TemporalPag};

use crate::discriminating_paths::{discriminating_implies_collider, find_discriminating_paths};
use crate::orientation::{OrientationError, OrientationQueue, OrientationState, RuleDelta};

/// Drain the orientation queue into a focus set, or scan all nodes when empty
/// (same contract as Meek [`crate::orientation`] rules).
fn focus_nodes(graph: &TemporalPag, queue: &mut OrientationQueue) -> Vec<DenseNodeId> {
    if queue.is_empty() {
        (0..graph.node_count()).map(|i| DenseNodeId::from_raw(i as u32)).collect()
    } else {
        let mut v = Vec::new();
        while let Some(n) = queue.pop() {
            v.push(n);
        }
        v
    }
}

/// Enqueue a changed node and its adjacency (local delta, not full-graph re-seed).
fn enqueue_local(graph: &TemporalPag, id: DenseNodeId, queue: &mut OrientationQueue) {
    queue.push(id);
    for (n, _, _) in graph.neighbors(id) {
        queue.push(n);
    }
}

/// Named LPCMCI orientation rule.
pub trait LpcmciOrientationRule {
    /// Rule id.
    fn id(&self) -> &'static str;

    /// Apply once; enqueue only locally affected nodes.
    ///
    /// # Errors
    ///
    /// Graph mutation failures.
    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError>;
}

/// Orient unshielded colliders from sepsets (circle→arrow into middle).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciOrientCollider;

impl LpcmciOrientationRule for LpcmciOrientCollider {
    fn id(&self) -> &'static str {
        "lpcmci.orient_collider"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        for b in focus {
            let nbrs: Vec<_> = graph.neighbors(b).map(|(x, _, _)| x).collect();
            for (ai, &a) in nbrs.iter().enumerate() {
                for &c in &nbrs[ai + 1..] {
                    if graph.has_edge(a, c) {
                        continue; // shielded
                    }
                    let Some(sep) = state.sepset(a, c) else {
                        continue;
                    };
                    if sep.iter().any(|&z| z == b) {
                        continue; // non-collider
                    }
                    // Orient a *→ b ←* c
                    if let Some(e) = graph.edge_between(a, b) {
                        let at_a = e.at_a;
                        let at_b = Endpoint::Arrow;
                        if !matches!(e.at_b, Endpoint::Arrow) {
                            graph.set_marks(a, b, at_a, at_b).map_err(OrientationError::from)?;
                            delta.edges_changed += 1;
                            enqueue_local(graph, a, queue);
                            enqueue_local(graph, b, queue);
                        }
                    }
                    if let Some(e) = graph.edge_between(c, b) {
                        let at_c = e.at_a;
                        let at_b = Endpoint::Arrow;
                        if !matches!(e.at_b, Endpoint::Arrow) {
                            graph.set_marks(c, b, at_c, at_b).map_err(OrientationError::from)?;
                            delta.edges_changed += 1;
                            enqueue_local(graph, c, queue);
                            enqueue_local(graph, b, queue);
                        }
                    }
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// FCI R1: if `a *→ b o–* c` (arrow at b on a–b; circle at b on b–c) and a,c nonadjacent,
/// orient `b → c` (tail at b and arrow at c).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR1;

impl LpcmciOrientationRule for LpcmciR1 {
    fn id(&self) -> &'static str {
        "lpcmci.r1"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        for b in focus {
            let nbrs: Vec<_> = graph.neighbors(b).map(|(x, _, _)| x).collect();
            for &a in &nbrs {
                let Some((_, at_b_ab)) = marks_between(graph, a, b) else {
                    continue;
                };
                // Premise: arrow into b on a–b (any mark at a: o→, →, or ↔).
                if !matches!(at_b_ab, Endpoint::Arrow) {
                    continue;
                }
                for &c in &nbrs {
                    if c == a || graph.has_edge(a, c) {
                        continue;
                    }
                    let Some((at_b_bc, _)) = marks_between(graph, b, c) else {
                        continue;
                    };
                    // b o–* c (circle at b).
                    if !matches!(at_b_bc, Endpoint::Circle) {
                        continue;
                    }
                    // Orient b → c (both endpoints).
                    set_marks_oriented(graph, b, c, Endpoint::Tail, Endpoint::Arrow)?;
                    delta.edges_changed += 1;
                    enqueue_local(graph, b, queue);
                    enqueue_local(graph, c, queue);
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// FCI R2: if `a → b *→ c` or `a *→ b → c`, and `a *–o c` (circle **at c**), orient
/// arrow at c on a–c. Never overwrites a Tail at c.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR2;

impl LpcmciOrientationRule for LpcmciR2 {
    fn id(&self) -> &'static str {
        "lpcmci.r2"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        for b in focus {
            let nbrs: Vec<_> = graph.neighbors(b).map(|(x, _, _)| x).collect();
            for &a in &nbrs {
                for &c in &nbrs {
                    if a == c {
                        continue;
                    }
                    let Some((at_a_ab, at_b_ab)) = marks_between(graph, a, b) else {
                        continue;
                    };
                    let Some((at_b_bc, at_c_bc)) = marks_between(graph, b, c) else {
                        continue;
                    };
                    // a → b *→ c  (definite directed a→b; arrow at c on b–c)
                    let case1 = matches!(at_a_ab, Endpoint::Tail)
                        && matches!(at_b_ab, Endpoint::Arrow)
                        && matches!(at_c_bc, Endpoint::Arrow);
                    // a *→ b → c  (arrow at b on a–b; definite directed b→c)
                    let case2 = matches!(at_b_ab, Endpoint::Arrow)
                        && matches!(at_b_bc, Endpoint::Tail)
                        && matches!(at_c_bc, Endpoint::Arrow);
                    if !(case1 || case2) {
                        continue;
                    }
                    let Some((at_a_ac, at_c_ac)) = marks_between(graph, a, c) else {
                        continue;
                    };
                    // Circle at c only — never overwrite Tail or re-orient Arrow.
                    if !matches!(at_c_ac, Endpoint::Circle) {
                        continue;
                    }
                    set_marks_oriented(graph, a, c, at_a_ac, Endpoint::Arrow)?;
                    delta.edges_changed += 1;
                    enqueue_local(graph, a, queue);
                    enqueue_local(graph, c, queue);
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// FCI R3: collider `a *→ b ←* c` with nonadjacent a,c and `a *–o d o–* c`, `d *–o b` → orient `d *→ b`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR3;

impl LpcmciOrientationRule for LpcmciR3 {
    fn id(&self) -> &'static str {
        "lpcmci.r3"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        let n = graph.node_count();
        for b in focus {
            let nbrs: Vec<_> = graph.neighbors(b).map(|(x, _, _)| x).collect();
            for (ai, &a) in nbrs.iter().enumerate() {
                for &c in &nbrs[ai + 1..] {
                    if graph.has_edge(a, c) {
                        continue;
                    }
                    let Some((_, at_b_ab)) = marks_between(graph, a, b) else {
                        continue;
                    };
                    let Some((_, at_b_cb)) = marks_between(graph, c, b) else {
                        continue;
                    };
                    if !matches!(at_b_ab, Endpoint::Arrow) || !matches!(at_b_cb, Endpoint::Arrow) {
                        continue;
                    }
                    // Find θ = d adjacent to a,c,b with circle marks into d from a/c and circle at b side.
                    for j in 0..n {
                        let d = DenseNodeId::from_raw(j as u32);
                        if d == a || d == b || d == c {
                            continue;
                        }
                        let Some((at_a_ad, at_d_ad)) = marks_between(graph, a, d) else {
                            continue;
                        };
                        let Some((at_c_cd, at_d_cd)) = marks_between(graph, c, d) else {
                            continue;
                        };
                        let Some((at_d_db, at_b_db)) = marks_between(graph, d, b) else {
                            continue;
                        };
                        let _ = (at_a_ad, at_c_cd);
                        if !matches!(at_d_ad, Endpoint::Circle)
                            || !matches!(at_d_cd, Endpoint::Circle)
                        {
                            continue;
                        }
                        if !matches!(at_b_db, Endpoint::Circle) {
                            continue;
                        }
                        if matches!(at_d_db, Endpoint::Arrow) && matches!(at_b_db, Endpoint::Arrow)
                        {
                            continue;
                        }
                        set_marks_oriented(graph, d, b, at_d_db, Endpoint::Arrow)?;
                        delta.edges_changed += 1;
                        enqueue_local(graph, d, queue);
                        enqueue_local(graph, b, queue);
                    }
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

fn marks_between(
    graph: &TemporalPag,
    a: DenseNodeId,
    b: DenseNodeId,
) -> Option<(Endpoint, Endpoint)> {
    let e = graph.edge_between(a, b)?;
    if e.a == a { Some((e.at_a, e.at_b)) } else { Some((e.at_b, e.at_a)) }
}

fn set_marks_oriented(
    graph: &mut TemporalPag,
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
) -> Result<(), OrientationError> {
    let Some(e) = graph.edge_between(a, b) else {
        return Err(OrientationError::msg("missing edge in set_marks_oriented"));
    };
    if e.a == a {
        graph.set_marks(a, b, at_a, at_b).map_err(OrientationError::from)
    } else {
        graph.set_marks(b, a, at_b, at_a).map_err(OrientationError::from)
    }
}

/// Apply discriminating-path orientations (Zhang FCI R4).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciDiscriminatingPathRule;

impl LpcmciOrientationRule for LpcmciDiscriminatingPathRule {
    fn id(&self) -> &'static str {
        "lpcmci.discriminating_path"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        let focus_set: HashSet<u32> = focus.iter().map(|n| n.raw()).collect();
        let paths = find_discriminating_paths(graph, 64, 8);
        for path in paths {
            if !path.nodes.iter().any(|n| focus_set.contains(&n.raw())) {
                continue;
            }
            let a = path.a();
            let c = path.c();
            let b = path.b();
            let d_k = path.d_k();
            // R4 consults Sep(a,b) — the non-adjacent endpoints — not Sep(a,c).
            let Some(sep) = state.sepset(a, b) else {
                continue;
            };
            let c_in_sep = sep.iter().any(|&z| z == c);
            let collider = discriminating_implies_collider(c_in_sep);
            let Some(e_cb) = graph.edge_between(c, b) else {
                continue;
            };
            // Premise: circle still at c on c–b.
            let mark_at_c = if e_cb.a == c { e_cb.at_a } else { e_cb.at_b };
            if !matches!(mark_at_c, Endpoint::Circle) {
                continue;
            }
            if collider {
                // dₖ *→ c ←* b (arrows into c on both edges; keep far-end marks).
                set_arrow_at(graph, c, d_k)?;
                set_arrow_at(graph, c, b)?;
            } else {
                // c → b (non-collider at c).
                graph
                    .set_marks(c, b, Endpoint::Tail, Endpoint::Arrow)
                    .map_err(OrientationError::from)?;
            }
            delta.edges_changed += 1;
            enqueue_local(graph, c, queue);
            enqueue_local(graph, b, queue);
            enqueue_local(graph, d_k, queue);
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// Set the mark at `at` on edge `{at, other}` to [`Endpoint::Arrow`], keeping the far mark.
fn set_arrow_at(
    graph: &mut TemporalPag,
    at: DenseNodeId,
    other: DenseNodeId,
) -> Result<(), OrientationError> {
    let e = graph.edge_between(at, other).ok_or(OrientationError::Precondition {
        message: "discriminating path missing edge",
    })?;
    let at_other = if e.a == other { e.at_a } else { e.at_b };
    graph.set_marks(at, other, Endpoint::Arrow, at_other).map_err(OrientationError::from)
}

/// Schedule LPCMCI rules to a fixed point using a local delta queue.
///
/// Seeds all nodes once. Subsequent rounds honor nodes enqueued by rules
/// (DESIGN.md §13.6 / §13.8) — no full-graph re-seed after each round.
///
/// # Errors
///
/// Rule application failures.
pub fn run_lpcmci_orientation(
    graph: &mut TemporalPag,
    rules: &[&dyn LpcmciOrientationRule],
    state: &mut OrientationState,
) -> Result<RuleDelta, OrientationError> {
    const MAX_ROUNDS: u32 = 10_000;
    let mut queue = OrientationQueue::new();
    for i in 0..graph.node_count() {
        queue.push(DenseNodeId::from_raw(i as u32));
    }
    let mut total = RuleDelta::default();
    let mut rounds = 0u32;
    while rounds < MAX_ROUNDS {
        rounds += 1;
        let mut any = false;
        for rule in rules {
            let d = rule.apply(graph, state, &mut queue)?;
            total.edges_changed += d.edges_changed;
            total.enqueued += d.enqueued;
            if d.edges_changed > 0 {
                any = true;
            }
        }
        if !any && queue.is_empty() {
            total.fixed_point = true;
            break;
        }
        if !any {
            // Idle queue with no further orientations — drain and stop.
            while queue.pop().is_some() {}
            total.fixed_point = true;
            break;
        }
        // Keep delta-queued nodes for the next round (do not re-seed all nodes).
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{Lag, VariableId};

    #[test]
    fn r2_orients_circle_into_arrow() {
        // a → b o→ c and a o-o c ⇒ a o→ c
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_circle_arrow(b, c).unwrap();
        g.insert_marked(causal_graph::MarkedEdge {
            a,
            b: c,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Circle,
        })
        .unwrap();
        let mut state = OrientationState::default();
        let mut queue = OrientationQueue::new();
        let d = LpcmciR2.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(d.edges_changed > 0);
        let (at_a, at_c) = marks_between(&g, a, c).unwrap();
        assert!(matches!(at_a, Endpoint::Circle));
        assert!(matches!(at_c, Endpoint::Arrow));
    }

    #[test]
    fn r2_fires_on_fully_directed_chain() {
        // a → b → c and a *–o c (circle at c) ⇒ orient arrow at c.
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_marked(causal_graph::MarkedEdge {
            a,
            b: c,
            at_a: Endpoint::Tail,
            at_b: Endpoint::Circle,
        })
        .unwrap();
        let mut state = OrientationState::default();
        let mut queue = OrientationQueue::new();
        let d = LpcmciR2.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(d.edges_changed > 0);
        let (at_a, at_c) = marks_between(&g, a, c).unwrap();
        assert!(matches!(at_a, Endpoint::Tail));
        assert!(matches!(at_c, Endpoint::Arrow));
    }

    #[test]
    fn r2_does_not_overwrite_tail_at_c() {
        // a → b → c and a → c already (tail at c would be illegal for R2 premise;
        // use a *– Tail at c to ensure we refuse to overwrite).
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        // Circle at a, Tail at c on a–c — R2 must not turn the Tail into an Arrow.
        g.insert_marked(causal_graph::MarkedEdge {
            a,
            b: c,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Tail,
        })
        .unwrap();
        let mut state = OrientationState::default();
        let mut queue = OrientationQueue::new();
        let d = LpcmciR2.apply(&mut g, &mut state, &mut queue).unwrap();
        assert_eq!(d.edges_changed, 0);
        let (_, at_c) = marks_between(&g, a, c).unwrap();
        assert!(matches!(at_c, Endpoint::Tail));
    }

    #[test]
    fn r1_orients_from_circle_arrow_premise() {
        // a o→ b o–o c, a ≁ c ⇒ b → c (both marks).
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_circle_arrow(a, b).unwrap();
        g.insert_marked(causal_graph::MarkedEdge {
            a: b,
            b: c,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Circle,
        })
        .unwrap();
        let mut state = OrientationState::default();
        let mut queue = OrientationQueue::new();
        let d = LpcmciR1.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(d.edges_changed > 0);
        let (at_b, at_c) = marks_between(&g, b, c).unwrap();
        assert!(matches!(at_b, Endpoint::Tail));
        assert!(matches!(at_c, Endpoint::Arrow));
    }

    #[test]
    fn rule_ids_cover_r1_r2_r3() {
        assert_eq!(LpcmciR1.id(), "lpcmci.r1");
        assert_eq!(LpcmciR2.id(), "lpcmci.r2");
        assert_eq!(LpcmciR3.id(), "lpcmci.r3");
    }

    #[test]
    fn scheduler_honors_delta_queue_without_reseed() {
        // a → b o→ c and a o-o c ⇒ R2 orients a o→ c; subsequent rounds must not
        // require a full-graph re-seed to finish.
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_circle_arrow(b, c).unwrap();
        g.insert_marked(causal_graph::MarkedEdge {
            a,
            b: c,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Circle,
        })
        .unwrap();
        let mut state = OrientationState::default();
        let rules: [&dyn LpcmciOrientationRule; 1] = [&LpcmciR2];
        let d = run_lpcmci_orientation(&mut g, &rules, &mut state).unwrap();
        assert!(d.edges_changed > 0);
        assert!(d.fixed_point);
        let (at_a, at_c) = marks_between(&g, a, c).unwrap();
        assert!(matches!(at_a, Endpoint::Circle));
        assert!(matches!(at_c, Endpoint::Arrow));
    }

    #[test]
    fn discriminating_r4_orients_collider_when_c_not_in_sep_ab() {
        // Zhang path ⟨a, d, c, b⟩: a → d ← c, d → b, c o→ b; c ∉ Sep(a,b) ⇒ d *→ c ←* b.
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let d = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, d).unwrap();
        g.insert_directed(c, d).unwrap();
        g.insert_directed(d, b).unwrap();
        g.insert_circle_arrow(c, b).unwrap();

        let mut state = OrientationState::default();
        state.set_sepset(a, b, std::sync::Arc::from([])); // c ∉ Sep(a,b)
        let mut queue = OrientationQueue::new();
        let delta = LpcmciDiscriminatingPathRule.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(delta.edges_changed > 0);

        let (at_c_cb, at_b) = marks_between(&g, c, b).unwrap();
        assert!(matches!(at_c_cb, Endpoint::Arrow), "arrow into c from b");
        assert!(matches!(at_b, Endpoint::Arrow));

        let (at_c_cd, at_d) = marks_between(&g, c, d).unwrap();
        assert!(matches!(at_c_cd, Endpoint::Arrow), "arrow into c from d");
        assert!(matches!(at_d, Endpoint::Arrow), "keep arrow at d");
    }

    #[test]
    fn discriminating_r4_orients_noncollider_when_c_in_sep_ab() {
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let d = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, d).unwrap();
        g.insert_directed(c, d).unwrap();
        g.insert_directed(d, b).unwrap();
        g.insert_circle_arrow(c, b).unwrap();

        let mut state = OrientationState::default();
        state.set_sepset(a, b, std::sync::Arc::from([c])); // c ∈ Sep(a,b)
        let mut queue = OrientationQueue::new();
        let delta = LpcmciDiscriminatingPathRule.apply(&mut g, &mut state, &mut queue).unwrap();
        assert!(delta.edges_changed > 0);

        let (at_c, at_b) = marks_between(&g, c, b).unwrap();
        assert!(matches!(at_c, Endpoint::Tail));
        assert!(matches!(at_b, Endpoint::Arrow));
    }
}
