//! LPCMCI orientation rule scheduling (DESIGN.md §13.6).
//!
//! Explicit scheduler module — rules are applied via a delta queue, not a
//! single procedural blob that rescans all edges after every change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use causal_graph::{DenseNodeId, Endpoint, TemporalPag};

use crate::discriminating_paths::{
    discriminating_implies_collider, find_discriminating_paths,
};
use crate::orientation::{
    OrientationError, OrientationQueue, OrientationState, RuleDelta,
};

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
        let n = graph.node_count();
        for i in 0..n {
            let b = DenseNodeId::from_raw(i as u32);
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
                            graph
                                .set_marks(a, b, at_a, at_b)
                                .map_err(|err| OrientationError::Graph(err.to_string()))?;
                            delta.edges_changed += 1;
                            queue.push(a);
                            queue.push(b);
                        }
                    }
                    if let Some(e) = graph.edge_between(c, b) {
                        let at_c = e.at_a;
                        let at_b = Endpoint::Arrow;
                        if !matches!(e.at_b, Endpoint::Arrow) {
                            graph
                                .set_marks(c, b, at_c, at_b)
                                .map_err(|err| OrientationError::Graph(err.to_string()))?;
                            delta.edges_changed += 1;
                            queue.push(c);
                            queue.push(b);
                        }
                    }
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// Propagate: if `a → b o-* c` and a,c nonadjacent, orient `b → c`.
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
        let n = graph.node_count();
        for i in 0..n {
            let b = DenseNodeId::from_raw(i as u32);
            let nbrs: Vec<_> = graph.neighbors(b).collect();
            for &(a, at_b_from_a, at_a) in &nbrs {
                // a → b : at_a Tail, at_b Arrow when viewed from a… use edge_between
                let Some(e_ab) = graph.edge_between(a, b) else {
                    continue;
                };
                let directed_ab = matches!(
                    (e_ab.at_a, e_ab.at_b),
                    (Endpoint::Tail, Endpoint::Arrow)
                );
                if !directed_ab {
                    let _ = (at_b_from_a, at_a);
                    continue;
                }
                for &(c, _, _) in &nbrs {
                    if c == a {
                        continue;
                    }
                    if graph.has_edge(a, c) {
                        continue;
                    }
                    let Some(e_bc) = graph.edge_between(b, c) else {
                        continue;
                    };
                    // b o-* c with circle at b
                    let mark_at_b = if e_bc.a == b { e_bc.at_a } else { e_bc.at_b };
                    let mark_at_c = if e_bc.a == c { e_bc.at_a } else { e_bc.at_b };
                    if !matches!(mark_at_b, Endpoint::Circle) {
                        continue;
                    }
                    // Orient b → c
                    let (at_b, at_c) = (Endpoint::Tail, mark_at_c);
                    // set from b's perspective
                    if e_bc.a == b {
                        graph
                            .set_marks(b, c, at_b, at_c)
                            .map_err(|err| OrientationError::Graph(err.to_string()))?;
                    } else {
                        graph
                            .set_marks(c, b, at_c, at_b)
                            .map_err(|err| OrientationError::Graph(err.to_string()))?;
                    }
                    delta.edges_changed += 1;
                    queue.push(b);
                    queue.push(c);
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// FCI R2: if `a → b o→ c` or `a o→ b → c`, and `a o-* c`, orient arrow into `c` on `a–c`.
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
        let n = graph.node_count();
        for i in 0..n {
            let b = DenseNodeId::from_raw(i as u32);
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
                    let case1 = matches!(at_a_ab, Endpoint::Tail)
                        && matches!(at_b_ab, Endpoint::Arrow)
                        && matches!(at_b_bc, Endpoint::Circle)
                        && matches!(at_c_bc, Endpoint::Arrow);
                    let case2 = matches!(at_a_ab, Endpoint::Circle)
                        && matches!(at_b_ab, Endpoint::Arrow)
                        && matches!(at_b_bc, Endpoint::Tail)
                        && matches!(at_c_bc, Endpoint::Arrow);
                    if !(case1 || case2) {
                        continue;
                    }
                    let Some((at_a_ac, at_c_ac)) = marks_between(graph, a, c) else {
                        continue;
                    };
                    if !matches!(at_a_ac, Endpoint::Circle) {
                        continue;
                    }
                    if matches!(at_c_ac, Endpoint::Arrow) {
                        continue; // already oriented
                    }
                    set_marks_oriented(graph, a, c, at_a_ac, Endpoint::Arrow)?;
                    delta.edges_changed += 1;
                    queue.push(a);
                    queue.push(c);
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
        let n = graph.node_count();
        for i in 0..n {
            let b = DenseNodeId::from_raw(i as u32);
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
                        if !matches!(at_d_ad, Endpoint::Circle) || !matches!(at_d_cd, Endpoint::Circle)
                        {
                            continue;
                        }
                        if !matches!(at_b_db, Endpoint::Circle) {
                            continue;
                        }
                        if matches!(at_d_db, Endpoint::Arrow) && matches!(at_b_db, Endpoint::Arrow) {
                            continue;
                        }
                        set_marks_oriented(graph, d, b, at_d_db, Endpoint::Arrow)?;
                        delta.edges_changed += 1;
                        queue.push(d);
                        queue.push(b);
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
    if e.a == a {
        Some((e.at_a, e.at_b))
    } else {
        Some((e.at_b, e.at_a))
    }
}

fn set_marks_oriented(
    graph: &mut TemporalPag,
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
) -> Result<(), OrientationError> {
    let Some(e) = graph.edge_between(a, b) else {
        return Err(OrientationError::Graph("missing edge in set_marks_oriented".into()));
    };
    if e.a == a {
        graph
            .set_marks(a, b, at_a, at_b)
            .map_err(|err| OrientationError::Graph(err.to_string()))
    } else {
        graph
            .set_marks(b, a, at_b, at_a)
            .map_err(|err| OrientationError::Graph(err.to_string()))
    }
}

/// Apply discriminating-path orientations.
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
        let paths = find_discriminating_paths(graph, 64, 8);
        for path in paths {
            if path.nodes.len() < 3 {
                continue;
            }
            let a = path.nodes[0];
            let c = path.nodes[path.nodes.len() - 2];
            let b = path.nodes[path.nodes.len() - 1];
            let Some(sep) = state.sepset(a, c) else {
                continue;
            };
            let b_in_sep = sep.iter().any(|&z| z == b);
            let collider = discriminating_implies_collider(&path, b_in_sep);
            let Some(e) = graph.edge_between(c, b) else {
                continue;
            };
            if collider {
                // c *→ b
                let at_c = if e.a == c { e.at_a } else { e.at_b };
                let at_b = Endpoint::Arrow;
                if e.a == c {
                    graph
                        .set_marks(c, b, at_c, at_b)
                        .map_err(|err| OrientationError::Graph(err.to_string()))?;
                } else {
                    graph
                        .set_marks(b, c, at_b, at_c)
                        .map_err(|err| OrientationError::Graph(err.to_string()))?;
                }
            } else {
                // c —* b with tail at c (non-collider)
                let at_c = Endpoint::Tail;
                let at_b = if e.a == b { e.at_a } else { e.at_b };
                if e.a == c {
                    graph
                        .set_marks(c, b, at_c, at_b)
                        .map_err(|err| OrientationError::Graph(err.to_string()))?;
                } else {
                    graph
                        .set_marks(b, c, at_b, at_c)
                        .map_err(|err| OrientationError::Graph(err.to_string()))?;
                }
            }
            delta.edges_changed += 1;
            queue.push(c);
            queue.push(b);
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// Schedule LPCMCI rules to a fixed point using a local delta queue.
///
/// # Errors
///
/// Rule application failures.
pub fn run_lpcmci_orientation(
    graph: &mut TemporalPag,
    rules: &[&dyn LpcmciOrientationRule],
    state: &mut OrientationState,
) -> Result<RuleDelta, OrientationError> {
    let mut queue = OrientationQueue::new();
    for i in 0..graph.node_count() {
        queue.push(DenseNodeId::from_raw(i as u32));
    }
    let mut total = RuleDelta::default();
    let mut rounds = 0u32;
    const MAX_ROUNDS: u32 = 10_000;
    while rounds < MAX_ROUNDS {
        rounds += 1;
        let mut any = false;
        for rule in rules {
            let d = rule.apply(graph, state, &mut queue)?;
            total.edges_changed += d.edges_changed;
            if d.edges_changed > 0 {
                any = true;
            }
        }
        if !any {
            total.fixed_point = true;
            break;
        }
        // Drain queue markers (rules already push affected locals).
        while queue.pop().is_some() {}
        for i in 0..graph.node_count() {
            queue.push(DenseNodeId::from_raw(i as u32));
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{Lag, VariableId};

    #[test]
    fn r2_orients_circle_into_arrow() {
        // a → b o→ c and a o-o c  ⇒  a o→ c
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
    fn rule_ids_cover_r1_r2_r3() {
        assert_eq!(LpcmciR1.id(), "lpcmci.r1");
        assert_eq!(LpcmciR2.id(), "lpcmci.r2");
        assert_eq!(LpcmciR3.id(), "lpcmci.r3");
    }
}

