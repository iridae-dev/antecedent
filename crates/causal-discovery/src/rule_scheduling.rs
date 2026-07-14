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
