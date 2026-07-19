//! FCI / LPCMCI orientation rule scheduling (DESIGN.md §13.6).
//!
//! Zhang FCI rules (collider, R1–R4, R8–R10) share [`PagOps`] bodies and run on
//! static [`Pag`] via [`FciOrientationRule`] or temporal [`TemporalPag`] via
//! [`LpcmciOrientationRule`]. APR/MMR stay temporal-only (middle marks + lag).
//!
//! Explicit scheduler module — rules are applied via a delta queue, not a
//! single procedural blob that rescans all edges after every change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names, clippy::similar_names)]

use std::collections::HashSet;

use causal_graph::{DenseNodeId, Endpoint, MiddleMark, NodeRef, Pag, TemporalPag};

use crate::discriminating_paths::{
    discriminating_implies_collider, find_discriminating_paths_with_budget,
};
use crate::orientation::{
    OrientationError, OrientationQueue, OrientationState, PagOps, RuleDelta,
};
use crate::uncovered_paths::{uncovered_pd_paths_with_budget, EndpointPattern};

/// Drain the orientation queue into a focus set, or scan all nodes when empty
/// (same contract as Meek [`crate::orientation`] rules).
fn focus_nodes<G: PagOps>(graph: &G, queue: &mut OrientationQueue) -> Vec<DenseNodeId> {
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
fn enqueue_local<G: PagOps>(graph: &G, id: DenseNodeId, queue: &mut OrientationQueue) {
    queue.push(id);
    for (n, _, _) in graph.neighbors(id) {
        queue.push(n);
    }
}

/// Named Zhang FCI orientation rule on a static [`Pag`].
pub trait FciOrientationRule {
    /// Rule id.
    fn id(&self) -> &'static str;

    /// Apply once; enqueue only locally affected nodes.
    ///
    /// # Errors
    ///
    /// Graph mutation failures.
    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError>;
}

/// Named LPCMCI orientation rule on a [`TemporalPag`].
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

fn marks_between<G: PagOps>(
    graph: &G,
    a: DenseNodeId,
    b: DenseNodeId,
) -> Option<(Endpoint, Endpoint)> {
    let e = graph.edge_between(a, b)?;
    if e.a == a { Some((e.at_a, e.at_b)) } else { Some((e.at_b, e.at_a)) }
}

fn set_marks_oriented<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    delta: &mut RuleDelta,
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
) -> Result<bool, OrientationError> {
    let Some(e) = graph.edge_between(a, b) else {
        return Err(OrientationError::msg("missing edge in set_marks_oriented"));
    };
    // pinned baseline: once an endpoint is `x`, further rules leave it alone.
    if matches!(e.at_a, Endpoint::Conflict) || matches!(e.at_b, Endpoint::Conflict) {
        return Ok(false);
    }
    let result = if e.a == a {
        graph.set_marks(a, b, at_a, at_b)
    } else {
        graph.set_marks(b, a, at_b, at_a)
    };
    match result {
        Ok(()) => Ok(true),
        Err(causal_graph::GraphError::Cycle { .. }) => {
            state.record_conflict(delta, a, b, "cycle");
            if graph.mark_conflict(a, b).is_ok() {
                delta.edges_changed += 1;
                delta.fixed_point = false;
            }
            Ok(false)
        }
        Err(err) => Err(OrientationError::from(err)),
    }
}

/// Set the mark at `at` on edge `{at, other}` to [`Endpoint::Arrow`], keeping the far mark.
fn set_arrow_at<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    delta: &mut RuleDelta,
    at: DenseNodeId,
    other: DenseNodeId,
) -> Result<bool, OrientationError> {
    let e = graph.edge_between(at, other).ok_or(OrientationError::Precondition {
        message: "discriminating path missing edge",
    })?;
    let at_other = if e.a == other { e.at_a } else { e.at_b };
    set_marks_oriented(graph, state, delta, at, other, Endpoint::Arrow, at_other)
}

fn apply_orient_collider<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    for b in focus {
        let nbrs: Vec<_> = graph.neighbors(b).into_iter().map(|(x, _, _)| x).collect();
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
                if let Some((at_a, at_b)) = marks_between(graph, a, b) {
                    if !matches!(at_b, Endpoint::Arrow)
                        && set_marks_oriented(
                            graph,
                            state,
                            &mut delta,
                            a,
                            b,
                            at_a,
                            Endpoint::Arrow,
                        )?
                    {
                        enqueue_local(graph, a, queue);
                        enqueue_local(graph, b, queue);
                        delta.edges_changed += 1;
                    }
                }
                if let Some((at_c, at_b)) = marks_between(graph, c, b) {
                    if !matches!(at_b, Endpoint::Arrow)
                        && set_marks_oriented(
                            graph,
                            state,
                            &mut delta,
                            c,
                            b,
                            at_c,
                            Endpoint::Arrow,
                        )?
                    {
                        enqueue_local(graph, c, queue);
                        enqueue_local(graph, b, queue);
                        delta.edges_changed += 1;
                    }
                }
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_r1<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    for b in focus {
        let nbrs: Vec<_> = graph.neighbors(b).into_iter().map(|(x, _, _)| x).collect();
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
                if set_marks_oriented(
                    graph,
                    state,
                    &mut delta,
                    b,
                    c,
                    Endpoint::Tail,
                    Endpoint::Arrow,
                )? {
                    delta.edges_changed += 1;
                    enqueue_local(graph, b, queue);
                    enqueue_local(graph, c, queue);
                }
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_r2<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    for b in focus {
        let nbrs: Vec<_> = graph.neighbors(b).into_iter().map(|(x, _, _)| x).collect();
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
                if set_marks_oriented(graph, state, &mut delta, a, c, at_a_ac, Endpoint::Arrow)? {
                    delta.edges_changed += 1;
                    enqueue_local(graph, a, queue);
                    enqueue_local(graph, c, queue);
                }
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_r3<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    let n = graph.node_count();
    for b in focus {
        let nbrs: Vec<_> = graph.neighbors(b).into_iter().map(|(x, _, _)| x).collect();
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
                // Find θ = d with circles at a, c, b: d *–o a, d *–o c, d *–o b.
                for j in 0..n {
                    let d = DenseNodeId::from_raw(j as u32);
                    if d == a || d == b || d == c {
                        continue;
                    }
                    let Some((at_a_ad, _)) = marks_between(graph, a, d) else {
                        continue;
                    };
                    let Some((at_c_cd, _)) = marks_between(graph, c, d) else {
                        continue;
                    };
                    let Some((at_d_db, at_b_db)) = marks_between(graph, d, b) else {
                        continue;
                    };
                    if !matches!(at_a_ad, Endpoint::Circle)
                        || !matches!(at_c_cd, Endpoint::Circle)
                        || !matches!(at_b_db, Endpoint::Circle)
                    {
                        continue;
                    }
                    if set_marks_oriented(
                        graph,
                        state,
                        &mut delta,
                        d,
                        b,
                        at_d_db,
                        Endpoint::Arrow,
                    )? {
                        delta.edges_changed += 1;
                        enqueue_local(graph, d, queue);
                        enqueue_local(graph, b, queue);
                    }
                }
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_discriminating_path<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
    rule_id: &'static str,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    let focus_set: HashSet<u32> = focus.iter().map(|n| n.raw()).collect();
    let (paths, truncated) = find_discriminating_paths_with_budget(graph, 64, 8);
    if truncated {
        return Err(OrientationError::SearchBudgetExhausted {
            rule: rule_id,
            max_paths: 64,
            max_len: 8,
        });
    }
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
            let mut changed = false;
            if set_arrow_at(graph, state, &mut delta, c, d_k)? {
                changed = true;
            }
            if set_arrow_at(graph, state, &mut delta, c, b)? {
                changed = true;
            }
            if !changed {
                continue;
            }
        } else if set_marks_oriented(
            graph,
            state,
            &mut delta,
            c,
            b,
            Endpoint::Tail,
            Endpoint::Arrow,
        )? {
            // c → b (non-collider at c).
        } else {
            continue;
        }
        delta.edges_changed += 1;
        enqueue_local(graph, c, queue);
        enqueue_local(graph, b, queue);
        enqueue_local(graph, d_k, queue);
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_r8<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    for b in focus {
        let nbrs: Vec<_> = graph.neighbors(b).into_iter().map(|(x, _, _)| x).collect();
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
                // a → b required; b → c or b o→ c.
                if !matches!(at_a_ab, Endpoint::Tail) || !matches!(at_b_ab, Endpoint::Arrow) {
                    continue;
                }
                if !matches!(at_c_bc, Endpoint::Arrow)
                    || !matches!(at_b_bc, Endpoint::Tail | Endpoint::Circle)
                {
                    continue;
                }
                let Some((at_a_ac, _)) = marks_between(graph, a, c) else {
                    continue;
                };
                if !matches!(at_a_ac, Endpoint::Circle) {
                    continue;
                }
                if set_marks_oriented(
                    graph,
                    state,
                    &mut delta,
                    a,
                    c,
                    Endpoint::Tail,
                    Endpoint::Arrow,
                )? {
                    delta.edges_changed += 1;
                    enqueue_local(graph, a, queue);
                    enqueue_local(graph, c, queue);
                }
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_r9<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
    rule_id: &'static str,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    for a in focus {
        let nbrs: Vec<_> = graph.neighbors(a).into_iter().map(|(x, _, _)| x).collect();
        for &c in &nbrs {
            let Some((at_a_ac, at_c_ac)) = marks_between(graph, a, c) else {
                continue;
            };
            if !matches!(at_a_ac, Endpoint::Circle) || !matches!(at_c_ac, Endpoint::Arrow) {
                continue;
            }
            // Need weakly-minimal sepset involving some B1 with a in Sep(B1, c).
            for &b1 in &nbrs {
                if b1 == c || graph.has_edge(b1, c) {
                    continue;
                }
                let Some(sep) = state.sepset(b1, c) else {
                    continue;
                };
                if !sep.iter().any(|&z| z == a) {
                    continue;
                }
                if !state.is_weakly_minimal(b1, c) && !sep.is_empty() {
                    // Prefer WM; allow empty/singleton sepsets as trivially WM.
                    if sep.len() > 1 {
                        continue;
                    }
                }
                let initial = [
                    EndpointPattern::circle_circle(),
                    EndpointPattern::circle_arrow(),
                    EndpointPattern::directed(),
                ];
                let (paths, truncated) =
                    uncovered_pd_paths_with_budget(graph, b1, c, &initial, 8, 8);
                if truncated {
                    return Err(OrientationError::SearchBudgetExhausted {
                        rule: rule_id,
                        max_paths: 8,
                        max_len: 8,
                    });
                }
                let qualifies = paths.iter().any(|path| {
                    path.len() >= 3 && !path.contains(&a) && !graph.has_edge(a, path[1])
                });
                if !qualifies {
                    continue;
                }
                if set_marks_oriented(
                    graph,
                    state,
                    &mut delta,
                    a,
                    c,
                    Endpoint::Tail,
                    Endpoint::Arrow,
                )? {
                    delta.edges_changed += 1;
                    enqueue_local(graph, a, queue);
                    enqueue_local(graph, c, queue);
                    break;
                }
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

fn apply_r10<G: PagOps>(
    graph: &mut G,
    state: &mut OrientationState,
    queue: &mut OrientationQueue,
    rule_id: &'static str,
) -> Result<RuleDelta, OrientationError> {
    let mut delta = RuleDelta::default();
    let focus = focus_nodes(graph, queue);
    for c in focus {
        let nbrs: Vec<_> = graph.neighbors(c).into_iter().map(|(x, _, _)| x).collect();
        // `a o→ c` candidates and parents into c with arrow at c and tail/circle at parent.
        let mut o_arrow_into: Vec<DenseNodeId> = Vec::new();
        let mut parents_into: Vec<DenseNodeId> = Vec::new();
        for &n in &nbrs {
            let Some((at_n, at_c)) = marks_between(graph, n, c) else {
                continue;
            };
            if matches!(at_c, Endpoint::Arrow) && matches!(at_n, Endpoint::Circle) {
                o_arrow_into.push(n);
            }
            if matches!(at_c, Endpoint::Arrow)
                && matches!(at_n, Endpoint::Tail | Endpoint::Circle)
            {
                parents_into.push(n);
            }
        }
        let initial = [
            EndpointPattern::circle_circle(),
            EndpointPattern::circle_arrow(),
            EndpointPattern::directed(),
        ];
        for &a in &o_arrow_into {
            // Need two distinct parents (≠ a) with node-disjoint uncovered PD paths from a.
            let mut path_for_parent: Vec<(DenseNodeId, Vec<DenseNodeId>)> = Vec::new();
            for &p in &parents_into {
                if p == a {
                    continue;
                }
                let (paths, truncated) =
                    uncovered_pd_paths_with_budget(graph, a, p, &initial, 8, 8);
                if truncated {
                    return Err(OrientationError::SearchBudgetExhausted {
                        rule: rule_id,
                        max_paths: 8,
                        max_len: 8,
                    });
                }
                if let Some(path) = paths.into_iter().find(|p| p.len() >= 3 && !p.contains(&c)) {
                    path_for_parent.push((p, path));
                }
            }
            let mut found_pair = false;
            'pair: for i in 0..path_for_parent.len() {
                for j in (i + 1)..path_for_parent.len() {
                    let (p1, ref path1) = path_for_parent[i];
                    let (p2, ref path2) = path_for_parent[j];
                    if p1 == p2 {
                        continue;
                    }
                    // Node-disjoint except shared endpoint `a` (and possibly the two parents).
                    if paths_node_disjoint_except_a(path1, path2, a) {
                        found_pair = true;
                        break 'pair;
                    }
                }
            }
            if !found_pair {
                continue;
            }
            if set_marks_oriented(
                graph,
                state,
                &mut delta,
                a,
                c,
                Endpoint::Tail,
                Endpoint::Arrow,
            )? {
                delta.edges_changed += 1;
                enqueue_local(graph, a, queue);
                enqueue_local(graph, c, queue);
            }
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

/// Two uncovered PD paths are node-disjoint except at the shared start `a`.
fn paths_node_disjoint_except_a(
    path1: &[DenseNodeId],
    path2: &[DenseNodeId],
    a: DenseNodeId,
) -> bool {
    for &n in path1 {
        if n == a {
            continue;
        }
        if path2.iter().any(|&m| m == n) {
            return false;
        }
    }
    true
}

/// Orient unshielded colliders from sepsets (circle→arrow into middle).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciOrientCollider;

impl FciOrientationRule for LpcmciOrientCollider {
    fn id(&self) -> &'static str {
        "fci.orient_collider"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_orient_collider(graph, state, queue)
    }
}

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
        apply_orient_collider(graph, state, queue)
    }
}

/// FCI R1: if `a *→ b o–* c` (arrow at b on a–b; circle at b on b–c) and a,c nonadjacent,
/// orient `b → c` (tail at b and arrow at c).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR1;

impl FciOrientationRule for LpcmciR1 {
    fn id(&self) -> &'static str {
        "fci.r1"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r1(graph, state, queue)
    }
}

impl LpcmciOrientationRule for LpcmciR1 {
    fn id(&self) -> &'static str {
        "lpcmci.r1"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r1(graph, state, queue)
    }
}

/// FCI R2: if `a → b *→ c` or `a *→ b → c`, and `a *–o c` (circle **at c**), orient
/// arrow at c on a–c. Never overwrites a Tail at c.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR2;

impl FciOrientationRule for LpcmciR2 {
    fn id(&self) -> &'static str {
        "fci.r2"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r2(graph, state, queue)
    }
}

impl LpcmciOrientationRule for LpcmciR2 {
    fn id(&self) -> &'static str {
        "lpcmci.r2"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r2(graph, state, queue)
    }
}

/// FCI R3: collider `a *→ b ←* c` with nonadjacent a,c and `d *–o a`, `d *–o c`, `d *–o b`
/// → orient `d *→ b` (Zhang: circles at a, c, and b — not at d).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR3;

impl FciOrientationRule for LpcmciR3 {
    fn id(&self) -> &'static str {
        "fci.r3"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r3(graph, state, queue)
    }
}

impl LpcmciOrientationRule for LpcmciR3 {
    fn id(&self) -> &'static str {
        "lpcmci.r3"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r3(graph, state, queue)
    }
}

/// Apply discriminating-path orientations (Zhang FCI R4).
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciDiscriminatingPathRule;

impl FciOrientationRule for LpcmciDiscriminatingPathRule {
    fn id(&self) -> &'static str {
        "fci.discriminating_path"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_discriminating_path(graph, state, queue, "fci.discriminating_path")
    }
}

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
        apply_discriminating_path(graph, state, queue, "lpcmci.discriminating_path")
    }
}

/// FCI R8′: `a → b → c` or `a → b o→ c`, and `a o–* c` → orient `a → c`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR8;

impl FciOrientationRule for LpcmciR8 {
    fn id(&self) -> &'static str {
        "fci.r8"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r8(graph, state, queue)
    }
}

impl LpcmciOrientationRule for LpcmciR8 {
    fn id(&self) -> &'static str {
        "lpcmci.r8"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r8(graph, state, queue)
    }
}

/// FCI R9′: uncovered PD path from neighbor of `a` to `c` with `a o→ c` → `a → c`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR9;

impl FciOrientationRule for LpcmciR9 {
    fn id(&self) -> &'static str {
        "fci.r9"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r9(graph, state, queue, "fci.r9")
    }
}

impl LpcmciOrientationRule for LpcmciR9 {
    fn id(&self) -> &'static str {
        "lpcmci.r9"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r9(graph, state, queue, "lpcmci.r9")
    }
}

/// FCI R10′: `a o→ c` with **two node-disjoint** uncovered PD paths into two
/// distinct parents of `c` (`→` or `o→`) → orient `a → c`.
///
/// A single path into one parent is not sufficient (Zhang 2008 R10′); the prior
/// one-path rule could over-orient.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciR10;

impl FciOrientationRule for LpcmciR10 {
    fn id(&self) -> &'static str {
        "fci.r10"
    }

    fn apply(
        &self,
        graph: &mut Pag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r10(graph, state, queue, "fci.r10")
    }
}

impl LpcmciOrientationRule for LpcmciR10 {
    fn id(&self) -> &'static str {
        "lpcmci.r10"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        apply_r10(graph, state, queue, "lpcmci.r10")
    }
}

/// Lag of a temporal node (`None` if missing). Smaller lag = later in time.
fn node_lag(graph: &TemporalPag, id: DenseNodeId) -> Option<u32> {
    match graph.nodes().get(id.as_usize())? {
        NodeRef::Lagged { lag, .. } => Some(lag.raw()),
        NodeRef::Static(_) | NodeRef::Context { .. } => Some(0),
    }
}

/// Whether `a` is strictly later in time than `b` (smaller lag).
fn is_later(graph: &TemporalPag, a: DenseNodeId, b: DenseNodeId) -> bool {
    match (node_lag(graph, a), node_lag(graph, b)) {
        (Some(la), Some(lb)) => la < lb,
        _ => false,
    }
}

/// Ancestor–parent rule (Lemma 1): clear middle marks on definite directed parents.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciApr;

impl LpcmciOrientationRule for LpcmciApr {
    fn id(&self) -> &'static str {
        "lpcmci.apr"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        for a in focus {
            let nbrs: Vec<_> = graph.neighbors(a).map(|(x, _, _)| x).collect();
            for &b in &nbrs {
                let Some(e) = graph.edge_between(a, b) else {
                    continue;
                };
                let (at_a, at_b, mid) = if e.a == a {
                    (e.at_a, e.at_b, e.middle)
                } else {
                    (e.at_b, e.at_a, e.middle)
                };
                // Definite a → b with non-empty middle that APR clears.
                if !matches!(at_a, Endpoint::Tail) || !matches!(at_b, Endpoint::Arrow) {
                    continue;
                }
                let clear = match mid {
                    MiddleMark::Both => true,
                    MiddleMark::Left | MiddleMark::Right => {
                        // Left = later-endpoint parent search; clear when a is later than b.
                        // Right = earlier-endpoint parent search; clear when a is earlier than b.
                        (matches!(mid, MiddleMark::Left) && is_later(graph, a, b))
                            || (matches!(mid, MiddleMark::Right) && is_later(graph, b, a))
                    }
                    _ => false,
                };
                if clear {
                    graph.set_middle(a, b, MiddleMark::Empty)?;
                    delta.edges_changed += 1;
                    enqueue_local(graph, a, queue);
                    enqueue_local(graph, b, queue);
                }
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// Middle-mark rule (MMR): on `*→` with `?`, set L/R by node order.
#[derive(Clone, Copy, Debug, Default)]
pub struct LpcmciMmr;

impl LpcmciOrientationRule for LpcmciMmr {
    fn id(&self) -> &'static str {
        "lpcmci.mmr"
    }

    fn apply(
        &self,
        graph: &mut TemporalPag,
        _state: &mut OrientationState,
        queue: &mut OrientationQueue,
    ) -> Result<RuleDelta, OrientationError> {
        let mut delta = RuleDelta::default();
        let focus = focus_nodes(graph, queue);
        for a in focus {
            let nbrs: Vec<_> = graph.neighbors(a).map(|(x, _, _)| x).collect();
            for &b in &nbrs {
                if a.raw() > b.raw() {
                    continue; // each edge once
                }
                let Some(e) = graph.edge_between(a, b) else {
                    continue;
                };
                if !matches!(e.middle, MiddleMark::Unknown) {
                    continue;
                }
                // Arrow at either end → apply L/R by time order (Left = later endpoint).
                let has_arrow =
                    matches!(e.at_a, Endpoint::Arrow) || matches!(e.at_b, Endpoint::Arrow);
                if !has_arrow {
                    continue;
                }
                let mark = if is_later(graph, a, b) {
                    // a later than b: arrow into a → Left; else Right.
                    if matches!(
                        if e.a == a { e.at_a } else { e.at_b },
                        Endpoint::Arrow
                    ) {
                        MiddleMark::Left
                    } else {
                        MiddleMark::Right
                    }
                } else if is_later(graph, b, a) {
                    if matches!(
                        if e.a == b { e.at_a } else { e.at_b },
                        Endpoint::Arrow
                    ) {
                        MiddleMark::Left
                    } else {
                        MiddleMark::Right
                    }
                } else {
                    // Contemporaneous: default Left (matches lagged-edge init convention).
                    MiddleMark::Left
                };
                graph.apply_middle(a, b, mark)?;
                delta.edges_changed += 1;
                enqueue_local(graph, a, queue);
                enqueue_local(graph, b, queue);
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }
}

/// Default Zhang FCI orientation rule list (collider + R1–R4 + R8–R10).
#[must_use]
pub fn default_fci_rules() -> [&'static dyn FciOrientationRule; 8] {
    [
        &LpcmciOrientCollider,
        &LpcmciR1,
        &LpcmciR2,
        &LpcmciR3,
        &LpcmciDiscriminatingPathRule,
        &LpcmciR8,
        &LpcmciR9,
        &LpcmciR10,
    ]
}

/// Default LPCMCI orientation rule list (collider + R1–R4 + R8–R10 + APR + MMR).
#[must_use]
pub fn default_lpcmci_rules() -> [&'static dyn LpcmciOrientationRule; 10] {
    [
        &LpcmciOrientCollider,
        &LpcmciR1,
        &LpcmciR2,
        &LpcmciR3,
        &LpcmciDiscriminatingPathRule,
        &LpcmciR8,
        &LpcmciR9,
        &LpcmciR10,
        &LpcmciApr,
        &LpcmciMmr,
    ]
}

/// Schedule Zhang FCI rules to a fixed point on a static [`Pag`].
///
/// Seeds all nodes once. Subsequent rounds honor nodes enqueued by rules
/// (DESIGN.md §13.6 / §13.8) — no full-graph re-seed after each round.
///
/// # Errors
///
/// Rule application failures.
pub fn run_fci_orientation_to_fixed_point(
    graph: &mut Pag,
    rules: &[&dyn FciOrientationRule],
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
            total.conflicts += d.conflicts;
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
            total.conflicts += d.conflicts;
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
#[path = "rule_scheduling_tests.rs"]
mod tests;
