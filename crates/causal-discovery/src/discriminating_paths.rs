//! Discriminating paths for FCI / LPCMCI PAG orientation.
//!
//! Zhang (2008) FCI R4: a path ⟨a, d₁, …, dₖ, c, b⟩ (k ≥ 1) is discriminating for `c`
//! between `a` and `b` when `a` and `b` are non-adjacent, every intermediate `dᵢ` is a
//! collider on the path and a definite parent of `b`, and the edge `c *-* b` has a circle
//! at `c`. Then if `c ∈ Sep(a,b)` orient `c → b`; otherwise orient `dₖ *→ c ←* b`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]

use causal_graph::{DenseNodeId, Endpoint, MarkedEdge};

use crate::orientation::PagOps;

/// A discriminating path `⟨a, …, c, b⟩` used by FCI / LPCMCI orientation rules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscriminatingPath {
    /// Path nodes from `a` to `b` inclusive (`len >= 4`).
    pub nodes: Vec<DenseNodeId>,
}

impl DiscriminatingPath {
    /// Endpoint `a` (first).
    #[must_use]
    pub fn a(&self) -> DenseNodeId {
        self.nodes[0]
    }

    /// Discriminated node `c` (second-to-last).
    #[must_use]
    pub fn c(&self) -> DenseNodeId {
        self.nodes[self.nodes.len() - 2]
    }

    /// Endpoint `b` (last).
    #[must_use]
    pub fn b(&self) -> DenseNodeId {
        self.nodes[self.nodes.len() - 1]
    }

    /// Predecessor `dₖ` of `c` on the path.
    #[must_use]
    pub fn d_k(&self) -> DenseNodeId {
        self.nodes[self.nodes.len() - 3]
    }
}

/// Find discriminating paths ending at edge `{c,b}` with a circle at `c`, bounded.
///
/// Returns `(paths, truncated)` when `max_paths` stopped further enumeration.
#[must_use]
pub fn find_discriminating_paths_with_budget<G: PagOps>(
    pag: &G,
    max_paths: usize,
    max_len: usize,
) -> (Vec<DiscriminatingPath>, bool) {
    let mut out = Vec::new();
    let mut truncated = false;
    if max_paths == 0 || max_len < 4 {
        return (out, false);
    }
    let n = pag.node_count();
    for i in 0..n {
        let b = DenseNodeId::from_raw(i as u32);
        for (c, _at_b, at_c) in pag.neighbors(b) {
            if !matches!(at_c, Endpoint::Circle) {
                continue;
            }
            // Grow prefixes ending at `c`; intermediates must be parents of `b`.
            let mut stack = vec![vec![c]];
            while let Some(path_to_c) = stack.pop() {
                if out.len() >= max_paths {
                    truncated = true;
                    return (out, truncated);
                }
                // Try to complete with endpoint `a` once we have ≥1 intermediate.
                if path_to_c.len() >= 2 {
                    let head = path_to_c[0];
                    for (a, _, _) in pag.neighbors(head) {
                        if out.len() >= max_paths {
                            truncated = true;
                            return (out, truncated);
                        }
                        if a == b || path_to_c.contains(&a) || pag.has_edge(a, b) {
                            continue;
                        }
                        let mut full = Vec::with_capacity(path_to_c.len() + 2);
                        full.push(a);
                        full.extend_from_slice(&path_to_c);
                        full.push(b);
                        if full.len() > max_len {
                            continue;
                        }
                        if is_discriminating_path(pag, &full) {
                            out.push(DiscriminatingPath { nodes: full });
                        }
                    }
                }
                // Extend leftward with another intermediate (parent of `b`).
                // Full path needs +2 for `a` and `b`.
                if path_to_c.len() + 2 >= max_len {
                    continue;
                }
                let head = path_to_c[0];
                for (pred, _, _) in pag.neighbors(head) {
                    if pred == b || path_to_c.contains(&pred) {
                        continue;
                    }
                    if !is_definite_parent(pag, pred, b) {
                        continue;
                    }
                    if path_to_c.len() >= 2 {
                        // `head` is already an intermediate: must stay a collider under prepend.
                        if !is_definite_parent(pag, head, b) {
                            continue;
                        }
                        if !is_collider_at(pag, head, pred, path_to_c[1]) {
                            continue;
                        }
                    }
                    let mut next = Vec::with_capacity(path_to_c.len() + 1);
                    next.push(pred);
                    next.extend_from_slice(&path_to_c);
                    stack.push(next);
                }
            }
        }
    }
    (out, truncated)
}

/// Find discriminating paths ending at edge `{c,b}` with a circle at `c`, bounded.
#[must_use]
pub fn find_discriminating_paths<G: PagOps>(
    pag: &G,
    max_paths: usize,
    max_len: usize,
) -> Vec<DiscriminatingPath> {
    find_discriminating_paths_with_budget(pag, max_paths, max_len).0
}

/// Whether `path` is a Zhang discriminating path for `c = path[n-2]` between `a` and `b`.
#[must_use]
pub fn is_discriminating_path<G: PagOps>(pag: &G, path: &[DenseNodeId]) -> bool {
    if path.len() < 4 {
        return false;
    }
    let a = path[0];
    let b = path[path.len() - 1];
    let c = path[path.len() - 2];
    if pag.has_edge(a, b) {
        return false;
    }
    let Some(e_cb) = pag.edge_between(c, b) else {
        return false;
    };
    if !matches!(mark_at(&e_cb, c), Endpoint::Circle) {
        return false;
    }
    // Every intermediate dᵢ (indices 1..len-3) is a collider on the path and parent of b.
    for i in 1..path.len() - 2 {
        let pred = path[i - 1];
        let v = path[i];
        let succ = path[i + 1];
        if !is_collider_at(pag, v, pred, succ) {
            return false;
        }
        if !is_definite_parent(pag, v, b) {
            return false;
        }
    }
    // Edges along the path must exist (collider checks cover intermediates; check a–d₁).
    if pag.edge_between(path[0], path[1]).is_none() {
        return false;
    }
    true
}

/// R4 collider branch: `c ∉ Sep(a,b)` ⇒ orient as collider at `c`.
///
/// `c_in_sepset_ab` is whether **`c ∈ Sep(a,b)`** (the non-adjacent endpoints).
#[must_use]
pub fn discriminating_implies_collider(c_in_sepset_ab: bool) -> bool {
    !c_in_sepset_ab
}

fn mark_at(edge: &MarkedEdge, node: DenseNodeId) -> Endpoint {
    if edge.a == node {
        edge.at_a
    } else {
        edge.at_b
    }
}

fn arrow_into<G: PagOps>(pag: &G, into: DenseNodeId, from: DenseNodeId) -> bool {
    pag.edge_between(into, from).is_some_and(|e| matches!(mark_at(&e, into), Endpoint::Arrow))
}

fn is_collider_at<G: PagOps>(
    pag: &G,
    v: DenseNodeId,
    pred: DenseNodeId,
    succ: DenseNodeId,
) -> bool {
    arrow_into(pag, v, pred) && arrow_into(pag, v, succ)
}

/// Definite directed parent: `parent → child` (tail at parent, arrow at child).
fn is_definite_parent<G: PagOps>(pag: &G, parent: DenseNodeId, child: DenseNodeId) -> bool {
    pag.edge_between(parent, child).is_some_and(|e| {
        matches!(mark_at(&e, parent), Endpoint::Tail) && matches!(mark_at(&e, child), Endpoint::Arrow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{Lag, VariableId};
    use causal_graph::{Pag, TemporalPag};

    /// Zhang minimal discriminating path ⟨a, d, c, b⟩ at lag 0.
    fn zhang_minimal() -> (TemporalPag, DenseNodeId, DenseNodeId, DenseNodeId, DenseNodeId) {
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(0)).unwrap();
        let d = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(0)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(0)).unwrap();
        // a → d ← c (collider at d), d → b, c o→ b
        g.insert_directed(a, d).unwrap();
        g.insert_directed(c, d).unwrap();
        g.insert_directed(d, b).unwrap();
        g.insert_circle_arrow(c, b).unwrap();
        (g, a, d, c, b)
    }

    fn zhang_minimal_static() -> (Pag, DenseNodeId, DenseNodeId, DenseNodeId, DenseNodeId) {
        let mut g = Pag::with_variables(4);
        let a = DenseNodeId::from_raw(0);
        let d = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        g.insert_directed(a, d).unwrap();
        g.insert_directed(c, d).unwrap();
        g.insert_directed(d, b).unwrap();
        g.insert_circle_arrow(c, b).unwrap();
        (g, a, d, c, b)
    }

    #[test]
    fn finds_zhang_minimal_discriminating_path() {
        let (g, a, d, c, b) = zhang_minimal();
        let paths = find_discriminating_paths(&g, 16, 8);
        assert!(
            paths.iter().any(|p| p.nodes == [a, d, c, b]),
            "paths={paths:?}"
        );
    }

    #[test]
    fn finds_zhang_minimal_on_static_pag() {
        let (g, a, d, c, b) = zhang_minimal_static();
        let paths = find_discriminating_paths(&g, 16, 8);
        assert!(
            paths.iter().any(|p| p.nodes == [a, d, c, b]),
            "paths={paths:?}"
        );
    }

    #[test]
    fn rejects_non_discriminating_directed_chain() {
        // Old buggy finder accepted a → c o→ b; that is not discriminating (no intermediate).
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let c = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(0)).unwrap();
        g.insert_directed(a, c).unwrap();
        g.insert_circle_arrow(c, b).unwrap();
        let paths = find_discriminating_paths(&g, 16, 8);
        assert!(paths.is_empty(), "spurious paths={paths:?}");
    }

    #[test]
    fn rejects_when_a_adjacent_to_b() {
        let (mut g, a, _d, _c, b) = zhang_minimal();
        g.insert_circle_arrow(a, b).unwrap();
        let paths = find_discriminating_paths(&g, 16, 8);
        assert!(paths.is_empty());
    }

    #[test]
    fn collider_implication_uses_c_in_sep_ab() {
        // c ∈ Sep(a,b) ⇒ non-collider; c ∉ Sep(a,b) ⇒ collider.
        assert!(!discriminating_implies_collider(true));
        assert!(discriminating_implies_collider(false));
    }
}
