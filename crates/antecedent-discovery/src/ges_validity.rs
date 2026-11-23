// SPDX-License-Identifier: MIT OR Apache-2.0
//! Validity conditions for GES insert and delete operators.

use std::collections::{HashSet, VecDeque};

use antecedent_graph::{Cpdag, DenseNodeId, Endpoint};

use super::Footprint;

pub(super) fn na_yx(cpdag: &Cpdag, y: DenseNodeId, x: DenseNodeId) -> Vec<DenseNodeId> {
    let mut out: Vec<_> =
        cpdag.undirected_neighbors(y).into_iter().filter(|&n| cpdag.has_edge(n, x)).collect();
    out.sort_unstable_by_key(|node| node.raw());
    out
}

pub(super) fn is_clique(cpdag: &Cpdag, nodes: &[DenseNodeId]) -> bool {
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !cpdag.has_edge(nodes[i], nodes[j]) {
                return false;
            }
        }
    }
    true
}

/// Semi-directed reachability: undirected either way; directed only forward.
///
/// Every node whose edges the search reads (each node it expands) is added to `footprint`.
fn semi_directed_reaches_avoiding(
    cpdag: &Cpdag,
    from: DenseNodeId,
    to: DenseNodeId,
    avoid: &HashSet<DenseNodeId>,
    mut footprint: Option<&mut Footprint>,
) -> bool {
    if from == to {
        return true;
    }
    let n = cpdag.node_count();
    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    q.push_back(from);
    seen[from.as_usize()] = true;
    while let Some(u) = q.pop_front() {
        if let Some(footprint) = footprint.as_deref_mut() {
            footprint.add(u);
        }
        for v in cpdag.children(u) {
            if avoid.contains(&v) && v != to {
                continue;
            }
            if v == to {
                return true;
            }
            if !seen[v.as_usize()] {
                seen[v.as_usize()] = true;
                q.push_back(v);
            }
        }
        for v in cpdag.undirected_neighbors(u) {
            if avoid.contains(&v) && v != to {
                continue;
            }
            if v == to {
                return true;
            }
            if !seen[v.as_usize()] {
                seen[v.as_usize()] = true;
                q.push_back(v);
            }
        }
    }
    false
}

#[cfg(test)]
pub(super) fn insert_valid(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    t: &[DenseNodeId],
) -> bool {
    if cpdag.has_edge(x, y) || x == y {
        return false;
    }
    insert_valid_given(cpdag, x, y, &na_yx(cpdag, y, x), t, None)
}

/// Insert validity for an `x`, `y` known to be non-adjacent, with `NA(Y,X)` already computed.
pub(super) fn insert_valid_given(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    na: &[DenseNodeId],
    t: &[DenseNodeId],
    footprint: Option<&mut Footprint>,
) -> bool {
    let mut clique_nodes = na.to_vec();
    clique_nodes.extend_from_slice(t);
    if !is_clique(cpdag, &clique_nodes) {
        return false;
    }
    let avoid: HashSet<DenseNodeId> = clique_nodes.into_iter().collect();
    // Every semi-directed path Y ↝ X hits NA∪T ⇔ no path avoiding NA∪T.
    !semi_directed_reaches_avoiding(cpdag, y, x, &avoid, footprint)
}

pub(super) fn delete_valid(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    h: &[DenseNodeId],
) -> bool {
    let Some(e) = cpdag.edge_between(x, y) else {
        return false;
    };
    // Edge must be X→Y or X—Y (not Y→X alone as X,Y ordered for operator).
    let xy_directed = matches!(
        (e.at_a, e.at_b),
        (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
    ) && e.parent_child() == Some((x, y));
    let undirected = e.is_undirected();
    if !xy_directed && !undirected {
        return false;
    }
    let na = na_yx(cpdag, y, x);
    let h_set: HashSet<DenseNodeId> = h.iter().copied().collect();
    if h.iter().any(|n| !na.contains(n)) {
        return false;
    }
    let rest: Vec<DenseNodeId> = na.into_iter().filter(|n| !h_set.contains(n)).collect();
    is_clique(cpdag, &rest)
}
