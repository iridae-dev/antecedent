//! Possible-D-Sep sets for classic FCI adjacency (Spirtes et al.).
//!
//! After the PC skeleton and unshielded-collider orientation, FCI removes further
//! edges by testing CI given subsets of Possible-D-Sep. A node `V` is in
//! Possible-D-Sep(`A`,`B`) when there is a path from `A` to `V` such that every
//! consecutive triple ⟨X,Y,Z⟩ has Y a collider on the path or X–Z adjacent
//! (triangle).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::collections::{HashSet, VecDeque};

use causal_graph::{DenseNodeId, Endpoint};

use crate::orientation::PagOps;

/// Whether `mid` is a collider on the path through `left`–`mid`–`right`
/// (arrow into `mid` from both sides).
#[must_use]
pub fn is_collider_on_path<G: PagOps>(
    graph: &G,
    left: DenseNodeId,
    mid: DenseNodeId,
    right: DenseNodeId,
) -> bool {
    arrow_into(graph, mid, left) && arrow_into(graph, mid, right)
}

fn arrow_into<G: PagOps>(graph: &G, into: DenseNodeId, from: DenseNodeId) -> bool {
    let Some(e) = graph.edge_between(into, from) else {
        return false;
    };
    let at_into = if e.a == into { e.at_a } else { e.at_b };
    matches!(at_into, Endpoint::Arrow)
}

/// Whether the consecutive triple may appear on a Possible-D-Sep path.
#[must_use]
pub fn pds_triple_ok<G: PagOps>(
    graph: &G,
    left: DenseNodeId,
    mid: DenseNodeId,
    right: DenseNodeId,
) -> bool {
    is_collider_on_path(graph, left, mid, right) || graph.has_edge(left, right)
}

/// Compute Possible-D-Sep(`from`, `wrt`) on a (partially oriented) PAG.
///
/// Excludes `from` and `wrt`. Neighbors of `from` are always included when the
/// single-edge path has no intermediate triple (vacuous condition).
///
/// `max_nodes` bounds the BFS frontier expansions (fail-closed when exceeded).
///
/// # Errors
///
/// Returns `Err(truncated)` when the search budget is exhausted before completion.
pub fn possible_d_sep<G: PagOps>(
    graph: &G,
    from: DenseNodeId,
    wrt: DenseNodeId,
    max_nodes: usize,
) -> Result<Vec<DenseNodeId>, PossibleDSepBudget> {
    let mut out: HashSet<u32> = HashSet::new();
    // Queue entries: (current, predecessor on the path from `from`).
    let mut queue: VecDeque<(DenseNodeId, DenseNodeId)> = VecDeque::new();
    let mut enqueued: HashSet<(u32, u32)> = HashSet::new();
    let mut expansions = 0usize;

    for (nbr, _, _) in graph.neighbors(from) {
        if nbr == wrt {
            continue;
        }
        out.insert(nbr.raw());
        let key = (nbr.raw(), from.raw());
        if enqueued.insert(key) {
            queue.push_back((nbr, from));
        }
    }

    while let Some((cur, pred)) = queue.pop_front() {
        expansions += 1;
        if expansions > max_nodes {
            return Err(PossibleDSepBudget { max_nodes });
        }
        for (next, _, _) in graph.neighbors(cur) {
            if next == from || next == pred {
                continue;
            }
            if !pds_triple_ok(graph, pred, cur, next) {
                continue;
            }
            if next != wrt {
                out.insert(next.raw());
            }
            let key = (next.raw(), cur.raw());
            if enqueued.insert(key) {
                queue.push_back((next, cur));
            }
        }
    }

    let mut ids: Vec<DenseNodeId> = out.into_iter().map(DenseNodeId::from_raw).collect();
    ids.sort_by_key(|id| id.raw());
    Ok(ids)
}

/// Possible-D-Sep BFS hit `max_nodes`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PossibleDSepBudget {
    /// Expansion budget that was exhausted.
    pub max_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_graph::Pag;

    #[test]
    fn neighbors_of_from_are_in_pds() {
        let mut g = Pag::with_variables(4);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        let d = DenseNodeId::from_raw(3);
        g.insert_circle_circle(a, c).unwrap();
        g.insert_circle_circle(a, d).unwrap();
        g.insert_circle_circle(a, b).unwrap();
        let pds = possible_d_sep(&g, a, b, 64).unwrap();
        assert!(pds.contains(&c));
        assert!(pds.contains(&d));
        assert!(!pds.contains(&a));
        assert!(!pds.contains(&b));
    }

    #[test]
    fn triangle_extends_pds_beyond_neighbors() {
        // Path a–c–d with triangle a–c–d (edge a–d): d enters via triple ok.
        // Also need a–e with collider path for a richer case:
        // a → c ← e, and c → d (wait). Simpler: undirected triangle a-c-d-a
        // then from a wrt b: c is neighbor; d reachable a-c-d with triangle a-c-d.
        let mut g = Pag::with_variables(4);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        let d = DenseNodeId::from_raw(3);
        g.insert_circle_circle(a, b).unwrap();
        g.insert_circle_circle(a, c).unwrap();
        g.insert_circle_circle(c, d).unwrap();
        g.insert_circle_circle(a, d).unwrap(); // triangle a-c-d
        let pds = possible_d_sep(&g, a, b, 64).unwrap();
        assert!(pds.contains(&c));
        assert!(pds.contains(&d), "d via triangle triple on a–c–d; pds={pds:?}");
    }

    #[test]
    fn collider_extends_pds() {
        // a → c ← d (collider at c), edge a–b. Path a–c–d: collider at c ⇒ d ∈ PDS(a,b).
        let mut g = Pag::with_variables(4);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        let d = DenseNodeId::from_raw(3);
        g.insert_circle_circle(a, b).unwrap();
        g.insert_directed(a, c).unwrap();
        g.insert_directed(d, c).unwrap();
        let pds = possible_d_sep(&g, a, b, 64).unwrap();
        assert!(pds.contains(&c));
        assert!(pds.contains(&d), "d via collider at c; pds={pds:?}");
    }
}
