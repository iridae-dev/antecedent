//! Shared graph algorithms (reachability, Kahn topo).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

/// Directed BFS reachability from `from` to `to` over `children`.
#[must_use]
pub fn bfs_reaches(
    children: &[Vec<DenseNodeId>],
    from: DenseNodeId,
    to: DenseNodeId,
    ws: &mut GraphWorkspace,
) -> bool {
    if from == to {
        return true;
    }
    if from.as_usize() >= children.len() || to.as_usize() >= children.len() {
        return false;
    }
    ws.prepare(children.len());
    ws.frontier.push(from);
    ws.visited.insert(from);
    while let Some(n) = ws.frontier.pop() {
        for &c in &children[n.as_usize()] {
            if c == to {
                return true;
            }
            if !ws.visited.contains(c) {
                ws.visited.insert(c);
                ws.frontier.push(c);
            }
        }
    }
    false
}

/// Closure of `seeds` along `next` (parents for ancestors, children for descendants).
///
/// `out` is resized to `n` and cleared, then receives the seeds (those below `n`) and every
/// node reached through an edge `keep(from, to)` accepts. A seed outside the closure's
/// admissible region must be filtered by the caller; `keep` gates only the traversed edges.
pub(crate) fn closure_into<'g>(
    n: usize,
    next: impl Fn(DenseNodeId) -> &'g [DenseNodeId],
    seeds: impl IntoIterator<Item = DenseNodeId>,
    out: &mut BitSet,
    ws: &mut GraphWorkspace,
    mut keep: impl FnMut(DenseNodeId, DenseNodeId) -> bool,
) {
    out.resize(n);
    out.clear();
    ws.prepare(n);
    for v in seeds {
        if v.as_usize() >= n {
            continue;
        }
        if !out.contains(v) {
            out.insert(v);
            ws.frontier.push(v);
        }
    }
    while let Some(u) = ws.frontier.pop() {
        for &w in next(u) {
            if !keep(u, w) || out.contains(w) {
                continue;
            }
            out.insert(w);
            ws.frontier.push(w);
        }
    }
}

/// Kahn topological order. Returns `None` if a directed cycle exists.
#[must_use]
pub fn kahn_order(
    parents: &[Vec<DenseNodeId>],
    children: &[Vec<DenseNodeId>],
) -> Option<Vec<DenseNodeId>> {
    let n = parents.len();
    debug_assert_eq!(n, children.len());
    let mut indeg = vec![0u32; n];
    for (i, p) in parents.iter().enumerate() {
        indeg[i] = u32::try_from(p.len()).ok()?;
    }
    let mut q: Vec<DenseNodeId> = indeg
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d == 0)
        .map(|(i, _)| DenseNodeId::from_raw(u32::try_from(i).expect("node fit")))
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = q.pop() {
        order.push(u);
        for &v in &children[u.as_usize()] {
            indeg[v.as_usize()] -= 1;
            if indeg[v.as_usize()] == 0 {
                q.push(v);
            }
        }
    }
    (order.len() == n).then_some(order)
}

/// Whether the directed graph is acyclic (Kahn count).
#[must_use]
pub fn is_dag(parents: &[Vec<DenseNodeId>], children: &[Vec<DenseNodeId>]) -> bool {
    kahn_order(parents, children).is_some()
}
