//! Shared graph algorithms (reachability, closure, Kahn topo).
//!
//! Three closure owners cover every representation callers hold:
//! [`closure_into`] over dense adjacency slices with a [`BitSet`] result,
//! [`mask_closure`] over per-node bitmask adjacency (graphs of at most the mask width),
//! and [`implicit_closure`] over a lazily expanded successor function on arbitrary ordered
//! keys (product graphs, indexed copies, class quotients).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::ops::{BitAnd, BitOr, Not};

use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

/// Insert `value` into the ascending `list`, keeping it sorted. Returns `false`, leaving the
/// list unchanged, when `value` is already present. The one sorted-row insertion every graph
/// store uses.
pub(crate) fn insert_sorted<T: Ord>(list: &mut Vec<T>, value: T) -> bool {
    match list.binary_search(&value) {
        Ok(_) => false,
        Err(at) => {
            list.insert(at, value);
            true
        }
    }
}

/// Directed BFS reachability from `from` to `to` over `children`.
#[must_use]
pub(crate) fn bfs_reaches(
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
pub fn closure_into<'g>(
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

/// An unsigned integer used as a node bitmask by [`mask_closure`].
pub trait MaskWord:
    Copy + Eq + BitOr<Output = Self> + BitAnd<Output = Self> + Not<Output = Self>
{
    /// The empty mask.
    const ZERO: Self;
    /// Index of the lowest set bit, or `None` for the empty mask.
    fn lowest_bit(self) -> Option<usize>;
    /// The mask with only bit `index` set.
    fn single(index: usize) -> Self;
}

macro_rules! mask_word {
    ($($ty:ty),*) => {$(
        impl MaskWord for $ty {
            const ZERO: Self = 0;
            fn lowest_bit(self) -> Option<usize> {
                (self != 0).then(|| self.trailing_zeros() as usize)
            }
            fn single(index: usize) -> Self {
                1 << index
            }
        }
    )*};
}
mask_word!(u8, u16, u32, u64);

/// Closure of the mask `seeds` along `next` (bit `u` of the graph's adjacency in `next[u]`),
/// for graphs small enough to keep one adjacency word per node.
///
/// The result holds the seeds and every node reached from a member whose bit is not in
/// `frozen`: a frozen member is included but not expanded, which is how "ancestors in the
/// graph without the edges into `frozen`" is read off the parent masks. Linear in the number
/// of members (each is expanded once).
///
/// # Panics
///
/// If a seed bit is at or beyond `next.len()` and is reached for expansion.
#[must_use]
pub fn mask_closure<M: MaskWord>(seeds: M, next: &[M], frozen: M) -> M {
    let mut closure = seeds;
    let mut pending = seeds;
    while let Some(u) = pending.lowest_bit() {
        pending = pending & !M::single(u);
        if frozen & M::single(u) != M::ZERO {
            continue;
        }
        let fresh = next[u] & !closure;
        closure = closure | fresh;
        pending = pending | fresh;
    }
    closure
}

/// Closure of `seeds` under `expand` over arbitrary ordered keys, seeds included.
///
/// For graphs that are never materialised as dense adjacency: `(variable, world)` product
/// nodes, `(variable, offset)` window nodes, or merged-class quotients. `expand` may carry
/// state (it is `FnMut`), e.g. to flag a truncated expansion.
pub fn implicit_closure<N: Ord + Clone, I: IntoIterator<Item = N>>(
    seeds: impl IntoIterator<Item = N>,
    mut expand: impl FnMut(&N) -> I,
) -> BTreeSet<N> {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<N> = seeds.into_iter().collect();
    while let Some(node) = stack.pop() {
        if seen.contains(&node) {
            continue;
        }
        stack.extend(expand(&node));
        seen.insert(node);
    }
    seen
}

/// Connected components of the undirected relation `neighbors` over nodes `0..n`.
///
/// Each component lists its members ascending, and components are ordered by their smallest
/// member. `neighbors(i)` may repeat nodes or name `i` itself. For graphs whose adjacency is
/// held in caller-side index form (a union of several edge kinds, an edge list) rather than
/// as dense adjacency slices.
pub fn components_by<I: IntoIterator<Item = usize>>(
    n: usize,
    mut neighbors: impl FnMut(usize) -> I,
) -> Vec<Vec<usize>> {
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    let mut stack = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut members = Vec::new();
        while let Some(u) = stack.pop() {
            members.push(u);
            for v in neighbors(u) {
                if v < n && !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }
    components
}

/// Kahn topological order. Returns `None` if a directed cycle exists.
#[must_use]
pub(crate) fn kahn_order(
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
        .map(|(i, _)| DenseNodeId::from_index(i))
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

/// Whether the directed graph is acyclic (Kahn count, without retaining the order).
#[must_use]
pub fn is_dag(parents: &[Vec<DenseNodeId>], children: &[Vec<DenseNodeId>]) -> bool {
    if parents.len() != children.len() {
        return false;
    }
    let n = parents.len();
    let mut indeg = vec![0u32; n];
    for (i, p) in parents.iter().enumerate() {
        let Ok(degree) = u32::try_from(p.len()) else {
            return false;
        };
        indeg[i] = degree;
    }
    let mut ready: Vec<DenseNodeId> = indeg
        .iter()
        .enumerate()
        .filter(|&(_, &degree)| degree == 0)
        .map(|(i, _)| DenseNodeId::from_index(i))
        .collect();
    let mut placed = 0usize;
    while let Some(node) = ready.pop() {
        placed += 1;
        for &child in &children[node.as_usize()] {
            let left = &mut indeg[child.as_usize()];
            *left -= 1;
            if *left == 0 {
                ready.push(child);
            }
        }
    }
    placed == n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    /// A small DAG with a diamond and a tail, as parent lists.
    fn parents() -> Vec<Vec<DenseNodeId>> {
        vec![vec![], vec![id(0)], vec![id(0)], vec![id(1), id(2)], vec![id(3)], vec![]]
    }

    #[test]
    fn mask_closure_agrees_with_closure_into_for_every_seed_and_frozen_set() {
        let parents = parents();
        let n = parents.len();
        let masks: Vec<u16> =
            parents.iter().map(|ps| ps.iter().fold(0u16, |m, p| m | (1 << p.as_usize()))).collect();
        let mut ws = GraphWorkspace::default();
        for seeds in 0u16..(1 << n) {
            for frozen in [0u16, 0b00_0100, 0b00_1010, 0b11_1111] {
                let mut dense = BitSet::with_len(n);
                closure_into(
                    n,
                    |u| parents[u.as_usize()].as_slice(),
                    (0..n).filter(|i| (seeds >> i) & 1 == 1).map(DenseNodeId::from_index),
                    &mut dense,
                    &mut ws,
                    |u, _| (frozen >> u.as_usize()) & 1 == 0,
                );
                let expected = dense.iter().fold(0u16, |m, v| m | (1 << v.as_usize()));
                assert_eq!(mask_closure(seeds, &masks, frozen), expected, "{seeds:b}/{frozen:b}");
            }
        }
    }

    #[test]
    fn components_by_orders_components_by_smallest_member() {
        // 0-3, 1-4-2, 5 alone.
        let adjacency = [vec![3], vec![4], vec![4], vec![0], vec![1, 2], vec![]];
        let components = components_by(6, |i| adjacency[i].iter().copied());
        assert_eq!(components, vec![vec![0, 3], vec![1, 2, 4], vec![5]]);
        assert!(components_by(0, |_| Vec::new()).is_empty());
    }

    #[test]
    fn implicit_closure_follows_a_stateful_expander_once_per_key() {
        let mut expansions = 0;
        let reached = implicit_closure([4u32], |&v| {
            expansions += 1;
            if v == 0 { vec![] } else { vec![v - 1, v / 2] }
        });
        assert_eq!(reached.into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert_eq!(expansions, 5);
    }
}
