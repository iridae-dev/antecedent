//! Reusable traversal workspace.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::types::DenseNodeId;

/// Bitset over dense node ids.
#[derive(Clone, Debug, Default)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// Create a bitset for `len` bits.
    #[must_use]
    pub fn with_len(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], len }
    }

    /// Clear all bits (retain capacity).
    pub fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// Set the number of addressable bits to `len`.
    ///
    /// Bits at or beyond `len` are cleared, so a shrink followed by a grow cannot re-expose
    /// stale members and equality / hashing see only the addressable bits.
    pub fn resize(&mut self, len: usize) {
        self.len = len;
        self.words.resize(len.div_ceil(64), 0);
        let tail = len % 64;
        if tail != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << tail) - 1;
            }
        }
    }

    /// Number of addressable bits.
    #[must_use]
    pub const fn bit_len(&self) -> usize {
        self.len
    }

    /// Borrow the underlying word storage (for hashing / memo keys).
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Set bit.
    pub fn insert(&mut self, id: DenseNodeId) {
        let i = id.as_usize();
        debug_assert!(i < self.len);
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    /// Clear one bit.
    pub fn remove(&mut self, id: DenseNodeId) {
        let i = id.as_usize();
        if i >= self.len {
            return;
        }
        self.words[i / 64] &= !(1u64 << (i % 64));
    }

    /// Test bit.
    #[must_use]
    pub fn contains(&self, id: DenseNodeId) -> bool {
        let i = id.as_usize();
        if i >= self.len {
            return false;
        }
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Whether any bit is set.
    #[must_use]
    pub fn any(&self) -> bool {
        self.words.iter().any(|w| *w != 0)
    }

    /// Number of set bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Collect set bit indices as dense node ids (ascending).
    #[must_use]
    pub fn to_dense_ids(&self) -> Vec<DenseNodeId> {
        let mut out = Vec::with_capacity(self.count_ones());
        for i in 0..self.len {
            let Ok(raw) = u32::try_from(i) else {
                break;
            };
            let id = DenseNodeId::from_raw(raw);
            if self.contains(id) {
                out.push(id);
            }
        }
        out
    }

    /// Union `other` into `self` (same length).
    pub fn union_with(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }

    /// Intersect `other` into `self` (same length).
    pub fn intersect_with(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= *b;
        }
    }

    /// Subtract `other` from `self` (same length).
    pub fn difference_with(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= !*b;
        }
    }

    /// Whether `self` is a subset of `other`.
    #[must_use]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        debug_assert_eq!(self.len, other.len);
        self.words.iter().zip(other.words.iter()).all(|(a, b)| a & !b == 0)
    }

    /// Whether `self` equals `other`.
    #[must_use]
    pub fn equal_set(&self, other: &Self) -> bool {
        self.len == other.len && self.words == other.words
    }
}

impl std::hash::Hash for BitSet {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        self.words.hash(state);
    }
}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        self.equal_set(other)
    }
}

impl Eq for BitSet {}

/// Scratch space for graph traversals; may grow but is reused.
#[derive(Clone, Debug, Default)]
pub struct GraphWorkspace {
    /// Visited set.
    pub visited: BitSet,
    /// BFS/DFS frontier.
    pub frontier: Vec<DenseNodeId>,
    /// Scratch node buffer.
    pub scratch_nodes: Vec<DenseNodeId>,
    /// Predecessor map (indexed by dense id).
    pub predecessor: Vec<Option<DenseNodeId>>,
}

impl GraphWorkspace {
    /// Prepare workspace for a graph with `n` nodes (visited set, frontier, scratch nodes).
    ///
    /// The predecessor map is not touched: traversals that need it call
    /// [`Self::prepare_with_predecessors`], so plain reachability does not pay an O(n) reset.
    pub fn prepare(&mut self, n: usize) {
        self.visited.resize(n);
        self.visited.clear();
        self.frontier.clear();
        self.scratch_nodes.clear();
    }

    /// [`Self::prepare`], plus a predecessor map of `n` empty slots.
    pub fn prepare_with_predecessors(&mut self, n: usize) {
        self.prepare(n);
        self.predecessor.clear();
        self.predecessor.resize(n, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    #[test]
    fn resize_clears_bits_beyond_the_new_length() {
        let mut a = BitSet::with_len(64);
        for i in [3, 9, 10, 40, 63] {
            a.insert(id(i));
        }
        a.resize(10);
        // Bits 10.. are gone: equal (and hash-equal) to a fresh set holding {3, 9}.
        let mut fresh = BitSet::with_len(10);
        fresh.insert(id(3));
        fresh.insert(id(9));
        assert_eq!(a, fresh);
        assert_eq!(a.words(), fresh.words());
        // Growing back must not resurrect the stale members.
        a.resize(64);
        assert_eq!(a.to_dense_ids(), vec![id(3), id(9)]);
        assert_eq!(a.count_ones(), 2);
    }

    #[test]
    fn prepare_only_sizes_predecessors_on_request() {
        let mut ws = GraphWorkspace::default();
        ws.prepare(5);
        assert!(ws.predecessor.is_empty());
        ws.prepare_with_predecessors(5);
        assert_eq!(ws.predecessor.len(), 5);
        assert!(ws.predecessor.iter().all(Option::is_none));
    }
}
