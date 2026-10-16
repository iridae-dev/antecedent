//! Reusable traversal workspace.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::GraphError;
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

    /// Create a bitset for `len` bits with every bit set.
    #[must_use]
    pub fn full(len: usize) -> Self {
        let mut set = Self { words: vec![u64::MAX; len.div_ceil(64)], len };
        let tail = len % 64;
        if tail != 0 {
            if let Some(last) = set.words.last_mut() {
                *last = (1u64 << tail) - 1;
            }
        }
        set
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

    /// Set bit `id`.
    ///
    /// # Errors
    ///
    /// `id` is outside `0..bit_len`. Callers that hold an arbitrary dense id use this
    /// instead of [`Self::insert`], which treats an out-of-range id as a programming bug.
    pub fn try_insert(&mut self, id: DenseNodeId) -> Result<(), GraphError> {
        let i = id.as_usize();
        if i >= self.len {
            return Err(GraphError::UnknownNode { id: id.raw() });
        }
        let word = i / 64;
        let bit = i % 64;
        let Some(slot) = self.words.get_mut(word) else {
            return Err(GraphError::UnknownNode { id: id.raw() });
        };
        *slot |= 1u64 << bit;
        Ok(())
    }

    /// Set bit.
    ///
    /// # Panics
    ///
    /// If `id` is outside `0..bit_len`. An out-of-range insert used to set a tail bit
    /// past the addressable length in release builds, so equal sets no longer compared
    /// equal when used as memo keys. In-range ids are unchanged. Arbitrary ids belong
    /// in [`Self::try_insert`].
    pub fn insert(&mut self, id: DenseNodeId) {
        assert!(
            self.try_insert(id).is_ok(),
            "BitSet::insert: dense id {} is outside 0..{} (programming bug: the id is not in this set's domain)",
            id.as_usize(),
            self.len
        );
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

    /// Iterate the set bits as dense node ids in ascending order.
    ///
    /// Word-wise (`trailing_zeros`): the cost is proportional to the number of words plus
    /// the number of members, not to the number of addressable bits.
    pub fn iter(&self) -> impl Iterator<Item = DenseNodeId> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            let base = w * 64;
            let mut rest = word;
            std::iter::from_fn(move || {
                if rest == 0 {
                    return None;
                }
                let bit = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                Some(DenseNodeId::from_index(base + bit))
            })
        })
    }

    /// Collect set bit indices as dense node ids (ascending).
    #[must_use]
    pub fn to_dense_ids(&self) -> Vec<DenseNodeId> {
        self.iter().collect()
    }

    /// Whether `self` and `other` share a member (same length).
    ///
    /// # Panics
    ///
    /// If the sets were sized for different node counts. Zipping unequal word
    /// vectors used to drop a tail word in release builds.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.require_same_domain(other);
        self.words.iter().zip(other.words.iter()).any(|(a, b)| a & b != 0)
    }

    /// Union `other` into `self` (same length).
    ///
    /// # Panics
    ///
    /// If the sets were sized for different node counts.
    pub fn union_with(&mut self, other: &Self) {
        self.require_same_domain(other);
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }

    /// Intersect `other` into `self` (same length).
    ///
    /// # Panics
    ///
    /// If the sets were sized for different node counts.
    pub fn intersect_with(&mut self, other: &Self) {
        self.require_same_domain(other);
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= *b;
        }
    }

    /// Subtract `other` from `self` (same length).
    ///
    /// # Panics
    ///
    /// If the sets were sized for different node counts.
    pub fn difference_with(&mut self, other: &Self) {
        self.require_same_domain(other);
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= !*b;
        }
    }

    /// Whether `self` is a subset of `other`.
    ///
    /// # Panics
    ///
    /// If the sets were sized for different node counts.
    #[must_use]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.require_same_domain(other);
        self.words.iter().zip(other.words.iter()).all(|(a, b)| a & !b == 0)
    }

    /// Both sets address the same bits, including the same word vector length.
    fn require_same_domain(&self, other: &Self) {
        assert!(
            self.len == other.len && self.words.len() == other.words.len(),
            "BitSet operation requires equal lengths ({} vs {})",
            self.len,
            other.len
        );
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
    use crate::error::GraphError;

    fn id(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    #[test]
    fn full_sets_exactly_the_addressable_bits() {
        for len in [0usize, 1, 5, 63, 64, 65, 130] {
            let full = BitSet::full(len);
            assert_eq!(full.count_ones(), len);
            let mut built = BitSet::with_len(len);
            for i in 0..len {
                built.insert(id(u32::try_from(i).unwrap()));
            }
            assert_eq!(full, built, "len {len}");
        }
    }

    #[test]
    fn iter_matches_membership_and_intersects_matches_a_scan() {
        let mut a = BitSet::with_len(130);
        let mut b = BitSet::with_len(130);
        for i in [0, 1, 63, 64, 100, 129] {
            a.insert(id(i));
        }
        for i in [2, 62, 65, 101] {
            b.insert(id(i));
        }
        let scanned: Vec<_> = (0..130).map(id).filter(|&node| a.contains(node)).collect();
        assert_eq!(a.iter().collect::<Vec<_>>(), scanned);
        assert_eq!(a.to_dense_ids(), scanned);
        assert!(!a.intersects(&b));
        b.insert(id(100));
        assert!(a.intersects(&b));
        assert_eq!(BitSet::with_len(0).iter().count(), 0);
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
    fn insert_rejects_a_tail_bit_past_the_addressable_length() {
        let mut set = BitSet::with_len(65);
        let err = set.try_insert(id(70)).unwrap_err();
        assert!(matches!(err, GraphError::UnknownNode { id: 70 }));
        assert_eq!(set.count_ones(), 0);
        assert_eq!(set.words()[1], 0, "an out-of-range insert must not dirty the tail word");
    }

    #[test]
    #[should_panic(expected = "outside 0..65")]
    fn insert_panics_on_a_tail_bit_past_the_addressable_length() {
        let mut set = BitSet::with_len(65);
        set.insert(id(70));
    }

    #[test]
    fn in_range_insert_still_sets_the_last_addressable_bit() {
        let mut set = BitSet::with_len(65);
        set.insert(id(64));
        assert!(set.contains(id(64)));
        assert_eq!(set.count_ones(), 1);
        assert_eq!(set.words()[1], 1);
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn union_rejects_mismatched_lengths() {
        let mut a = BitSet::with_len(64);
        a.insert(id(1));
        let b = BitSet::with_len(128);
        a.union_with(&b);
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
