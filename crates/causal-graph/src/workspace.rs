//! Reusable traversal workspace (DESIGN.md §6.3).
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

    /// Ensure capacity for `len` bits.
    pub fn resize(&mut self, len: usize) {
        self.len = len;
        self.words.resize(len.div_ceil(64), 0);
    }

    /// Set bit.
    pub fn insert(&mut self, id: DenseNodeId) {
        let i = id.as_usize();
        debug_assert!(i < self.len);
        self.words[i / 64] |= 1u64 << (i % 64);
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
}

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
    /// Prepare workspace for a graph with `n` nodes.
    pub fn prepare(&mut self, n: usize) {
        self.visited.resize(n);
        self.visited.clear();
        self.frontier.clear();
        self.scratch_nodes.clear();
        self.predecessor.clear();
        self.predecessor.resize(n, None);
    }
}
