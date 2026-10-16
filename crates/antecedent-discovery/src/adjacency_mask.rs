//! Packed directed-edge set of one graph-posterior atom.
//!
//! Bit `edge_bit(n, from, to)` of the set is the edge `from → to`. The bit layout is the one
//! the single-`u64` adjacency used, so a graph on at most eight nodes occupies word 0 only and
//! that word is the earlier encoding unchanged: [`AdjacencyMask::from_word`] is the decoder for
//! a stored `u64`. Larger graphs continue into the following words.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::{self, Write as _};

use antecedent_core::Fnv1a64;

/// Most labeled nodes a graph-posterior atom can carry (`16 * 15 = 240` edge bits).
pub const MAX_POSTERIOR_NODES: usize = 16;

/// Words of the packed edge set; `MAX_POSTERIOR_NODES * (MAX_POSTERIOR_NODES - 1)` bits fit.
const WORDS: usize = 4;

/// Directed-edge set over at most [`MAX_POSTERIOR_NODES`] labeled nodes.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdjacencyMask([u64; WORDS]);

impl AdjacencyMask {
    /// No edges.
    pub const EMPTY: Self = Self([0; WORDS]);

    /// Decode a single-word packing (a graph on at most eight nodes).
    #[must_use]
    pub const fn from_word(word: u64) -> Self {
        Self([word, 0, 0, 0])
    }

    /// Build from little-endian words (bit `b` is bit `b % 64` of word `b / 64`).
    #[must_use]
    pub const fn from_words(words: [u64; WORDS]) -> Self {
        Self(words)
    }

    /// The little-endian words.
    #[must_use]
    pub const fn words(self) -> [u64; WORDS] {
        self.0
    }

    /// The single-word packing, or `None` when an edge bit beyond word 0 is set.
    #[must_use]
    pub fn to_word(self) -> Option<u64> {
        (self.0[1..].iter().all(|&w| w == 0)).then_some(self.0[0])
    }

    /// Little-endian bytes of the whole set, the form a big-integer consumer reads.
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; WORDS * 8] {
        let mut bytes = [0u8; WORDS * 8];
        for (chunk, word) in bytes.chunks_exact_mut(8).zip(self.0) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Inverse of [`Self::to_le_bytes`].
    #[must_use]
    pub fn from_le_bytes(bytes: [u8; WORDS * 8]) -> Self {
        let mut words = [0u64; WORDS];
        for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut le = [0u8; 8];
            le.copy_from_slice(chunk);
            *word = u64::from_le_bytes(le);
        }
        Self(words)
    }

    /// Whether bit `bit` is set; bits past the packing are never set.
    #[must_use]
    pub fn bit(self, bit: u32) -> bool {
        self.0.get(bit as usize / 64).is_some_and(|word| (word >> (bit % 64)) & 1 == 1)
    }

    /// Set or clear bit `bit`. Bits past the packing are not representable and are left
    /// alone; every constructor refuses a node count whose edge bits would reach them.
    #[must_use]
    pub fn with_bit(mut self, bit: u32, present: bool) -> Self {
        debug_assert!((bit as usize) < WORDS * 64, "edge bit {bit} is past the packing");
        if let Some(word) = self.0.get_mut(bit as usize / 64) {
            let flag = 1u64 << (bit % 64);
            if present {
                *word |= flag;
            } else {
                *word &= !flag;
            }
        }
        self
    }

    /// Whether the two sets share an edge.
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        self.0.iter().zip(other.0).any(|(a, b)| a & b != 0)
    }

    /// Whether every edge of `other` is in this set.
    #[must_use]
    pub fn contains_all(self, other: Self) -> bool {
        self.0.iter().zip(other.0).all(|(a, b)| b & !a == 0)
    }

    /// Opaque 64-bit key of the set.
    ///
    /// A single-word set keys as its word, which is the key the earlier encoding used. A wider
    /// set keys as an FNV-1a digest of its words. Distinct sets can share a key only by hash
    /// collision, which [`crate::GraphPosterior::new`] refuses.
    #[must_use]
    pub fn key(self) -> u64 {
        if let Some(word) = self.to_word() {
            return word;
        }
        let mut hash = Fnv1a64::new();
        for word in self.0 {
            hash.word(word);
        }
        hash.finish()
    }
}

impl From<u64> for AdjacencyMask {
    fn from(word: u64) -> Self {
        Self::from_word(word)
    }
}

impl PartialEq<u64> for AdjacencyMask {
    fn eq(&self, other: &u64) -> bool {
        self.to_word() == Some(*other)
    }
}

impl PartialEq<AdjacencyMask> for u64 {
    fn eq(&self, other: &AdjacencyMask) -> bool {
        other == self
    }
}

impl fmt::LowerHex for AdjacencyMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = self.0.iter().rposition(|&w| w != 0).unwrap_or(0);
        let mut digits = format!("{:x}", self.0[top]);
        for word in self.0[..top].iter().rev() {
            write!(digits, "{word:016x}")?;
        }
        f.pad_integral(true, "0x", &digits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_word_set_is_the_earlier_encoding() {
        let mask = AdjacencyMask::from_word(0b1010_0110);
        assert_eq!(mask.to_word(), Some(0b1010_0110));
        assert_eq!(mask.key(), 0b1010_0110);
        assert_eq!(mask, 0b1010_0110_u64);
        assert!(mask.bit(1) && mask.bit(2) && !mask.bit(0));
    }

    #[test]
    fn bits_past_word_zero_round_trip_and_leave_the_single_word_form() {
        let mask = AdjacencyMask::EMPTY.with_bit(3, true).with_bit(70, true).with_bit(239, true);
        assert!(mask.bit(3) && mask.bit(70) && mask.bit(239));
        assert!(!mask.bit(71) && !mask.bit(240));
        assert_eq!(mask.to_word(), None);
        assert_eq!(AdjacencyMask::from_le_bytes(mask.to_le_bytes()), mask);
        let cleared = mask.with_bit(70, false).with_bit(239, false);
        assert_eq!(cleared.to_word(), Some(1 << 3));
        assert_eq!(
            format!("{mask:#x}"),
            format!("0x{:x}{:016x}{:016x}{:016x}", 1u64 << 47, 0, 1u64 << 6, 8)
        );
    }

    #[test]
    fn subset_and_overlap_follow_set_semantics() {
        let a = AdjacencyMask::EMPTY.with_bit(5, true).with_bit(100, true);
        let b = AdjacencyMask::EMPTY.with_bit(100, true);
        let c = AdjacencyMask::EMPTY.with_bit(101, true);
        assert!(a.contains_all(b) && !b.contains_all(a));
        assert!(a.intersects(b) && !a.intersects(c));
        assert!(a.contains_all(AdjacencyMask::EMPTY));
    }

    #[test]
    fn wide_sets_that_differ_get_different_keys() {
        let a = AdjacencyMask::EMPTY.with_bit(100, true);
        let b = AdjacencyMask::EMPTY.with_bit(101, true);
        assert_ne!(a.key(), b.key());
        assert_eq!(a.key(), a.key());
    }
}
