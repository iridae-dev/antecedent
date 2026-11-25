//! Packed directed-edge set of one graph-posterior atom.
//!
//! Bit `edge_bit(n, from, to)` of the set is the edge `from → to`. The bit layout is the one
//! the single-`u64` adjacency used, so a graph on at most eight nodes occupies word 0 only and
//! that word is the earlier encoding unchanged: [`AdjacencyMask::from_word`] is the decoder for
//! a stored `u64`.
//!
//! The set grows with the graph. Up to [`INLINE_WORDS`] words (a graph on at most sixteen
//! nodes) live inline with no allocation; a wider set is held on the heap, trimmed of its
//! trailing zero words, so the representation of a set is unique and equality, ordering and
//! hashing are those of the bit set. The only bound on the node count is that an edge bit must
//! be addressable as a `u32` ([`edge_bits_fit`]).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::{self, Write as _};
use std::sync::Arc;

use crate::Fnv1a64;

/// Words held inline, with no allocation: `4 * 64 = 256` bits cover the `16 * 15 = 240` edge
/// bits of a sixteen-node graph.
const INLINE_WORDS: usize = 4;

/// Whether every edge bit of an `n`-node graph is addressable as a `u32` (`n * (n - 1)` fits).
///
/// This is the only limit on the node count of an [`AdjacencyMask`]; it is a width of the bit
/// index, not a property of any algorithm.
#[must_use]
pub fn edge_bits_fit(n: usize) -> bool {
    u32::try_from(n_directed_edges(n)).is_ok()
}

/// Storage of an [`AdjacencyMask`]. Canonical: `Heap` only when the highest set bit lies beyond
/// the inline words, and then with no trailing zero word.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Words {
    Inline([u64; INLINE_WORDS]),
    Heap(Arc<[u64]>),
}

/// Directed-edge set over any number of labeled nodes.
///
/// Also the bit-set type of the lag-edge templates ([`LagMask`]) and of the node sets the
/// samplers use for parent sets.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AdjacencyMask(Words);

/// Packed lagged-edge set of a template: bit `(lag - 1) * n² + from * n + to` is the edge
/// `from(t - lag) → to(t)`. The same growable bit set as [`AdjacencyMask`].
pub type LagMask = AdjacencyMask;

impl Default for AdjacencyMask {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl AdjacencyMask {
    /// No edges.
    pub const EMPTY: Self = Self(Words::Inline([0; INLINE_WORDS]));

    /// Decode a single-word packing (a graph on at most eight nodes).
    #[must_use]
    pub const fn from_word(word: u64) -> Self {
        Self(Words::Inline([word, 0, 0, 0]))
    }

    /// Build from little-endian words (bit `b` is bit `b % 64` of word `b / 64`); trailing zero
    /// words are immaterial.
    #[must_use]
    pub fn from_words(words: &[u64]) -> Self {
        let used = words.iter().rposition(|&w| w != 0).map_or(0, |top| top + 1);
        if used <= INLINE_WORDS {
            let mut inline = [0u64; INLINE_WORDS];
            inline[..used].copy_from_slice(&words[..used]);
            Self(Words::Inline(inline))
        } else {
            Self(Words::Heap(Arc::from(&words[..used])))
        }
    }

    /// The little-endian words: at least [`INLINE_WORDS`] of them for a set held inline, and
    /// the significant words only for a wider set. Bits past the last word are zero.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        match &self.0 {
            Words::Inline(words) => words,
            Words::Heap(words) => words,
        }
    }

    /// The single-word packing, or `None` when an edge bit beyond word 0 is set.
    #[must_use]
    pub fn to_word(&self) -> Option<u64> {
        let words = self.words();
        words[1..].iter().all(|&w| w == 0).then_some(words[0])
    }

    /// Little-endian bytes of the whole set, the form a big-integer consumer reads.
    ///
    /// Exactly 32 bytes for a set held inline (the encoding every graph on at most sixteen
    /// nodes has always had); a wider set continues with one 8-byte word per further word,
    /// trimmed of trailing zero words. [`Self::from_le_bytes`] is the inverse.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.words().iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// Bytes the execution pool admits for this mask: eight bytes for each stored word.
    ///
    /// An inline mask, including every graph whose edges fit one `u64`, is four words and
    /// 32 bytes, the same length as [`Self::to_le_bytes`].
    #[must_use]
    pub fn admission_bytes(&self) -> u64 {
        u64::try_from(self.words().len()).unwrap_or(u64::MAX).saturating_mul(8)
    }

    /// Inverse of [`Self::to_le_bytes`]; a length that is not a multiple of eight is padded
    /// with zero bytes, and trailing zero words are immaterial.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let words: Vec<u64> = bytes
            .chunks(8)
            .map(|chunk| {
                let mut le = [0u8; 8];
                le[..chunk.len()].copy_from_slice(chunk);
                u64::from_le_bytes(le)
            })
            .collect();
        Self::from_words(&words)
    }

    /// Whether the set uses only the directed-edge bits of an `n`-node graph.
    #[must_use]
    pub fn fits_graph(&self, n: usize) -> bool {
        let words = self.words();
        let bits = n_directed_edges(n);
        words.iter().rposition(|&word| word != 0).is_none_or(|last| {
            let highest_in_word = 63 - words[last].leading_zeros() as usize;
            last.saturating_mul(64).saturating_add(highest_in_word) < bits
        })
    }

    /// Whether bit `bit` is set; bits past the stored words are never set.
    #[must_use]
    pub fn bit(&self, bit: u32) -> bool {
        self.words().get(bit as usize / 64).is_some_and(|word| (word >> (bit % 64)) & 1 == 1)
    }

    /// Set or clear bit `bit`, growing the set when a bit past its words is set.
    pub fn set_bit(&mut self, bit: u32, present: bool) {
        let (index, flag) = (bit as usize / 64, 1u64 << (bit % 64));
        let apply = |word: &mut u64| {
            if present {
                *word |= flag;
            } else {
                *word &= !flag;
            }
        };
        if self.words().get(index).is_none() {
            if !present {
                return;
            }
            let mut words = self.words().to_vec();
            words.resize(index + 1, 0);
            apply(&mut words[index]);
            *self = Self::from_words(&words);
            return;
        }
        let mut trim = false;
        match &mut self.0 {
            Words::Inline(words) => apply(&mut words[index]),
            Words::Heap(words) => {
                if let Some(owned) = Arc::get_mut(words) {
                    apply(&mut owned[index]);
                } else {
                    let mut copy = words.to_vec();
                    apply(&mut copy[index]);
                    *words = Arc::from(copy);
                }
                trim = !present && words.last() == Some(&0);
            }
        }
        if trim {
            *self = Self::from_words(self.words());
        }
    }

    /// This set with bit `bit` set or cleared.
    #[must_use]
    pub fn with_bit(mut self, bit: u32, present: bool) -> Self {
        self.set_bit(bit, present);
        self
    }

    /// Whether no bit is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words().iter().all(|&w| w == 0)
    }

    /// Number of set bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words().iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Indexes of the set bits in ascending order.
    pub fn set_bits(&self) -> impl Iterator<Item = u32> + '_ {
        self.words().iter().enumerate().flat_map(|(index, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let bit = word.trailing_zeros();
                word &= word - 1;
                Some(u32::try_from(index).expect("bit word index fits u32") * 64 + bit)
            })
        })
    }

    /// The edges `(from, to)` of a mask on `n` nodes, in bit order (the inverse of
    /// [`edge_bit`]).
    pub fn edges(&self, n: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let width = n.saturating_sub(1).max(1);
        self.set_bits().filter_map(move |bit| {
            let bit = bit as usize;
            let (from, rest) = (bit / width, bit % width);
            let to = if rest < from { rest } else { rest + 1 };
            (from < n && to < n && n > 1).then_some((from, to))
        })
    }

    /// Whether directed edge `from → to` of an `n`-node graph is in the set.
    #[must_use]
    pub fn has_edge(&self, n: usize, from: usize, to: usize) -> bool {
        from != to && from < n && to < n && self.bit(edge_bit(n, from, to))
    }

    /// Set or clear directed edge `from → to` of an `n`-node graph.
    pub fn set_edge(&mut self, n: usize, from: usize, to: usize, present: bool) {
        self.set_bit(edge_bit(n, from, to), present);
    }

    /// Whether the two sets share an edge.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.words().iter().zip(other.words()).any(|(a, b)| a & b != 0)
    }

    /// Whether every edge of `other` is in this set.
    #[must_use]
    pub fn contains_all(&self, other: &Self) -> bool {
        let mine = self.words();
        other.words().iter().enumerate().all(|(i, &b)| b & !mine.get(i).copied().unwrap_or(0) == 0)
    }

    /// The union of the two sets.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a | b)
    }

    /// The edges of this set that are not in `other`.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        self.combine(other, |a, b| a & !b)
    }

    fn combine(&self, other: &Self, op: impl Fn(u64, u64) -> u64) -> Self {
        let (a, b) = (self.words(), other.words());
        let words: Vec<u64> = (0..a.len().max(b.len()))
            .map(|i| op(a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0)))
            .collect();
        Self::from_words(&words)
    }

    /// Opaque 64-bit key of the set.
    ///
    /// A single-word set keys as its word, which is the key the earlier encoding used. A wider
    /// set keys as an FNV-1a digest of its words. Distinct sets can share a key only by hash
    /// collision, which graph posterior construction refuses.
    #[must_use]
    pub fn key(&self) -> u64 {
        if let Some(word) = self.to_word() {
            return word;
        }
        let mut hash = Fnv1a64::new();
        for &word in self.words() {
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

impl From<&u64> for AdjacencyMask {
    fn from(word: &u64) -> Self {
        Self::from_word(*word)
    }
}

impl From<&AdjacencyMask> for AdjacencyMask {
    fn from(mask: &AdjacencyMask) -> Self {
        mask.clone()
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
        let words = self.words();
        let top = words.iter().rposition(|&w| w != 0).unwrap_or(0);
        let mut digits = format!("{:x}", words[top]);
        for word in words[..top].iter().rev() {
            write!(digits, "{word:016x}")?;
        }
        f.pad_integral(true, "0x", &digits)
    }
}

/// Bit index for directed edge `from → to` among `n` labeled nodes (`from != to`).
///
/// # Panics
///
/// Panics when the edge index exceeds `u32`.
#[must_use]
pub fn edge_bit(n: usize, from: usize, to: usize) -> u32 {
    debug_assert_ne!(from, to);
    debug_assert!(from < n && to < n);
    let index = from * (n - 1) + if to < from { to } else { to - 1 };
    u32::try_from(index).expect("directed edge index fits u32")
}

/// Number of possible directed edges on `n` labeled nodes.
#[must_use]
pub const fn n_directed_edges(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1))
}

impl fmt::Display for AdjacencyMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_word() {
            Some(word) => fmt::Display::fmt(&word, f),
            None => write!(f, "{self:#x}"),
        }
    }
}
