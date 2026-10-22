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
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Endpoint, Pag};

use crate::error::DiscoveryError;
use crate::graph_posterior::edge_bit;

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

// Graphs on up to `MAX_POSTERIOR_NODES` nodes: edge-level access and conversion to and from
// the graph types. The single-word functions of the same names in `graph_posterior` and at the
// crate root are the earlier `u64` API and cover graphs on at most eight nodes.

/// Whether bit `from → to` is set.
#[must_use]
pub fn has_edge(mask: impl Into<AdjacencyMask>, n: usize, from: usize, to: usize) -> bool {
    if from == to || from >= n || to >= n {
        return false;
    }
    mask.into().bit(edge_bit(n, from, to))
}

/// Set or clear directed edge `from → to`.
#[must_use]
pub fn set_edge(
    mask: impl Into<AdjacencyMask>,
    n: usize,
    from: usize,
    to: usize,
    present: bool,
) -> AdjacencyMask {
    mask.into().with_bit(edge_bit(n, from, to), present)
}

/// Parent set of `node` under `mask` as a bitset over the nodes (bit `p` is `p → node`).
#[must_use]
pub(crate) fn parent_bits(mask: AdjacencyMask, n: usize, node: usize) -> u16 {
    debug_assert!(n <= MAX_POSTERIOR_NODES && node < n);
    (0..n)
        .filter(|&p| p != node && mask.bit(edge_bit(n, p, node)))
        .fold(0u16, |bits, p| bits | (1 << p))
}

/// Parent indices of `node` under `mask`, in increasing parent index.
#[must_use]
pub fn parents_of(mask: impl Into<AdjacencyMask>, n: usize, node: usize) -> Vec<u32> {
    let mut pa = Vec::new();
    fill_parents(mask.into(), n, node, &mut pa);
    pa
}

/// Write parent indices of `node` into `parents`, reusing its allocation.
pub(crate) fn fill_parents(mask: AdjacencyMask, n: usize, node: usize, parents: &mut Vec<u32>) {
    parents.clear();
    for p in 0..n {
        if p != node && has_edge(mask, n, p, node) {
            parents.push(crate::indexing::dense_u32(p));
        }
    }
}

/// Whether `mask` encodes a DAG on `n` nodes.
///
/// An allocation-free Kahn peel over per-node parent bitsets. A node count past
/// [`MAX_POSTERIOR_NODES`] has no representable mask and is not a DAG.
#[must_use]
pub fn mask_is_dag(mask: impl Into<AdjacencyMask>, n: usize) -> bool {
    if n > MAX_POSTERIOR_NODES {
        return false;
    }
    let mask = mask.into();
    let mut parents = [0u16; MAX_POSTERIOR_NODES];
    for (node, pa) in parents.iter_mut().enumerate().take(n) {
        *pa = parent_bits(mask, n, node);
    }
    let mut remaining: u16 = if n == 0 { 0 } else { u16::MAX >> (MAX_POSTERIOR_NODES - n) };
    loop {
        let before = remaining;
        for (v, &pa) in parents.iter().enumerate().take(n) {
            if remaining & (1 << v) != 0 && pa & remaining == 0 {
                remaining &= !(1 << v);
            }
        }
        if remaining == 0 {
            return true;
        }
        if remaining == before {
            return false;
        }
    }
}

/// Build a [`Dag`] from a packed adjacency set.
///
/// # Errors
///
/// When `mask` is not a DAG on `n_vars` nodes, or `n_vars` exceeds packing capacity.
pub fn dag_from_adjacency_mask(
    mask: impl Into<AdjacencyMask>,
    n_vars: usize,
) -> Result<Dag, DiscoveryError> {
    let mask = mask.into();
    require_mask_capacity(n_vars, "dag_from_adjacency_mask")?;
    if !mask_is_dag(mask, n_vars) {
        let refusal = antecedent_graph::error::CyclicDagRefusal::condensation(
            antecedent_graph::error::DagOperation::DagProjection,
        );
        return Err(DiscoveryError::data_msg(format!(
            "adjacency mask {mask:#x} is not a DAG on {n_vars} nodes; {refusal}. The mask was \
             not passed to Dag::insert_directed"
        )));
    }
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut dag = Dag::with_variables(n_u32);
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i != j && has_edge(mask, n_vars, i, j) {
                dag.insert_directed(
                    DenseNodeId::from_raw(crate::indexing::dense_u32(i)),
                    DenseNodeId::from_raw(crate::indexing::dense_u32(j)),
                )?;
            }
        }
    }
    Ok(dag)
}

fn require_mask_capacity(n_vars: usize, what: &str) -> Result<(), DiscoveryError> {
    if n_vars == 0 {
        return Err(DiscoveryError::data_msg(format!("{what}: n_vars must be > 0")));
    }
    if n_vars > MAX_POSTERIOR_NODES {
        return Err(DiscoveryError::data_msg(format!(
            "{what}: n_vars={n_vars} exceeds the {MAX_POSTERIOR_NODES}-variable adjacency packing"
        )));
    }
    Ok(())
}

/// Build a [`Cpdag`] from a packed adjacency set.
///
/// Both directed bits of an unordered pair encode an undirected edge.
///
/// # Errors
///
/// Packing overflow, or an insert the CPDAG rejects.
pub fn cpdag_from_adjacency_mask(
    mask: impl Into<AdjacencyMask>,
    n_vars: usize,
) -> Result<Cpdag, DiscoveryError> {
    let mask = mask.into();
    require_mask_capacity(n_vars, "cpdag_from_adjacency_mask")?;
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut cpdag = Cpdag::with_variables(n_u32);
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let ij = has_edge(mask, n_vars, i, j);
            let ji = has_edge(mask, n_vars, j, i);
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            if ij && ji {
                cpdag.insert_undirected(a, b)?;
            } else if ij {
                cpdag.insert_directed(a, b)?;
            } else if ji {
                cpdag.insert_directed(b, a)?;
            }
        }
    }
    Ok(cpdag)
}

/// Pack a [`Cpdag`] into a directed-edge set (undirected = both bits).
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_mask_from_cpdag(cpdag: &Cpdag) -> Result<AdjacencyMask, DiscoveryError> {
    let n_vars = cpdag.node_count();
    require_mask_capacity(n_vars, "adjacency_mask_from_cpdag")?;
    let mut mask = AdjacencyMask::EMPTY;
    for edge in cpdag.edges() {
        let i = edge.a.as_usize();
        let j = edge.b.as_usize();
        if edge.is_undirected() {
            mask = set_edge(mask, n_vars, i, j, true);
            mask = set_edge(mask, n_vars, j, i, true);
        } else if edge.at_a == Endpoint::Tail && edge.at_b == Endpoint::Arrow {
            mask = set_edge(mask, n_vars, i, j, true);
        } else if edge.at_a == Endpoint::Arrow && edge.at_b == Endpoint::Tail {
            mask = set_edge(mask, n_vars, j, i, true);
        }
    }
    Ok(mask)
}

/// Build a [`Pag`] from packed adjacency and optional circle-endpoint sets.
///
/// Both directed bits encode a bidirected edge when neither end is circled.
/// A circle bit on `(from, to)` marks the `from` endpoint as a circle.
///
/// # Errors
///
/// Packing overflow, or an insert the PAG rejects.
pub fn pag_from_adjacency_mask(
    mask: impl Into<AdjacencyMask>,
    mark_mask: impl Into<AdjacencyMask>,
    n_vars: usize,
) -> Result<Pag, DiscoveryError> {
    let (mask, mark_mask) = (mask.into(), mark_mask.into());
    require_mask_capacity(n_vars, "pag_from_adjacency_mask")?;
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut pag = Pag::with_variables(n_u32);
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let ij = has_edge(mask, n_vars, i, j);
            let ji = has_edge(mask, n_vars, j, i);
            if !ij && !ji {
                continue;
            }
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            let circle_a = has_edge(mark_mask, n_vars, i, j);
            let circle_b = has_edge(mark_mask, n_vars, j, i);
            if ij && ji {
                if circle_a && circle_b {
                    pag.insert_circle_circle(a, b)?;
                } else if !circle_a && !circle_b {
                    pag.insert_bidirected(a, b)?;
                } else if circle_a {
                    pag.insert_circle_arrow(a, b)?;
                } else {
                    pag.insert_circle_arrow(b, a)?;
                }
            } else if ij {
                if circle_a {
                    pag.insert_circle_arrow(a, b)?;
                } else {
                    pag.insert_directed(a, b)?;
                }
            } else if circle_b {
                pag.insert_circle_arrow(b, a)?;
            } else {
                pag.insert_directed(b, a)?;
            }
        }
    }
    Ok(pag)
}

/// Pack a [`Pag`] into adjacency and circle-endpoint sets.
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_masks_from_pag(
    pag: &Pag,
) -> Result<(AdjacencyMask, AdjacencyMask), DiscoveryError> {
    let n_vars = pag.node_count();
    require_mask_capacity(n_vars, "adjacency_masks_from_pag")?;
    let mut adj = AdjacencyMask::EMPTY;
    let mut marks = AdjacencyMask::EMPTY;
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            let Some(edge) = pag.edge_between(a, b) else {
                continue;
            };
            let (at_i, at_j) = if edge.a.as_usize() == i {
                (edge.at_a, edge.at_b)
            } else {
                (edge.at_b, edge.at_a)
            };
            match (at_i, at_j) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    adj = set_edge(adj, n_vars, j, i, true);
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                    adj = set_edge(adj, n_vars, j, i, true);
                }
                (Endpoint::Circle, Endpoint::Arrow) | (Endpoint::Tail, Endpoint::Circle) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                    marks = set_edge(marks, n_vars, i, j, true);
                }
                (Endpoint::Arrow, Endpoint::Circle) | (Endpoint::Circle, Endpoint::Tail) => {
                    adj = set_edge(adj, n_vars, j, i, true);
                    marks = set_edge(marks, n_vars, j, i, true);
                }
                (Endpoint::Circle, Endpoint::Circle) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                    adj = set_edge(adj, n_vars, j, i, true);
                    marks = set_edge(marks, n_vars, i, j, true);
                    marks = set_edge(marks, n_vars, j, i, true);
                }
                (Endpoint::Tail, Endpoint::Tail) => {
                    return Err(DiscoveryError::unsupported(
                        "PAG packing cannot represent a tail-tail edge; refusing to drop it",
                    ));
                }
                (Endpoint::Conflict, _) | (_, Endpoint::Conflict) => {
                    return Err(DiscoveryError::unsupported(
                        "PAG packing cannot represent a conflict endpoint; refusing to drop it",
                    ));
                }
            }
        }
    }
    Ok((adj, marks))
}

/// Build an [`Admg`] from a packed adjacency set.
///
/// Both directed bits of an unordered pair encode a bidirected edge. A single
/// directed bit encodes a directed edge. Circle marks are ignored: this is
/// ADMG semantics, not PAG.
///
/// A pair that is both directed and bidirected cannot be packed into one
/// bitmask (same limitation as PAG packing). Front-door `T→M→Y` with `T↔Y`
/// packs cleanly because the bidirected pair is not also directed.
///
/// # Errors
///
/// Packing overflow, a directed cycle, or an insert the ADMG rejects.
pub fn admg_from_adjacency_mask(
    mask: impl Into<AdjacencyMask>,
    n_vars: usize,
) -> Result<Admg, DiscoveryError> {
    let mask = mask.into();
    require_mask_capacity(n_vars, "admg_from_adjacency_mask")?;
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut admg = Admg::with_variables(n_u32);
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let ij = has_edge(mask, n_vars, i, j);
            let ji = has_edge(mask, n_vars, j, i);
            if !ij && !ji {
                continue;
            }
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            if ij && ji {
                admg.insert_bidirected(a, b)?;
            } else if ij {
                admg.insert_directed(a, b)?;
            } else {
                admg.insert_directed(b, a)?;
            }
        }
    }
    Ok(admg)
}

/// Pack an [`Admg`] into a directed-edge set (bidirected = both bits).
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_mask_from_admg(admg: &Admg) -> Result<AdjacencyMask, DiscoveryError> {
    let n_vars = admg.node_count();
    require_mask_capacity(n_vars, "adjacency_mask_from_admg")?;
    let mut mask = AdjacencyMask::EMPTY;
    for i in 0..n_vars {
        let from = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
        for &to in admg.children(from) {
            mask = set_edge(mask, n_vars, i, to.as_usize(), true);
        }
        for &nbr in admg.bidirected_neighbors(from) {
            let j = nbr.as_usize();
            if i < j {
                mask = set_edge(mask, n_vars, i, j, true);
                mask = set_edge(mask, n_vars, j, i, true);
            }
        }
    }
    Ok(mask)
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
