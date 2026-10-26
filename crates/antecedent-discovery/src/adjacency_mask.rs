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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::{self, Write as _};
use std::sync::Arc;

use antecedent_core::Fnv1a64;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Endpoint, Pag};

use crate::error::DiscoveryError;
use crate::graph_posterior::{edge_bit, n_directed_edges};

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
            *self = Self::from_words(&self.words().to_vec());
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
                Some(crate::indexing::dense_u32(index) * 64 + bit)
            })
        })
    }

    /// The edges `(from, to)` of a mask on `n` nodes, in bit order (the inverse of
    /// [`edge_bit`]).
    pub(crate) fn edges(&self, n: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
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
    /// collision, which [`crate::GraphPosterior::from_atoms`] refuses.
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

// Edge-level access and conversion to and from the graph types, for any node count whose edge
// bits are addressable (see [`edge_bits_fit`]). The single-word functions of the same names in
// `graph_posterior` and at the crate root are the earlier `u64` API and cover graphs on at most
// eight nodes.

/// Whether bit `from → to` is set.
#[must_use]
pub fn has_edge(mask: impl Into<AdjacencyMask>, n: usize, from: usize, to: usize) -> bool {
    mask.into().has_edge(n, from, to)
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
pub(crate) fn parent_bits(mask: &AdjacencyMask, n: usize, node: usize) -> AdjacencyMask {
    debug_assert!(node < n);
    let mut parents = AdjacencyMask::EMPTY;
    for p in (0..n).filter(|&p| p != node && mask.bit(edge_bit(n, p, node))) {
        parents.set_bit(crate::indexing::dense_u32(p), true);
    }
    parents
}

/// Parent indices of `node` under `mask`, in increasing parent index.
#[must_use]
pub fn parents_of(mask: impl Into<AdjacencyMask>, n: usize, node: usize) -> Vec<u32> {
    let mut pa = Vec::new();
    fill_parents(&mask.into(), n, node, &mut pa);
    pa
}

/// Write parent indices of `node` into `parents`, reusing its allocation.
pub(crate) fn fill_parents(mask: &AdjacencyMask, n: usize, node: usize, parents: &mut Vec<u32>) {
    parents.clear();
    for p in 0..n {
        if p != node && mask.has_edge(n, p, node) {
            parents.push(crate::indexing::dense_u32(p));
        }
    }
}

/// Whether `mask` encodes a DAG on `n` nodes.
///
/// A Kahn peel over the set edges, `O(n + edges)`.
#[must_use]
pub fn mask_is_dag(mask: impl Into<AdjacencyMask>, n: usize) -> bool {
    let mask = mask.into();
    let edges: Vec<(usize, usize)> = mask.edges(n).collect();
    // `edges` is ordered by tail, so the children of a node are one contiguous run.
    let mut start = vec![0usize; n + 1];
    let mut in_degree = vec![0usize; n];
    for &(from, to) in &edges {
        start[from + 1] += 1;
        in_degree[to] += 1;
    }
    for node in 0..n {
        start[node + 1] += start[node];
    }
    let mut ready: Vec<usize> = (0..n).filter(|&node| in_degree[node] == 0).collect();
    let mut peeled = 0usize;
    while let Some(node) = ready.pop() {
        peeled += 1;
        for &(_, child) in &edges[start[node]..start[node + 1]] {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                ready.push(child);
            }
        }
    }
    peeled == n
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
    if !mask_is_dag(&mask, n_vars) {
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
            if i != j && mask.has_edge(n_vars, i, j) {
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
    if !edge_bits_fit(n_vars) {
        return Err(DiscoveryError::data_msg(format!(
            "{what}: n_vars={n_vars} has more edge bits than a u32 bit index addresses"
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
            let ij = mask.has_edge(n_vars, i, j);
            let ji = mask.has_edge(n_vars, j, i);
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
            mask.set_edge(n_vars, i, j, true);
            mask.set_edge(n_vars, j, i, true);
        } else if edge.at_a == Endpoint::Tail && edge.at_b == Endpoint::Arrow {
            mask.set_edge(n_vars, i, j, true);
        } else if edge.at_a == Endpoint::Arrow && edge.at_b == Endpoint::Tail {
            mask.set_edge(n_vars, j, i, true);
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
            let ij = mask.has_edge(n_vars, i, j);
            let ji = mask.has_edge(n_vars, j, i);
            if !ij && !ji {
                continue;
            }
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            let circle_a = mark_mask.has_edge(n_vars, i, j);
            let circle_b = mark_mask.has_edge(n_vars, j, i);
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
                    adj.set_edge(n_vars, i, j, true);
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    adj.set_edge(n_vars, j, i, true);
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    adj.set_edge(n_vars, i, j, true);
                    adj.set_edge(n_vars, j, i, true);
                }
                (Endpoint::Circle, Endpoint::Arrow) | (Endpoint::Tail, Endpoint::Circle) => {
                    adj.set_edge(n_vars, i, j, true);
                    marks.set_edge(n_vars, i, j, true);
                }
                (Endpoint::Arrow, Endpoint::Circle) | (Endpoint::Circle, Endpoint::Tail) => {
                    adj.set_edge(n_vars, j, i, true);
                    marks.set_edge(n_vars, j, i, true);
                }
                (Endpoint::Circle, Endpoint::Circle) => {
                    adj.set_edge(n_vars, i, j, true);
                    adj.set_edge(n_vars, j, i, true);
                    marks.set_edge(n_vars, i, j, true);
                    marks.set_edge(n_vars, j, i, true);
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
            let ij = mask.has_edge(n_vars, i, j);
            let ji = mask.has_edge(n_vars, j, i);
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
            mask.set_edge(n_vars, i, to.as_usize(), true);
        }
        for &nbr in admg.bidirected_neighbors(from) {
            let j = nbr.as_usize();
            if i < j {
                mask.set_edge(n_vars, i, j, true);
                mask.set_edge(n_vars, j, i, true);
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
        assert_eq!(mask.to_le_bytes().len(), 32, "a sixteen-node set keeps its 32-byte wire form");
        assert_eq!(AdjacencyMask::from_le_bytes(&mask.to_le_bytes()), mask);
        let cleared = mask.clone().with_bit(70, false).with_bit(239, false);
        assert_eq!(cleared.to_word(), Some(1 << 3));
        assert_eq!(
            format!("{mask:#x}"),
            format!("0x{:x}{:016x}{:016x}{:016x}", 1u64 << 47, 0, 1u64 << 6, 8)
        );
    }

    #[test]
    fn a_set_grows_past_the_inline_words_and_shrinks_back() {
        let wide = AdjacencyMask::EMPTY.with_bit(5, true).with_bit(1000, true).with_bit(40_000, true);
        assert!(wide.bit(5) && wide.bit(1000) && wide.bit(40_000));
        assert!(!wide.bit(999) && !wide.bit(40_001) && !wide.bit(1_000_000));
        assert_eq!(wide.count_ones(), 3);
        assert_eq!(wide.set_bits().collect::<Vec<_>>(), vec![5, 1000, 40_000]);
        let bytes = wide.to_le_bytes();
        assert_eq!(bytes.len(), (40_000 / 64 + 1) * 8);
        assert_eq!(AdjacencyMask::from_le_bytes(&bytes), wide);
        let shrunk = wide.clone().with_bit(40_000, false).with_bit(1000, false);
        assert_eq!(shrunk, AdjacencyMask::EMPTY.with_bit(5, true));
        assert_eq!(shrunk.to_le_bytes().len(), 32);
        assert_eq!(wide.with_bit(7, false).with_bit(40_000, true).count_ones(), 3);
    }

    #[test]
    fn equal_sets_hash_and_order_alike_whatever_their_history() {
        let built = AdjacencyMask::EMPTY.with_bit(900, true).with_bit(2, true);
        let reached = AdjacencyMask::from_words(&[4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 4])
            .with_bit(2, false)
            .with_bit(2, true);
        assert_eq!(built, reached);
        assert_eq!(built.key(), reached.key());
        assert_eq!(built.cmp(&reached), std::cmp::Ordering::Equal);
        assert!(AdjacencyMask::from_word(7) < AdjacencyMask::from_word(8));
    }

    #[test]
    fn subset_and_overlap_follow_set_semantics() {
        let a = AdjacencyMask::EMPTY.with_bit(5, true).with_bit(100, true);
        let b = AdjacencyMask::EMPTY.with_bit(100, true);
        let c = AdjacencyMask::EMPTY.with_bit(101, true);
        assert!(a.contains_all(&b) && !b.contains_all(&a));
        assert!(a.intersects(&b) && !a.intersects(&c));
        assert!(a.contains_all(&AdjacencyMask::EMPTY));
        assert_eq!(a.union(&c).count_ones(), 3);
        assert_eq!(a.difference(&b), AdjacencyMask::EMPTY.with_bit(5, true));
    }

    #[test]
    fn wide_sets_that_differ_get_different_keys() {
        let a = AdjacencyMask::EMPTY.with_bit(100, true);
        let b = AdjacencyMask::EMPTY.with_bit(101, true);
        assert_ne!(a.key(), b.key());
        assert_eq!(a.key(), a.key());
    }

    #[test]
    fn a_chain_on_many_nodes_is_a_dag_and_closing_it_is_not() {
        let n = 40;
        let chain = (0..n - 1).fold(AdjacencyMask::EMPTY, |mask, i| set_edge(mask, n, i, i + 1, true));
        assert!(mask_is_dag(&chain, n));
        assert_eq!(chain.edges(n).count(), n - 1);
        assert!(has_edge(&chain, n, 38, 39) && !has_edge(&chain, n, 39, 38));
        let closed = set_edge(chain, n, n - 1, 0, true);
        assert!(!mask_is_dag(&closed, n));
        assert_eq!(parents_of(&closed, n, 0), vec![u32::try_from(n - 1).unwrap()]);
    }
}
