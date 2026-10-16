//! Identity encoding and request plumbing shared by the `recalc_*` routes.
//!
//! A stage identity is what resume and reuse are keyed on, so every input is folded as a
//! typed, length-delimited field: never through `Debug` or `Display`, whose output is not a
//! stability contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::recalc_receipt::RecalcRunError;
use crate::graph::{Dag, DenseNodeId};
use antecedent_core::NodeRef;
use antecedent_core::recalc::StageIdentity;
use antecedent_data::{DataError, TabularData};
use std::sync::atomic::{AtomicU64, Ordering};

/// Identity of a fixed text under `label`.
pub(super) fn literal(label: &str, text: &str) -> StageIdentity {
    StageIdentity::of(label, &[text.as_bytes()])
}

/// Identity of byte parts under `label`.
pub(super) fn identity(label: &str, parts: &[Vec<u8>]) -> StageIdentity {
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    StageIdentity::of(label, &refs)
}

/// IEEE-754 bit patterns, little endian, concatenated.
pub(super) fn float_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}

/// Canonical CBOR of a wire value.
///
/// A value that cannot be encoded gets a process-unique marker, so it never shares an
/// identity (and so never reuses a retained stage) with anything else.
pub(super) fn wire_bytes<T: serde::Serialize>(value: &T) -> Vec<u8> {
    static UNENCODABLE: AtomicU64 = AtomicU64::new(0);
    antecedent_io::to_cbor(value).unwrap_or_else(|_| {
        let mut marker = b"unencodable:".to_vec();
        marker.extend_from_slice(&UNENCODABLE.fetch_add(1, Ordering::Relaxed).to_le_bytes());
        marker
    })
}

/// A directed graph over `columns` variables with the declared `edges`, deduplicated.
///
/// `code` is the refusal for a graph the request cannot form.
pub(super) fn dag_from_edges(
    columns: usize,
    edges: &[(u32, u32)],
    code: &'static str,
) -> Result<Dag, RecalcRunError> {
    let n = u32::try_from(columns).map_err(|_| RecalcRunError::Request(code))?;
    let mut graph = Dag::with_variables(n);
    let mut edges = edges.to_vec();
    edges.sort_unstable();
    edges.dedup();
    for (from, to) in edges {
        graph
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(|_| RecalcRunError::Request(code))?;
    }
    Ok(graph)
}

/// The table of named numeric columns.
pub(super) fn table_from_columns(columns: &[(String, Vec<f64>)]) -> Result<TabularData, DataError> {
    TabularData::from_f64_columns(
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect::<Vec<_>>(),
    )
}

/// Typed, length-delimited identity input. Each field carries a kind byte, and each
/// variable-length field its length, so no two field sequences encode alike.
#[derive(Default)]
pub(super) struct Fields(Vec<u8>);

impl Fields {
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn put(mut self, kind: u8, payload: &[u8]) -> Self {
        self.0.push(kind);
        self.0.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        self.0.extend_from_slice(payload);
        self
    }

    /// A text field.
    pub(super) fn text(self, value: &str) -> Self {
        self.put(b's', value.as_bytes())
    }

    /// An unsigned integer field.
    pub(super) fn word(self, value: u64) -> Self {
        self.put(b'u', &value.to_le_bytes())
    }

    /// A count or length.
    pub(super) fn len(self, value: usize) -> Self {
        self.word(u64::try_from(value).unwrap_or(u64::MAX))
    }

    /// A boolean field.
    pub(super) fn flag(self, value: bool) -> Self {
        self.put(b'b', &[u8::from(value)])
    }

    /// A float, by bit pattern.
    pub(super) fn real(self, value: f64) -> Self {
        self.put(b'f', &value.to_bits().to_le_bytes())
    }

    /// A float sequence, by bit patterns.
    pub(super) fn reals(self, values: &[f64]) -> Self {
        self.put(b'F', &float_bytes(values))
    }

    /// A sequence of raw ids.
    pub(super) fn ids(self, values: &[u32]) -> Self {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.put(b'U', &bytes)
    }

    /// A sequence of `(from, to)` pairs, in the given order.
    pub(super) fn pairs(self, values: &[(u32, u32)]) -> Self {
        let bytes: Vec<u8> =
            values.iter().flat_map(|(a, b)| [a.to_le_bytes(), b.to_le_bytes()]).flatten().collect();
        self.put(b'P', &bytes)
    }

    /// A sequence of texts, each length-delimited.
    pub(super) fn texts<S: AsRef<str>>(self, values: &[S]) -> Self {
        let mut this = self.len(values.len());
        for value in values {
            this = this.text(value.as_ref());
        }
        this
    }

    /// Graph node references, each by kind and coordinates.
    pub(super) fn nodes(self, nodes: &[NodeRef]) -> Self {
        let mut this = self.len(nodes.len());
        for node in nodes {
            this = match *node {
                NodeRef::Static(v) => this.text("static").word(u64::from(v.raw())),
                NodeRef::Lagged { variable, lag } => {
                    this.text("lagged").word(u64::from(variable.raw())).word(u64::from(lag.raw()))
                }
                NodeRef::Unfolded { variable, offset } => this
                    .text("unfolded")
                    .word(u64::from(variable.raw()))
                    .word(u64::from(u32::from_le_bytes(offset.to_le_bytes()))),
                NodeRef::Context { variable, environment } => this
                    .text("context")
                    .word(u64::from(variable.raw()))
                    .flag(environment.is_some())
                    .word(environment.map_or(0, |e| u64::from(e.raw()))),
            };
        }
        this
    }

    /// Raw bytes, such as an artifact.
    pub(super) fn bytes(self, value: &[u8]) -> Self {
        self.put(b'x', value)
    }

    /// A wire value, by its canonical CBOR.
    pub(super) fn wire<T: serde::Serialize>(self, value: &T) -> Self {
        self.put(b'w', &wire_bytes(value))
    }

    /// The encoded bytes, for a caller that folds them into a larger identity.
    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// Identity of the fields under `label`.
    pub(super) fn identity(self, label: &str) -> StageIdentity {
        StageIdentity::of(label, &[&self.0])
    }
}
