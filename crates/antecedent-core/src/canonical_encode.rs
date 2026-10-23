//! Canonical encoding shared by every type that enters a semantic identity.
//!
//! A type that contributes to an identity implements [`CanonicalEncode`] beside
//! its own definition. Numeric tags live next to the enums they number, in
//! exhaustive matches, so a new variant fails to compile until it is given a tag
//! and a canonical form.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// A value with one canonical byte encoding on a [`CanonicalWriter`].
///
/// Implementations must be injective on the scientific content they stand for
/// and must not depend on allocation, ordering of unordered containers, or
/// caller-chosen labels that are not part of that content.
pub trait CanonicalEncode {
    /// Append the canonical encoding to `writer`.
    fn encode(&self, writer: &mut CanonicalWriter);
}

impl CanonicalEncode for VariableId {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(u64::from(self.raw()));
    }
}

/// A length as a canonical word.
pub(crate) fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// A signed 32-bit integer as the word of its two's-complement bits.
pub(crate) fn i32_bits(value: i32) -> u64 {
    u64::from(u32::from_le_bytes(value.to_le_bytes()))
}

/// A signed 64-bit integer as the word of its two's-complement bits.
pub(crate) fn i64_bits(value: i64) -> u64 {
    u64::from_le_bytes(value.to_le_bytes())
}

/// A flag as a canonical word.
pub(crate) fn flag(value: bool) -> u64 {
    u64::from(u8::from(value))
}

/// A length-prefixed list of variables in the order given.
pub(crate) fn encode_variables(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    writer.word(count(variables.len()));
    for variable in variables {
        variable.encode(writer);
    }
}

/// A length-prefixed variable set: the variables in ascending id order, so the
/// order a caller listed them in does not reach the bytes. Repeats are kept, so
/// two lists encode alike only when one is a permutation of the other.
pub(crate) fn encode_variable_set(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    let mut sorted = variables.to_vec();
    sorted.sort_unstable();
    encode_variables(writer, &sorted);
}

/// A length-prefixed list of encodable items in the order given.
pub(crate) fn encode_list<T: CanonicalEncode>(writer: &mut CanonicalWriter, items: &[T]) {
    writer.word(count(items.len()));
    for item in items {
        item.encode(writer);
    }
}

/// Domain tag of the per-item digests [`encode_set`] sorts.
const SET_ITEM_TAG: &str = "antecedent.canonical_set_item";

/// A length-prefixed unordered collection: each item is digested on its own,
/// and the digests are written in ascending order. Repeats are kept, so two
/// collections encode alike only when one is a permutation of the other.
pub(crate) fn encode_set<T: CanonicalEncode>(writer: &mut CanonicalWriter, items: &[T]) {
    let mut digests = items
        .iter()
        .map(|item| {
            let mut item_writer = CanonicalWriter::new(SET_ITEM_TAG);
            item.encode(&mut item_writer);
            item_writer.finish()
        })
        .collect::<Vec<[u8; 32]>>();
    digests.sort_unstable();
    writer.word(count(digests.len()));
    for digest in &digests {
        writer.bytes(digest);
    }
}

/// An optional encodable value: a presence word (`0` or `1`), then the value.
pub(crate) fn encode_option<T: CanonicalEncode>(writer: &mut CanonicalWriter, value: Option<&T>) {
    match value {
        Some(value) => {
            writer.word(1);
            value.encode(writer);
        }
        None => {
            writer.word(0);
        }
    }
}

/// A length-prefixed list of indices or counts in the order given.
pub(crate) fn encode_usizes(writer: &mut CanonicalWriter, values: &[usize]) {
    writer.word(count(values.len()));
    for value in values {
        writer.word(count(*value));
    }
}

/// A length-prefixed list of flags in the order given.
pub(crate) fn encode_flags(writer: &mut CanonicalWriter, values: &[bool]) {
    writer.word(count(values.len()));
    for value in values {
        writer.word(flag(*value));
    }
}

/// A length-prefixed list of unsigned 32-bit values in the order given.
pub(crate) fn encode_u32s(writer: &mut CanonicalWriter, values: &[u32]) {
    writer.word(count(values.len()));
    for value in values {
        writer.word(u64::from(*value));
    }
}

/// A length-prefixed list of signed 64-bit values in the order given.
pub(crate) fn encode_i64s(writer: &mut CanonicalWriter, values: &[i64]) {
    writer.word(count(values.len()));
    for value in values {
        writer.word(i64_bits(*value));
    }
}
