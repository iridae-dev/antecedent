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
