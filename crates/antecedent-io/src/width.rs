//! Lossless `usize` counts for artifact builders that do not return [`Result`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// `usize` as `u64` without saturating.
///
/// Supported targets have `usize` no wider than `u64`, so this is the word
/// `u64::try_from` would produce. A wider `usize` fails to compile instead of
/// aliasing `u64::MAX`.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "usize::BITS <= 64 is asserted; the cast cannot truncate on a target this crate builds for"
)]
pub(crate) const fn lossless_u64(value: usize) -> u64 {
    const {
        assert!(usize::BITS <= 64, "refusing to narrow a usize that does not fit in u64");
    }
    value as u64
}
