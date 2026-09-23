//! Index conversions shared by the discovery algorithms.

/// Converts a variable, node or column position to the `u32` used by dense ids.
///
/// Dense node ids, lags and score columns are `u32` throughout the workspace, so any position
/// derived from a graph or design already lies in the `u32` range.
#[inline]
#[must_use]
pub(crate) fn dense_u32(index: usize) -> u32 {
    debug_assert!(u32::try_from(index).is_ok(), "dense id index exceeds u32");
    #[allow(
        clippy::cast_possible_truncation,
        reason = "positions index u32-addressed dense ids, so they are below u32::MAX"
    )]
    let narrowed = index as u32;
    narrowed
}

/// Reduces a raw 64-bit random draw modulo `bound` and returns it as a `usize` index.
///
/// Any high bits lost on narrow targets only discard uniform entropy; the remainder is always
/// strictly below `bound`.
#[inline]
#[must_use]
pub(crate) fn bounded_index(draw: u64, bound: usize) -> usize {
    debug_assert!(bound > 0, "bounded_index requires a positive bound");
    #[allow(
        clippy::cast_possible_truncation,
        reason = "dropping high bits of a uniform draw keeps it uniform, and the modulo keeps the result below `bound`"
    )]
    let index = (draw as usize) % bound;
    index
}
