//! Shared identification helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_graph::DenseNodeId;

/// Invoke `visit` for each size-`size` subset of `candidates` (bitmask order).
///
/// Returns `true` if `visit` ever returned `true` (early-stop signal).
pub(crate) fn for_each_mask_of_size(
    candidates: &[DenseNodeId],
    size: usize,
    mut visit: impl FnMut(&[DenseNodeId]) -> bool,
) -> bool {
    let m = candidates.len();
    if size > m {
        return false;
    }
    let total_masks = 1usize << m;
    let mut z = Vec::with_capacity(size);
    for mask in 0..total_masks {
        if mask.count_ones() as usize != size {
            continue;
        }
        z.clear();
        for i in 0..m {
            if (mask & (1 << i)) != 0 {
                z.push(candidates[i]);
            }
        }
        if visit(&z) {
            return true;
        }
    }
    false
}
