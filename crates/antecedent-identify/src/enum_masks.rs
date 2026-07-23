//! Shared identification helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::DenseNodeId;

/// Invoke `visit` for each size-`size` subset of `candidates` (lexicographic index order).
///
/// Returns `true` if `visit` ever returned `true` (early-stop signal).
///
/// Enumerates only the `C(m, size)` subsets — not all `2^m` bitmasks — so large
/// candidate pools remain tractable when [`AdjustmentSearchConfig::max_results`]
/// or minimal-set early-stop limits how many sizes are fully scanned.
pub(crate) fn for_each_mask_of_size(
    candidates: &[DenseNodeId],
    size: usize,
    mut visit: impl FnMut(&[DenseNodeId]) -> bool,
) -> bool {
    let m = candidates.len();
    if size > m {
        return false;
    }
    if size == 0 {
        return visit(&[]);
    }

    // Combinatorial number system: indices[0] < indices[1] < … < indices[size-1] < m
    let mut indices: Vec<usize> = (0..size).collect();
    let mut z = Vec::with_capacity(size);
    loop {
        z.clear();
        for &i in &indices {
            z.push(candidates[i]);
        }
        if visit(&z) {
            return true;
        }
        // Advance to next combination.
        let mut i = size;
        while i > 0 {
            i -= 1;
            if indices[i] != i + m - size {
                indices[i] += 1;
                for j in i + 1..size {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_combinations_not_powerset() {
        let c: Vec<DenseNodeId> = (0..5).map(DenseNodeId::from_raw).collect();
        let mut n = 0usize;
        for_each_mask_of_size(&c, 2, |_| {
            n += 1;
            false
        });
        assert_eq!(n, 10); // C(5,2)
    }

    #[test]
    fn early_stop_halts() {
        let c: Vec<DenseNodeId> = (0..8).map(DenseNodeId::from_raw).collect();
        let mut n = 0usize;
        let stopped = for_each_mask_of_size(&c, 3, |_| {
            n += 1;
            n >= 3
        });
        assert!(stopped);
        assert_eq!(n, 3);
    }
}
