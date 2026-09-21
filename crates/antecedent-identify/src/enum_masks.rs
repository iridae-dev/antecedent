//! Shared identification helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::DenseNodeId;

use crate::error::IdentificationError;

/// Result of [`first_set_by_size`].
pub(crate) struct FirstSet {
    /// Smallest passing subset in size-then-index order, if the search reached one.
    pub found: Option<Vec<DenseNodeId>>,
    /// Tests run, counting the `spent` already charged to the caller.
    pub examined: u64,
    /// The budget refused a test while subsets remained untested and none had passed.
    pub budget_exhausted: bool,
}

/// Smallest subset of `candidates` that `test` accepts, searching sizes ascending.
///
/// `spent` tests are already charged against `max_examinations`. The budget is noticed by
/// the first subset it refuses to test, so a search that ends exactly on its budget is
/// complete rather than bounded. Errors from `test` end the search.
pub(crate) fn first_set_by_size(
    candidates: &[DenseNodeId],
    spent: u64,
    max_examinations: u64,
    mut test: impl FnMut(&[DenseNodeId]) -> Result<bool, IdentificationError>,
) -> Result<FirstSet, IdentificationError> {
    let mut examined = spent;
    let mut found = None;
    let mut budget_exhausted = false;
    for size in 0..=candidates.len() {
        let mut error = None;
        for_each_mask_of_size(candidates, size, |z| {
            if examined >= max_examinations {
                budget_exhausted = true;
                return true;
            }
            examined += 1;
            match test(z) {
                Ok(true) => {
                    found = Some(z.to_vec());
                    true
                }
                Ok(false) => false,
                Err(e) => {
                    error = Some(e);
                    true
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        if found.is_some() || budget_exhausted {
            break;
        }
    }
    Ok(FirstSet { found, examined, budget_exhausted })
}

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
    fn first_set_by_size_returns_smallest_in_index_order() {
        let c: Vec<DenseNodeId> = (0..5).map(DenseNodeId::from_raw).collect();
        // Accept the first subset containing both 1 and 3: the size-2 subset {1, 3}.
        let out = first_set_by_size(&c, 0, 1_000, |z| {
            Ok(z.contains(&DenseNodeId::from_raw(1)) && z.contains(&DenseNodeId::from_raw(3)))
        })
        .unwrap();
        assert_eq!(out.found, Some(vec![DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)]));
        assert!(!out.budget_exhausted);
        // 1 (empty) + 5 (singletons) + the size-2 subsets up to and including {1, 3}: {0,1},
        // {0,2}, {0,3}, {0,4}, {1,2}, {1,3}.
        assert_eq!(out.examined, 12);
    }

    #[test]
    fn first_set_by_size_ending_exactly_on_budget_is_complete() {
        let c: Vec<DenseNodeId> = (0..3).map(DenseNodeId::from_raw).collect();
        // 2^3 = 8 subsets, all rejected, budget 8: complete, not bounded.
        let complete = first_set_by_size(&c, 0, 8, |_| Ok(false)).unwrap();
        assert!(complete.found.is_none());
        assert!(!complete.budget_exhausted);
        assert_eq!(complete.examined, 8);
        // Budget 7 refuses the eighth subset.
        let bounded = first_set_by_size(&c, 0, 7, |_| Ok(false)).unwrap();
        assert!(bounded.budget_exhausted);
        assert_eq!(bounded.examined, 7);
    }

    #[test]
    fn first_set_by_size_charges_spent_tests() {
        let c: Vec<DenseNodeId> = (0..3).map(DenseNodeId::from_raw).collect();
        let out = first_set_by_size(&c, 5, 6, |_| Ok(false)).unwrap();
        assert!(out.budget_exhausted);
        assert_eq!(out.examined, 6);
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
