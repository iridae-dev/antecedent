//! Combinatorial helpers for PC conditioning sets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, VariableId};

/// Lexicographic combinations of `k` items from `items` (allocating).
///
/// Prefer [`for_each_combination`] on hot paths.
#[must_use]
pub fn combinations(items: &[(VariableId, Lag)], k: usize) -> Vec<Vec<(VariableId, Lag)>> {
    let mut out = Vec::new();
    for_each_combination(items, k, &mut Vec::new(), |combo| {
        out.push(combo.to_vec());
        true
    });
    out
}

/// Invoke `visit` for each lexicographic `k`-combination, writing into `scratch`.
///
/// `visit` returns `false` to stop early. `scratch` is resized to `k` and reused.
/// Combination order is the same for every `T: Copy`: index tuples `(0,1,…,k-1)`,
/// then the rightmost index that can advance.
pub(crate) fn for_each_combination<T: Copy>(
    items: &[T],
    k: usize,
    scratch: &mut Vec<T>,
    mut visit: impl FnMut(&[T]) -> bool,
) {
    if k == 0 {
        scratch.clear();
        let _ = visit(scratch);
        return;
    }
    if k > items.len() {
        return;
    }
    scratch.clear();
    scratch.extend(items[..k].iter().copied());
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        if !visit(scratch) {
            return;
        }
        if !advance_combination(&mut idx, items.len()) {
            return;
        }
        for (slot, &i) in idx.iter().enumerate() {
            scratch[slot] = items[i];
        }
    }
}

/// Advance a lexicographic combination index. `false` when `idx` was the last one.
fn advance_combination(idx: &mut [usize], n: usize) -> bool {
    let k = idx.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if idx[i] != i + n - k {
            idx[i] += 1;
            for j in i + 1..k {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Lazily generated lexicographic `k`-combinations of `items` (positions ascending), each
/// yielded as a fresh `Vec`. Equal in content and order to [`for_each_combination`].
pub(crate) struct Combinations<'a, T> {
    items: &'a [T],
    idx: Vec<usize>,
    first: bool,
    done: bool,
}

impl<'a, T: Copy> Combinations<'a, T> {
    pub(crate) fn new(items: &'a [T], k: usize) -> Self {
        Self { items, idx: (0..k).collect(), first: true, done: k > items.len() }
    }
}

impl<T: Copy> Iterator for Combinations<'_, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        if self.done {
            return None;
        }
        if self.first {
            self.first = false;
        } else if !advance_combination(&mut self.idx, self.items.len()) {
            self.done = true;
            return None;
        }
        Some(self.idx.iter().map(|&i| self.items[i]).collect())
    }
}

/// Lexicographic combinations of static [`VariableId`]s.
#[must_use]
#[cfg(test)]
pub(crate) fn combinations_vars(items: &[VariableId], k: usize) -> Vec<Vec<VariableId>> {
    let mut out = Vec::new();
    for_each_combination_vars(items, k, &mut Vec::new(), |combo| {
        out.push(combo.to_vec());
        true
    });
    out
}

/// Invoke `visit` for each lexicographic `k`-combination of static variables.
pub(crate) fn for_each_combination_vars(
    items: &[VariableId],
    k: usize,
    scratch: &mut Vec<VariableId>,
    visit: impl FnMut(&[VariableId]) -> bool,
) {
    for_each_combination(items, k, scratch, visit);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_match_iterator() {
        let items = [
            (VariableId::from_raw(0), Lag::from_raw(1)),
            (VariableId::from_raw(1), Lag::from_raw(1)),
            (VariableId::from_raw(2), Lag::from_raw(2)),
        ];
        let alloc = combinations(&items, 2);
        let mut via = Vec::new();
        for_each_combination(&items, 2, &mut Vec::new(), |c| {
            via.push(c.to_vec());
            true
        });
        assert_eq!(alloc, via);
        assert_eq!(alloc.len(), 3);
    }

    #[test]
    fn lazy_combinations_match_the_visitor_order() {
        let items: Vec<_> = (0..6).map(VariableId::from_raw).collect();
        for k in 0..=7 {
            let lazy: Vec<_> = Combinations::new(&items, k).collect();
            assert_eq!(lazy, combinations_vars(&items, k), "k={k}");
        }
    }

    #[test]
    fn combinations_vars_match() {
        let items = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let alloc = combinations_vars(&items, 2);
        assert_eq!(alloc.len(), 3);
    }
}
