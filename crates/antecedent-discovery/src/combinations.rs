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
pub fn for_each_combination(
    items: &[(VariableId, Lag)],
    k: usize,
    scratch: &mut Vec<(VariableId, Lag)>,
    mut visit: impl FnMut(&[(VariableId, Lag)]) -> bool,
) {
    if k == 0 {
        scratch.clear();
        let _ = visit(scratch);
        return;
    }
    if k > items.len() {
        return;
    }
    scratch.resize(k, (VariableId::from_raw(0), Lag::CONTEMPORANEOUS));
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        for (slot, &i) in idx.iter().enumerate() {
            scratch[slot] = items[i];
        }
        if !visit(scratch) {
            return;
        }
        let mut i = k;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if idx[i] != i + items.len() - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Lexicographic combinations of static [`VariableId`]s.
#[must_use]
pub fn combinations_vars(items: &[VariableId], k: usize) -> Vec<Vec<VariableId>> {
    let mut out = Vec::new();
    for_each_combination_vars(items, k, &mut Vec::new(), |combo| {
        out.push(combo.to_vec());
        true
    });
    out
}

/// Invoke `visit` for each lexicographic `k`-combination of static variables.
pub fn for_each_combination_vars(
    items: &[VariableId],
    k: usize,
    scratch: &mut Vec<VariableId>,
    mut visit: impl FnMut(&[VariableId]) -> bool,
) {
    if k == 0 {
        scratch.clear();
        let _ = visit(scratch);
        return;
    }
    if k > items.len() {
        return;
    }
    scratch.resize(k, VariableId::from_raw(0));
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        for (slot, &i) in idx.iter().enumerate() {
            scratch[slot] = items[i];
        }
        if !visit(scratch) {
            return;
        }
        let mut i = k;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if idx[i] != i + items.len() - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
    }
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
    fn combinations_vars_match() {
        let items = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let alloc = combinations_vars(&items, 2);
        assert_eq!(alloc.len(), 3);
    }
}
