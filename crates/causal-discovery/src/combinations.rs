//! Combinatorial helpers for PC conditioning sets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{Lag, VariableId};

/// Lexicographic combinations of `k` items from `items`.
pub(crate) fn combinations(items: &[(VariableId, Lag)], k: usize) -> Vec<Vec<(VariableId, Lag)>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] != i + items.len() - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
    }
}
