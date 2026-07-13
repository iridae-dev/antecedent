//! False discovery rate helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

/// Benjamini–Hochberg adjusted p-values (input order preserved).
#[must_use]
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| {
        p_values[a].partial_cmp(&p_values[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut adj = vec![0.0; m];
    let mut running: f64 = 1.0;
    for (rank_rev, &i) in idx.iter().rev().enumerate() {
        let rank = m - rank_rev; // 1..=m from largest p
        let candidate = (p_values[i] * m as f64 / rank as f64).min(1.0);
        running = running.min(candidate);
        adj[i] = running;
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_preserves_length_and_bounds() {
        let p = [0.001, 0.04, 0.5, 0.02];
        let a = benjamini_hochberg(&p);
        assert_eq!(a.len(), 4);
        assert!(a.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert!(a[0] <= a[2]);
    }
}
