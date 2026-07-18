//! Portable optimized kernels (safe auto-vectorization friendly loops).
//!
//! These share the scalar semantic contract and are selected once per batch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)] // n as f64 for means/variances is intentional

use crate::view::{BitMaskView, F64VectorView};

/// Contiguous fast path for masked sum when unit-stride and no mask.
#[must_use]
pub fn masked_sum(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> f64 {
    if mask.is_none() {
        if let Some(slice) = x.as_slice() {
            return slice.iter().sum();
        }
    }
    crate::scalar::masked_sum(x, mask)
}

/// Contiguous fast path for masked mean.
#[must_use]
pub fn masked_mean(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Option<f64> {
    if mask.is_none() {
        if let Some(slice) = x.as_slice() {
            if slice.is_empty() {
                return None;
            }
            let sum: f64 = slice.iter().sum();
            return Some(sum / slice.len() as f64);
        }
    }
    crate::scalar::masked_mean(x, mask)
}

/// Contiguous fast path for population variance.
#[must_use]
pub fn masked_variance(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Option<f64> {
    if mask.is_none() {
        if let Some(slice) = x.as_slice() {
            if slice.is_empty() {
                return None;
            }
            let n = slice.len() as f64;
            let mean = slice.iter().sum::<f64>() / n;
            let var = slice
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            return Some(var);
        }
    }
    crate::scalar::masked_variance(x, mask)
}

/// Gather with contiguous source fast path.
pub fn gather(src: F64VectorView<'_>, indices: &[usize], out: &mut [f64]) {
    assert_eq!(out.len(), indices.len());
    if let Some(slice) = src.as_slice() {
        for (dst, &idx) in out.iter_mut().zip(indices.iter()) {
            *dst = slice[idx];
        }
        return;
    }
    crate::scalar::gather(src, indices, out);
}

/// Copy with contiguous source fast path.
pub fn copy_vec(src: F64VectorView<'_>, dst: &mut [f64]) {
    assert_eq!(dst.len(), src.len());
    if let Some(slice) = src.as_slice() {
        dst.copy_from_slice(slice);
        return;
    }
    crate::scalar::copy_vec(src, dst);
}

/// Contiguous fast path for population covariance.
#[must_use]
pub fn masked_covariance(
    x: F64VectorView<'_>,
    y: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    assert_eq!(x.len(), y.len(), "covariance views must share length");
    if mask.is_none() {
        if let (Some(xs), Some(ys)) = (x.as_slice(), y.as_slice()) {
            let n = xs.len();
            if n == 0 {
                return None;
            }
            let nf = n as f64;
            let mx = xs.iter().sum::<f64>() / nf;
            let my = ys.iter().sum::<f64>() / nf;
            let mut acc = 0.0;
            for i in 0..n {
                acc += (xs[i] - mx) * (ys[i] - my);
            }
            return Some(acc / nf);
        }
    }
    crate::scalar::masked_covariance(x, y, mask)
}

/// Contiguous standardize (same contract as scalar).
#[must_use]
pub fn standardize_inplace(x: &mut [f64], eps: f64) -> (f64, f64) {
    crate::scalar::standardize_inplace(x, eps)
}

/// Pairwise L1 fill (same contract; contiguous loops for auto-vectorization).
pub fn pairwise_l1_fill(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert_eq!(out.len(), n.saturating_mul(n), "pairwise out must be n*n");
    for i in 0..n {
        let xi = x[i];
        let row = &mut out[i * n..(i + 1) * n];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = (xi - x[j]).abs();
        }
    }
}

/// Contingency accumulation (portable = scalar; scatter-add resists vectorization).
pub fn accumulate_contingency(
    x_codes: &[u32],
    y_codes: &[u32],
    out: &mut [f64],
    n_y_levels: usize,
) {
    crate::scalar::accumulate_contingency(x_codes, y_codes, out, n_y_levels);
}

/// Row-subset contingency accumulation.
pub fn accumulate_contingency_rows(
    x_codes: &[u32],
    y_codes: &[u32],
    rows: &[usize],
    out: &mut [f64],
    n_y_levels: usize,
) {
    crate::scalar::accumulate_contingency_rows(x_codes, y_codes, rows, out, n_y_levels);
}

/// Weighted sum with contiguous zip.
#[must_use]
pub fn weighted_sum(x: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(x.len(), weights.len(), "weighted_sum length mismatch");
    x.iter()
        .zip(weights.iter())
        .map(|(xi, &w)| crate::scalar::sanitize_weight(w) * xi)
        .sum()
}

/// Weighted mean with contiguous zip.
#[must_use]
pub fn weighted_mean(x: &[f64], weights: &[f64]) -> Option<f64> {
    crate::scalar::weighted_mean(x, weights)
}

/// Weighted dot with contiguous zip.
#[must_use]
pub fn weighted_dot(x: &[f64], y: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "weighted_dot xy length");
    assert_eq!(x.len(), weights.len(), "weighted_dot weight length");
    let mut acc = 0.0;
    for i in 0..x.len() {
        acc += crate::scalar::sanitize_weight(weights[i]) * x[i] * y[i];
    }
    acc
}
