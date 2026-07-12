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
