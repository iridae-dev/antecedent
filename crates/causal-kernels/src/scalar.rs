//! Scalar reference kernels (correctness baseline).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)] // n as f64 for means/variances is intentional

use crate::view::{BitMaskView, F64VectorView};

/// Masked sum of a vector. When `mask` is `None`, all elements are included.
#[must_use]
pub fn masked_sum(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        // SAFETY: i < x.len()
        acc += unsafe { x.get_unchecked(i) };
    }
    acc
}

/// Masked mean. Returns `None` when no valid observations.
#[must_use]
pub fn masked_mean(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Option<f64> {
    let mut acc = 0.0;
    let mut n = 0usize;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        acc += unsafe { x.get_unchecked(i) };
        n += 1;
    }
    if n == 0 { None } else { Some(acc / n as f64) }
}

/// Population variance with optional mask. Returns `None` when `n == 0`.
#[must_use]
pub fn masked_variance(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Option<f64> {
    let mean = masked_mean(x, mask)?;
    let mut acc = 0.0;
    let mut n = 0usize;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        let d = unsafe { x.get_unchecked(i) } - mean;
        acc += d * d;
        n += 1;
    }
    if n == 0 { None } else { Some(acc / n as f64) }
}

/// Gather values at `indices` into `out` (must have `out.len() == indices.len()`).
///
/// # Panics
///
/// Panics if `out.len() != indices.len()` or any index is out of bounds.
pub fn gather(src: F64VectorView<'_>, indices: &[usize], out: &mut [f64]) {
    assert_eq!(out.len(), indices.len());
    for (dst, &idx) in out.iter_mut().zip(indices.iter()) {
        *dst = src.get(idx).expect("gather index in bounds");
    }
}

/// Copy `src` into `dst` (same length).
///
/// # Panics
///
/// Panics if lengths differ.
pub fn copy_vec(src: F64VectorView<'_>, dst: &mut [f64]) {
    assert_eq!(dst.len(), src.len());
    for (i, slot) in dst.iter_mut().enumerate() {
        *slot = unsafe { src.get_unchecked(i) };
    }
}
