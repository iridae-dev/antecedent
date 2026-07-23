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

/// Population covariance of paired observations. Returns `None` when no valid pairs.
///
/// Deterministic left-to-right reduction. Tolerance class: `StableFloat`.
#[must_use]
pub fn masked_covariance(
    x: F64VectorView<'_>,
    y: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Option<f64> {
    assert_eq!(x.len(), y.len(), "covariance views must share length");
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0usize;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        // SAFETY: i < len for both views
        sx += unsafe { x.get_unchecked(i) };
        sy += unsafe { y.get_unchecked(i) };
        n += 1;
    }
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    let mut acc = 0.0;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        let dx = unsafe { x.get_unchecked(i) } - mx;
        let dy = unsafe { y.get_unchecked(i) } - my;
        acc += dx * dy;
    }
    Some(acc / nf)
}

/// Standardize `x` in place: subtract mean, divide by `max(sample_sd, eps)`.
///
/// Sample SD uses `n - 1` when `n > 1`; otherwise scale is `max(1, eps)`.
/// Returns `(mean, scale)` applied. Tolerance class: `StableFloat`.
///
/// # Panics
///
/// Never panics for finite `eps`; empty slice returns `(0.0, eps.max(1.0))` without mutation.
#[must_use]
pub fn standardize_inplace(x: &mut [f64], eps: f64) -> (f64, f64) {
    let eps = if eps.is_finite() && eps > 0.0 { eps } else { 1e-12 };
    let n = x.len();
    if n == 0 {
        return (0.0, eps.max(1.0));
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut var = 0.0;
    for &v in x.iter() {
        let d = v - mean;
        var += d * d;
    }
    let scale = if n > 1 { (var / (n - 1) as f64).sqrt().max(eps) } else { 1.0_f64.max(eps) };
    for v in x.iter_mut() {
        *v = (*v - mean) / scale;
    }
    (mean, scale)
}

/// Fill `out` (`n * n`) with pairwise L1 distances `|x_i - x_j|`.
///
/// Deterministic row-major fill. Tolerance class: `Exact` for finite inputs.
///
/// # Panics
///
/// Panics if `out.len() != x.len() * x.len()`.
pub fn pairwise_l1_fill(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    assert_eq!(out.len(), n.saturating_mul(n), "pairwise out must be n*n");
    for i in 0..n {
        let xi = x[i];
        let row = i * n;
        for j in 0..n {
            out[row + j] = (xi - x[j]).abs();
        }
    }
}

/// Accumulate contingency counts: `out[x_code * n_y_levels + y_code] += 1` for each row.
///
/// `x_codes` / `y_codes` are level indexes (same length). Counts stored as `f64`.
/// Tolerance class: `Exact`.
///
/// # Panics
///
/// Panics if code lengths differ, `n_y_levels == 0`, `out` is too short, or a code is
/// out of range for the table shape.
pub fn accumulate_contingency(
    x_codes: &[u32],
    y_codes: &[u32],
    out: &mut [f64],
    n_y_levels: usize,
) {
    assert_eq!(x_codes.len(), y_codes.len(), "contingency code lengths");
    assert!(n_y_levels > 0, "n_y_levels must be > 0");
    let x_cardinality = out.len() / n_y_levels;
    assert_eq!(out.len(), x_cardinality.saturating_mul(n_y_levels), "contingency out shape");
    for (&xc, &yc) in x_codes.iter().zip(y_codes.iter()) {
        let ix = xc as usize;
        let iy = yc as usize;
        assert!(ix < x_cardinality && iy < n_y_levels, "contingency code out of range");
        out[ix * n_y_levels + iy] += 1.0;
    }
}

/// Accumulate contingency over a row subset (indexes into `x_codes` / `y_codes`).
///
/// # Panics
///
/// Same as [`accumulate_contingency`], plus out-of-range row indexes.
pub fn accumulate_contingency_rows(
    x_codes: &[u32],
    y_codes: &[u32],
    rows: &[usize],
    out: &mut [f64],
    n_y_levels: usize,
) {
    assert_eq!(x_codes.len(), y_codes.len(), "contingency code lengths");
    assert!(n_y_levels > 0, "n_y_levels must be > 0");
    let x_cardinality = out.len() / n_y_levels;
    assert_eq!(out.len(), x_cardinality.saturating_mul(n_y_levels), "contingency out shape");
    for &r in rows {
        let xc = x_codes[r] as usize;
        let yc = y_codes[r] as usize;
        assert!(xc < x_cardinality && yc < n_y_levels, "contingency code out of range");
        out[xc * n_y_levels + yc] += 1.0;
    }
}

/// Bootstrap / weighted-CI contract: non-finite and negative weights become `0`.
#[inline]
#[must_use]
pub fn sanitize_weight(w: f64) -> f64 {
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

/// Weighted sum `Σ w_i x_i` (non-finite / negative weights treated as 0).
///
/// Deterministic left-to-right reduction. Tolerance class: `StableFloat`.
///
/// # Panics
///
/// Panics if lengths differ.
#[must_use]
pub fn weighted_sum(x: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(x.len(), weights.len(), "weighted_sum length mismatch");
    let mut acc = 0.0;
    for (xi, &w) in x.iter().zip(weights.iter()) {
        acc += sanitize_weight(w) * xi;
    }
    acc
}

/// Weighted mean. Returns `None` when total weight is 0.
///
/// # Panics
///
/// Panics if lengths differ.
#[must_use]
pub fn weighted_mean(x: &[f64], weights: &[f64]) -> Option<f64> {
    assert_eq!(x.len(), weights.len(), "weighted_mean length mismatch");
    let mut sw = 0.0;
    let mut sx = 0.0;
    for (xi, &w) in x.iter().zip(weights.iter()) {
        let ww = sanitize_weight(w);
        sw += ww;
        sx += ww * xi;
    }
    if sw <= 0.0 { None } else { Some(sx / sw) }
}

/// Weighted dot product `Σ w_i x_i y_i`.
///
/// # Panics
///
/// Panics if lengths differ.
#[must_use]
pub fn weighted_dot(x: &[f64], y: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "weighted_dot xy length");
    assert_eq!(x.len(), weights.len(), "weighted_dot weight length");
    let mut acc = 0.0;
    for i in 0..x.len() {
        acc += sanitize_weight(weights[i]) * x[i] * y[i];
    }
    acc
}
