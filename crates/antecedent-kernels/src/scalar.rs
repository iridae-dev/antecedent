//! Scalar reference kernels (correctness baseline).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "n as f64 for means/variances is intentional")]

use crate::view::{BitMaskView, F64VectorView};

/// A mask must cover exactly the vector it masks: `BitMaskView::get` reads out of
/// range as "excluded", so a short mask would silently drop the tail.
fn check_mask_len(len: usize, mask: Option<BitMaskView<'_>>) {
    if let Some(m) = mask {
        assert_eq!(m.len(), len, "mask length must equal vector length");
    }
}

/// Masked sum of a vector. When `mask` is `None`, all elements are included.
///
/// # Panics
///
/// Panics if `mask` is present and its length differs from `x`.
#[must_use]
pub fn masked_sum(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> f64 {
    check_mask_len(x.len(), mask);
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
///
/// # Panics
///
/// Panics if `mask` is present and its length differs from `x`.
#[must_use]
pub fn masked_mean(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Option<f64> {
    check_mask_len(x.len(), mask);
    let mut acc = 0.0;
    let mut n = 0usize;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        // SAFETY: i < x.len()
        acc += unsafe { x.get_unchecked(i) };
        n += 1;
    }
    if n == 0 { None } else { Some(acc / n as f64) }
}

/// Population variance with optional mask. Returns `None` when `n == 0`.
///
/// # Panics
///
/// Panics if `mask` is present and its length differs from `x`.
#[must_use]
pub fn masked_variance(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Option<f64> {
    let mean = masked_mean(x, mask)?;
    let mut acc = 0.0;
    let mut n = 0usize;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        // SAFETY: i < x.len()
        let d = unsafe { x.get_unchecked(i) } - mean;
        acc += d * d;
        n += 1;
    }
    if n == 0 { None } else { Some(acc / n as f64) }
}

/// Gather values at `indices` into `out` (`out.len()` must equal `indices.len()`).
///
/// # Errors
///
/// [`crate::view::ViewError::InvalidShape`] when `out.len() != indices.len()`, or
/// [`crate::view::ViewError::OutOfBounds`] when an index does not address `src`.
pub fn gather(
    src: F64VectorView<'_>,
    indices: &[usize],
    out: &mut [f64],
) -> Result<(), crate::view::ViewError> {
    if out.len() != indices.len() {
        return Err(crate::view::ViewError::InvalidShape {
            message: "gather out.len() must equal indices.len()",
        });
    }
    for (dst, &idx) in out.iter_mut().zip(indices.iter()) {
        *dst = src.get(idx)?;
    }
    Ok(())
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
    check_mask_len(x.len(), mask);
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0usize;
    for i in 0..x.len() {
        if mask.is_some_and(|m| !m.get(i)) {
            continue;
        }
        // SAFETY: i < x.len(), and y.len() == x.len() is asserted above
        sx += unsafe { x.get_unchecked(i) };
        // SAFETY: as above, i < y.len()
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
        // SAFETY: i < x.len(), and y.len() == x.len() is asserted above
        let dx = unsafe { x.get_unchecked(i) } - mx;
        // SAFETY: as above, i < y.len()
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

/// Bootstrap / weighted-CI contract: non-finite and negative weights become `0`.
#[inline]
#[must_use]
pub fn sanitize_weight(w: f64) -> f64 {
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

/// Kish effective sample size `(Σw)² / Σw²` of a weight vector.
///
/// This is the single workspace owner of the Kish ESS. Contract:
///
/// * normalised and unnormalised weights are both accepted (the ratio is
///   scale-free); weights are taken as given, with no sanitising;
/// * any non-finite weight (`NaN`, `±∞`) yields `NaN`: impossible weights
///   must never be mistaken for a uniform, healthy sample;
/// * an empty slice or an all-zero vector carries no mass and yields `0.0`;
/// * finite weights whose squares overflow are rescaled by their largest
///   magnitude, so the result stays finite and exact to rounding.
///
/// A caller that gates on `ess.is_finite() && ess >= floor` therefore refuses
/// both the `NaN` and the zero-mass cases.
#[must_use]
pub fn kish_ess(weights: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut all_finite = true;
    for &w in weights {
        all_finite &= w.is_finite();
        sum += w;
        sum_sq += w * w;
    }
    if !all_finite {
        return f64::NAN;
    }
    if sum.is_finite() && sum_sq.is_finite() && sum_sq > 0.0 {
        return (sum * sum) / sum_sq;
    }
    // Zero mass, or finite weights whose sum or squares overflow: rescale by
    // the largest magnitude (the ratio is scale-free).
    let scale = weights.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    let (s, q) = weights.iter().fold((0.0_f64, 0.0_f64), |(s, q), &w| {
        let u = w / scale;
        (s + u, q + u * u)
    });
    (s * s) / q
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kish_ess_hand_values_and_zero_guard() {
        assert!((kish_ess(&[0.5, 0.25, 0.25]) - 8.0 / 3.0).abs() < 1e-12);
        assert!((kish_ess(&[3.0, 3.0, 3.0, 3.0]) - 4.0).abs() < 1e-12);
        assert!(kish_ess(&[0.0, 0.0]).abs() < f64::MIN_POSITIVE);
        assert!(kish_ess(&[]).abs() < f64::MIN_POSITIVE);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn kish_ess_boundary_contract() {
        // Impossible weights never look uniform.
        assert!(kish_ess(&[f64::NAN, f64::NAN]).is_nan());
        assert!(kish_ess(&[0.5, f64::NAN]).is_nan());
        assert!(kish_ess(&[1.0, f64::INFINITY]).is_nan());
        assert!(kish_ess(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        // No mass.
        assert_eq!(kish_ess(&[0.0, 0.0, 0.0]), 0.0);
        // One-hot, normalised or not.
        assert!((kish_ess(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-15);
        assert!((kish_ess(&[0.0, 7.5, 0.0]) - 1.0).abs() < 1e-15);
        // Uniform, normalised or not.
        assert!((kish_ess(&[0.25; 4]) - 4.0).abs() < 1e-15);
        assert!((kish_ess(&[9.0; 5]) - 5.0).abs() < 1e-14);
        // Squares overflow: the rescaled path stays exact.
        let big = f64::MAX / 2.0;
        assert!((kish_ess(&[big, big, big]) - 3.0).abs() < 1e-12);
        // Scale-free.
        let w = [0.1, 0.4, 0.2, 0.3];
        let scaled: Vec<f64> = w.iter().map(|x| x * 1e6).collect();
        assert!((kish_ess(&w) - kish_ess(&scaled)).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "mask length must equal vector length")]
    fn short_mask_is_a_shape_error_not_a_silent_tail_drop() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bytes = [0xFFu8];
        let mask = BitMaskView::new(&bytes, 8).unwrap();
        let _ = masked_sum(F64VectorView::contiguous(&x), Some(mask));
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the masked sum of small integers is exact in f64")]
    fn full_length_mask_selects_exactly_its_bits() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bytes = [0b0000_0101u8, 0b0000_0010];
        let mask = BitMaskView::new(&bytes, 10).unwrap();
        // Bits 0, 2 and 9 are set: 1 + 3 + 10.
        assert_eq!(masked_sum(F64VectorView::contiguous(&x), Some(mask)), 14.0);
        assert_eq!(masked_mean(F64VectorView::contiguous(&x), Some(mask)), Some(14.0 / 3.0));
    }
}
