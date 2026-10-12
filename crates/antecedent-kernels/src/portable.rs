//! Portable optimized kernels (safe auto-vectorization friendly loops).
//!
//! These share the scalar semantic contract and are selected once per batch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "n as f64 for means/variances is intentional")]

use crate::view::{BitMaskView, F64VectorView, ViewError};

/// Contiguous fast path for masked sum when unit-stride and no mask.
///
/// # Errors
///
/// Same contract as [`crate::scalar::masked_sum`].
pub fn masked_sum(x: F64VectorView<'_>, mask: Option<BitMaskView<'_>>) -> Result<f64, ViewError> {
    if mask.is_none() {
        if let Some(slice) = x.as_slice() {
            return Ok(slice.iter().sum());
        }
    }
    crate::scalar::masked_sum(x, mask)
}

/// Contiguous fast path for masked mean.
///
/// # Errors
///
/// Same contract as [`crate::scalar::masked_mean`].
pub fn masked_mean(
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Result<Option<f64>, ViewError> {
    if mask.is_none() {
        if let Some(slice) = x.as_slice() {
            if slice.is_empty() {
                return Ok(None);
            }
            let sum: f64 = slice.iter().sum();
            return Ok(Some(sum / slice.len() as f64));
        }
    }
    crate::scalar::masked_mean(x, mask)
}

/// Contiguous fast path for population variance.
///
/// # Errors
///
/// Same contract as [`crate::scalar::masked_variance`].
pub fn masked_variance(
    x: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Result<Option<f64>, ViewError> {
    if mask.is_none() {
        if let Some(slice) = x.as_slice() {
            if slice.is_empty() {
                return Ok(None);
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
            return Ok(Some(var));
        }
    }
    crate::scalar::masked_variance(x, mask)
}

/// Gather with contiguous source fast path.
/// # Errors
///
/// Same contract as [`crate::scalar::gather`].
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
    if let Some(slice) = src.as_slice() {
        for (dst, &idx) in out.iter_mut().zip(indices.iter()) {
            *dst = *slice
                .get(idx)
                .ok_or(crate::view::ViewError::OutOfBounds { index: idx, len: slice.len() })?;
        }
        return Ok(());
    }
    crate::scalar::gather(src, indices, out)
}

/// Contiguous fast path for population covariance.
///
/// # Errors
///
/// Same contract as [`crate::scalar::masked_covariance`].
pub fn masked_covariance(
    x: F64VectorView<'_>,
    y: F64VectorView<'_>,
    mask: Option<BitMaskView<'_>>,
) -> Result<Option<f64>, ViewError> {
    if x.len() != y.len() {
        return Err(ViewError::InvalidShape { message: "covariance views must share length" });
    }
    if mask.is_none() {
        if let (Some(xs), Some(ys)) = (x.as_slice(), y.as_slice()) {
            let n = xs.len();
            if n == 0 {
                return Ok(None);
            }
            let nf = n as f64;
            let mx = xs.iter().sum::<f64>() / nf;
            let my = ys.iter().sum::<f64>() / nf;
            let mut acc = 0.0;
            for i in 0..n {
                acc += (xs[i] - mx) * (ys[i] - my);
            }
            return Ok(Some(acc / nf));
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
///
/// # Errors
///
/// Same contract as [`crate::scalar::pairwise_l1_fill`]. On error `out` is left untouched.
pub fn pairwise_l1_fill(x: &[f64], out: &mut [f64]) -> Result<(), ViewError> {
    let n = x.len();
    if out.len() != n.saturating_mul(n) {
        return Err(ViewError::InvalidShape { message: "pairwise out must be n*n" });
    }
    for i in 0..n {
        let xi = x[i];
        let row = &mut out[i * n..(i + 1) * n];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = (xi - x[j]).abs();
        }
    }
    Ok(())
}

/// Contingency accumulation (portable = scalar; scatter-add resists vectorization).
///
/// # Errors
///
/// Same contract as [`crate::scalar::accumulate_contingency`].
pub fn accumulate_contingency(
    x_codes: &[u32],
    y_codes: &[u32],
    out: &mut [f64],
    n_y_levels: usize,
) -> Result<(), ViewError> {
    crate::scalar::accumulate_contingency(x_codes, y_codes, out, n_y_levels)
}

/// Weighted sum with contiguous zip.
///
/// # Errors
///
/// Same contract as [`crate::scalar::weighted_sum`].
pub fn weighted_sum(x: &[f64], weights: &[f64]) -> Result<f64, ViewError> {
    if x.len() != weights.len() {
        return Err(ViewError::InvalidShape { message: "weighted_sum length mismatch" });
    }
    Ok(x.iter().zip(weights.iter()).map(|(xi, &w)| crate::scalar::sanitize_weight(w) * xi).sum())
}

/// Weighted mean with contiguous zip.
///
/// # Errors
///
/// Same contract as [`crate::scalar::weighted_mean`].
pub fn weighted_mean(x: &[f64], weights: &[f64]) -> Result<Option<f64>, ViewError> {
    crate::scalar::weighted_mean(x, weights)
}
