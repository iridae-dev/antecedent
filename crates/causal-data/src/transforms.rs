//! Column transforms for discovery / symbolic CI .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::error::DataError;

fn require_finite(col: &[f64], what: &str) -> Result<(), DataError> {
    if let Some(i) = col.iter().position(|v| !v.is_finite()) {
        return Err(DataError::InvalidArgument {
            message: format!("{what}: non-finite value at index {i}"),
        });
    }
    Ok(())
}

/// Equal-width binning of a float column into `n_bins` integer codes in `0..n_bins`.
///
/// Non-finite values are rejected (no silent map to bin 0).
///
/// # Errors
///
/// Empty `n_bins`, length mismatch, or non-finite input.
pub fn equal_width_bin(col: &[f64], n_bins: usize, out: &mut [f64]) -> Result<(), DataError> {
    if n_bins == 0 {
        return Err(DataError::InvalidArgument { message: "n_bins must be > 0".into() });
    }
    if col.len() != out.len() {
        return Err(DataError::InvalidArgument { message: "out length != col length".into() });
    }
    if col.is_empty() {
        return Ok(());
    }
    require_finite(col, "equal_width_bin")?;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in col {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let width = (max_v - min_v).max(1e-12);
    let last_bin = (n_bins - 1) as f64;
    for (slot, &v) in out.iter_mut().zip(col.iter()) {
        let b = ((v - min_v) / width * n_bins as f64).floor();
        *slot = b.clamp(0.0, last_bin);
    }
    Ok(())
}

/// Ordinal pattern of embedding dimension `m` with delay `tau` (Bandt–Pompe).
///
/// Writes one pattern code per valid window into `out` (length `col.len() - (m-1)*tau`).
/// Non-finite values in any window are rejected (no silent tie treatment of NaN).
///
/// # Errors
///
/// Bad shape or non-finite input.
pub fn ordinal_patterns(
    col: &[f64],
    m: usize,
    tau: usize,
    out: &mut [f64],
) -> Result<usize, DataError> {
    if m < 2 || tau == 0 {
        return Err(DataError::InvalidArgument { message: "need m>=2 and tau>=1".into() });
    }
    let need = col.len().saturating_sub((m - 1) * tau);
    if out.len() < need {
        return Err(DataError::InvalidArgument { message: "out buffer too short".into() });
    }
    require_finite(col, "ordinal_patterns")?;
    let mut idx = vec![0usize; m];
    for t in 0..need {
        for (k, slot) in idx.iter_mut().enumerate() {
            *slot = k;
        }
        idx.sort_by(|&a, &b| {
            col[t + a * tau]
                .partial_cmp(&col[t + b * tau])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Lehmer code
        let mut code = 0usize;
        for i in 0..m {
            let mut smaller = 0usize;
            for j in (i + 1)..m {
                if idx[j] < idx[i] {
                    smaller += 1;
                }
            }
            code = code * (m - i) + smaller;
        }
        out[t] = code as f64;
    }
    Ok(need)
}

/// Simple moving-average smoother (odd window).
///
/// # Errors
///
/// Even/zero window, shape mismatch, or non-finite input.
pub fn moving_average(col: &[f64], window: usize, out: &mut [f64]) -> Result<(), DataError> {
    if window == 0 || window % 2 == 0 {
        return Err(DataError::InvalidArgument { message: "window must be odd and > 0".into() });
    }
    if col.len() != out.len() {
        return Err(DataError::InvalidArgument { message: "out length != col length".into() });
    }
    require_finite(col, "moving_average")?;
    let half = window / 2;
    let n = col.len();
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let s: f64 = col[lo..hi].iter().sum();
        *slot = s / (hi - lo) as f64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_two_bins() {
        let col = [0.0, 1.0, 2.0, 3.0];
        let mut out = [0.0; 4];
        equal_width_bin(&col, 2, &mut out).unwrap();
        assert!((out[0] - 0.0).abs() < f64::EPSILON);
        assert!((out[3] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn binning_rejects_nan() {
        let col = [0.0, f64::NAN, 2.0];
        let mut out = [0.0; 3];
        assert!(equal_width_bin(&col, 2, &mut out).is_err());
    }

    #[test]
    fn ordinal_runs() {
        let col: Vec<f64> = (0..20).map(|i| f64::from(i).sin()).collect();
        let mut out = vec![0.0; 20];
        let n = ordinal_patterns(&col, 3, 1, &mut out).unwrap();
        assert!(n > 0);
    }

    #[test]
    fn ordinal_rejects_nan() {
        let col = [0.0, 1.0, f64::NAN, 3.0, 4.0];
        let mut out = vec![0.0; 5];
        assert!(ordinal_patterns(&col, 3, 1, &mut out).is_err());
    }
}
