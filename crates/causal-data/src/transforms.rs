//! Column transforms for discovery / symbolic CI (Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::all, clippy::pedantic, clippy::restriction)]

use crate::error::DataError;

/// Equal-width binning of a float column into `n_bins` integer codes in `0..n_bins`.
///
/// # Errors
///
/// Empty input or `n_bins == 0`.
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
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in col {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let width = (max_v - min_v).max(1e-12);
    for (i, &v) in col.iter().enumerate() {
        let b = ((v - min_v) / width * n_bins as f64).floor() as usize;
        out[i] = b.min(n_bins - 1) as f64;
    }
    Ok(())
}

/// Ordinal pattern of embedding dimension `m` with delay `tau` (Bandt–Pompe).
///
/// Writes one pattern code per valid window into `out` (length `col.len() - (m-1)*tau`).
///
/// # Errors
///
/// Bad shape.
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
/// Even/zero window or shape mismatch.
pub fn moving_average(col: &[f64], window: usize, out: &mut [f64]) -> Result<(), DataError> {
    if window == 0 || window % 2 == 0 {
        return Err(DataError::InvalidArgument { message: "window must be odd and > 0".into() });
    }
    if col.len() != out.len() {
        return Err(DataError::InvalidArgument { message: "out length != col length".into() });
    }
    let half = window / 2;
    let n = col.len();
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let s: f64 = col[lo..hi].iter().sum();
        out[i] = s / (hi - lo) as f64;
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
        assert_eq!(out[0], 0.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn ordinal_runs() {
        let col: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
        let mut out = vec![0.0; 20];
        let n = ordinal_patterns(&col, 3, 1, &mut out).unwrap();
        assert!(n > 0);
    }
}
