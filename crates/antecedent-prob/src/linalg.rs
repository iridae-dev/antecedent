//! Shared dense Cholesky / solve helpers for Bayesian backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use antecedent_kernels::CholeskyError;

use crate::error::ProbError;

/// Lower-triangular Cholesky of an SPD matrix (row-major `n×n`).
///
/// Delegates to [`antecedent_kernels::cholesky_spd_into`], which refuses NaN and
/// numerically singular pivots.
///
/// # Errors
///
/// NaN, non-positive, or numerically singular pivot, or a buffer shorter than `n²`.
pub fn cholesky_spd_into(a: &[f64], n: usize, l: &mut [f64]) -> Result<(), ProbError> {
    antecedent_kernels::cholesky_spd_into(a, n, l).map_err(|e| match e {
        CholeskyError::BufferTooShort => {
            ProbError::Numerical { message: "Cholesky buffer shorter than n²".into() }
        }
        CholeskyError::NotPositiveDefinite { index } => {
            ProbError::Numerical { message: format!("Cholesky failed at diagonal {index}") }
        }
    })
}

/// Lower-triangular Cholesky of an SPD matrix (row-major `n×n`).
///
/// # Errors
///
/// Non-positive pivot.
pub fn cholesky_spd(a: &[f64], n: usize) -> Result<Vec<f64>, ProbError> {
    let mut l = vec![0.0; n.saturating_mul(n)];
    cholesky_spd_into(a, n, &mut l)?;
    Ok(l)
}

/// Invert SPD via Cholesky.
///
/// # Errors
///
/// Cholesky failure.
pub fn invert_spd(a: &[f64], n: usize) -> Result<Vec<f64>, ProbError> {
    let chol = cholesky_spd(a, n)?;
    Ok(invert_spd_from_chol(&chol, n))
}

/// Inverse from an existing Cholesky factor (skips the redundant O(n³/3)
/// refactorization when the caller already holds one).
#[must_use]
pub fn invert_spd_from_chol(chol: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    let mut eye_col = vec![0.0; n];
    let mut y = vec![0.0; n];
    for col in 0..n {
        eye_col.fill(0.0);
        eye_col[col] = 1.0;
        for i in 0..n {
            let mut acc = eye_col[i];
            for j in 0..i {
                acc -= chol[i * n + j] * y[j];
            }
            y[i] = acc / chol[i * n + i];
        }
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in (i + 1)..n {
                acc -= chol[j * n + i] * inv[j * n + col];
            }
            inv[i * n + col] = acc / chol[i * n + i];
        }
    }
    inv
}

/// Solve `L L' x = b` given a lower-triangular Cholesky factor; writes into `x`.
///
/// `y` is forward-substitution scratch (length `n`).
pub fn solve_chol_into(chol: &[f64], n: usize, b: &[f64], y: &mut [f64], x: &mut [f64]) {
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= chol[i * n + j] * y[j];
        }
        y[i] = acc / chol[i * n + i];
    }
    for i in (0..n).rev() {
        let mut acc = y[i];
        for j in (i + 1)..n {
            acc -= chol[j * n + i] * x[j];
        }
        x[i] = acc / chol[i * n + i];
    }
}

/// Solve `A x = b` for SPD `A` via Cholesky, using caller buffers for the factor and `y`.
///
/// # Errors
///
/// Cholesky failure.
pub fn solve_spd_into(
    a: &[f64],
    n: usize,
    b: &[f64],
    x: &mut [f64],
    factor: &mut [f64],
    y: &mut [f64],
) -> Result<(), ProbError> {
    cholesky_spd_into(a, n, factor)?;
    solve_chol_into(factor, n, b, y, x);
    Ok(())
}

/// Solve `A x = b` for SPD `A` via Cholesky; writes into `x`.
///
/// Allocating wrapper around [`solve_spd_into`]. Prefer the into-variant on
/// hot paths that already own factor / `y` buffers.
///
/// # Errors
///
/// Cholesky failure.
#[allow(dead_code)] // allocating wrapper; tests and one-shot callers
pub fn solve_spd(a: &[f64], n: usize, b: &[f64], x: &mut [f64]) -> Result<(), ProbError> {
    let mut factor = vec![0.0; n.saturating_mul(n)];
    let mut y = vec![0.0; n];
    solve_spd_into(a, n, b, x, &mut factor, &mut y)
}

/// Lower bound on the Hessian condition number from its Cholesky diagonal;
/// `+∞` for a NaN or non-positive diagonal. See
/// [`antecedent_kernels::cholesky_condition_lower_bound`].
#[must_use]
pub fn condition_from_chol(chol: &[f64], n: usize) -> f64 {
    antecedent_kernels::cholesky_condition_lower_bound(chol, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_spd_into_matches_allocating_solve() {
        // SPD: A = [[4, 1], [1, 3]]
        let a = [4.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        let mut x_alloc = [0.0; 2];
        solve_spd(&a, 2, &b, &mut x_alloc).unwrap();
        let mut x_into = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut y = [0.0; 2];
        solve_spd_into(&a, 2, &b, &mut x_into, &mut factor, &mut y).unwrap();
        for i in 0..2 {
            assert!((x_alloc[i] - x_into[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_refuses_nan_hessian() {
        let mut l = [0.0; 4];
        assert!(cholesky_spd_into(&[f64::NAN, 0.0, 0.0, 1.0], 2, &mut l).is_err());
        assert!(cholesky_spd(&[1.0, f64::NAN, f64::NAN, 1.0], 2).is_err());
    }

    #[test]
    fn condition_is_infinite_for_nan_factor() {
        assert!(condition_from_chol(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_infinite());
    }
}
