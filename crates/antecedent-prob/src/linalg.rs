//! Shared dense Cholesky / solve helpers for Bayesian backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use crate::error::ProbError;

/// Lower-triangular Cholesky of an SPD matrix (row-major `n×n`).
///
/// # Errors
///
/// Non-positive pivot.
pub fn cholesky_spd(a: &[f64], n: usize) -> Result<Vec<f64>, ProbError> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(ProbError::Numerical {
                        message: format!("Cholesky failed at diagonal {i}"),
                    });
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Invert SPD via Cholesky.
///
/// # Errors
///
/// Cholesky failure.
pub fn invert_spd(a: &[f64], n: usize) -> Result<Vec<f64>, ProbError> {
    let chol = cholesky_spd(a, n)?;
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
    Ok(inv)
}

/// Solve `A x = b` for SPD `A` via Cholesky; writes into `x`.
///
/// # Errors
///
/// Cholesky failure.
pub fn solve_spd(a: &[f64], n: usize, b: &[f64], x: &mut [f64]) -> Result<(), ProbError> {
    let chol = cholesky_spd(a, n)?;
    let mut y = vec![0.0; n];
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
    Ok(())
}

/// LDLT factorization fallback for indefinite / poorly conditioned matrices.
/// Returns `(diag, lower)` where `A ≈ L diag L'` with unit lower `L`.
///
/// # Errors
///
/// Zero pivot.
pub fn ldlt_decompose(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), ProbError> {
    let mut l = vec![0.0; n * n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        l[i * n + i] = 1.0;
        let mut di = a[i * n + i];
        for k in 0..i {
            di -= l[i * n + k] * l[i * n + k] * d[k];
        }
        if di.abs() < 1e-14 {
            return Err(ProbError::Numerical { message: format!("LDLT zero pivot at {i}") });
        }
        d[i] = di;
        for j in (i + 1)..n {
            let mut lij = a[j * n + i];
            for k in 0..i {
                lij -= l[j * n + k] * l[i * n + k] * d[k];
            }
            l[j * n + i] = lij / di;
        }
    }
    Ok((d, l))
}

/// Approximate condition number from Cholesky diagonals (κ ≈ (max/min)²).
#[must_use]
pub fn condition_from_chol(chol: &[f64], n: usize) -> f64 {
    let mut min_d = f64::INFINITY;
    let mut max_d: f64 = 0.0;
    for i in 0..n {
        let d = chol[i * n + i].abs();
        min_d = min_d.min(d);
        max_d = max_d.max(d);
    }
    if min_d <= 0.0 {
        return f64::INFINITY;
    }
    let ratio = max_d / min_d;
    ratio * ratio
}
