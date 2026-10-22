// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small dense row-major linear algebra shared by the linear-equilibrium modules.
//!
//! Every product checks that its entries stay finite, so a caller gets a typed
//! failure rather than a silent `inf` or `NaN` in a covariance.

/// Why a dense workspace or product could not be formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatrixError {
    /// The requested workspace length does not fit in `usize`.
    Overflow,
    /// A product entry is not finite.
    NonFinite,
}

/// A zeroed `rows x cols` row-major workspace.
pub(crate) fn zeros(rows: usize, cols: usize) -> Result<Vec<f64>, MatrixError> {
    let len = rows.checked_mul(cols).ok_or(MatrixError::Overflow)?;
    Ok(vec![0.0; len])
}

/// `out += left * rightᵀ` for `left`, `right` of shape `rows x mid` and `out` of shape
/// `rows x rows`.
pub(crate) fn add_times_transpose(
    left: &[f64],
    right: &[f64],
    out: &mut [f64],
    rows: usize,
    mid: usize,
) -> Result<(), MatrixError> {
    for row in 0..rows {
        for col in 0..rows {
            let mut sum = 0.0;
            for k in 0..mid {
                sum += left[row * mid + k] * right[col * mid + k];
            }
            if !sum.is_finite() {
                return Err(MatrixError::NonFinite);
            }
            out[row * rows + col] += sum;
        }
    }
    Ok(())
}

/// `left * right` for `left` of shape `rows x mid` and `right` of shape `mid x cols`.
pub(crate) fn matmul(
    left: &[f64],
    rows: usize,
    mid: usize,
    right: &[f64],
    cols: usize,
) -> Result<Vec<f64>, MatrixError> {
    let mut out = zeros(rows, cols)?;
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for k in 0..mid {
                sum += left[row * mid + k] * right[k * cols + col];
            }
            if !sum.is_finite() {
                return Err(MatrixError::NonFinite);
            }
            out[row * cols + col] = sum;
        }
    }
    Ok(out)
}

/// Transpose of a `rows x cols` matrix.
pub(crate) fn transpose(matrix: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, MatrixError> {
    let mut out = zeros(cols, rows)?;
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = matrix[row * cols + col];
        }
    }
    Ok(out)
}

/// Largest absolute row sum (the infinity norm) of a square `n x n` matrix.
pub(crate) fn max_abs_row_sum(matrix: &[f64], n: usize) -> f64 {
    matrix
        .chunks_exact(n)
        .map(|row| row.iter().map(|entry| entry.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// Product of the off-diagonal entries of a 2x2 update `K` and the determinant
/// `1 - K₀₁ K₁₀` of `I - K`, as `(product, determinant)`.
pub(crate) fn eye_minus_determinant2(off_diagonal: [f64; 2]) -> (f64, f64) {
    let product = off_diagonal[0] * off_diagonal[1];
    (product, 1.0 - product)
}

/// Determinant of `I - K` for a row-major 3x3 update `K`.
pub(crate) fn eye_minus_determinant3(k: &[f64]) -> f64 {
    let mut eye_minus = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            let coefficient = k[row * 3 + col];
            eye_minus[row * 3 + col] = if row == col { 1.0 - coefficient } else { -coefficient };
        }
    }
    determinant3(&eye_minus)
}

fn determinant3(m: &[f64; 9]) -> f64 {
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[3], m[4], m[5]);
    let (g, h, i) = (m[6], m[7], m[8]);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Infinity-norm condition number `‖A‖∞ ‖A⁻¹‖∞ = (1 + max|aᵢⱼ|)² / |det|` of the unit-diagonal
/// 2x2 matrix whose off-diagonal entries have the given magnitudes (the sign of an entry does
/// not matter). Not finite when the matrix is singular or the arithmetic overflows.
pub(crate) fn unit_diagonal_infinity_kappa2(off_diagonal: [f64; 2], determinant: f64) -> f64 {
    let row_sum = (1.0 + off_diagonal[0].abs()).max(1.0 + off_diagonal[1].abs());
    (row_sum * row_sum) / determinant.abs()
}
