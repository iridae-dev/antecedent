//! Matrix factorization, residual and covariance checks for SCC compilation.

use super::{
    GaussianDisturbance, LinearStructuralMap, SccCompileError, eye_minus_determinant2,
    gauss_jordan_solve, max_abs_row_sum, transpose, unit_diagonal_infinity_kappa2, zeros,
};

/// `I - K_SS` for a block of the free support. `LinearStructuralMap::try_new` guarantees the
/// zero diagonal, so the diagonal here is exactly one.
pub(crate) fn eye_minus_block(
    model: &LinearStructuralMap,
    members: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let mut matrix = zeros(m, m)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &from_node) in members.iter().enumerate() {
            matrix[row * m + col] = if row == col {
                1.0
            } else {
                -structural_coefficient(model, into_node, from_node)?
            };
        }
    }
    finite_matrix(&matrix)?;
    Ok(matrix)
}

pub(super) fn condition_of(
    matrix: &[f64],
    m: usize,
    members: &[usize],
) -> Result<(f64, bool), SccCompileError> {
    if m == 0 {
        return Ok((1.0, true));
    }
    if let Some(kappa) = exact_infinity_kappa(matrix, m) {
        return Ok((kappa, true));
    }
    let transposed = transpose(matrix, m, m)?;
    let inverse_inf = hager_one_norm_inverse(&transposed, members)?;
    let kappa = max_abs_row_sum(matrix, m) * inverse_inf;
    if !kappa.is_finite() {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    Ok((kappa, false))
}

/// Hager's 1-norm estimate of `||A^{-1}||_1` (Higham, 1988).
///
/// Applied to `A^T`, the estimate is `||A^{-1}||_∞`.
fn hager_one_norm_inverse(matrix: &[f64], members: &[usize]) -> Result<f64, SccCompileError> {
    let m = members.len();
    let Ok(count) = u32::try_from(m) else {
        return Err(SccCompileError::NonFiniteEquilibrium);
    };
    if count == 0 {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    let mut x = vec![1.0 / f64::from(count); m];
    let transposed = transpose(matrix, m, m)?;
    let mut estimate = 0.0_f64;
    for _step in 0..4 {
        let y = solve_vector(matrix, &x, members)?;
        let norm_y = y.iter().fold(0.0_f64, |acc, value| acc + value.abs());
        if norm_y <= estimate {
            break;
        }
        estimate = norm_y;
        let mut xi = Vec::with_capacity(m);
        for value in &y {
            xi.push(if *value >= 0.0 { 1.0 } else { -1.0 });
        }
        let z = solve_vector(&transposed, &xi, members)?;
        let mut dot = 0.0_f64;
        let mut peak = 0.0_f64;
        let mut peak_at = 0_usize;
        for (index, value) in z.iter().enumerate() {
            dot += value * x[index];
            if value.abs() > peak {
                peak = value.abs();
                peak_at = index;
            }
        }
        if peak <= dot {
            break;
        }
        x = vec![0.0; m];
        x[peak_at] = 1.0;
    }
    if !estimate.is_finite() {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    Ok(estimate)
}

fn solve_vector(
    matrix: &[f64],
    rhs: &[f64],
    members: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    solve_row_major(matrix, rhs, m, 1, members)
}

fn exact_infinity_kappa(matrix: &[f64], m: usize) -> Option<f64> {
    match m {
        1 => Some(1.0),
        2 => {
            let (_, det) = eye_minus_determinant2([matrix[1], matrix[2]]);
            let kappa = unit_diagonal_infinity_kappa2([matrix[1], matrix[2]], det);
            kappa.is_finite().then_some(kappa)
        }
        _ => None,
    }
}

pub(super) fn block_residual(
    matrix: &[f64],
    solved: &[f64],
    cols: usize,
    rhs: &[f64],
    m: usize,
) -> Result<f64, SccCompileError> {
    let mut residual = 0.0_f64;
    for row in 0..m {
        let mut image = 0.0;
        for col in 0..m {
            image += matrix[row * m + col] * solved[col * cols];
        }
        if !image.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        let error = image - rhs[row];
        if !error.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        residual = residual.max(error.abs());
    }
    Ok(residual)
}

pub(super) fn structural_residual(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    mean: &[f64],
    free: &[usize],
) -> Result<f64, SccCompileError> {
    let n = model.dimension();
    let mut residual = 0.0_f64;
    for &node in free {
        let mut image = mean[node] - model.intercept()[node] - disturbance.mean()[node];
        for (other, &value) in mean.iter().enumerate().take(n) {
            if other == node {
                continue;
            }
            image -= structural_coefficient(model, node, other)? * value;
        }
        if !image.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        residual = residual.max(image.abs());
    }
    Ok(residual)
}

pub(super) fn solve_row_major(
    matrix: &[f64],
    rhs_row_major: &[f64],
    m: usize,
    cols: usize,
    members: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    if m == 0 {
        return Ok(Vec::new());
    }
    let mut factor = matrix.to_vec();
    let mut packed = zeros(cols, m)?;
    for col in 0..cols {
        for row in 0..m {
            packed[col * m + row] = rhs_row_major[row * cols + col];
        }
    }
    if !gauss_jordan_solve(&mut factor, &mut packed, m, cols) {
        return Err(SccCompileError::NumericallySingularBlock { members: members.to_vec() });
    }
    let mut solved = zeros(m, cols)?;
    for col in 0..cols {
        for row in 0..m {
            let value = packed[col * m + row];
            if !value.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            solved[row * cols + col] = value;
        }
    }
    Ok(solved)
}

/// Refuse a covariance that is not positive semidefinite.
///
/// The test runs on the correlation matrix `R = D^{-1/2} Σ D^{-1/2}` with
/// `D = diag(Σ)`, so the tolerance is relative to each variable's own scale rather than
/// to the largest entry of `Σ`. A variance that is negative beyond a relative tolerance of
/// its own row is refused, so `diag(1e6, -1e-9)` is not hidden by the `1e6` entry. A
/// variable of zero variance must have a zero row.
#[allow(clippy::float_cmp, reason = "a zero-variance row must be exactly zero off the diagonal")]
pub(crate) fn require_psd(covariance: &[f64], n: usize) -> Result<(), SccCompileError> {
    let tolerance = antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR;
    let mut scale = vec![0.0_f64; n];
    for row in 0..n {
        let variance = covariance[row * n + row];
        let row_scale = (0..n).fold(0.0_f64, |best, col| best.max(covariance[row * n + col].abs()));
        if variance < -tolerance * row_scale {
            return Err(SccCompileError::NotPositiveSemidefinite { index: row });
        }
        scale[row] = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    }
    let mut corr = zeros(n, n)?;
    for row in 0..n {
        for col in 0..n {
            let entry = covariance[row * n + col];
            if scale[row] > 0.0 && scale[col] > 0.0 {
                corr[row * n + col] = entry / scale[row] / scale[col];
            } else if entry != 0.0 && row != col {
                // A zero-variance variable cannot covary with anything.
                return Err(SccCompileError::NotPositiveSemidefinite { index: row.max(col) });
            }
        }
    }
    let mut factor = zeros(n, n)?;
    for row in 0..n {
        for col in 0..row {
            let mut sum = corr[row * n + col];
            for k in 0..col {
                sum -= factor[row * n + k] * factor[col * n + k];
            }
            if !sum.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            let diag = factor[col * n + col];
            if diag > 0.0 {
                factor[row * n + col] = sum / diag;
            } else if sum.abs() > tolerance {
                return Err(SccCompileError::NotPositiveSemidefinite { index: row });
            } else {
                factor[row * n + col] = 0.0;
            }
        }
        let mut diag = corr[row * n + row];
        for k in 0..row {
            diag -= factor[row * n + k] * factor[row * n + k];
        }
        if !diag.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        if diag < -tolerance {
            return Err(SccCompileError::NotPositiveSemidefinite { index: row });
        }
        factor[row * n + row] = if diag > 0.0 { diag.sqrt() } else { 0.0 };
    }
    Ok(())
}

pub(super) fn symmetrize(covariance: &mut [f64], n: usize) -> Result<(), SccCompileError> {
    for row in 0..n {
        for col in (row + 1)..n {
            let value = 0.5 * (covariance[row * n + col] + covariance[col * n + row]);
            if !value.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            covariance[row * n + col] = value;
            covariance[col * n + row] = value;
        }
    }
    Ok(())
}

pub(super) fn finite_matrix(matrix: &[f64]) -> Result<(), SccCompileError> {
    if matrix.iter().any(|entry| !entry.is_finite()) {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    Ok(())
}

pub(super) fn structural_coefficient(
    model: &LinearStructuralMap,
    into_node: usize,
    from_node: usize,
) -> Result<f64, SccCompileError> {
    model.coefficient(into_node, from_node).ok_or(SccCompileError::Shape {
        message: "structural coefficient index is outside the model",
    })
}

pub(super) fn positions(n: usize, nodes: &[usize]) -> Result<Vec<Option<usize>>, SccCompileError> {
    let mut pos = vec![None; n];
    for (index, &node) in nodes.iter().enumerate() {
        if node >= n {
            return Err(SccCompileError::Shape { message: "block member is outside the model" });
        }
        pos[node] = Some(index);
    }
    Ok(pos)
}
