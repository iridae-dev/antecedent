//! Blockwise propagation of equilibrium means and covariance.

use super::{
    BlockDiagnostic, GaussianDisturbance, LinearStructuralMap, SccCompileError,
    add_times_transpose, block_residual, condition_of, eye_minus_block, finite_matrix, matmul,
    pivots_regular, positions, solve_row_major, structural_coefficient, structural_residual,
    symmetrize, transpose, zeros,
};

pub(super) struct Propagated {
    pub(super) mean: Vec<f64>,
    pub(super) covariance: Vec<f64>,
    pub(super) residual_inf: f64,
    pub(super) blocks: Vec<BlockDiagnostic>,
}

pub(super) fn propagate(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    assigned: &[Option<f64>],
    blocks: &[Vec<usize>],
) -> Result<Propagated, SccCompileError> {
    let n = model.dimension();
    let mut mean = vec![0.0; n];
    for (node, value) in assigned.iter().enumerate() {
        if let Some(clamped) = value {
            mean[node] = *clamped;
        }
    }
    let mut covariance = zeros(n, n)?;
    let mut eps_x = zeros(n, n)?;
    let mut diagnostics = Vec::with_capacity(blocks.len());
    let mut earlier: Vec<usize> = Vec::new();
    for members in blocks {
        let diagnostic = solve_block(
            model,
            disturbance,
            assigned,
            members,
            &earlier,
            &mut Moments { mean: &mut mean, covariance: &mut covariance, eps_x: &mut eps_x },
        )?;
        earlier.extend(members.iter().copied());
        diagnostics.push(diagnostic);
    }
    symmetrize(&mut covariance, n)?;
    let mean_bad = mean.iter().any(|value| !value.is_finite());
    let covariance_bad = covariance.iter().any(|value| !value.is_finite());
    if mean_bad || covariance_bad {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    let free = blocks.iter().flatten().copied().collect::<Vec<_>>();
    let residual_inf = structural_residual(model, disturbance, &mean, &free)?;
    Ok(Propagated { mean, covariance, residual_inf, blocks: diagnostics })
}

struct Moments<'a> {
    mean: &'a mut [f64],
    covariance: &'a mut [f64],
    eps_x: &'a mut [f64],
}

fn solve_block(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    assigned: &[Option<f64>],
    members: &[usize],
    earlier: &[usize],
    moments: &mut Moments<'_>,
) -> Result<BlockDiagnostic, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let n = model.dimension();
    let matrix = eye_minus_block(model, members)?;
    if !pivots_regular(&matrix, m) {
        return Err(SccCompileError::NumericallySingularBlock { members: members.to_vec() });
    }
    let mean_rhs =
        mean_right_hand_side(model, disturbance, assigned, members, earlier, moments.mean)?;
    // Parent coefficients and the parent covariance are shared by the three inputs below.
    let k_fe = parent_coefficients(model, members, earlier)?;
    let c_ee = earlier_covariance(earlier, moments.covariance, n)?;
    let parents = ParentInputs { k_fe: &k_fe, c_ee: &c_ee };
    let cov_u = disturbance_block(disturbance, members, earlier, &parents, moments.eps_x, n)?;
    let cov_u_earlier = cross_input(members, earlier, &parents, moments.eps_x, n)?;
    let cov_u_eps = disturbance_cross(disturbance, members, earlier, &k_fe, moments.eps_x, n)?;
    let cols = batch_columns(m, p, n)?;
    let mut rhs = zeros(m, cols)?;
    let cov_at = 1;
    let cross_at = cov_at + m;
    let eps_at = cross_at + p;
    for row in 0..m {
        let base = row * cols;
        rhs[base] = mean_rhs[row];
        for col in 0..m {
            rhs[base + cov_at + col] = cov_u[row * m + col];
        }
        for col in 0..p {
            rhs[base + cross_at + col] = cov_u_earlier[row * p + col];
        }
        for col in 0..n {
            rhs[base + eps_at + col] = cov_u_eps[row * n + col];
        }
    }
    let solved = solve_row_major(&matrix, &rhs, m, cols, members)?;
    for (local, &node) in members.iter().enumerate() {
        moments.mean[node] = solved[local * cols];
    }
    let mut solved_cov_u = zeros(m, m)?;
    for row in 0..m {
        for col in 0..m {
            solved_cov_u[row * m + col] = solved[row * cols + cov_at + col];
        }
    }
    // `W = A^{-1} Cov(u)`. `Cov(X_S) = W A^{-T} = (A^{-1} W^T)^T`.
    let cov_factor = transpose(&solved_cov_u, m, m)?;
    let solved_cov = solve_row_major(&matrix, &cov_factor, m, m, members)?;
    let cov_block = transpose(&solved_cov, m, m)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &from_node) in members.iter().enumerate() {
            moments.covariance[into_node * n + from_node] = cov_block[row * m + col];
        }
        for (col, &prior) in earlier.iter().enumerate() {
            let cross = solved[row * cols + cross_at + col];
            moments.covariance[into_node * n + prior] = cross;
            moments.covariance[prior * n + into_node] = cross;
        }
        for noise in 0..n {
            moments.eps_x[noise * n + into_node] = solved[row * cols + eps_at + noise];
        }
    }
    let residual_inf = block_residual(&matrix, &solved, cols, &mean_rhs, m)?;
    let (kappa_inf, exact) = condition_of(&matrix, m, members)?;
    Ok(BlockDiagnostic { members: members.to_vec(), residual_inf, kappa_inf, exact })
}

fn batch_columns(m: usize, earlier: usize, n: usize) -> Result<usize, SccCompileError> {
    m.checked_add(earlier)
        .and_then(|sum| sum.checked_add(n))
        .and_then(|sum| sum.checked_add(1))
        .ok_or(SccCompileError::Shape { message: "block right-hand side width overflows usize" })
}

fn mean_right_hand_side(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    assigned: &[Option<f64>],
    members: &[usize],
    earlier: &[usize],
    mean: &[f64],
) -> Result<Vec<f64>, SccCompileError> {
    let n = model.dimension();
    let member_at = positions(n, members)?;
    let earlier_at = positions(n, earlier)?;
    let mut rhs = vec![0.0; members.len()];
    for (local, &node) in members.iter().enumerate() {
        let mut value = model.intercept()[node] + disturbance.mean()[node];
        for other in 0..n {
            if other == node {
                continue;
            }
            let coefficient = structural_coefficient(model, node, other)?;
            #[allow(
                clippy::float_cmp,
                reason = "a structural zero does not enter the right-hand side"
            )]
            if coefficient == 0.0 {
                continue;
            }
            if let Some(clamped) = assigned[other] {
                value += coefficient * clamped;
            } else if earlier_at[other].is_some() {
                value += coefficient * mean[other];
            } else if member_at[other].is_none() {
                return Err(SccCompileError::Shape {
                    message: "structural edge enters a free variable before its block is solved",
                });
            }
        }
        if !value.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        rhs[local] = value;
    }
    Ok(rhs)
}

/// Coefficients of the earlier blocks into this block, and the earlier blocks' covariance.
struct ParentInputs<'a> {
    /// `K[members, earlier]`, row-major `m × p`.
    k_fe: &'a [f64],
    /// `Cov(X_earlier)`, row-major `p × p`.
    c_ee: &'a [f64],
}

fn disturbance_block(
    disturbance: &GaussianDisturbance,
    members: &[usize],
    earlier: &[usize],
    parents: &ParentInputs<'_>,
    eps_x: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let sigma = disturbance.covariance();
    let mut cov_u = zeros(m, m)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &from_node) in members.iter().enumerate() {
            cov_u[row * m + col] = sigma[into_node * n + from_node];
        }
    }
    if p == 0 {
        return Ok(cov_u);
    }
    let mut c_ex = zeros(m, p)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &prior) in earlier.iter().enumerate() {
            c_ex[row * p + col] = eps_x[into_node * n + prior];
        }
    }
    // `Cov(u) = Σ_SS + K C(ε, X)^T + C(ε, X) K^T + K C(X) K^T`.
    add_times_transpose(parents.k_fe, &c_ex, &mut cov_u, m, p)?;
    add_times_transpose(&c_ex, parents.k_fe, &mut cov_u, m, p)?;
    let through_parents = matmul(parents.k_fe, m, p, parents.c_ee, p)?;
    add_times_transpose(&through_parents, parents.k_fe, &mut cov_u, m, p)?;
    finite_matrix(&cov_u)?;
    Ok(cov_u)
}

fn cross_input(
    members: &[usize],
    earlier: &[usize],
    parents: &ParentInputs<'_>,
    eps_x: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    if p == 0 {
        return Ok(Vec::new());
    }
    let mut cov_u = zeros(m, p)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &prior) in earlier.iter().enumerate() {
            cov_u[row * p + col] = eps_x[into_node * n + prior];
        }
    }
    let through_parents = matmul(parents.k_fe, m, p, parents.c_ee, p)?;
    for (index, value) in through_parents.iter().copied().enumerate() {
        cov_u[index] += value;
    }
    finite_matrix(&cov_u)?;
    Ok(cov_u)
}

fn disturbance_cross(
    disturbance: &GaussianDisturbance,
    members: &[usize],
    earlier: &[usize],
    k_fe: &[f64],
    eps_x: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let sigma = disturbance.covariance();
    let mut cov_u = zeros(m, n)?;
    for (row, &into_node) in members.iter().enumerate() {
        for noise in 0..n {
            let mut value = sigma[into_node * n + noise];
            for (col, &prior) in earlier.iter().enumerate() {
                value += k_fe[row * p + col] * eps_x[noise * n + prior];
            }
            cov_u[row * n + noise] = value;
        }
    }
    finite_matrix(&cov_u)?;
    Ok(cov_u)
}

fn parent_coefficients(
    model: &LinearStructuralMap,
    members: &[usize],
    earlier: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let mut coefficients = zeros(m, p)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &prior) in earlier.iter().enumerate() {
            coefficients[row * p + col] = structural_coefficient(model, into_node, prior)?;
        }
    }
    Ok(coefficients)
}

fn earlier_covariance(
    earlier: &[usize],
    covariance: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let p = earlier.len();
    let mut c_ee = zeros(p, p)?;
    for (row, &left) in earlier.iter().enumerate() {
        for (col, &right) in earlier.iter().enumerate() {
            c_ee[row * p + col] = covariance[left * n + right];
        }
    }
    Ok(c_ee)
}
