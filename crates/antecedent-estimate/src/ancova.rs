//! Independent-unit randomized ANCOVA point and HC0 sandwich variance.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Fitted treatment coefficient, HC0 variance, and covariate coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct AncovaFit {
    /// Coefficient on realized assignment.
    pub effect: f64,
    /// Independent-row HC0 sandwich variance; no calibrated interval follows.
    pub hc0_variance: f64,
    /// Coefficients in the supplied covariate order.
    pub covariate_coefficients: Vec<f64>,
    /// Number assigned treatment.
    pub treated: usize,
    /// Number assigned control.
    pub control: usize,
}

/// Fit OLS with intercept, assignment, and pre-assignment covariates.
pub fn fit_ancova(
    outcome: &[f64], assignment: &[bool], covariates: &[&[f64]],
) -> Result<AncovaFit, &'static str> {
    let n = outcome.len();
    let k = covariates.len();
    let q = k + 2;
    if n <= q || assignment.len() != n || k == 0 || covariates.iter().any(|x| x.len() != n) {
        return Err("ANCOVA requires aligned rows, at least one covariate, and residual degrees of freedom");
    }
    if outcome.iter().chain(covariates.iter().flat_map(|x| x.iter())).any(|v| !v.is_finite()) {
        return Err("outcomes and covariates must be finite");
    }
    let treated = assignment.iter().filter(|&&value| value).count();
    let control = n - treated;
    if treated == 0 || control == 0 {
        return Err("ANCOVA requires observed treated and control units");
    }
    let rows = (0..n).map(|i| {
        let mut row = Vec::with_capacity(q);
        row.extend([1.0, f64::from(assignment[i])]);
        row.extend(covariates.iter().map(|x| x[i]));
        row
    }).collect::<Vec<_>>();
    let mut gram = vec![vec![0.0; q]; q];
    let mut rhs = vec![0.0; q];
    for (row, &target) in rows.iter().zip(outcome) {
        for j in 0..q {
            rhs[j] += row[j] * target;
            for l in 0..q { gram[j][l] += row[j] * row[l]; }
        }
    }
    let inverse = invert_gram(gram)?;
    let beta = (0..q).map(|j| (0..q).map(|l| inverse[j][l] * rhs[l]).sum::<f64>()).collect::<Vec<_>>();
    let mut meat = vec![vec![0.0; q]; q];
    for (row, &target) in rows.iter().zip(outcome) {
        let residual = target - row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
        for j in 0..q { for l in 0..q { meat[j][l] += row[j] * row[l] * residual.powi(2); } }
    }
    let mut variance = 0.0;
    for j in 0..q {
        for l in 0..q {
            variance += inverse[1][j] * meat[j][l] * inverse[l][1];
        }
    }
    let variance = variance.max(0.0);
    if !beta.iter().all(|v| v.is_finite()) || !variance.is_finite() {
        return Err("ANCOVA coefficients or variance are not finite");
    }
    Ok(AncovaFit { effect: beta[1], hc0_variance: variance, covariate_coefficients: beta[2..].to_vec(), treated, control })
}

fn invert_gram(mut matrix: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, &'static str> {
    let n = matrix.len();
    let max_diagonal = (0..n).map(|i| matrix[i][i].abs()).fold(0.0_f64, f64::max);
    let mut inverse = vec![vec![0.0; n]; n];
    for (i, row) in inverse.iter_mut().enumerate() { row[i] = 1.0; }
    for column in 0..n {
        let pivot_row = (column..n).max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs())).expect("non-empty pivot range");
        let pivot_scale = matrix[pivot_row][column].abs();
        if !pivot_scale.is_finite() || pivot_scale <= f64::EPSILON * 16.0 || pivot_scale <= max_diagonal.max(1.0) * 1e-12 {
            return Err("ANCOVA design is rank deficient or covariates are collinear");
        }
        matrix.swap(column, pivot_row);
        inverse.swap(column, pivot_row);
        let pivot = matrix[column][column];
        for j in 0..n { matrix[column][j] /= pivot; inverse[column][j] /= pivot; }
        for row in 0..n {
            if row == column { continue; }
            let multiplier = matrix[row][column];
            for j in 0..n { matrix[row][j] -= multiplier * matrix[column][j]; inverse[row][j] -= multiplier * inverse[column][j]; }
        }
    }
    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_linear_effect_and_refuses_collinearity() {
        let x1 = [0., 1., 2., 3., 4., 5., 6., 7.];
        let x2 = [1., 0., 1., 0., 1., 0., 1., 0.];
        let assignment = [false, true, false, true, true, false, true, false];
        let y = (0..8).map(|i| 3. + 2. * f64::from(assignment[i]) + 4. * x1[i] - x2[i]).collect::<Vec<_>>();
        let fit = fit_ancova(&y, &assignment, &[&x1, &x2]).unwrap();
        assert!((fit.effect - 2.).abs() < 1e-10);
        assert!(fit.hc0_variance < 1e-20);
        assert!(fit_ancova(&y, &assignment, &[&x1, &x1]).is_err());
    }
}
