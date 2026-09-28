//! Independent-unit randomized ANCOVA point and HC0 sandwich variance.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Fitted treatment coefficient, HC0 variance, and covariate coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct AncovaFit {
    /// Coefficient on realized assignment.
    pub effect: f64,
    /// Independent-row HC0 sandwich variance for the treatment coefficient.
    pub hc0_variance: f64,
    /// Coefficients in the supplied covariate order.
    pub covariate_coefficients: Vec<f64>,
    /// Number assigned treatment.
    pub treated: usize,
    /// Number assigned control.
    pub control: usize,
}

/// Pointwise interval calibrated for independently assigned Bernoulli rows.
///
/// The coefficient and HC0 variance remain available below this measured
/// support boundary; an interval does not.
#[must_use]
pub fn calibrated_bernoulli_interval_95(fit: &AncovaFit, probability: f64) -> Option<[f64; 2]> {
    const NORMAL_95: f64 = 1.959_963_984_540_054;
    if fit.covariate_coefficients.is_empty()
        || fit.covariate_coefficients.len() > 2
        || fit.treated + fit.control < 400
        || fit.treated < 30
        || fit.control < 30
        || !probability.is_finite()
        || probability + 1e-12 < 0.2
        || probability > 0.8 + 1e-12
        || !fit.hc0_variance.is_finite()
        || fit.hc0_variance <= 0.0
    {
        return None;
    }
    let radius = NORMAL_95 * fit.hc0_variance.sqrt();
    Some([fit.effect - radius, fit.effect + radius])
}

/// Fit OLS with intercept, assignment, and pre-assignment covariates.
pub fn fit_ancova(
    outcome: &[f64],
    assignment: &[bool],
    covariates: &[&[f64]],
) -> Result<AncovaFit, &'static str> {
    let n = outcome.len();
    let k = covariates.len();
    let q = k + 2;
    if n <= q || assignment.len() != n || k == 0 || covariates.iter().any(|x| x.len() != n) {
        return Err(
            "ANCOVA requires aligned rows, at least one covariate, and residual degrees of freedom",
        );
    }
    if outcome.iter().chain(covariates.iter().flat_map(|x| x.iter())).any(|v| !v.is_finite()) {
        return Err("outcomes and covariates must be finite");
    }
    let treated = assignment.iter().filter(|&&value| value).count();
    let control = n - treated;
    if treated == 0 || control == 0 {
        return Err("ANCOVA requires observed treated and control units");
    }
    let rows = (0..n)
        .map(|i| {
            let mut row = Vec::with_capacity(q);
            row.extend([1.0, f64::from(assignment[i])]);
            row.extend(covariates.iter().map(|x| x[i]));
            row
        })
        .collect::<Vec<_>>();
    let mut gram = vec![vec![0.0; q]; q];
    let mut rhs = vec![0.0; q];
    for (row, &target) in rows.iter().zip(outcome) {
        for j in 0..q {
            rhs[j] += row[j] * target;
            for l in 0..q {
                gram[j][l] += row[j] * row[l];
            }
        }
    }
    let inverse = invert_gram(gram)?;
    let beta =
        (0..q).map(|j| (0..q).map(|l| inverse[j][l] * rhs[l]).sum::<f64>()).collect::<Vec<_>>();
    let mut meat = vec![vec![0.0; q]; q];
    for (row, &target) in rows.iter().zip(outcome) {
        let residual = target - row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
        for j in 0..q {
            for l in 0..q {
                meat[j][l] += row[j] * row[l] * residual.powi(2);
            }
        }
    }
    let mut variance = 0.0;
    #[allow(clippy::needless_range_loop, reason = "index used for multiple aligned slices")]
    for j in 0..q {
        for l in 0..q {
            variance += inverse[1][j] * meat[j][l] * inverse[l][1];
        }
    }
    let variance = variance.max(0.0);
    if !beta.iter().all(|v| v.is_finite()) || !variance.is_finite() {
        return Err("ANCOVA coefficients or variance are not finite");
    }
    Ok(AncovaFit {
        effect: beta[1],
        hc0_variance: variance,
        covariate_coefficients: beta[2..].to_vec(),
        treated,
        control,
    })
}

fn invert_gram(mut matrix: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, &'static str> {
    let n = matrix.len();
    let max_diagonal = (0..n).map(|i| matrix[i][i].abs()).fold(0.0_f64, f64::max);
    let mut inverse = vec![vec![0.0; n]; n];
    for (i, row) in inverse.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for column in 0..n {
        let pivot_row = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .expect("non-empty pivot range");
        let pivot_scale = matrix[pivot_row][column].abs();
        if !pivot_scale.is_finite()
            || pivot_scale <= f64::EPSILON * 16.0
            || pivot_scale <= max_diagonal.max(1.0) * 1e-12
        {
            return Err("ANCOVA design is rank deficient or covariates are collinear");
        }
        matrix.swap(column, pivot_row);
        inverse.swap(column, pivot_row);
        let pivot = matrix[column][column];
        for j in 0..n {
            matrix[column][j] /= pivot;
            inverse[column][j] /= pivot;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let multiplier = matrix[row][column];
            for j in 0..n {
                matrix[row][j] -= multiplier * matrix[column][j];
                inverse[row][j] -= multiplier * inverse[column][j];
            }
        }
    }
    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(mut state: u64) -> f64 {
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        ((state ^ (state >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    #[test]
    fn recovers_known_linear_effect_and_refuses_collinearity() {
        let x1 = [0., 1., 2., 3., 4., 5., 6., 7.];
        let x2 = [1., 0., 1., 0., 1., 0., 1., 0.];
        let assignment = [false, true, false, true, true, false, true, false];
        let y = (0..8)
            .map(|i| 3. + 2. * f64::from(assignment[i]) + 4. * x1[i] - x2[i])
            .collect::<Vec<_>>();
        let fit = fit_ancova(&y, &assignment, &[&x1, &x2]).unwrap();
        assert!((fit.effect - 2.).abs() < 1e-10);
        assert!(fit.hc0_variance < 1e-20);
        assert!(fit_ancova(&y, &assignment, &[&x1, &x1]).is_err());
    }

    #[test]
    fn bernoulli_ancova_hc0_covers_known_finite_population_effect() {
        const REPLICATES: usize = 2_000;
        const N: usize = 400;
        let x1 = (0..N).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
        let x2 = (0..N).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
        let effect = (0..N).map(|i| 1.4 + 0.2 * x1[i]).sum::<f64>() / N as f64;
        for (covariate_count, probability) in
            [(1, 0.2), (1, 0.5), (1, 0.8), (2, 0.2), (2, 0.5), (2, 0.8)]
        {
            let mut covered = 0;
            for rep in 0..REPLICATES {
                let assignment = (0..N)
                    .map(|i| {
                        uniform(
                            (rep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
                        ) < probability
                    })
                    .collect::<Vec<_>>();
                let outcome = (0..N)
                    .map(|i| {
                        2.0 + 1.2 * x1[i] - 0.7 * x2[i]
                            + 0.5 * (0.29 * i as f64).sin()
                            + f64::from(assignment[i]) * (1.4 + 0.2 * x1[i])
                    })
                    .collect::<Vec<_>>();
                let covariates: Vec<&[f64]> =
                    if covariate_count == 1 { vec![&x1] } else { vec![&x1, &x2] };
                let fit = fit_ancova(&outcome, &assignment, &covariates).unwrap();
                assert!(fit.hc0_variance > 0.0);
                let [lower, upper] = calibrated_bernoulli_interval_95(&fit, probability).unwrap();
                covered += usize::from(lower <= effect && effect <= upper);
            }
            let rate = covered as f64 / REPLICATES as f64;
            eprintln!(
                "Bernoulli ANCOVA HC0 covariates={covariate_count} p={probability} known-truth coverage: {covered}/{REPLICATES} = {rate:.4}"
            );
            assert!((0.93..=0.985).contains(&rate));
        }
    }

    #[test]
    fn calibrated_interval_refuses_sparse_low_probability_and_degenerate_variance() {
        let fit = AncovaFit {
            effect: 1.0,
            hc0_variance: 0.1,
            covariate_coefficients: vec![2.0],
            treated: 29,
            control: 371,
        };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
        let fit = AncovaFit { treated: 200, control: 200, ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.1).is_none());
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_some());
        let fit = AncovaFit { hc0_variance: 0.0, ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
        let fit = AncovaFit { hc0_variance: 0.1, covariate_coefficients: vec![0.0; 3], ..fit };
        assert!(calibrated_bernoulli_interval_95(&fit, 0.5).is_none());
    }
}
