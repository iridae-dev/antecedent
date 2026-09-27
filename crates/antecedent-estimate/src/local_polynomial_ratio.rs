//! Shared native local-polynomial fuzzy RD and regression-kink kernel.
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Bias-corrected local ratio and HC0 delta-method uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalPolynomialRatioFit {
    /// Reduced-form to first-stage ratio.
    pub estimate: f64,
    /// Corrected outcome discontinuity or kink.
    pub reduced_form: f64,
    /// Corrected treatment discontinuity or kink.
    pub first_stage: f64,
    /// Window observations to the left.
    pub n_left: usize,
    /// Window observations to the right.
    pub n_right: usize,
    /// HC0 delta-method standard error.
    pub standard_error: f64,
    /// Normal interval lower endpoint at the fixed bandwidth.
    pub ci_lower: f64,
    /// Normal interval upper endpoint at the fixed bandwidth.
    pub ci_upper: f64,
    /// Reduced-form HC0 standard error.
    pub reduced_form_standard_error: f64,
    /// First-stage HC0 standard error.
    pub first_stage_standard_error: f64,
}

fn invert_three(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut augmented = [[0.0; 6]; 3];
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&matrix[row]);
        augmented[row][row + 3] = 1.0;
    }
    for column in 0..3 {
        let pivot = (column..3).max_by(|left, right| {
            augmented[*left][column].abs().total_cmp(&augmented[*right][column].abs())
        })?;
        if augmented[pivot][column].abs() < 1e-12 {
            return None;
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= scale;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let scale = augmented[row][column];
            for index in 0..6 {
                augmented[row][index] -= scale * augmented[column][index];
            }
        }
    }
    let mut inverse = [[0.0; 3]; 3];
    for row in 0..3 {
        inverse[row].copy_from_slice(&augmented[row][3..]);
    }
    Some(inverse)
}

fn local_quadratic_side(
    running: &[f64],
    outcome: &[f64],
    treatment: &[f64],
    cutoff: f64,
    bandwidth: f64,
    right_side: bool,
) -> Option<([f64; 3], [f64; 3], usize)> {
    let mut gram = [[0.0; 3]; 3];
    let mut rhs_y = [0.0; 3];
    let mut rhs_t = [0.0; 3];
    let mut count = 0usize;
    for index in 0..running.len() {
        let distance = (running[index] - cutoff) / bandwidth;
        if distance == 0.0
            || distance.abs() >= 1.0
            || (right_side && distance < 0.0)
            || (!right_side && distance > 0.0)
        {
            continue;
        }
        let basis = [1.0, distance, distance * distance];
        let weight = 1.0 - distance.abs();
        count += 1;
        for row in 0..3 {
            rhs_y[row] += weight * basis[row] * outcome[index];
            rhs_t[row] += weight * basis[row] * treatment[index];
            for column in 0..3 {
                gram[row][column] += weight * basis[row] * basis[column];
            }
        }
    }
    let inverse = invert_three(gram)?;
    let mut beta_y = [0.0; 3];
    let mut beta_t = [0.0; 3];
    for row in 0..3 {
        for column in 0..3 {
            beta_y[row] += inverse[row][column] * rhs_y[column];
            beta_t[row] += inverse[row][column] * rhs_t[column];
        }
    }
    Some((beta_y, beta_t, count))
}

#[derive(Clone, Copy)]
struct LocalCubicFit {
    beta_y: [f64; 4],
    beta_t: [f64; 4],
    var_y: [[f64; 4]; 4],
    var_t: [[f64; 4]; 4],
    cov_yt: [[f64; 4]; 4],
    count: usize,
}

fn invert_four(matrix: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut augmented = [[0.0; 8]; 4];
    for row in 0..4 {
        augmented[row][..4].copy_from_slice(&matrix[row]);
        augmented[row][row + 4] = 1.0;
    }
    for column in 0..4 {
        let pivot = (column..4).max_by(|left, right| {
            augmented[*left][column].abs().total_cmp(&augmented[*right][column].abs())
        })?;
        if augmented[pivot][column].abs() < 1e-12 {
            return None;
        }
        augmented.swap(column, pivot);
        let scale = augmented[column][column];
        for value in &mut augmented[column] {
            *value /= scale;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let scale = augmented[row][column];
            for index in 0..8 {
                augmented[row][index] -= scale * augmented[column][index];
            }
        }
    }
    let mut inverse = [[0.0; 4]; 4];
    for row in 0..4 {
        inverse[row].copy_from_slice(&augmented[row][4..]);
    }
    Some(inverse)
}

fn sandwich_four(inverse: [[f64; 4]; 4], meat: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut covariance = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            #[allow(clippy::needless_range_loop, reason = "index used for multiple aligned matrices")]
            for left in 0..4 {
                for right in 0..4 {
                    covariance[row][column] +=
                        inverse[row][left] * meat[left][right] * inverse[right][column];
                }
            }
        }
    }
    covariance
}

#[derive(Clone, Copy)]
struct LocalQuarticSlopeFit {
    slope_y: f64,
    slope_t: f64,
    var_y: f64,
    var_t: f64,
    cov_yt: f64,
    count: usize,
}

fn invert_five(mut matrix: [[f64; 5]; 5]) -> Option<[[f64; 5]; 5]> {
    let mut inverse = [[0.0; 5]; 5];
    #[allow(clippy::needless_range_loop, reason = "index sets the matrix diagonal inverse[index][index]")]
    for index in 0..5 { inverse[index][index] = 1.0; }
    for column in 0..5 {
        let pivot = (column..5).max_by(|left, right| {
            matrix[*left][column].abs().total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() < 1e-12 { return None; }
        matrix.swap(column, pivot);
        inverse.swap(column, pivot);
        let scale = matrix[column][column];
        for index in 0..5 {
            matrix[column][index] /= scale;
            inverse[column][index] /= scale;
        }
        for row in 0..5 {
            if row == column { continue; }
            let scale = matrix[row][column];
            for index in 0..5 {
                matrix[row][index] -= scale * matrix[column][index];
                inverse[row][index] -= scale * inverse[column][index];
            }
        }
    }
    Some(inverse)
}

fn local_quartic_slope_side(
    running: &[f64], outcome: &[f64], treatment: &[f64], cutoff: f64,
    bandwidth: f64, right_side: bool,
) -> Option<LocalQuarticSlopeFit> {
    let mut gram = [[0.0; 5]; 5];
    let mut rhs_y = [0.0; 5];
    let mut rhs_t = [0.0; 5];
    let mut rows = Vec::new();
    for index in 0..running.len() {
        let distance = (running[index] - cutoff) / bandwidth;
        if distance == 0.0 || distance.abs() >= 1.0
            || (right_side && distance < 0.0) || (!right_side && distance > 0.0) {
            continue;
        }
        let basis = [1.0, distance, distance.powi(2), distance.powi(3), distance.powi(4)];
        let weight = 1.0 - distance.abs();
        rows.push((basis, weight, outcome[index], treatment[index]));
        for row in 0..5 {
            rhs_y[row] += weight * basis[row] * outcome[index];
            rhs_t[row] += weight * basis[row] * treatment[index];
            for column in 0..5 { gram[row][column] += weight * basis[row] * basis[column]; }
        }
    }
    let inverse = invert_five(gram)?;
    let mut beta_y = [0.0; 5];
    let mut beta_t = [0.0; 5];
    for row in 0..5 {
        for column in 0..5 {
            beta_y[row] += inverse[row][column] * rhs_y[column];
            beta_t[row] += inverse[row][column] * rhs_t[column];
        }
    }
    let mut var_y = 0.0;
    let mut var_t = 0.0;
    let mut cov_yt = 0.0;
    for (basis, weight, value_y, value_t) in &rows {
        let leverage = (0..5).map(|index| inverse[1][index] * basis[index]).sum::<f64>();
        let score_y = weight * leverage
            * (value_y - (0..5).map(|index| beta_y[index] * basis[index]).sum::<f64>());
        let score_t = weight * leverage
            * (value_t - (0..5).map(|index| beta_t[index] * basis[index]).sum::<f64>());
        var_y += score_y * score_y;
        var_t += score_t * score_t;
        cov_yt += score_y * score_t;
    }
    Some(LocalQuarticSlopeFit {
        slope_y: beta_y[1], slope_t: beta_t[1], var_y, var_t, cov_yt,
        count: rows.len(),
    })
}

fn local_cubic_side(
    running: &[f64],
    outcome: &[f64],
    treatment: &[f64],
    cutoff: f64,
    bandwidth: f64,
    right_side: bool,
) -> Option<LocalCubicFit> {
    let mut gram = [[0.0; 4]; 4];
    let mut rhs_y = [0.0; 4];
    let mut rhs_t = [0.0; 4];
    let mut rows = Vec::new();
    for index in 0..running.len() {
        let distance = (running[index] - cutoff) / bandwidth;
        if distance == 0.0
            || distance.abs() >= 1.0
            || (right_side && distance < 0.0)
            || (!right_side && distance > 0.0)
        {
            continue;
        }
        let basis = [1.0, distance, distance.powi(2), distance.powi(3)];
        let weight = 1.0 - distance.abs();
        rows.push((basis, weight, outcome[index], treatment[index]));
        for row in 0..4 {
            rhs_y[row] += weight * basis[row] * outcome[index];
            rhs_t[row] += weight * basis[row] * treatment[index];
            for column in 0..4 {
                gram[row][column] += weight * basis[row] * basis[column];
            }
        }
    }
    let inverse = invert_four(gram)?;
    let mut beta_y = [0.0; 4];
    let mut beta_t = [0.0; 4];
    for row in 0..4 {
        for column in 0..4 {
            beta_y[row] += inverse[row][column] * rhs_y[column];
            beta_t[row] += inverse[row][column] * rhs_t[column];
        }
    }
    let mut meat_y = [[0.0; 4]; 4];
    let mut meat_t = [[0.0; 4]; 4];
    let mut meat_yt = [[0.0; 4]; 4];
    for (basis, weight, value_y, value_t) in &rows {
        let residual_y = *value_y - (0..4).map(|i| beta_y[i] * basis[i]).sum::<f64>();
        let residual_t = *value_t - (0..4).map(|i| beta_t[i] * basis[i]).sum::<f64>();
        for row in 0..4 {
            for column in 0..4 {
                let scale = *weight * *weight * basis[row] * basis[column];
                meat_y[row][column] += scale * residual_y * residual_y;
                meat_t[row][column] += scale * residual_t * residual_t;
                meat_yt[row][column] += scale * residual_y * residual_t;
            }
        }
    }
    Some(LocalCubicFit {
        beta_y,
        beta_t,
        var_y: sandwich_four(inverse, meat_y),
        var_t: sandwich_four(inverse, meat_t),
        cov_yt: sandwich_four(inverse, meat_yt),
        count: rows.len(),
    })
}

fn cubic_bias_projection(
    running: &[f64],
    cutoff: f64,
    bandwidth: f64,
    right_side: bool,
) -> Option<[f64; 3]> {
    let mut gram = [[0.0; 3]; 3];
    let mut cross = [0.0; 3];
    for value in running {
        let distance = (*value - cutoff) / bandwidth;
        if distance == 0.0
            || distance.abs() >= 1.0
            || (right_side && distance < 0.0)
            || (!right_side && distance > 0.0)
        {
            continue;
        }
        let basis = [1.0, distance, distance * distance];
        let weight = 1.0 - distance.abs();
        for row in 0..3 {
            cross[row] += weight * basis[row] * distance.powi(3);
            for column in 0..3 {
                gram[row][column] += weight * basis[row] * basis[column];
            }
        }
    }
    let inverse = invert_three(gram)?;
    let mut projection = [0.0; 3];
    for row in 0..3 {
        #[allow(clippy::needless_range_loop, reason = "index used for multiple aligned matrices")]
        for column in 0..3 {
            projection[row] += inverse[row][column] * cross[column];
        }
    }
    Some(projection)
}

/// Fit local quadratic jumps with a cubic pilot, or kinks with a quartic pilot.
// length reflects the estimator's fixed statistical contract; refactor would change behavior
#[allow(clippy::too_many_lines)]
pub fn fit_local_polynomial_ratio(
    running: &[f64],
    outcome: &[f64],
    treatment: &[f64],
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
) -> Result<LocalPolynomialRatioFit, String> {
    let n = running.len();
    if n == 0 || outcome.len() != n || treatment.len() != n {
        return Err(String::from(
            "running, outcome, and treatment vectors must have equal non-zero length",
        ));
    }
    if !cutoff.is_finite() || !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(String::from("cutoff must be finite and bandwidth positive"));
    }
    if running.iter().chain(outcome.iter()).chain(treatment.iter()).any(|x| !x.is_finite()) {
        return Err(String::from("running, outcome, and treatment values must be finite"));
    }
    let running: Vec<f64> = running.to_vec();
    let outcome: Vec<f64> = outcome.to_vec();
    let treatment: Vec<f64> = treatment.to_vec();
    let (left_y_p, left_t_p, n_left) =
        local_quadratic_side(&running, &outcome, &treatment, cutoff, bandwidth, false)
            .filter(|(_, _, count)| *count >= 4)
            .ok_or_else(|| {
                String::from("local polynomial lacks full-rank support on the left side")
            })?;
    let (right_y_p, right_t_p, n_right) =
        local_quadratic_side(&running, &outcome, &treatment, cutoff, bandwidth, true)
            .filter(|(_, _, count)| *count >= 4)
            .ok_or_else(|| {
                String::from("local polynomial lacks full-rank support on the right side")
            })?;
    let left_q = local_cubic_side(&running, &outcome, &treatment, cutoff, bandwidth, false)
        .filter(|fit| fit.count >= 5)
        .ok_or_else(|| {
            String::from("robust bias correction lacks cubic support on the left side")
        })?;
    let right_q = local_cubic_side(&running, &outcome, &treatment, cutoff, bandwidth, true)
        .filter(|fit| fit.count >= 5)
        .ok_or_else(|| {
            String::from("robust bias correction lacks cubic support on the right side")
        })?;
    let left_projection = cubic_bias_projection(&running, cutoff, bandwidth, false)
        .ok_or_else(|| String::from("bias correction lacks left-side support"))?;
    let right_projection = cubic_bias_projection(&running, cutoff, bandwidth, true)
        .ok_or_else(|| String::from("bias correction lacks right-side support"))?;
    let coefficient = usize::from(kink);
    let scale = if kink { 1.0 / bandwidth } else { 1.0 };
    // The p=2 local-polynomial coefficient's leading omitted-cubic bias is
    // projection[j] * beta_3. Subtracting it gives the RBC coefficient; the
    // same identity makes the q=3 HC0 covariance the corrected-coefficient
    // covariance (including the covariance of the estimated bias term).
    let corrected_y_left = left_y_p[coefficient] - left_projection[coefficient] * left_q.beta_y[3];
    let corrected_y_right =
        right_y_p[coefficient] - right_projection[coefficient] * right_q.beta_y[3];
    let corrected_t_left = left_t_p[coefficient] - left_projection[coefficient] * left_q.beta_t[3];
    let corrected_t_right =
        right_t_p[coefficient] - right_projection[coefficient] * right_q.beta_t[3];
    let mut reduced_form = (corrected_y_right - corrected_y_left) * scale;
    let mut first_stage = (corrected_t_right - corrected_t_left) * scale;
    if first_stage.abs() < 1e-10 {
        return Err(String::from(
            "local treatment discontinuity is too small to form a fuzzy design estimate",
        ));
    }
    let mut variance_y = (left_q.var_y[coefficient][coefficient]
        + right_q.var_y[coefficient][coefficient])
        * scale
        * scale;
    let mut variance_t = (left_q.var_t[coefficient][coefficient]
        + right_q.var_t[coefficient][coefficient])
        * scale
        * scale;
    let mut covariance_yt = (left_q.cov_yt[coefficient][coefficient]
        + right_q.cov_yt[coefficient][coefficient])
        * scale
        * scale;
    if kink {
        // A fixed, wide bandwidth can leave a common quartic outcome trend
        // in the q=3 local slopes. A q=4 pilot removes that term from the
        // slope contrast and supplies the matching HC0 joint covariance.
        let left = local_quartic_slope_side(&running, &outcome, &treatment, cutoff, bandwidth, false)
            .filter(|fit| fit.count >= 6)
            .ok_or_else(|| String::from("kink bias correction lacks quartic support on the left side"))?;
        let right = local_quartic_slope_side(&running, &outcome, &treatment, cutoff, bandwidth, true)
            .filter(|fit| fit.count >= 6)
            .ok_or_else(|| String::from("kink bias correction lacks quartic support on the right side"))?;
        reduced_form = (right.slope_y - left.slope_y) / bandwidth;
        first_stage = (right.slope_t - left.slope_t) / bandwidth;
        variance_y = (left.var_y + right.var_y) / bandwidth.powi(2);
        variance_t = (left.var_t + right.var_t) / bandwidth.powi(2);
        covariance_yt = (left.cov_yt + right.cov_yt) / bandwidth.powi(2);
    }
    let se_first = variance_t.max(0.0).sqrt();
    if first_stage.abs() <= 1.96 * se_first {
        return Err(String::from(
            "weak first stage: local treatment change is not separated from zero by its HC0 95% interval",
        ));
    }
    let estimate = reduced_form / first_stage;
    let variance_estimate = variance_y / first_stage.powi(2)
        + reduced_form.powi(2) * variance_t / first_stage.powi(4)
        - 2.0 * reduced_form * covariance_yt / first_stage.powi(3);
    if variance_estimate < -1e-10 || !variance_estimate.is_finite() {
        return Err(String::from("robust ratio variance is not finite and non-negative"));
    }
    let standard_error = variance_estimate.max(0.0).sqrt();
    let critical = 1.959_963_984_540_054;
    Ok(LocalPolynomialRatioFit {
        estimate, reduced_form, first_stage, n_left, n_right, standard_error,
        ci_lower: estimate - critical * standard_error,
        ci_upper: estimate + critical * standard_error,
        reduced_form_standard_error: variance_y.max(0.0).sqrt(),
        first_stage_standard_error: se_first,
    })
}

#[cfg(test)]
mod tests {
    use super::fit_local_polynomial_ratio;

    #[test]
    fn fuzzy_jump_recovers_known_local_effect_and_refuses_sparse_support() {
        let mut running = Vec::new();
        let mut treatment = Vec::new();
        let mut outcome = Vec::new();
        for step in 1..80 {
            for (sign, assignments) in [(-1.0, [1.0, 0.0, 0.0, 0.0]), (1.0, [1.0, 1.0, 1.0, 0.0])] {
                let score = sign * f64::from(step) / 80.0;
                for (assigned, noise) in assignments.into_iter().zip([-0.02, -0.01, 0.01, 0.02]) {
                    running.push(score);
                    treatment.push(assigned);
                    outcome.push(1.0 + 2.0 * score + 0.5 * score * score + 3.0 * assigned + noise);
                }
            }
        }
        let fit = fit_local_polynomial_ratio(&running, &outcome, &treatment, 0.0, 1.0, false).unwrap();
        assert!((fit.estimate - 3.0).abs() < 1e-10);
        assert!((fit.first_stage - 0.5).abs() < 1e-10);
        assert_eq!((fit.n_left, fit.n_right), (316, 316));
        assert!(fit.standard_error > 0.0);
        let sparse_running = [-0.2, -0.1, 0.1, 0.2];
        assert!(fit_local_polynomial_ratio(&sparse_running, &[0.0, 0.0, 1.0, 1.0],
            &[0.0, 0.0, 1.0, 1.0], 0.0, 1.0, false).unwrap_err().contains("full-rank support"));
    }
}
