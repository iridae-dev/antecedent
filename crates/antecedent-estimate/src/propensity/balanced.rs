//! Exponential covariate-balance calibration of inverse-probability weights and the exact
//! linearization of its calibrated Hájek contrast.
//!
//! Each arm's weights are calibrated to the sampling-weighted population's covariate moments;
//! the influence of the calibrated contrast is a diagonal-plus-low-rank map of the outcomes,
//! which gives the working-Gaussian quadratic-form reference and the exact conditional noise
//! component of a linear point.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::arm::TreatmentArm;
use super::prepare::PreparedPropensityProblem;
use super::weighting::{PopulationLinearVariance, hajek_iid_se};
use crate::error::EstimationError;

/// Positive exponential calibration of each arm to the full weighted covariate moments.
pub(super) fn balance_arm_weights(
    x: &[f64],
    p: usize,
    treatment: &[f64],
    sampling: Option<&[f64]>,
    weights: &mut [f64],
) -> Result<(), EstimationError> {
    let n = treatment.len();
    let mass = |i: usize| sampling.map_or(1.0, |s| s[i]);
    let target: Vec<f64> = (0..p).map(|c| (0..n).map(|i| mass(i) * x[c * n + i]).sum()).collect();
    let scale = (0..n).map(mass).sum::<f64>();
    for active in [false, true] {
        let rows: Vec<usize> = (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(treatment[i])) == active).collect();
        let base: Vec<f64> = rows.iter().map(|&i| weights[i]).collect();
        let mut lambda = vec![0.0; p];
        let evaluate = |lambda: &[f64]| -> Option<(f64, Vec<f64>)> {
            let calibrated: Vec<f64> = rows
                .iter()
                .zip(&base)
                .map(|(&i, base)| {
                    base * (0..p).map(|c| lambda[c] * x[c * n + i]).sum::<f64>().exp()
                })
                .collect();
            let objective = calibrated.iter().sum::<f64>()
                - lambda.iter().zip(&target).map(|(a, b)| a * b).sum::<f64>();
            objective.is_finite().then_some((objective, calibrated))
        };
        let mut converged = false;
        for _ in 0..60 {
            let (objective, calibrated) = evaluate(&lambda).ok_or_else(|| {
                EstimationError::stats_msg("nonfinite exponential calibration weights")
            })?;
            let mut residual = target.clone();
            let mut hessian = vec![0.0; p * p];
            for (&i, &w) in rows.iter().zip(&calibrated) {
                for c in 0..p {
                    residual[c] -= w * x[c * n + i];
                    for d in 0..p {
                        hessian[c * p + d] += w * x[c * n + i] * x[d * n + i];
                    }
                }
            }
            if residual.iter().all(|r| r.abs() <= 1e-10 * scale) {
                for (&i, &w) in rows.iter().zip(&calibrated) {
                    weights[i] = w;
                }
                converged = true;
                break;
            }
            let step = crate::util::solve_spd(&hessian, &residual, p).ok_or_else(|| {
                EstimationError::stats_msg("singular covariate-balance information")
            })?;
            let descent = residual.iter().zip(&step).map(|(r, s)| r * s).sum::<f64>();
            let mut length = 1.0;
            let mut accepted = false;
            for _ in 0..30 {
                let candidate: Vec<f64> =
                    lambda.iter().zip(&step).map(|(l, s)| l + length * s).collect();
                if evaluate(&candidate).is_some_and(|(value, calibrated)| {
                    // Near the optimum, objective reductions round away before the moment
                    // equations reach their requested tolerance. Accept a Newton step that
                    // halves their residual as an independent, well-scaled merit criterion.
                    let candidate_residual = (0..p)
                        .map(|c| {
                            (target[c]
                                - rows
                                    .iter()
                                    .zip(&calibrated)
                                    .map(|(&i, w)| w * x[c * n + i])
                                    .sum::<f64>())
                            .abs()
                        })
                        .fold(0.0_f64, f64::max);
                    value <= objective - 1e-4 * length * descent
                        || candidate_residual
                            <= 0.5 * residual.iter().map(|r| r.abs()).fold(0.0_f64, f64::max)
                }) {
                    lambda = candidate;
                    accepted = true;
                    break;
                }
                length *= 0.5;
            }
            if !accepted {
                return Err(EstimationError::stats_msg("covariate-balance line search failed"));
            }
        }
        if !converged {
            return Err(EstimationError::stats_msg("covariate-balance fit did not converge"));
        }
    }
    Ok(())
}

/// A centered outcome-to-influence map `L = diag(d) + U V'`.
pub(super) struct LowRankInfluence {
    diagonal: Vec<f64>,
    left: Vec<f64>,
    right: Vec<f64>,
    rank: usize,
}

impl LowRankInfluence {
    pub(super) fn apply(&self, y: &[f64]) -> Vec<f64> {
        let reduced: Vec<f64> = (0..self.rank)
            .map(|c| (0..y.len()).map(|i| self.right[i * self.rank + c] * y[i]).sum())
            .collect();
        (0..y.len())
            .map(|i| {
                self.diagonal[i] * y[i]
                    + (0..self.rank).map(|c| self.left[i * self.rank + c] * reduced[c]).sum::<f64>()
            })
            .collect()
    }
    fn transpose_apply(&self, y: &[f64]) -> Vec<f64> {
        let reduced: Vec<f64> = (0..self.rank)
            .map(|c| (0..y.len()).map(|i| self.left[i * self.rank + c] * y[i]).sum())
            .collect();
        (0..y.len())
            .map(|i| {
                self.diagonal[i] * y[i]
                    + (0..self.rank)
                        .map(|c| self.right[i * self.rank + c] * reduced[c])
                        .sum::<f64>()
            })
            .collect()
    }
    fn quadratic_traces(&self) -> Result<(f64, f64), EstimationError> {
        let n = self.diagonal.len();
        let k = self.rank;
        let rank = 2 * k;
        let diagonal: Vec<f64> = self.diagonal.iter().map(|d| d * d).collect();
        let mut basis = vec![0.0; n * rank];
        let mut core = vec![0.0; rank * rank];
        for i in 0..n {
            for c in 0..k {
                basis[i * rank + c] = self.diagonal[i] * self.left[i * k + c];
                basis[i * rank + k + c] = self.right[i * k + c];
            }
        }
        for c in 0..k {
            core[c * rank + k + c] = 1.0;
            core[(k + c) * rank + c] = 1.0;
            for d in 0..k {
                core[(k + c) * rank + k + d] =
                    (0..n).map(|i| self.left[i * k + c] * self.left[i * k + d]).sum();
            }
        }
        Ok(antecedent_stats::diagonal_low_rank_quadratic_traces(&diagonal, &basis, &core, rank)?)
    }
}

/// Exact conditional noise component for an outcome-linear point h'Y. The caller must
/// ensure h is outcome-independent, rows are independent, and the declared outcome model
/// is homoskedastic Gaussian with a mean linear in each arm's prepared design.
pub(crate) struct PopulationLinearNoise {
    pub variance: f64,
    pub scale: f64,
    pub degrees_of_freedom: usize,
    pub working_mean: Vec<f64>,
    pub contrast_mean: Vec<f64>,
}

pub(crate) fn population_linear_noise(
    problem: &PreparedPropensityProblem,
    h: &[f64],
) -> Result<PopulationLinearNoise, EstimationError> {
    let n = problem.nrows;
    let p = problem.design_ncols;
    if n <= 2 * p + 1 {
        return Err(EstimationError::data_msg(
            "population linear score requires residual degrees of freedom",
        ));
    }
    if h.len() != n || h.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::data_msg("invalid linear point coefficients"));
    }
    let (working_mean, contrast_mean, _) =
        gaussian_working_outcomes(&problem.design_matrix, p, &problem.treatment, &problem.outcome)?;
    let x = &problem.design_matrix;
    // v=P h, with P the orthogonal (unweighted) arm-OLS residual projector.
    let mut residual_direction = h.to_vec();
    for active in [false, true] {
        let mut gram = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for i in (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(problem.treatment[i])) == active) {
            for c in 0..p {
                rhs[c] += x[c * n + i] * h[i];
                for d in 0..p {
                    gram[c * p + d] += x[c * n + i] * x[d * n + i];
                }
            }
        }
        let coefficients = crate::util::solve_spd(&gram, &rhs, p)
            .ok_or_else(|| EstimationError::stats_msg("singular linear noise projection"))?;
        for i in (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(problem.treatment[i])) == active) {
            residual_direction[i] -= (0..p).map(|c| x[c * n + i] * coefficients[c]).sum::<f64>();
        }
    }
    let norm = residual_direction.iter().map(|v| v * v).sum::<f64>();
    let coefficient_norm = h.iter().map(|v| v * v).sum::<f64>();
    let residuals: Vec<f64> = (0..n).map(|i| problem.outcome[i] - working_mean[i]).collect();
    let (residual_sum, noise_degrees_of_freedom) = if norm > 64.0 * f64::EPSILON * coefficient_norm
    {
        let dot = residual_direction.iter().zip(&residuals).map(|(v, e)| v * e).sum::<f64>();
        // P-vv'/||v||² annihilates both the arm-linear mean and h. Its
        // Gaussian quadratic is independent of h'Y and has rank n-2p-1. The quadratic is
        // summed from the projected residual e - (v'e/||v||²) v itself, never as the difference
        // `e'e - (v'e)²/||v||²` of two nearly equal sums, which cancels for a near-perfect fit.
        let slope = dot / norm;
        let projected = residuals
            .iter()
            .zip(&residual_direction)
            .map(|(e, v)| {
                let r = e - slope * v;
                r * r
            })
            .sum::<f64>();
        (projected, n - 2 * p - 1)
    } else {
        (residuals.iter().map(|e| e * e).sum::<f64>(), n - 2 * p)
    };
    let noise_scale = residual_sum / noise_degrees_of_freedom as f64;
    Ok(PopulationLinearNoise {
        variance: noise_scale * coefficient_norm,
        scale: noise_scale,
        degrees_of_freedom: noise_degrees_of_freedom,
        working_mean,
        contrast_mean,
    })
}

pub(super) struct BalancedLinearization {
    pub(super) operator: LowRankInfluence,
    point_coefficients: Vec<f64>,
    pub(super) working_mean: Vec<f64>,
    noise_variance: f64,
}

impl BalancedLinearization {
    pub(super) fn population_linear_variance(
        &self,
        problem: &PreparedPropensityProblem,
    ) -> Result<PopulationLinearVariance, EstimationError> {
        let noise = super::population_linear_noise(problem, &self.point_coefficients)?;
        if noise.contrast_mean.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::stats_msg("non-finite linear target contrast"));
        }
        let signal = self.operator.apply(&noise.working_mean);
        Ok(PopulationLinearVariance {
            outcome_model: "independent_homoskedastic_gaussian_arm_linear",
            noise: noise.variance,
            target: hajek_iid_se(&signal, problem.design_ncols).powi(2),
            noise_scale: noise.scale,
            noise_degrees_of_freedom: noise.degrees_of_freedom,
        })
    }

    pub(super) fn reference_df(&self) -> Result<f64, EstimationError> {
        let (trace, squared_trace) = self.operator.quadratic_traces()?;
        let signal = self.operator.apply(&self.working_mean);
        let second = self.operator.transpose_apply(&signal);
        let noncentral_mean = signal.iter().map(|x| x * x).sum::<f64>();
        let noncentral_variance = second.iter().map(|x| x * x).sum::<f64>();
        // For Gaussian Y~N(mu,sigma²I), E[Y'QY]=sigma²trQ+mu'Qmu and
        // Var[Y'QY]=2sigma⁴tr(Q²)+4sigma²mu'Q²mu. Match those two moments to a
        // scaled chi-square rather than pretending that a high-weight row contributes n df.
        let mean = self.noise_variance * trace + noncentral_mean;
        let half_variance = self.noise_variance.powi(2) * squared_trace
            + 2.0 * self.noise_variance * noncentral_variance;
        if half_variance <= 0.0 {
            return Ok(1e12);
        }
        let df = mean * mean / half_variance;
        if !df.is_finite() || df <= 0.0 {
            return Err(EstimationError::stats_msg("invalid calibrated quadratic reference df"));
        }
        Ok(df.clamp(1.0, 1e12))
    }
}

/// Sandwich of logistic scores, both calibrated-arm moment equations and the population
/// contrast. Eliminating the calibration coefficients gives outcome-model residuals plus
/// the estimated target's conditional contrast; the remaining logistic derivative is added
/// explicitly. HC3 corrects the fitted weighted residuals for their case-deletion leverage.
#[allow(clippy::too_many_arguments, reason = "aligned calibrated weighted estimating equations")]
pub(super) fn balanced_influence_values(
    x: &[f64],
    p: usize,
    treatment: &[f64],
    outcome: &[f64],
    sampling: Option<&[f64]>,
    propensity: &[f64],
    clip: Option<f64>,
    weights: &[f64],
    ate: f64,
) -> Result<(Vec<f64>, BalancedLinearization), EstimationError> {
    let n = treatment.len();
    let mass = |i: usize| sampling.map_or(1.0, |s| s[i]);
    let total = (0..n).map(mass).sum::<f64>();
    let mut beta = Vec::new();
    let mut inverse = Vec::new();
    for active in [false, true] {
        let mut information = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for i in (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(treatment[i])) == active) {
            for c in 0..p {
                rhs[c] += weights[i] * x[c * n + i] * outcome[i];
                for d in 0..p {
                    information[c * p + d] += weights[i] * x[c * n + i] * x[d * n + i];
                }
            }
        }
        let solve = |rhs: &[f64]| {
            crate::util::solve_spd(&information, rhs, p)
                .ok_or_else(|| EstimationError::stats_msg("singular calibrated outcome moment"))
        };
        beta.push(solve(&rhs)?);
        let mut inv = vec![0.0; p * p];
        for c in 0..p {
            let mut unit = vec![0.0; p];
            unit[c] = 1.0;
            let column = solve(&unit)?;
            for d in 0..p {
                inv[d * p + c] = column[d];
            }
        }
        inverse.push(inv);
    }
    let mut psi = vec![0.0; n];
    let mut score = vec![0.0; n];
    let mut information = vec![0.0; n];
    let mut derivative = vec![0.0; n];
    let mut leverages = vec![0.0; n];
    for i in 0..n {
        let arm = usize::from(TreatmentArm::is_treated_indicator(treatment[i]));
        let sign = if arm == 1 { 1.0 } else { -1.0 };
        let prediction = |a: usize| (0..p).map(|c| beta[a][c] * x[c * n + i]).sum::<f64>();
        let residual = outcome[i] - prediction(arm);
        let leverage = weights[i]
            * (0..p)
                .map(|c| {
                    (0..p)
                        .map(|d| x[c * n + i] * inverse[arm][c * p + d] * x[d * n + i])
                        .sum::<f64>()
                })
                .sum::<f64>();
        if leverage >= 1.0 - 1e-10 {
            return Err(EstimationError::stats_msg("unit-leverage calibrated outcome moment"));
        }
        leverages[i] = leverage;
        psi[i] = n as f64 / total
            * (sign * weights[i] * residual / (1.0 - leverage)
                + mass(i) * (prediction(1) - prediction(0) - ate));
        let e = propensity[i];
        let log_derivative = if clip.is_some_and(|c| e <= c || e >= 1.0 - c) {
            0.0
        } else if arm == 1 {
            -(1.0 - e)
        } else {
            e
        };
        derivative[i] = n as f64 / total * sign * weights[i] * residual * log_derivative;
        score[i] = mass(i) * (treatment[i] - e);
        information[i] = mass(i) * e * (1.0 - e);
    }
    crate::se::add_nuisance_correction(
        &mut psi,
        x,
        p,
        &score,
        &information,
        &derivative,
        "singular propensity information in balanced weighting",
    )?;
    let linearization = balanced_linearization(
        x,
        p,
        treatment,
        outcome,
        sampling,
        propensity,
        clip,
        weights,
        &score,
        &information,
        &inverse,
        &leverages,
    )?;
    Ok((psi, linearization))
}

/// Assemble the exact diagonal-plus-low-rank map of outcome noise into the calibrated IF.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the low-rank blocks must share the calibrated and propensity information matrices"
)]
fn balanced_linearization(
    x: &[f64],
    p: usize,
    treatment: &[f64],
    outcome: &[f64],
    sampling: Option<&[f64]>,
    propensity: &[f64],
    clip: Option<f64>,
    weights: &[f64],
    score: &[f64],
    information: &[f64],
    inverse: &[Vec<f64>],
    leverages: &[f64],
) -> Result<BalancedLinearization, EstimationError> {
    let n = treatment.len();
    let mass = |i: usize| sampling.map_or(1.0, |s| s[i]);
    let total = (0..n).map(mass).sum::<f64>();
    let invert = |matrix: &[f64]| -> Result<Vec<f64>, EstimationError> {
        let mut inverse = vec![0.0; p * p];
        for c in 0..p {
            let mut unit = vec![0.0; p];
            unit[c] = 1.0;
            let column = crate::util::solve_spd(matrix, &unit, p).ok_or_else(|| {
                EstimationError::stats_msg("singular calibrated variance operator")
            })?;
            for d in 0..p {
                inverse[d * p + c] = column[d];
            }
        }
        Ok(inverse)
    };
    let mut prop_information = vec![0.0; p * p];
    let mut calibration_derivative = vec![vec![0.0; p * p]; 2];
    let mut log_weight = vec![0.0; n];
    for i in 0..n {
        let arm = usize::from(TreatmentArm::is_treated_indicator(treatment[i]));
        let e = propensity[i];
        log_weight[i] = if clip.is_some_and(|c| e <= c || e >= 1.0 - c) {
            0.0
        } else if arm == 1 {
            -(1.0 - e)
        } else {
            e
        };
        let sign = if arm == 1 { 1.0 } else { -1.0 };
        for c in 0..p {
            for d in 0..p {
                prop_information[c * p + d] += information[i] * x[c * n + i] * x[d * n + i];
                calibration_derivative[arm][c * p + d] +=
                    sign * weights[i] * log_weight[i] * x[c * n + i] * x[d * n + i];
            }
        }
    }
    let prop_inverse = invert(&prop_information)?;
    let rank = 3 * p + 2;
    let mut diagonal = vec![0.0; n];
    let mut left = vec![0.0; n * rank];
    let mut right = vec![0.0; n * rank];
    let scaling = n as f64 / total;
    let mut prop_row = vec![0.0; p];
    for i in 0..n {
        let arm = usize::from(TreatmentArm::is_treated_indicator(treatment[i]));
        let sign = if arm == 1 { 1.0 } else { -1.0 };
        let leverage = leverages[i];
        let raw = sign * weights[i] / (1.0 - leverage);
        diagonal[i] = scaling * raw;
        for (c, slot) in prop_row.iter_mut().enumerate() {
            *slot =
                score[i] * (0..p).map(|d| x[d * n + i] * prop_inverse[d * p + c]).sum::<f64>();
        }
        for a in 0..2 {
            for c in 0..p {
                let direction = if a == 1 { 1.0 } else { -1.0 };
                left[i * rank + a * p + c] = scaling
                    * (direction * mass(i) * x[c * n + i]
                        - if arm == a { raw * x[c * n + i] } else { 0.0 }
                        - (0..p)
                            .map(|d| prop_row[d] * calibration_derivative[a][d * p + c])
                            .sum::<f64>());
                right[i * rank + a * p + c] = if arm == a {
                    weights[i] * (0..p).map(|d| inverse[a][c * p + d] * x[d * n + i]).sum::<f64>()
                } else {
                    0.0
                };
            }
        }
        left[i * rank + 2 * p] = -scaling * mass(i);
        right[i * rank + 2 * p] = sign * weights[i] / total;
        for c in 0..p {
            left[i * rank + 2 * p + 1 + c] = scaling * prop_row[c];
            right[i * rank + 2 * p + 1 + c] = sign * weights[i] * log_weight[i] * x[c * n + i];
        }
    }
    // Appending -1 * mean(L)' centers every influence column without materializing n².
    let mean_left: Vec<f64> =
        (0..rank - 1).map(|c| (0..n).map(|i| left[i * rank + c]).sum::<f64>() / n as f64).collect();
    for i in 0..n {
        let mean = diagonal[i] / n as f64
            + (0..rank - 1).map(|c| mean_left[c] * right[i * rank + c]).sum::<f64>();
        left[i * rank + rank - 1] = -1.0;
        right[i * rank + rank - 1] = mean;
    }
    let (working_mean, _, noise_variance) = gaussian_working_outcomes(x, p, treatment, outcome)?;
    let point_coefficients = (0..n).map(|i| right[i * rank + 2 * p]).collect();
    Ok(BalancedLinearization {
        operator: LowRankInfluence { diagonal, left, right, rank },
        point_coefficients,
        working_mean,
        noise_variance,
    })
}

/// Unweighted arm OLS is an orthogonal projection: its residual RSS/(n-2p) estimates the
/// homoskedastic working noise variance without the nonsymmetric weighted-smoother bias.
fn gaussian_working_outcomes(
    x: &[f64],
    p: usize,
    treatment: &[f64],
    outcome: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, f64), EstimationError> {
    let n = treatment.len();
    let mut working_mean = vec![0.0; n];
    let mut contrast_mean = vec![0.0; n];
    let mut rss = 0.0;
    for active in [false, true] {
        let mut gram = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for i in (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(treatment[i])) == active) {
            for c in 0..p {
                rhs[c] += x[c * n + i] * outcome[i];
                for d in 0..p {
                    gram[c * p + d] += x[c * n + i] * x[d * n + i];
                }
            }
        }
        let coefficients = crate::util::solve_spd(&gram, &rhs, p)
            .ok_or_else(|| EstimationError::stats_msg("singular Gaussian working outcome model"))?;
        let sign = if active { 1.0 } else { -1.0 };
        for i in 0..n {
            contrast_mean[i] += sign * (0..p).map(|c| coefficients[c] * x[c * n + i]).sum::<f64>();
        }
        for i in (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(treatment[i])) == active) {
            working_mean[i] = (0..p).map(|c| coefficients[c] * x[c * n + i]).sum();
            rss += (outcome[i] - working_mean[i]).powi(2);
        }
    }
    if n <= 2 * p {
        return Err(EstimationError::data_msg(
            "calibrated Gaussian reference requires residual degrees of freedom",
        ));
    }
    Ok((working_mean, contrast_mean, rss / (n - 2 * p) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::IpwTarget;
    use crate::propensity::ipw_oracle::oracle_fit;
    use crate::propensity::weighting::hajek_difference;

    #[test]
    fn calibrated_variance_operator_and_noncentral_df_match_dense_noise_basis() {
        let n = 20;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 - 10.0) / 5.0).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(i % 3 == 0)).collect();
        let y: Vec<f64> =
            (0..n).map(|i| z[i] + t[i] * (1.0 + 0.5 * z[i]) + (i as f64).sin()).collect();
        let mass: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&z);
        let (_, e) = oracle_fit(&t, &y, &z, &mass, IpwTarget::Ate, None, None);
        let mut weights: Vec<f64> =
            (0..n).map(|i| mass[i] / if TreatmentArm::is_treated_indicator(t[i]) { e[i] } else { 1.0 - e[i] }).collect();
        balance_arm_weights(&design, 2, &t, Some(&mass), &mut weights).unwrap();
        let point = hajek_difference(&TreatmentArm::classify(&t), &y, &weights).unwrap();
        let (psi, linear) = balanced_influence_values(
            &design,
            2,
            &t,
            &y,
            Some(&mass),
            &e,
            None,
            &weights,
            point,
        )
        .unwrap();
        let mean = psi.iter().sum::<f64>() / n as f64;
        let mapped = linear.operator.apply(&y);
        for i in 0..n {
            assert!((mapped[i] - (psi[i] - mean)).abs() < 1e-9);
        }
        let mut dense = vec![0.0; n * n];
        for column in 0..n {
            let mut outcome = vec![0.0; n];
            outcome[column] = 1.0;
            let point = hajek_difference(&TreatmentArm::classify(&t), &outcome, &weights).unwrap();
            assert!((linear.point_coefficients[column] - point).abs() < 1e-10);
            let (values, _) = balanced_influence_values(
                &design,
                2,
                &t,
                &outcome,
                Some(&mass),
                &e,
                None,
                &weights,
                point,
            )
            .unwrap();
            let mean = values.iter().sum::<f64>() / n as f64;
            for row in 0..n {
                dense[row * n + column] = values[row] - mean;
            }
        }
        let q: Vec<f64> = (0..n * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                (0..n).map(|row| dense[row * n + i] * dense[row * n + j]).sum()
            })
            .collect();
        let trace = (0..n).map(|i| q[i * n + i]).sum::<f64>();
        let squared = q.iter().map(|x| x * x).sum::<f64>();
        let actual = linear.operator.quadratic_traces().unwrap();
        assert!((actual.0 - trace).abs() < 1e-8 * trace);
        assert!((actual.1 - squared).abs() < 1e-8 * squared);
        let qmean: Vec<f64> =
            (0..n).map(|i| (0..n).map(|j| q[i * n + j] * linear.working_mean[j]).sum()).collect();
        let noncentral = linear.working_mean.iter().zip(&qmean).map(|(a, b)| a * b).sum::<f64>();
        let expected = (linear.noise_variance * trace + noncentral).powi(2)
            / (linear.noise_variance.powi(2) * squared
                + 2.0 * linear.noise_variance * qmean.iter().map(|x| x * x).sum::<f64>());
        assert!((linear.reference_df().unwrap() - expected).abs() < 1e-8 * expected);
    }
}
