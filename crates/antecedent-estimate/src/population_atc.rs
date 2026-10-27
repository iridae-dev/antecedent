//! Arm-linear heteroskedastic inference for the AIPW control-population target.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::arm::TreatmentArm;
use crate::propensity::PreparedPropensityProblem;
use crate::{EffectEstimate, EstimationError};

/// A control-population interval using the complete outcome coefficient and arm-wise HC3.
///
/// This explicit model construction assumes independent rows and arm-linear conditional
/// means. Outcome variances may differ across rows. The complete coefficient accounts for
/// reusing the treated arm to train the counterfactual mean; arm-wise unweighted OLS HC3
/// residuals estimate its conditional noise variance. A separate weighted control-target
/// heterogeneity term remains in the asymptotic normal reference.
#[derive(Clone, Debug)]
pub struct PopulationAtcInterval {
    /// Unchanged AIPW ATC point, with this construction's analytic standard error.
    pub estimate: EffectEstimate,
    /// Lower confidence endpoint.
    pub lower: f64,
    /// Upper confidence endpoint.
    pub upper: f64,
    /// Confidence level.
    pub level: f64,
    /// Complete-coefficient heteroskedastic outcome-noise variance.
    pub noise_variance: f64,
    /// Weighted control-target heterogeneity variance.
    pub target_variance: f64,
}

pub(crate) fn coefficient(
    problem: &PreparedPropensityProblem,
    propensity: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let n = problem.nrows;
    let p = problem.design_ncols;
    let x = &problem.design_matrix;
    if propensity.len() != n
        || weights.len() != n
        || propensity.iter().any(|e| !e.is_finite() || *e <= 0.0 || *e >= 1.0)
    {
        return Err(EstimationError::data_msg("invalid ATC linearization inputs"));
    }
    let controls = weights
        .iter()
        .zip(problem.treatment.iter())
        .filter(|(_, t)| !TreatmentArm::is_treated_indicator(**t))
        .map(|(w, _)| w)
        .sum::<f64>();
    if controls <= 0.0 {
        return Err(EstimationError::data_msg("ATC requires controls"));
    }
    let mut h = Vec::with_capacity(n);
    let mut gram = vec![0.0; p * p];
    let mut load = vec![0.0; p];
    for i in 0..n {
        let treated = TreatmentArm::is_treated_indicator(problem.treatment[i]);
        let r = if treated { (1.0 - propensity[i]) / propensity[i] } else { -1.0 };
        h.push(weights[i] / controls * r);
        for c in 0..p {
            load[c] -= weights[i] / controls * r * x[c * n + i];
            if treated {
                for d in 0..p {
                    gram[c * p + d] += weights[i] * x[c * n + i] * x[d * n + i];
                }
            }
        }
    }
    let beta = crate::util::solve_spd(&gram, &load, p)
        .ok_or_else(|| EstimationError::stats_msg("singular ATC treated training design"))?;
    for i in (0..n).filter(|&i| TreatmentArm::is_treated_indicator(problem.treatment[i])) {
        h[i] += weights[i] * (0..p).map(|c| x[c * n + i] * beta[c]).sum::<f64>();
    }
    Ok(h)
}

pub(crate) fn interval(
    problem: &PreparedPropensityProblem,
    mut estimate: EffectEstimate,
    propensity: &[f64],
    level: f64,
) -> Result<PopulationAtcInterval, EstimationError> {
    if !matches!(problem.target_population, antecedent_core::TargetPopulation::Untreated)
        || crate::propensity::trim_of(problem.overlap).is_some()
        || problem.fold_units.is_some()
        || !level.is_finite()
        || level <= 0.0
        || level >= 1.0
    {
        return Err(EstimationError::unsupported(
            "arm-linear ATC requires independent untrimmed control target and valid level",
        ));
    }
    let n = problem.nrows;
    let weights =
        problem.sampling_weights.as_ref().map_or_else(|| vec![1.0; n], |s| s.weights.to_vec());
    let h = coefficient(problem, propensity, &weights)?;
    let point = h.iter().zip(problem.outcome.iter()).map(|(a, b)| a * b).sum::<f64>();
    if (point - estimate.ate).abs() > 1e-7 * (1.0 + estimate.ate.abs()) {
        return Err(EstimationError::stats_msg("ATC linear coefficient does not reproduce point"));
    }
    let (contrast, noise_variance) = arm_residual_variance(problem, &h)?;
    let mass =
        weights.iter().zip(problem.treatment.iter()).map(|(w, t)| w * (1.0 - t)).sum::<f64>();
    let target = weights
        .iter()
        .zip(problem.treatment.iter())
        .zip(&contrast)
        .map(|((w, t), v)| w * (1.0 - t) * v)
        .sum::<f64>()
        / mass;
    let target_variance = weights
        .iter()
        .zip(problem.treatment.iter())
        .zip(&contrast)
        .map(|((w, t), v)| (w * (1.0 - t) / mass * (v - target)).powi(2))
        .sum::<f64>()
        * n as f64
        / (n - 1) as f64;
    let variance = noise_variance + target_variance;
    if !variance.is_finite() {
        return Err(EstimationError::stats_msg("invalid ATC model variance"));
    }
    let radius = antecedent_stats::normal_ppf((1.0 + level) / 2.0) * variance.sqrt();
    estimate.uncertainty.analytic.se = variance.sqrt();
    estimate.uncertainty.analytic.kind = Some(crate::AnalyticSeKind::Hc3);
    estimate.uncertainty.analytic.reference_df = None;
    estimate.scores = crate::EstimateScores::default();
    estimate.joint.covariance = None;
    Ok(PopulationAtcInterval {
        lower: estimate.ate - radius,
        upper: estimate.ate + radius,
        estimate,
        level,
        noise_variance,
        target_variance,
    })
}

fn arm_residual_variance(
    problem: &PreparedPropensityProblem,
    h: &[f64],
) -> Result<(Vec<f64>, f64), EstimationError> {
    let n = problem.nrows;
    let p = problem.design_ncols;
    let x = &problem.design_matrix;
    let mut contrast = vec![0.0; n];
    let mut noise_variance = 0.0;
    for active in [false, true] {
        let mut gram = vec![0.0; p * p];
        let mut rhs = vec![0.0; p];
        for i in
            (0..n).filter(|&i| (TreatmentArm::is_treated_indicator(problem.treatment[i])) == active)
        {
            for c in 0..p {
                rhs[c] += x[c * n + i] * problem.outcome[i];
                for d in 0..p {
                    gram[c * p + d] += x[c * n + i] * x[d * n + i];
                }
            }
        }
        let beta = crate::util::solve_spd(&gram, &rhs, p)
            .ok_or_else(|| EstimationError::stats_msg("singular ATC residual arm design"))?;
        let mut inverse = vec![0.0; p * p];
        for c in 0..p {
            let mut unit = vec![0.0; p];
            unit[c] = 1.0;
            let column = crate::util::solve_spd(&gram, &unit, p)
                .ok_or_else(|| EstimationError::stats_msg("singular ATC leverage design"))?;
            for d in 0..p {
                inverse[d * p + c] = column[d];
            }
        }
        for i in 0..n {
            let fitted = (0..p).map(|c| x[c * n + i] * beta[c]).sum::<f64>();
            contrast[i] += if active { fitted } else { -fitted };
            if (TreatmentArm::is_treated_indicator(problem.treatment[i])) == active {
                let mut leverage = 0.0;
                for c in 0..p {
                    for d in 0..p {
                        leverage += x[c * n + i] * inverse[c * p + d] * x[d * n + i];
                    }
                }
                if leverage >= 1.0 {
                    return Err(EstimationError::stats_msg("ATC arm has unit leverage"));
                }
                noise_variance += (h[i] * (problem.outcome[i] - fitted) / (1.0 - leverage)).powi(2);
            }
        }
    }
    Ok((contrast, noise_variance))
}
