//! Bayesian target-native checks: prior sensitivity and posterior-predictive checks on the
//! target quantity itself.
//!
//! * [`prior_sensitivity_target_check`] re-reads the posterior of the **whole target vector**
//!   under alternative priors. The shift of each alternative's posterior mean from the
//!   baseline posterior mean is measured as the Mahalanobis length under the baseline
//!   posterior covariance, `sqrt((m_a − m_0)ᵀ Σ_0⁻¹ (m_a − m_0))`: a shift in posterior
//!   standard deviations that accounts for the correlation between coordinates. The check
//!   passes when the largest shift is at most the declared tolerance.
//! * [`posterior_predictive_target_check`] compares the target computed on the observed data
//!   with the target recomputed on data replicated from each posterior draw. For draw `s`
//!   with target value `θ_s`, the discrepancy is `D(T, θ_s) = (T − θ_s)ᵀ V⁻¹ (T − θ_s)`, `V`
//!   the covariance of `T_rep − θ` across draws; the posterior-predictive p-value is the
//!   share of draws with `D(T_rep,s, θ_s) ≥ D(T_obs, θ_s)` (Gelman, Meng and Stern 1996). A
//!   small value says the model does not reproduce the target it reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::joint::{JointStatistic, JointStatisticKind, mahalanobis_squared, mean_and_covariance};
use super::{RefutationTarget, TargetRefutationReport, TargetRefuterKind};
use crate::ValidationError;

/// Inputs of [`prior_sensitivity_target_check`].
#[derive(Clone, Debug, PartialEq)]
pub struct PriorSensitivityTarget<'a> {
    /// Target the draws are of.
    pub target: &'a RefutationTarget,
    /// Baseline posterior draws of the target (`s × k`, row-major).
    pub baseline: &'a [f64],
    /// Posterior draws of the target under each alternative prior, with the alternative's
    /// label (each `s_a × k`).
    pub alternatives: &'a [(Arc<str>, Vec<f64>)],
    /// Largest acceptable shift, in baseline posterior standard deviations.
    pub tolerance: f64,
}

/// Prior sensitivity of the whole target vector.
///
/// # Errors
///
/// No alternative, too few baseline draws for a covariance, mismatched dimensions, a
/// non-positive tolerance, or non-finite draws.
pub fn prior_sensitivity_target_check(
    input: &PriorSensitivityTarget<'_>,
) -> Result<TargetRefutationReport, ValidationError> {
    let k = input.target.dimension();
    if input.alternatives.is_empty() || !(input.tolerance.is_finite() && input.tolerance > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "prior sensitivity needs at least one alternative prior and a positive \
                      tolerance",
        });
    }
    let (base_mean, base_cov) = draw_moments(input.baseline, k)?;
    let mut worst = 0.0_f64;
    let mut worst_label: Option<Arc<str>> = None;
    let mut worst_mean = base_mean.clone();
    for (label, draws) in input.alternatives {
        let (mean, _) = draw_moments(draws, k)?;
        let diff: Vec<f64> = mean.iter().zip(&base_mean).map(|(a, b)| a - b).collect();
        let mut cov = base_cov.clone();
        let shift = mahalanobis_squared(&diff, &mut cov, k)?.sqrt();
        if shift > worst || worst_label.is_none() {
            worst = shift;
            worst_label = Some(Arc::clone(label));
            worst_mean = mean;
        }
    }
    let passed = worst <= input.tolerance;
    Ok(TargetRefutationReport {
        target: input.target.clone(),
        kind: TargetRefuterKind::PriorSensitivity,
        original: base_mean.into(),
        refuted: worst_mean.into(),
        statistic: JointStatistic {
            kind: JointStatisticKind::PosteriorShift,
            value: worst,
            df: k,
            p_value: None,
        },
        passed,
        informative: true,
        replicates: u32::try_from(input.alternatives.len()).unwrap_or(u32::MAX),
        failure: (!passed).then(|| {
            Arc::from(format!(
                "under prior `{}` the posterior of the target moves by {worst:.3} posterior \
                 standard deviations (tolerance {})",
                worst_label.as_deref().unwrap_or("?"),
                input.tolerance
            ))
        }),
    })
}

/// Inputs of [`posterior_predictive_target_check`].
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorPredictiveTarget<'a> {
    /// Target.
    pub target: &'a RefutationTarget,
    /// Target recomputed on the observed data by the route's point procedure (`k`).
    pub observed: &'a [f64],
    /// Posterior draws of the target (`s × k`).
    pub draws: &'a [f64],
    /// For each draw, the target recomputed on data replicated from that draw (`s × k`).
    pub replicated: &'a [f64],
    /// Fail when the posterior-predictive p-value is below this.
    pub alpha: f64,
}

/// Posterior-predictive check of the target quantity.
///
/// # Errors
///
/// Mismatched dimensions, too few draws, a degenerate replicate covariance, or `alpha`
/// outside `(0, 1)`.
pub fn posterior_predictive_target_check(
    input: &PosteriorPredictiveTarget<'_>,
) -> Result<TargetRefutationReport, ValidationError> {
    let k = input.target.dimension();
    if input.observed.len() != k
        || input.draws.len() != input.replicated.len()
        || input.draws.len() % k != 0
        || !(input.alpha > 0.0 && input.alpha < 1.0)
    {
        return Err(ValidationError::NotApplicable {
            message: "posterior-predictive target check: draws, replicates and the observed \
                      target must share the target dimension, and alpha must lie in (0, 1)",
        });
    }
    let s = input.draws.len() / k;
    if s < k + 2 {
        return Err(ValidationError::NotApplicable {
            message: "posterior-predictive target check needs at least target dimension + 2 \
                      draws",
        });
    }
    let residuals: Vec<f64> =
        input.replicated.iter().zip(input.draws).map(|(rep, theta)| rep - theta).collect();
    if residuals.iter().any(|v| !v.is_finite()) || input.observed.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::estimation_msg("posterior-predictive target is not finite"));
    }
    let (_, cov) = mean_and_covariance(&residuals, k);
    let mut exceed = 0_usize;
    let mut observed_sum = 0.0;
    for (theta, rep) in input.draws.chunks_exact(k).zip(residuals.chunks_exact(k)) {
        let obs: Vec<f64> = input.observed.iter().zip(theta).map(|(t, th)| t - th).collect();
        let d_obs = mahalanobis_squared(&obs, &mut cov.clone(), k)?;
        let d_rep = mahalanobis_squared(rep, &mut cov.clone(), k)?;
        observed_sum += d_obs;
        if d_rep >= d_obs {
            exceed += 1;
        }
    }
    let p = exceed as f64 / s as f64;
    let passed = p >= input.alpha;
    let mut posterior_mean = vec![0.0; k];
    for row in input.draws.chunks_exact(k) {
        for (m, v) in posterior_mean.iter_mut().zip(row) {
            *m += v / s as f64;
        }
    }
    Ok(TargetRefutationReport {
        target: input.target.clone(),
        kind: TargetRefuterKind::PosteriorPredictive,
        original: posterior_mean.into(),
        refuted: input.observed.to_vec().into(),
        statistic: JointStatistic {
            kind: JointStatisticKind::PredictiveTail,
            value: observed_sum / s as f64,
            df: k,
            p_value: Some(p),
        },
        passed,
        informative: true,
        replicates: u32::try_from(s).unwrap_or(u32::MAX),
        failure: (!passed).then(|| {
            Arc::from(format!(
                "data replicated from the posterior reproduce a target as far from the draws as \
                 the observed one in only {p:.3} of draws (alpha {})",
                input.alpha
            ))
        }),
    })
}

fn draw_moments(draws: &[f64], k: usize) -> Result<(Vec<f64>, Vec<f64>), ValidationError> {
    if k == 0 || draws.len() % k != 0 || draws.len() / k < k + 2 {
        return Err(ValidationError::NotApplicable {
            message: "posterior draws do not match the target dimension, or are too few for a \
                      covariance (at least dimension + 2)",
        });
    }
    if draws.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::estimation_msg(
            "a posterior draw of the target is not finite",
        ));
    }
    Ok(mean_and_covariance(draws, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target_native::{TargetFamily, TargetQuadrant};

    fn target(k: usize) -> RefutationTarget {
        let labels: Vec<String> = (0..k).map(|i| format!("c{i}")).collect();
        RefutationTarget::new(
            TargetFamily::WorldValued,
            TargetQuadrant::STATIC_BAYESIAN,
            labels,
            [7; 32],
        )
        .unwrap()
    }

    fn grid_draws(center: [f64; 2]) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..40 {
            let u = f64::from(i % 8) / 4.0 - 0.875;
            let v = f64::from(i / 8) / 2.0 - 1.0;
            out.push(center[0] + u);
            out.push(center[1] + v);
        }
        out
    }

    #[test]
    fn a_small_prior_shift_passes_and_a_large_one_fails() {
        let t = target(2);
        let base = grid_draws([0.0, 0.0]);
        let near: Vec<(Arc<str>, Vec<f64>)> = vec![(Arc::from("wide"), grid_draws([0.05, 0.0]))];
        let report = prior_sensitivity_target_check(&PriorSensitivityTarget {
            target: &t,
            baseline: &base,
            alternatives: &near,
            tolerance: 0.5,
        })
        .unwrap();
        assert!(report.passed);
        let far: Vec<(Arc<str>, Vec<f64>)> = vec![(Arc::from("tight"), grid_draws([3.0, 0.0]))];
        let report = prior_sensitivity_target_check(&PriorSensitivityTarget {
            target: &t,
            baseline: &base,
            alternatives: &far,
            tolerance: 0.5,
        })
        .unwrap();
        assert!(!report.passed);
    }

    #[test]
    fn an_observed_target_far_outside_the_replicates_fails() {
        let t = target(1);
        let draws: Vec<f64> = (0..50).map(|i| f64::from(i % 5) * 0.01).collect();
        let replicated: Vec<f64> =
            draws.iter().enumerate().map(|(i, d)| d + (i as f64 / 49.0 - 0.5)).collect();
        let ok = posterior_predictive_target_check(&PosteriorPredictiveTarget {
            target: &t,
            observed: &[0.02],
            draws: &draws,
            replicated: &replicated,
            alpha: 0.05,
        })
        .unwrap();
        assert!(ok.passed);
        let bad = posterior_predictive_target_check(&PosteriorPredictiveTarget {
            target: &t,
            observed: &[10.0],
            draws: &draws,
            replicated: &replicated,
            alpha: 0.05,
        })
        .unwrap();
        assert!(!bad.passed);
        assert_eq!(bad.statistic.p_value, Some(0.0));
    }
}
