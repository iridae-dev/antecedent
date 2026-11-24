//! A Bayesian reading of the weighted design of a marginal structural model, and the posterior
//! sensitivity ranges of its period parameters.
//!
//! The longitudinal regime route fits `outcome ~ 1 + A_1 + ... + A_T` by weighted least squares
//! with stabilized sequential weights `w` (see the `msm_parameters` module); its result is a
//! point. [`msm_parameter_posterior`] reads the same design, on the same observed subjects and the
//! same weights, as a weighted (power) likelihood: with the weights normalized to sum to the
//! effective sample size `n_eff = (sum w)^2 / sum w^2`, `w~ = w n_eff / sum w`, the outcome model is
//! `prod_i N(y_i; x_i beta, sigma^2)^{w~_i}` under the reference prior `p(beta, sigma^2) ∝ 1 /
//! sigma^2`. The posterior is conjugate:
//!
//! * `sigma^2 = RSS~ / chi2_{n_eff - p}`, `RSS~ = sum w~ (y - x beta_hat)^2`;
//! * `beta | sigma^2 ~ N(beta_hat, sigma^2 (X^T W~ X)^-1)`, `beta_hat` the weighted least-squares
//!   fit the route reports;
//!
//! so a parameter's posterior is `beta_hat_t + sqrt(RSS~ / (n_eff - p)) [(X^T W~ X)^-1]_tt^(1/2)`
//! times a Student-t on `n_eff - p` degrees of freedom. `sigma` is the residual standard
//! deviation of the weighted pseudo-population, the scale `sqrt(sum w r^2 / sum w)` the
//! frequentist E-value of the same parameters standardizes by. Draws are taken from the keyed
//! stream [`MSM_POSTERIOR_STREAM`] of the execution context's seed: `chi2` first, then `p`
//! standard normals, per draw.
//!
//! * [`SensitivityRangeRequest::MsmParameterEValuePosterior`]: each parameter draw over the
//!   `sigma` of the same draw, read by [`component_evalue_posterior_range`] on the
//!   standardized-mean-difference scale under the marginal structural model;
//! * [`SensitivityRangeRequest::MsmParametersOvbPosterior`]: the parameter draws with bias scale
//!   `sigma / sd(A_t | other periods)`, `sd(A_t | other periods)^2 = RSS~(A_t ~ 1 + A_-t) /
//!   (n_eff - p) = 1 / ([(X^T W~ X)^-1]_tt (n_eff - p))`, read by
//!   [`component_ovb_posterior_range`] (Cinelli and Hazlett 2020).
//!
//! The weights are held fixed: the posterior does not carry the cost of estimating them.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_kernels::{design_gram_cholesky, sample_gamma, standard_normal};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::target_native::RefutationTarget;
use antecedent_validate::{
    ComponentModel, EffectScale, OvbPosterior, PosteriorEValue, ValidationError,
    component_evalue_posterior_range, component_ovb_posterior_range,
};

use super::msm_parameters::MsmDesign;
use super::{SensitivityRangeOutcome, SensitivityRangeRequest};
use crate::error::CausalError;

/// Keyed stream of the marginal structural model posterior.
pub const MSM_POSTERIOR_STREAM: u64 = 0x5345_4E53_4D53_4D31;
/// Most posterior draws one analysis takes.
const MAX_MSM_POSTERIOR_DRAWS: usize = 100_000;

/// Draws of the posterior of a marginal structural model's weighted design.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MsmParameterPosterior {
    /// `coefficients[t][d]`: draw `d` of the parameter of period `t`.
    pub coefficients: Vec<Vec<f64>>,
    /// Draw `d` of the residual standard deviation of the weighted pseudo-population.
    pub sigma: Vec<f64>,
    /// The weighted least-squares fit of every parameter (the route's point estimates).
    pub estimates: Vec<f64>,
    /// Per period, `sd(A_t | other periods)` on the weighted design's residual degrees of
    /// freedom, the scale of the parameter's omitted-variable bias.
    pub exposure_scale: Vec<f64>,
    /// Residual degrees of freedom `n_eff - p` of the posterior of `sigma^2`.
    pub dof: f64,
    /// Weighted residual sum of squares `RSS~` under the normalized weights.
    pub rss: f64,
    /// The stabilized weight of each observed subject, in row order.
    pub weights: Vec<f64>,
}

impl MsmParameterPosterior {
    /// Each parameter draw over the `sigma` of the same draw.
    #[must_use]
    pub fn standardized(&self) -> Vec<Vec<f64>> {
        self.coefficients
            .iter()
            .map(|column| column.iter().zip(&self.sigma).map(|(b, s)| b / s).collect())
            .collect()
    }

    /// Per period and draw, `sigma / sd(A_t | other periods)`.
    #[must_use]
    pub fn bias_scale(&self) -> Vec<Vec<f64>> {
        self.exposure_scale
            .iter()
            .map(|scale| self.sigma.iter().map(|s| s / scale).collect())
            .collect()
    }
}

/// The posterior of `design` (see the module documentation), `draws` draws.
///
/// # Errors
/// A draw count outside `[2, 100000]`, no positively weighted subject, fewer than two effective
/// residual degrees of freedom, a rank-deficient weighted design, and cancellation.
#[allow(clippy::cast_precision_loss, reason = "the row and draw counts are far below 2^53")]
pub(super) fn msm_parameter_posterior(
    design: &MsmDesign,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<MsmParameterPosterior, CausalError> {
    if !(2..=MAX_MSM_POSTERIOR_DRAWS).contains(&draws) {
        return Err(CausalError::Unsupported {
            message: "the marginal structural model posterior takes between 2 and 100000 draws",
        });
    }
    let rows: Vec<usize> = (0..design.weights.len()).filter(|&i| design.weights[i] > 0.0).collect();
    let n = rows.len();
    let periods = design.exposures.len();
    let p = 1 + periods;
    let total: f64 = rows.iter().map(|&i| design.weights[i]).sum();
    let total_squares: f64 = rows.iter().map(|&i| design.weights[i] * design.weights[i]).sum();
    let effective = total * total / total_squares;
    let dof = effective - p as f64;
    if n <= p || !(dof.is_finite() && dof >= 2.0) {
        return Err(CausalError::Unsupported {
            message: "the effective sample size of the weights leaves fewer than two residual \
                      degrees of freedom for the marginal structural model posterior",
        });
    }
    let root: Vec<f64> =
        rows.iter().map(|&i| (design.weights[i] * effective / total).sqrt()).collect();
    let mut matrix = root.clone();
    for column in &design.exposures {
        matrix.extend(rows.iter().zip(&root).map(|(&i, r)| column[i] * r));
    }
    let scaled: Vec<f64> = rows.iter().zip(&root).map(|(&i, r)| design.outcome[i] * r).collect();
    let fit = FaerBackend
        .least_squares(&matrix, n, p, &scaled, &mut LeastSquaresWorkspace::default())
        .map_err(ValidationError::from)?;
    let l = design_gram_cholesky(&matrix, n, p).filter(|_| fit.rank == p).ok_or(
        CausalError::Unsupported {
            message: "the weighted marginal structural model design is rank deficient",
        },
    )?;
    if !(fit.rss.is_finite() && fit.rss > 0.0) {
        return Err(CausalError::Unsupported {
            message: "the weighted marginal structural model leaves no residual variation",
        });
    }
    // [(X^T W~ X)^-1]_jj = |L^-1 e_j|^2, by forward substitution.
    let inverse_diagonal = |j: usize| -> f64 {
        let mut y = vec![0.0; p];
        for i in j..p {
            let mut acc = if i == j { 1.0 } else { 0.0 };
            for q in j..i {
                acc -= l[i * p + q] * y[q];
            }
            y[i] = acc / l[i * p + i];
        }
        y.iter().map(|v| v * v).sum()
    };
    let exposure_scale: Vec<f64> =
        (1..p).map(|j| (1.0 / (inverse_diagonal(j) * dof)).sqrt()).collect();
    let mut rng = ctx.rng.stream(MSM_POSTERIOR_STREAM);
    let mut posterior = MsmParameterPosterior {
        coefficients: vec![Vec::with_capacity(draws); periods],
        sigma: Vec::with_capacity(draws),
        estimates: fit.coefficients[1..].to_vec(),
        exposure_scale,
        dof,
        rss: fit.rss,
        weights: design.weights.clone(),
    };
    for _ in 0..draws {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "marginal structural model posterior" });
        }
        let chi2 = sample_gamma(dof / 2.0, 0.5, &mut rng);
        let sigma = (fit.rss / chi2.max(f64::MIN_POSITIVE)).sqrt();
        // beta = beta_hat + sigma L^-T z.
        let mut w: Vec<f64> = (0..p).map(|_| standard_normal(&mut rng)).collect();
        for i in (0..p).rev() {
            let mut acc = w[i];
            for q in (i + 1)..p {
                acc -= l[q * p + i] * w[q];
            }
            w[i] = acc / l[i * p + i];
        }
        for (t, column) in posterior.coefficients.iter_mut().enumerate() {
            column.push(fit.coefficients[t + 1] + sigma * w[t + 1]);
        }
        posterior.sigma.push(sigma);
    }
    Ok(posterior)
}

/// The posterior analysis `request` of the parameters on `target`.
pub(super) fn msm_posterior_range(
    design: &MsmDesign,
    target: RefutationTarget,
    request: &SensitivityRangeRequest,
    ctx: &ExecutionContext,
) -> Result<SensitivityRangeOutcome, CausalError> {
    let model = ComponentModel::MarginalStructural;
    match request {
        SensitivityRangeRequest::MsmParameterEValuePosterior { level, draws } => {
            let posterior = msm_parameter_posterior(design, *draws, ctx)?;
            let (summary, range): (PosteriorEValue, _) = component_evalue_posterior_range(
                target,
                &posterior.standardized(),
                EffectScale::StandardizedMeanDifference,
                *level,
                model,
            )?;
            Ok(SensitivityRangeOutcome::MsmParameterPosterior {
                posterior,
                evalue: Some(summary),
                range,
            })
        }
        SensitivityRangeRequest::MsmParametersOvbPosterior { level, r2yz_extreme, draws } => {
            let posterior = msm_parameter_posterior(design, *draws, ctx)?;
            let range = component_ovb_posterior_range(
                target,
                &OvbPosterior {
                    coefficients: &posterior.coefficients,
                    bias_scale: &posterior.bias_scale(),
                },
                *level,
                *r2yz_extreme,
                model,
            )?;
            Ok(SensitivityRangeOutcome::MsmParameterPosterior { posterior, evalue: None, range })
        }
        _ => Err(CausalError::Unsupported {
            message: "not a marginal structural model posterior analysis",
        }),
    }
}
