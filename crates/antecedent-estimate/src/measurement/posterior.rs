//! Linear outcome model over an externally supplied latent-variable posterior.
//!
//! The provider's posterior is consumed as joint draws. Two combinations are
//! distinct and chosen by the model's declaration, never by the data:
//!
//! * the posterior **does not** condition on the outcome: the regressor is
//!   replaced by the per-unit posterior mean. For a linear outcome model this
//!   is exact (regression calibration); regressing on a *draw* instead adds
//!   Berkson noise and attenuates the slope. A mean of `S` draws still carries
//!   the Monte Carlo error `v / S`, which attenuates the slope by
//!   `1 - (v / S) / Var(mean | covariates)`; the mean is therefore replaced by
//!   its best linear predictor given the covariates (`calibrated_means`), on
//!   which least squares is the moment-corrected slope (Fuller 1987, section
//!   1.2; Carroll, Ruppert, Stefanski and Crainiceanu 2006, chapter 4). The
//!   provider's own parameter uncertainty is not representable from per-unit
//!   means, so the record says [`ParameterUncertainty::NotPropagated`];
//! * the posterior **does** condition on the outcome: each joint draw is a
//!   proper imputation, one linear fit per draw, combined with Rubin's rules;
//!   the between-imputation variance carries the provider's uncertainty.
//!
//! The caller must first have passed the provider object through
//! `antecedent_contract::check_latent_posterior`; the model's posterior
//! reference binds the object identity and this module has no trust to grant.

use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, MeasurementModel, MeasurementParameters, MeasurementRoute,
    ParameterUncertainty,
};
use antecedent_stats::measurement_error::{
    CombinedImputations, ErrorVariance, LinearErrorData, LinearMoments, combine_imputations_linear,
    posterior_means,
};

use super::{
    DataBinding, MeasurementError, RecordParts, build_record, check_binding, check_cancelled,
    component, digest, require_non_differential, require_route, stats_error,
};

/// Outcome, covariates and joint posterior draws of the latent regressor.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorLinearData<'a> {
    /// Outcome.
    pub outcome: &'a [f64],
    /// Joint posterior draws, draw-major: draw `s` is `draws[s*n..(s+1)*n]`.
    pub draws: &'a [f64],
    /// Number of joint draws.
    pub draw_count: usize,
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: &'a [f64],
    /// Number of covariate columns.
    pub covariate_count: usize,
}

/// How the draws were combined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PosteriorCombination {
    /// Per-unit posterior mean substituted for the latent regressor.
    PosteriorMeanSubstitution,
    /// One fit per outcome-conditioned draw, combined by Rubin's rules.
    MultipleImputation,
}

/// A linear effect over a latent posterior, bound to the model.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedPosteriorLinearEffect {
    /// Which combination was applied.
    pub combination: PosteriorCombination,
    /// Coefficients, intercept first and the latent regressor second.
    pub coefficients: Vec<f64>,
    /// Coefficient covariance, row-major.
    pub covariance: Vec<f64>,
    /// Sufficient moments of the substitution fit.
    pub moments: Option<LinearMoments>,
    /// Per-imputation fits and their combination.
    pub imputations: Option<CombinedImputations>,
    /// The latent regressor's coefficient, bound to the model identity.
    pub record: CorrectedFunctionalRecord,
}

/// Consume a latent posterior under the model's declared conditioning.
///
/// # Errors
/// A malformed model, binding, route or structure mismatch, fewer draws than
/// the model's minimum, a shape mismatch, a rank-deficient fit, or
/// cancellation refuse.
#[allow(
    clippy::too_many_lines,
    reason = "one flat sequence of independent refusal checks followed by the single corrected-effect assembly"
)]
pub fn correct_latent_posterior_linear_effect(
    model: &MeasurementModel,
    binding: &DataBinding,
    data: &PosteriorLinearData<'_>,
    ctx: &ExecutionContext,
) -> Result<CorrectedPosteriorLinearEffect, MeasurementError> {
    check_cancelled(ctx)?;
    check_binding(model, binding)?;
    require_route(model, MeasurementRoute::ExternalLatentPosterior)?;
    require_non_differential(model)?;
    let MeasurementParameters::ExternalPosterior { conditioned_on_outcome, min_draws, .. } =
        &model.parameters
    else {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            antecedent_core::reason_details::measurement_model::INVALID_MODEL,
            "posterior route needs external-posterior parameters",
        ));
    };
    if data.draw_count < *min_draws as usize {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            antecedent_core::reason_details::measurement_model::INSUFFICIENT_DRAWS,
            format!(
                "{} joint draws supplied; the model declares at least {min_draws}",
                data.draw_count
            ),
        ));
    }
    let n = data.outcome.len();
    let identity = digest(&[
        ("outcome", data.outcome),
        ("draws", data.draws),
        ("covariates", data.covariates),
    ]);
    if *conditioned_on_outcome {
        let combined = combine_imputations_linear(
            data.outcome,
            data.draws,
            data.draw_count,
            data.covariates,
            data.covariate_count,
        )
        .map_err(|e| stats_error(&e))?;
        let d = combined.dim;
        let total = combined.covariance[d + 1];
        let record = build_record(RecordParts {
            model,
            functional_id: "linear_coefficient",
            point: combined.coefficients[1],
            naive_point: None,
            variance: Some(total),
            components: vec![
                component("within_imputation", combined.within[d + 1]),
                component(
                    "between_imputation",
                    (1.0 + 1.0 / combined.draws as f64) * combined.between[d + 1],
                ),
            ],
            rows: n as u64,
            data_identity: identity,
            parameter_uncertainty: ParameterUncertainty::Propagated,
        });
        return Ok(CorrectedPosteriorLinearEffect {
            combination: PosteriorCombination::MultipleImputation,
            coefficients: combined.coefficients.clone(),
            covariance: combined.covariance.clone(),
            moments: None,
            imputations: Some(combined),
            record,
        });
    }
    let means = posterior_means(data.draws, n, data.draw_count).map_err(|e| stats_error(&e))?;
    let means = calibrated_means(&means, data)?;
    let exact = ErrorVariance::known(0.0, 0.0).map_err(|e| stats_error(&e))?;
    let raw = LinearErrorData {
        y: data.outcome,
        w: &means,
        covariates: data.covariates,
        covariate_count: data.covariate_count,
        replicate_counts: None,
    };
    let moments = LinearMoments::accumulate(&raw, &exact, 0.0).map_err(|e| stats_error(&e))?;
    let fit = moments.solve().map_err(|e| stats_error(&e))?;
    let d = fit.dim;
    let variance = fit.naive_covariance[d + 1];
    let record = build_record(RecordParts {
        model,
        functional_id: "linear_coefficient",
        point: fit.naive_coefficients[1],
        naive_point: None,
        variance: Some(variance),
        components: vec![component("sampling_given_posterior", variance)],
        rows: n as u64,
        data_identity: identity,
        parameter_uncertainty: ParameterUncertainty::NotPropagated,
    });
    Ok(CorrectedPosteriorLinearEffect {
        combination: PosteriorCombination::PosteriorMeanSubstitution,
        coefficients: fit.naive_coefficients.clone(),
        covariance: fit.naive_covariance.clone(),
        moments: Some(moments),
        imputations: None,
        record,
    })
}

/// The least-squares fit of `means` on an intercept and the covariates, row by row.
fn fitted_on_covariates(
    means: &[f64],
    covariates: &[f64],
    covariate_count: usize,
) -> Result<Vec<f64>, MeasurementError> {
    let n = means.len();
    if covariate_count == 0 {
        let mean = means.iter().sum::<f64>() / n as f64;
        return Ok(vec![mean; n]);
    }
    let (first, rest) = covariates.split_at(n);
    let exact = ErrorVariance::known(0.0, 0.0).map_err(|e| stats_error(&e))?;
    let design = LinearErrorData {
        y: means,
        w: first,
        covariates: rest,
        covariate_count: covariate_count - 1,
        replicate_counts: None,
    };
    let moments = LinearMoments::accumulate(&design, &exact, 0.0).map_err(|e| stats_error(&e))?;
    let fit = moments.solve().map_err(|e| stats_error(&e))?;
    let beta = &fit.naive_coefficients;
    Ok((0..n)
        .map(|i| {
            beta[0]
                + beta[1] * first[i]
                + (0..covariate_count - 1).map(|j| beta[2 + j] * rest[j * n + i]).sum::<f64>()
        })
        .collect())
}

/// The per-unit posterior means with the Monte Carlo error of a finite number of draws
/// calibrated out.
///
/// A mean of `S` draws carries the error variance `v / S`, estimated by the pooled within-unit
/// variance of the draws over `S`. Each mean is replaced by `fitted + lambda (mean - fitted)`,
/// with `fitted` its regression on the covariates and
/// `lambda = 1 - (v / S) / Var(mean | covariates)`: the best linear predictor of the exact
/// posterior mean. The draws of one unit are read as independent draws of its posterior.
fn calibrated_means(
    means: &[f64],
    data: &PosteriorLinearData<'_>,
) -> Result<Vec<f64>, MeasurementError> {
    let n = means.len();
    let draw_count = data.draw_count;
    // One draw has no within-unit variance, and a malformed covariate block is refused by the
    // fit that follows.
    if draw_count < 2 || data.covariate_count.checked_mul(n) != Some(data.covariates.len()) {
        return Ok(means.to_vec());
    }
    let within: f64 = data
        .draws
        .chunks_exact(n)
        .map(|draw| draw.iter().zip(means).map(|(x, m)| (x - m) * (x - m)).sum::<f64>())
        .sum();
    let monte_carlo = within / (n as f64 * (draw_count as f64 - 1.0)) / draw_count as f64;
    if monte_carlo <= 0.0 {
        return Ok(means.to_vec());
    }
    let fitted = fitted_on_covariates(means, data.covariates, data.covariate_count)?;
    let residual_variance =
        means.iter().zip(&fitted).map(|(m, f)| (m - f) * (m - f)).sum::<f64>() / n as f64;
    let lambda = 1.0 - monte_carlo / residual_variance;
    if lambda.is_nan() || lambda <= 0.0 {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            antecedent_core::reason_details::measurement_model::INSUFFICIENT_DRAWS,
            format!(
                "the Monte Carlo error of the posterior means over {draw_count} draws is not \
                 below their spread given the covariates; supply more joint draws"
            ),
        ));
    }
    Ok(means.iter().zip(&fitted).map(|(m, f)| f + lambda * (m - f)).collect())
}
