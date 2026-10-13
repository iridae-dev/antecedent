//! Linear outcome model over an externally supplied latent-variable posterior.
//!
//! The provider's posterior is consumed as joint draws. Two combinations are
//! distinct and chosen by the model's declaration, never by the data:
//!
//! * the posterior **does not** condition on the outcome: the regressor is
//!   replaced by the per-unit posterior mean. For a linear outcome model this
//!   is exact (regression calibration); regressing on a *draw* instead adds
//!   Berkson noise and attenuates the slope. The provider's own parameter
//!   uncertainty is not representable from per-unit means, so the record says
//!   [`ParameterUncertainty::NotPropagated`];
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
            "measurement_model.invalid_model",
            "posterior route needs external-posterior parameters",
        ));
    };
    if data.draw_count < *min_draws as usize {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.insufficient_draws",
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
    let means =
        posterior_means(data.draws, n, data.draw_count).map_err(|e| stats_error(&e))?;
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
