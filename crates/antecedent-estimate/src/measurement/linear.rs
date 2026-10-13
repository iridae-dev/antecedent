//! Corrected linear outcome model with an error-prone regressor.
//!
//! Classical continuous error and repeated measurements share one numerical
//! correction ([`antecedent_stats::measurement_error::LinearMoments`]) and are
//! distinct routes: they differ in where the error variance comes from.

use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, MeasurementModel, MeasurementParameters, MeasurementRoute,
    ParameterUncertainty,
};
use antecedent_stats::measurement_error::{
    CorrectedLinear, ErrorVariance, LinearErrorData, LinearMoments, VarianceComponents,
};

use super::{
    DataBinding, MeasurementError, RecordParts, build_record, check_binding, check_cancelled,
    component, counts_as_reals, digest, require_non_differential, require_route, stats_error,
};

/// Columns of one linear analysis with an error-prone regressor.
#[derive(Clone, Copy, Debug)]
pub struct LinearMeasuredData<'a> {
    /// Outcome.
    pub outcome: &'a [f64],
    /// Error-prone regressor; the row mean of replicates when replicated.
    pub regressor: &'a [f64],
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: &'a [f64],
    /// Number of covariate columns.
    pub covariate_count: usize,
    /// Replicates averaged into each regressor value; `None` means one each.
    pub replicate_counts: Option<&'a [u32]>,
}

/// A corrected linear effect, its sufficient moments and its bound record.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedLinearEffect {
    /// Corrected coefficients and covariance components.
    pub fit: CorrectedLinear,
    /// Sufficient moments: [`LinearMoments::solve`] on these reproduces `fit`.
    pub moments: LinearMoments,
    /// Within-unit and between-unit variance components (repeated route only).
    pub components: Option<VarianceComponents>,
    /// The regressor's corrected coefficient, bound to the model identity.
    pub record: CorrectedFunctionalRecord,
}

impl CorrectedLinearEffect {
    /// Value and variance of a linear functional of the corrected coefficient
    /// vector (intercept first, regressor second, then the covariates).
    ///
    /// The variance is the quadratic form under the joint covariance that
    /// includes the error-variance estimate's uncertainty.
    ///
    /// # Errors
    /// A weight vector of the wrong width refuses.
    pub fn downstream_linear_functional(
        &self,
        weights: &[f64],
    ) -> Result<(f64, f64), MeasurementError> {
        self.fit.linear_functional(weights).map_err(|e| stats_error(&e))
    }
}

fn finish(
    model: &MeasurementModel,
    data: &LinearMeasuredData<'_>,
    replicate_values: &[f64],
    moments: LinearMoments,
    components: Option<VarianceComponents>,
    uncertainty: ParameterUncertainty,
) -> Result<CorrectedLinearEffect, MeasurementError> {
    let fit = moments.solve().map_err(|e| stats_error(&e))?;
    let d = fit.dim;
    let total = fit.covariance[d + 1];
    let given = fit.covariance_given_error_variance[d + 1];
    let counts = data.replicate_counts.map(counts_as_reals).unwrap_or_default();
    let identity = digest(&[
        ("outcome", data.outcome),
        ("regressor", data.regressor),
        ("covariates", data.covariates),
        ("replicate_counts", &counts),
        ("replicate_values", replicate_values),
    ]);
    let record = build_record(RecordParts {
        model,
        functional_id: "linear_coefficient",
        point: fit.coefficients[1],
        naive_point: Some(fit.naive_coefficients[1]),
        variance: Some(total),
        components: vec![
            component("sampling_given_measurement_parameters", given),
            component("measurement_parameter_uncertainty", total - given),
        ],
        rows: fit.n as u64,
        data_identity: identity,
        parameter_uncertainty: uncertainty,
    });
    Ok(CorrectedLinearEffect { fit, moments, components, record })
}

/// Classical additive error with a declared or validated error variance.
///
/// Corrects the coefficient of the error-prone regressor in a linear outcome
/// model with error-free covariates. When rows average replicates, the model's
/// error variance is the variance of **one** measurement and each row's error
/// variance is that divided by its replicate count.
///
/// # Errors
/// A malformed model, a binding mismatch, a route or structure mismatch, an
/// implied reliability at or below the model's floor, a rank-deficient
/// design, or cancellation refuse; see [`MeasurementError`].
pub fn correct_classical_linear_effect(
    model: &MeasurementModel,
    binding: &DataBinding,
    data: &LinearMeasuredData<'_>,
    ctx: &ExecutionContext,
) -> Result<CorrectedLinearEffect, MeasurementError> {
    check_cancelled(ctx)?;
    check_binding(model, binding)?;
    require_route(model, MeasurementRoute::ClassicalContinuousError)?;
    require_non_differential(model)?;
    let MeasurementParameters::ClassicalError {
        error_variance,
        error_variance_standard_error,
        min_reliability,
    } = &model.parameters
    else {
        return Err(super::MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.invalid_model",
            "classical route needs classical error parameters",
        ));
    };
    let error = ErrorVariance::known(*error_variance, *error_variance_standard_error)
        .map_err(|e| stats_error(&e))?;
    let raw = LinearErrorData {
        y: data.outcome,
        w: data.regressor,
        covariates: data.covariates,
        covariate_count: data.covariate_count,
        replicate_counts: data.replicate_counts,
    };
    let moments =
        LinearMoments::accumulate(&raw, &error, *min_reliability).map_err(|e| stats_error(&e))?;
    let uncertainty = if *error_variance_standard_error > 0.0 {
        ParameterUncertainty::Propagated
    } else {
        ParameterUncertainty::DeclaredExact
    };
    finish(model, data, &[], moments, None, uncertainty)
}

/// Repeated measurements: the error variance is estimated from the analysis
/// sample's own replicates, keeping its dependence with the outcome score.
///
/// `replicate_values` holds every unit's replicates back to back;
/// `replicate_counts[i]` is unit `i`'s count. The regressor is the row mean.
/// Units with one replicate contribute to the outcome model and not to the
/// within variance.
///
/// # Errors
/// As [`correct_classical_linear_effect`], and fewer units with two or more
/// replicates than the model declares.
#[allow(
    clippy::too_many_arguments,
    reason = "public signature shared with the facade and its tests; the columns mirror the classical route's argument order"
)]
pub fn correct_repeated_linear_effect(
    model: &MeasurementModel,
    binding: &DataBinding,
    outcome: &[f64],
    replicate_values: &[f64],
    replicate_counts: &[u32],
    covariates: &[f64],
    covariate_count: usize,
    ctx: &ExecutionContext,
) -> Result<CorrectedLinearEffect, MeasurementError> {
    check_cancelled(ctx)?;
    check_binding(model, binding)?;
    require_route(model, MeasurementRoute::RepeatedMeasurements)?;
    require_non_differential(model)?;
    let MeasurementParameters::Repeated { min_replicated_units, min_reliability } =
        &model.parameters
    else {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.invalid_model",
            "repeated route needs repeated-measurement parameters",
        ));
    };
    if outcome.len() != replicate_counts.len() {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "measurement_model.input_invalid",
            "outcome and replicate counts differ in length",
        ));
    }
    let replicated = replicate_counts.iter().filter(|&&k| k >= 2).count();
    if replicated < *min_replicated_units as usize {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.insufficient_replicates",
            format!(
                "{replicated} units carry two or more replicates; the model declares at least {min_replicated_units}"
            ),
        ));
    }
    let (error, components) =
        ErrorVariance::from_main_sample_replicates(replicate_values, replicate_counts)
            .map_err(|e| stats_error(&e))?;
    let mut means = Vec::with_capacity(replicate_counts.len());
    let mut at = 0_usize;
    for &k in replicate_counts {
        let k = k as usize;
        let group = &replicate_values[at..at + k];
        means.push(group.iter().sum::<f64>() / k as f64);
        at += k;
    }
    let data = LinearMeasuredData {
        outcome,
        regressor: &means,
        covariates,
        covariate_count,
        replicate_counts: Some(replicate_counts),
    };
    let raw = LinearErrorData {
        y: outcome,
        w: &means,
        covariates,
        covariate_count,
        replicate_counts: Some(replicate_counts),
    };
    let moments =
        LinearMoments::accumulate(&raw, &error, *min_reliability).map_err(|e| stats_error(&e))?;
    finish(
        model,
        &data,
        replicate_values,
        moments,
        Some(components),
        ParameterUncertainty::Propagated,
    )
}
