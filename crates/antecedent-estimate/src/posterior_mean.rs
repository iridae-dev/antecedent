//! Point-only posterior means of promotion routes that already publish a point.
//!
//! A mapped coefficient prior, an external power prior, and a request for an
//! unmeasured band refuse. The returned number is the posterior mean. Draws
//! are not an interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;

use crate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, summarize_categorical_design,
};
use crate::error::EstimationError;
use crate::mechanism_discrepancy::{MechanismSummary, mechanism_design_summary};
use crate::vector_treatment::{
    DesignSummary, VectorTreatmentInput, coefficient_posterior_mean,
    full_coefficient_posterior_mean, summarize_vector_design,
};

/// What a caller may ask a posterior-mean route to condition on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PosteriorRequest {
    /// Independent normal priors of this scale on the slope coefficients.
    IndependentNormal {
        /// Prior standard deviation. Must be finite and positive.
        scale: f64,
    },
    /// A prior transported from another coefficient's posterior.
    MappedCoefficient,
    /// A power prior taken from an external sample.
    ExternalPower,
    /// A request to publish an unmeasured interval around the mean.
    UnmeasuredBand,
}

fn refuse(code: &'static str, detail: &'static str) -> EstimationError {
    EstimationError::refused_detail(code, detail, "posterior mean only")
}

fn scale_of(
    request: PosteriorRequest,
    non_positive: &'static str,
    mapped: &'static str,
    power: &'static str,
    band: &'static str,
) -> Result<f64, EstimationError> {
    match request {
        PosteriorRequest::IndependentNormal { scale } if scale.is_finite() && scale > 0.0 => {
            Ok(scale)
        }
        PosteriorRequest::IndependentNormal { .. } => {
            Err(refuse(reason_code!("invalid_argument"), non_positive))
        }
        PosteriorRequest::MappedCoefficient => {
            Err(refuse(reason_code!("route_not_supported"), mapped))
        }
        PosteriorRequest::ExternalPower => Err(refuse(reason_code!("route_not_supported"), power)),
        PosteriorRequest::UnmeasuredBand => Err(refuse(reason_code!("cell_not_licensed"), band)),
    }
}

/// Posterior mean of the joint treatment coefficients.
///
/// # Errors
///
/// `vector_posterior.non_positive_scale`, `vector_posterior.mapped_prior`,
/// `vector_posterior.external_power_prior`, or `vector_posterior.unmeasured_band`,
/// plus the design refusals of the joint fit.
pub fn vector_treatment_posterior(
    input: &VectorTreatmentInput,
    request: PosteriorRequest,
) -> Result<Vec<f64>, EstimationError> {
    let scale = scale_of(
        request,
        "vector_posterior.non_positive_scale",
        "vector_posterior.mapped_prior",
        "vector_posterior.external_power_prior",
        "vector_posterior.unmeasured_band",
    )?;
    coefficient_posterior_mean(&summarize_vector_design(input)?, scale)
}

/// Posterior mean of the dummy-coded categorical coefficients, reference omitted.
///
/// # Errors
///
/// `categorical_posterior.non_positive_scale`, `categorical_posterior.mapped_prior`,
/// `categorical_posterior.external_power_prior`, or `categorical_posterior.unmeasured_band`,
/// plus the design refusals of the categorical fit.
pub fn categorical_treatment_posterior(
    input: &CategoricalTreatmentInput,
    spec: &CategoricalTreatmentSpec,
    request: PosteriorRequest,
) -> Result<Vec<f64>, EstimationError> {
    let scale = scale_of(
        request,
        "categorical_posterior.non_positive_scale",
        "categorical_posterior.mapped_prior",
        "categorical_posterior.external_power_prior",
        "categorical_posterior.unmeasured_band",
    )?;
    let summary = summarize_categorical_design(input, spec)?;
    coefficient_posterior_mean(&summary.design, scale)
}

/// Replay a posterior mean from normal equations the caller retained.
///
/// This is the consumer: it never sees the rows.
///
/// # Errors
///
/// The summary refusals of [`coefficient_posterior_mean`].
pub fn consume_coefficient_posterior(
    summary: &DesignSummary,
    prior_scale: f64,
) -> Result<Vec<f64>, EstimationError> {
    coefficient_posterior_mean(summary, prior_scale)
}

/// Local-quadratic posterior mean at `at`: level and first derivative.
///
/// Independent normal priors on the linear and quadratic coefficients, intercept
/// unpenalized, plug-in weighted residual variance. No interval.
pub fn local_quadratic_posterior(
    dose: &[f64],
    outcome: &[f64],
    at: f64,
    bandwidth: f64,
    request: PosteriorRequest,
    namespace: LocalQuadraticNamespace,
) -> Result<(f64, f64), EstimationError> {
    let (non_positive, mapped, power, band, weight) = namespace.details();
    let scale = scale_of(request, non_positive, mapped, power, band)?;
    local_quadratic_mean(dose, outcome, at, bandwidth, scale)
        .ok_or_else(|| refuse(reason_code!("cell_not_licensed"), weight))
}

/// Which local-quadratic route is refusing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalQuadraticNamespace {
    /// `estimate.quadratic_dose`.
    QuadraticDose,
    /// `estimate.dose_grid_functional`.
    DoseGrid,
}

impl LocalQuadraticNamespace {
    fn details(self) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::QuadraticDose => (
                "quadratic_posterior.non_positive_scale",
                "quadratic_posterior.mapped_prior",
                "quadratic_posterior.external_power_prior",
                "quadratic_posterior.unmeasured_band",
                "quadratic_posterior.insufficient_weight",
            ),
            Self::DoseGrid => (
                "dose_grid_posterior.non_positive_scale",
                "dose_grid_posterior.mapped_prior",
                "dose_grid_posterior.external_power_prior",
                "dose_grid_posterior.unmeasured_band",
                "dose_grid_posterior.insufficient_weight",
            ),
        }
    }
}

/// Replay a local-quadratic posterior mean from the weighted normal equations.
///
/// `gram` is the unpenalized `X'WX`, `rhs` is `X'Wy`, `weighted_rss` is the
/// unpenalized weighted residual sum of squares, and `weight_sum` is the sum of
/// the kernel weights. The consumer does not see the rows.
#[must_use]
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
pub fn consume_local_quadratic_posterior(
    gram: [[f64; 3]; 3],
    rhs: [f64; 3],
    weight_sum: f64,
    weighted_rss: f64,
    prior_scale: f64,
) -> Option<(f64, f64)> {
    if !(weight_sum > 3.0) || !(prior_scale > 0.0) || !weighted_rss.is_finite() {
        return None;
    }
    let sigma2 = weighted_rss / (weight_sum - 3.0);
    let penalty = sigma2 / (prior_scale * prior_scale);
    let mut penalized = gram;
    penalized[1][1] += penalty;
    penalized[2][2] += penalty;
    let inverse = crate::util::invert_fixed(&penalized)?;
    let beta = matvec(inverse, rhs);
    (beta[0].is_finite() && beta[1].is_finite()).then_some((beta[0], beta[1]))
}

#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
fn local_quadratic_mean(
    dose: &[f64],
    outcome: &[f64],
    at: f64,
    bandwidth: f64,
    prior_scale: f64,
) -> Option<(f64, f64)> {
    if dose.len() != outcome.len() || dose.len() < 4 || !at.is_finite() || !(bandwidth > 0.0) {
        return None;
    }
    let mut gram = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    let mut weight_sum = 0.0;
    let mut rows = Vec::with_capacity(dose.len());
    for (&x, &y) in dose.iter().zip(outcome) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let dx = x - at;
        let w = (-0.5 * (dx / bandwidth).powi(2)).exp();
        let row = [1.0, dx, dx * dx];
        for j in 0..3 {
            rhs[j] += w * row[j] * y;
            for k in 0..3 {
                gram[j][k] += w * row[j] * row[k];
            }
        }
        weight_sum += w;
        rows.push((w, row, y));
    }
    if weight_sum <= 3.0 {
        return None;
    }
    let inverse = crate::util::invert_fixed(&gram)?;
    let beta = matvec(inverse, rhs);
    let mut rss = 0.0;
    for (w, row, y) in &rows {
        let fitted = row[0] * beta[0] + row[1] * beta[1] + row[2] * beta[2];
        rss += w * (y - fitted).powi(2);
    }
    let sigma2 = rss / (weight_sum - 3.0);
    let penalty = sigma2 / (prior_scale * prior_scale);
    gram[1][1] += penalty;
    gram[2][2] += penalty;
    let inverse = crate::util::invert_fixed(&gram)?;
    let beta = matvec(inverse, rhs);
    (beta[0].is_finite() && beta[1].is_finite()).then_some((beta[0], beta[1]))
}

fn matvec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

/// Which cross-fitted score a posterior mean belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreNamespace {
    /// Penalized-propensity AIPW.
    Penalized,
    /// Clustered double machine learning.
    Clustered,
    /// Batch retarget of a cross-fitted score.
    BatchRetarget,
    /// Learned continuous trial transport.
    LearnedContinuous,
    /// Smoothed dose-response transport.
    SmoothedDose,
}

impl ScoreNamespace {
    fn details(self) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Penalized => (
                "penalized_posterior.non_positive_scale",
                "penalized_posterior.mapped_prior",
                "penalized_posterior.external_power_prior",
                "penalized_posterior.unmeasured_band",
                "penalized_posterior.empty_scores",
            ),
            Self::Clustered => (
                "clustered_posterior.non_positive_scale",
                "clustered_posterior.mapped_prior",
                "clustered_posterior.external_power_prior",
                "clustered_posterior.unmeasured_band",
                "clustered_posterior.empty_scores",
            ),
            Self::BatchRetarget => (
                "batch_retarget_posterior.non_positive_scale",
                "batch_retarget_posterior.mapped_prior",
                "batch_retarget_posterior.external_power_prior",
                "batch_retarget_posterior.unmeasured_band",
                "batch_retarget_posterior.empty_scores",
            ),
            Self::LearnedContinuous => (
                "learned_continuous_posterior.non_positive_scale",
                "learned_continuous_posterior.mapped_prior",
                "learned_continuous_posterior.external_power_prior",
                "learned_continuous_posterior.unmeasured_band",
                "learned_continuous_posterior.empty_scores",
            ),
            Self::SmoothedDose => (
                "smoothed_dose_posterior.non_positive_scale",
                "smoothed_dose_posterior.mapped_prior",
                "smoothed_dose_posterior.external_power_prior",
                "smoothed_dose_posterior.unmeasured_band",
                "smoothed_dose_posterior.empty_scores",
            ),
        }
    }
}

/// Posterior mean of a cross-fitted contrast under an independent normal prior on the mean.
///
/// The contrast is treated minus control. The plug-in score variance is the likelihood.
/// A wide scale stays at the score mean. No interval is returned.
///
/// # Errors
///
/// The namespace's non-positive scale, mapped prior, external power prior, unmeasured
/// band, or empty score table.
pub fn score_contrast_posterior(
    control: &[f64],
    treated: &[f64],
    request: PosteriorRequest,
    namespace: ScoreNamespace,
) -> Result<f64, EstimationError> {
    let (non_positive, mapped, power, band, empty) = namespace.details();
    let scale = scale_of(request, non_positive, mapped, power, band)?;
    if control.len() != treated.len() || control.len() < 2 {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    let contrast: Vec<f64> = treated.iter().zip(control).map(|(one, zero)| one - zero).collect();
    if contrast.iter().any(|value| !value.is_finite()) {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    let n = contrast.len() as f64;
    let mean = contrast.iter().sum::<f64>() / n;
    let rss = contrast.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>();
    let variance = rss / (n - 1.0);
    if !variance.is_finite() || variance <= 0.0 {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    let weight = (n / variance) / (n / variance + 1.0 / (scale * scale));
    Ok(weight * mean)
}

/// Dirichlet posterior mean of a finite weight table.
///
/// `concentration` is the prior mass on each atom. A non-positive concentration,
/// a mapped prior, an external power prior, and an unmeasured band refuse under
/// `namespace`.
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
pub fn dirichlet_posterior_mean(
    weights: &[f64],
    request: PosteriorRequest,
    namespace: DirichletNamespace,
) -> Result<Vec<f64>, EstimationError> {
    let (non_positive, mapped, power, band, empty) = namespace.details();
    let concentration = scale_of(request, non_positive, mapped, power, band)?;
    if weights.is_empty() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    let total = weights.iter().sum::<f64>() + concentration * weights.len() as f64;
    if !(total > 0.0) {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    Ok(weights.iter().map(|w| (w + concentration) / total).collect())
}

/// Which finite-law route owns the Dirichlet refusal namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirichletNamespace {
    /// Decision functionals.
    Decision,
    /// Decision functional inverse.
    FunctionalInverse,
    /// Finite-action inverse.
    FiniteActionInverse,
    /// Binary observation recovery.
    BinaryRecovery,
    /// Chain observation recovery.
    ChainRecovery,
    /// Multi-source limited experiment.
    MultiSource,
    /// Two-step temporal transport.
    TwoStepTemporal,
    /// Conditional ADMG transport.
    AdmgConditional,
    /// Uncertain initial-state law.
    InitialState,
    /// Marginal structural model strata.
    Msm,
}

impl DirichletNamespace {
    fn details(self) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Decision => (
                "decision_posterior.non_positive_scale",
                "decision_posterior.mapped_prior",
                "decision_posterior.external_power_prior",
                "decision_posterior.unmeasured_band",
                "decision_posterior.empty_law",
            ),
            Self::FunctionalInverse => (
                "functional_inverse_posterior.non_positive_scale",
                "functional_inverse_posterior.mapped_prior",
                "functional_inverse_posterior.external_power_prior",
                "functional_inverse_posterior.unmeasured_band",
                "functional_inverse_posterior.empty_law",
            ),
            Self::FiniteActionInverse => (
                "finite_action_posterior.non_positive_scale",
                "finite_action_posterior.mapped_prior",
                "finite_action_posterior.external_power_prior",
                "finite_action_posterior.unmeasured_band",
                "finite_action_posterior.empty_law",
            ),
            Self::BinaryRecovery => (
                "binary_recovery_posterior.non_positive_scale",
                "binary_recovery_posterior.mapped_prior",
                "binary_recovery_posterior.external_power_prior",
                "binary_recovery_posterior.unmeasured_band",
                "binary_recovery_posterior.empty_law",
            ),
            Self::ChainRecovery => (
                "chain_recovery_posterior.non_positive_scale",
                "chain_recovery_posterior.mapped_prior",
                "chain_recovery_posterior.external_power_prior",
                "chain_recovery_posterior.unmeasured_band",
                "chain_recovery_posterior.empty_law",
            ),
            Self::MultiSource => (
                "multi_source_posterior.non_positive_scale",
                "multi_source_posterior.mapped_prior",
                "multi_source_posterior.external_power_prior",
                "multi_source_posterior.unmeasured_band",
                "multi_source_posterior.empty_law",
            ),
            Self::TwoStepTemporal => (
                "two_step_posterior.non_positive_scale",
                "two_step_posterior.mapped_prior",
                "two_step_posterior.external_power_prior",
                "two_step_posterior.unmeasured_band",
                "two_step_posterior.empty_law",
            ),
            Self::AdmgConditional => (
                "admg_conditional_posterior.non_positive_scale",
                "admg_conditional_posterior.mapped_prior",
                "admg_conditional_posterior.external_power_prior",
                "admg_conditional_posterior.unmeasured_band",
                "admg_conditional_posterior.empty_law",
            ),
            Self::InitialState => (
                "initial_state_posterior.non_positive_scale",
                "initial_state_posterior.mapped_prior",
                "initial_state_posterior.external_power_prior",
                "initial_state_posterior.unmeasured_band",
                "initial_state_posterior.empty_law",
            ),
            Self::Msm => (
                "msm_posterior.non_positive_scale",
                "msm_posterior.mapped_prior",
                "msm_posterior.external_power_prior",
                "msm_posterior.unmeasured_band",
                "msm_posterior.empty_law",
            ),
        }
    }
}

/// Dirichlet posterior mean that leaves structural zeros at zero.
///
/// Positive cells are updated as their own finite law. A zero cell is a support
/// constraint, not a prior atom.
///
/// # Errors
///
/// The namespace's non-positive scale, mapped prior, external power prior,
/// unmeasured band, or empty support.
pub fn dirichlet_supported_mean(
    probabilities: &[f64],
    request: PosteriorRequest,
    namespace: DirichletNamespace,
) -> Result<Vec<f64>, EstimationError> {
    let free: Vec<f64> = probabilities.iter().copied().filter(|p| *p > 0.0).collect();
    let mean = dirichlet_posterior_mean(&free, request, namespace)?;
    let mut filled = Vec::with_capacity(probabilities.len());
    let mut cursor = 0;
    for probability in probabilities {
        if *probability > 0.0 {
            filled.push(mean[cursor]);
            cursor += 1;
        } else if *probability == 0.0 {
            filled.push(0.0);
        } else {
            let empty = namespace.details().4;
            return Err(refuse(reason_code!("invalid_argument"), empty));
        }
    }
    Ok(filled)
}

/// Which scalar posterior mean a caller is publishing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarNamespace {
    /// Refreshed period response.
    Refresh,
    /// Effect-constancy contrast.
    Constancy,
    /// Scenario covariance entries.
    ScenarioCovariance,
}

impl ScalarNamespace {
    fn details(self) -> (&'static str, &'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::Refresh => (
                "refresh_posterior.non_positive_scale",
                "refresh_posterior.mapped_prior",
                "refresh_posterior.external_power_prior",
                "refresh_posterior.unmeasured_band",
                "refresh_posterior.empty_values",
            ),
            Self::Constancy => (
                "constancy_posterior.non_positive_scale",
                "constancy_posterior.mapped_prior",
                "constancy_posterior.external_power_prior",
                "constancy_posterior.unmeasured_band",
                "constancy_posterior.empty_contrast",
            ),
            Self::ScenarioCovariance => (
                "scenario_covariance_posterior.non_positive_scale",
                "scenario_covariance_posterior.mapped_prior",
                "scenario_covariance_posterior.external_power_prior",
                "scenario_covariance_posterior.unmeasured_band",
                "scenario_covariance_posterior.empty_matrix",
            ),
        }
    }
}

fn shrink(point: f64, precision: f64, scale: f64) -> f64 {
    let prior = 1.0 / (scale * scale);
    (precision / (precision + prior)) * point
}

/// Posterior mean of an observed sample under an independent normal prior at zero.
///
/// # Errors
///
/// The namespace's non-positive scale, mapped prior, external power prior,
/// unmeasured band, or a sample that cannot support a variance.
pub fn normal_mean_posterior(
    values: &[f64],
    request: PosteriorRequest,
    namespace: ScalarNamespace,
) -> Result<f64, EstimationError> {
    let (non_positive, mapped, power, band, empty) = namespace.details();
    let scale = scale_of(request, non_positive, mapped, power, band)?;
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let rss = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>();
    let variance = rss / (n - 1.0);
    if !variance.is_finite() || variance <= 0.0 {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    Ok(shrink(mean, n / variance, scale))
}

/// Posterior mean of one contrast whose sampling variance is already known.
///
/// # Errors
///
/// The namespace's non-positive scale, mapped prior, external power prior,
/// unmeasured band, or a non-positive variance.
pub fn known_variance_posterior(
    point: f64,
    variance: f64,
    request: PosteriorRequest,
    namespace: ScalarNamespace,
) -> Result<f64, EstimationError> {
    let (non_positive, mapped, power, band, empty) = namespace.details();
    let scale = scale_of(request, non_positive, mapped, power, band)?;
    if !point.is_finite() || !variance.is_finite() || variance <= 0.0 {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    Ok(shrink(point, 1.0 / variance, scale))
}

/// Posterior mean of covariance entries under independent normal priors at zero.
///
/// Each entry is shrunk toward 0 with precision `1 / v`, where `v` is the
/// sampling variance of that covariance estimator
/// ([`crate::scenario_covariance::covariance_estimator_moments`]): `correction²`
/// times the weighted second moment of the centered replicate products, divided
/// by the number of rows. The spread of the entries around their own mean is
/// not used, so a constant matrix still updates. A wide scale stays at the
/// supplied matrix. A sampling variance of 0 leaves the entry unchanged. No
/// interval is returned.
///
/// # Errors
///
/// The scenario-covariance namespace's non-positive scale, mapped prior, external
/// power prior, unmeasured band, or entries and variances that are empty, of
/// different lengths, non-finite, or negative.
pub fn covariance_entry_posterior(
    entries: &[f64],
    sampling_variances: &[f64],
    request: PosteriorRequest,
) -> Result<Vec<f64>, EstimationError> {
    let namespace = ScalarNamespace::ScenarioCovariance;
    let (non_positive, mapped, power, band, empty) = namespace.details();
    let scale = scale_of(request, non_positive, mapped, power, band)?;
    if entries.is_empty()
        || entries.len() != sampling_variances.len()
        || entries.iter().any(|value| !value.is_finite())
        || sampling_variances.iter().any(|variance| !variance.is_finite() || *variance < 0.0)
    {
        return Err(refuse(reason_code!("invalid_argument"), empty));
    }
    Ok(entries
        .iter()
        .zip(sampling_variances)
        .map(
            |(&entry, &variance)| {
                if variance == 0.0 { entry } else { shrink(entry, 1.0 / variance, scale) }
            },
        )
        .collect())
}

/// Posterior mean of the mediator and outcome coefficient vectors.
///
/// Independent normal priors on every non-intercept column, intercept unpenalized,
/// plug-in residual variance. A wide scale stays at the ordinary-least-squares
/// coefficients. No interval is returned.
///
/// # Errors
///
/// `mediation_posterior.non_positive_scale`, `mediation_posterior.mapped_prior`,
/// `mediation_posterior.external_power_prior`, `mediation_posterior.unmeasured_band`,
/// or `mediation_posterior.empty_design`.
pub fn mediation_coefficient_posterior(
    mediator: &DesignSummary,
    outcome: &DesignSummary,
    request: PosteriorRequest,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let scale = scale_of(
        request,
        "mediation_posterior.non_positive_scale",
        "mediation_posterior.mapped_prior",
        "mediation_posterior.external_power_prior",
        "mediation_posterior.unmeasured_band",
    )?;
    if mediator.xty.is_empty() || outcome.xty.is_empty() {
        return Err(refuse(reason_code!("invalid_argument"), "mediation_posterior.empty_design"));
    }
    Ok((
        full_coefficient_posterior_mean(mediator, scale)?,
        full_coefficient_posterior_mean(outcome, scale)?,
    ))
}

/// Posterior mean of each class's treatment coefficient.
///
/// Each design is the class's weighted normal equations, with the treatment in the
/// first slope column. A wide scale stays at that column's ordinary-least-squares
/// value. No interval is returned.
///
/// # Errors
///
/// `latent_class_posterior.non_positive_scale`, `latent_class_posterior.mapped_prior`,
/// `latent_class_posterior.external_power_prior`, `latent_class_posterior.unmeasured_band`,
/// or `latent_class_posterior.empty_design`.
pub fn latent_class_effect_posterior(
    designs: &[DesignSummary],
    request: PosteriorRequest,
) -> Result<Vec<f64>, EstimationError> {
    let scale = scale_of(
        request,
        "latent_class_posterior.non_positive_scale",
        "latent_class_posterior.mapped_prior",
        "latent_class_posterior.external_power_prior",
        "latent_class_posterior.unmeasured_band",
    )?;
    if designs.is_empty() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "latent_class_posterior.empty_design",
        ));
    }
    designs
        .iter()
        .map(|design| {
            let slopes = coefficient_posterior_mean(design, scale)?;
            slopes.first().copied().filter(|value| value.is_finite()).ok_or_else(|| {
                refuse(reason_code!("invalid_argument"), "latent_class_posterior.empty_design")
            })
        })
        .collect()
}

/// Posterior mean of one population's mechanism coefficients, intercept included.
///
/// Independent normal priors on the parent slopes, intercept unpenalized, plug-in
/// residual variance from `X'X`, `X'y` and `y'y`. This is not a posterior of the
/// Wald statistic. No interval is returned.
///
/// # Errors
///
/// `mechanism_posterior.non_positive_scale`, `mechanism_posterior.mapped_prior`,
/// `mechanism_posterior.external_power_prior`, `mechanism_posterior.unmeasured_band`,
/// or `mechanism_posterior.empty_summary`.
pub fn mechanism_coefficient_posterior(
    summary: &MechanismSummary,
    request: PosteriorRequest,
) -> Result<Vec<f64>, EstimationError> {
    let scale = scale_of(
        request,
        "mechanism_posterior.non_positive_scale",
        "mechanism_posterior.mapped_prior",
        "mechanism_posterior.external_power_prior",
        "mechanism_posterior.unmeasured_band",
    )?;
    if summary.xty.len() < 2 || summary.xtx.len() != summary.xty.len() * summary.xty.len() {
        return Err(refuse(reason_code!("invalid_argument"), "mechanism_posterior.empty_summary"));
    }
    let design = mechanism_design_summary(summary)?;
    full_coefficient_posterior_mean(&design, scale)
}

/// Posterior mean of a learned-continuous AIPW score table.
///
/// The table is scaled so its sample mean is the augmented estimate. A wide scale
/// stays at that mean. No interval is returned.
///
/// # Errors
///
/// The learned-continuous score namespace's non-positive scale, mapped prior,
/// external power prior, unmeasured band, or empty score table.
pub fn learned_continuous_score_posterior(
    scores: &[f64],
    request: PosteriorRequest,
) -> Result<f64, EstimationError> {
    let zeros = vec![0.0; scores.len()];
    score_contrast_posterior(&zeros, scores, request, ScoreNamespace::LearnedContinuous)
}

/// Posterior mean of each smoothed-dose grid point's per-row scores.
///
/// Each table is scaled so its sample mean is that dose's estimate. A wide scale
/// stays at those means. No interval is returned.
///
/// # Errors
///
/// The smoothed-dose score namespace's non-positive scale, mapped prior, external
/// power prior, unmeasured band, or empty score table.
pub fn smoothed_dose_score_posterior(
    grids: &[Vec<f64>],
    request: PosteriorRequest,
) -> Result<Vec<f64>, EstimationError> {
    if grids.is_empty() {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "smoothed_dose_posterior.empty_scores",
        ));
    }
    grids
        .iter()
        .map(|scores| {
            let zeros = vec![0.0; scores.len()];
            score_contrast_posterior(&zeros, scores, request, ScoreNamespace::SmoothedDose)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::consume_local_quadratic_posterior;

    #[test]
    fn the_local_quadratic_solve_is_scale_free_and_refuses_collinear_moments() {
        // The determinant of this Gram is 1e-15, below the former absolute 1e-12 cutoff, yet
        // the matrix is perfectly conditioned: the posterior mean is the exact coefficient.
        let gram = [[1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0], [0.0, 0.0, 1e-5]];
        let (level, slope) =
            consume_local_quadratic_posterior(gram, [1e-5, 2e-5, 3e-5], 10.0, 0.0, 1.0).unwrap();
        assert!((level - 1.0).abs() < 1e-12 && (slope - 2.0).abs() < 1e-12);
        // A rank-one Gram with no residual variance to penalise stays refused.
        let collinear = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        assert!(
            consume_local_quadratic_posterior(collinear, [1.0, 1.0, 1.0], 10.0, 0.0, 1.0).is_none()
        );
    }
}
