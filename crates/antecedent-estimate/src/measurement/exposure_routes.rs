//! Corrections declared by their own assumptions rather than by a [`MeasurementModel`]:
//! exposure misclassification, per-unit soft exposures, covariate-dependent detection and
//! latent-class repeat codings.
//!
//! These entry points take the declarations as arguments (rates or validation counts, the
//! detection function or its envelope, the codings' constraints) and return the corrected
//! functional with its variance components, the data identity and whether the uncertainty of
//! the declared parameters is in the variance. Each refuses what it cannot identify with a
//! coded [`MeasurementError`]; none gates on a diagnostic.
//!
//! [`MeasurementModel`]: antecedent_core::measurement_model::MeasurementModel

use antecedent_core::ExecutionContext;
use antecedent_core::identity::SemanticDigest;
use antecedent_core::measurement_model::{
    MeasurementInference, ParameterUncertainty, VarianceComponent,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, CombinedImputations, CorrectedExposureAssociation, DetectionAdjusted,
    DetectionData, DetectionDeclaration, ExposureErrorStructure, ExposureOutcomeCounts,
    FalsePositiveRate, LatentClassConstraints, LatentClassData, LatentClassFit,
    LatentClassSettings, SoftExposureData, SoftExposureReliability, SoftExposureSubstitution,
    corrected_exposure_association, detection_adjusted_prevalence, fit_latent_class,
    soft_exposure_imputation, soft_exposure_outcome_conditioned_draws, soft_exposure_reliability,
    soft_exposure_substitution,
};

use super::{MeasurementError, check_cancelled, component, digest, stats_error};

/// A corrected scalar functional of a declaration-bound route.
#[derive(Clone, Debug, PartialEq)]
pub struct DeclaredEstimate {
    /// Stable functional id.
    pub functional_id: &'static str,
    /// Point value.
    pub point: f64,
    /// The recorded (uncorrected) value of the same functional, when it is defined.
    pub naive_point: Option<f64>,
    /// Delta-method (or Rubin) standard error; uncalibrated.
    pub standard_error: Option<f64>,
    /// Named variance components.
    pub variance_components: Vec<VarianceComponent>,
    /// Rows used.
    pub rows: u64,
    /// Identity of the data the value was computed from.
    pub data_identity: SemanticDigest,
    /// Whether the declared parameters' uncertainty is in the variance.
    pub parameter_uncertainty: ParameterUncertainty,
    /// What the value reports about its uncertainty.
    pub inference: MeasurementInference,
}

impl DeclaredEstimate {
    /// Equal-tailed Wald interval `point -/+ z * standard_error`, never calibrated.
    ///
    /// # Errors
    /// A level outside `(0, 1)`, or an estimate without a standard error, refuses.
    pub fn wald_interval(&self, level: f64) -> Result<(f64, f64), MeasurementError> {
        if !(level > 0.0 && level < 1.0) {
            return Err(MeasurementError::new(
                antecedent_core::reason_code!("invalid_argument"),
                antecedent_core::reason_details::measurement_model::INTERVAL_LEVEL,
                "the interval level must lie strictly between 0 and 1",
            ));
        }
        let Some(se) = self.standard_error else {
            return Err(super::interval_not_supported());
        };
        let half = antecedent_stats::special::normal_ppf(0.5 + level / 2.0) * se;
        Ok((self.point - half, self.point + half))
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "one value per field of the record, all of them required"
)]
fn estimate(
    functional_id: &'static str,
    point: f64,
    naive_point: Option<f64>,
    variance: f64,
    components: Vec<VarianceComponent>,
    rows: u64,
    data_identity: SemanticDigest,
    parameter_uncertainty: ParameterUncertainty,
) -> DeclaredEstimate {
    DeclaredEstimate {
        functional_id,
        point,
        naive_point,
        standard_error: Some(variance.max(0.0).sqrt()),
        variance_components: components,
        rows,
        data_identity,
        parameter_uncertainty,
        inference: MeasurementInference::UncalibratedStandardError,
    }
}

// ---------------------------------------------------------------------------
// Exposure misclassification
// ---------------------------------------------------------------------------

/// A corrected exposure association and its declared-estimate record.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedExposure {
    /// Corrected shares, odds ratio and variance parts.
    pub association: CorrectedExposureAssociation,
    /// The corrected log odds ratio.
    pub estimate: DeclaredEstimate,
}

/// Correct a binary exposure's misclassification, including error that differs by outcome.
///
/// `structure` is [`ExposureErrorStructure::Shared`] for error that does not depend on the
/// outcome and [`ExposureErrorStructure::ByOutcome`] for outcome-specific rates, each declared
/// or estimated from validation samples (their binomial variances are then propagated).
/// Rates whose Youden index in either outcome level is at or below `min_youden` are not
/// identified and refuse.
///
/// # Errors
/// Invalid counts, rates that are not identified, a corrected exposure share outside `(0, 1)`
/// or cancellation refuse.
pub fn correct_exposure_misclassification(
    counts: &ExposureOutcomeCounts,
    structure: &ExposureErrorStructure,
    min_youden: f64,
    ctx: &ExecutionContext,
) -> Result<CorrectedExposure, MeasurementError> {
    check_cancelled(ctx)?;
    let association = corrected_exposure_association(counts, structure, min_youden)
        .map_err(|e| stats_error(&e))?;
    let rate_variance = |r: &BinaryErrorRates| r.sensitivity_variance + r.specificity_variance;
    let uncertain = match structure {
        ExposureErrorStructure::Shared(r) => rate_variance(r) > 0.0,
        ExposureErrorStructure::ByOutcome { controls, cases } => {
            rate_variance(controls) > 0.0 || rate_variance(cases) > 0.0
        }
    };
    let identity = digest(&[(
        "counts",
        &[
            counts.exposed_cases as f64,
            counts.cases as f64,
            counts.exposed_controls as f64,
            counts.controls as f64,
        ],
    )]);
    let estimate = estimate(
        "exposure_log_odds_ratio",
        association.log_odds_ratio,
        association.naive_log_odds_ratio,
        association.variance,
        vec![
            component("sampling_of_recorded_exposure_shares", association.variance_sampling),
            component("sensitivity_specificity_uncertainty", association.variance_rates),
        ],
        counts.cases + counts.controls,
        identity,
        if uncertain {
            ParameterUncertainty::Propagated
        } else {
            ParameterUncertainty::DeclaredExact
        },
    );
    Ok(CorrectedExposure { association, estimate })
}

// ---------------------------------------------------------------------------
// Soft exposures
// ---------------------------------------------------------------------------

/// How per-unit exposure probabilities enter the outcome model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoftExposureCombination<'a> {
    /// The per-unit posterior mean replaces the exposure.
    PosteriorMean,
    /// Outcome-conditioned draws are generated and combined by Rubin's rules.
    MultipleImputation {
        /// Imputations, at least two.
        draw_count: usize,
        /// Seed of the draws.
        seed: u64,
    },
    /// Caller-supplied outcome-conditioned `{0, 1}` draws, draw-major, combined by Rubin's rules.
    SuppliedDraws {
        /// Draws, `draw_count * n` values.
        draws: &'a [f64],
        /// Number of draws.
        draw_count: usize,
    },
}

/// A linear exposure effect over per-unit exposure probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftExposureEffect {
    /// Coefficients, intercept first and the exposure second.
    pub coefficients: Vec<f64>,
    /// Coefficient covariance, row-major.
    pub covariance: Vec<f64>,
    /// The posterior-mean fit with its exposure-uncertainty term.
    pub substitution: Option<SoftExposureSubstitution>,
    /// The per-imputation fits and their combination.
    pub imputations: Option<CombinedImputations>,
    /// The exposure's coefficient.
    pub estimate: DeclaredEstimate,
}

/// Fit a linear outcome model on a per-unit probability of exposure.
///
/// Posterior-mean substitution reports a variance term for the exposure uncertainty; multiple
/// imputation reports within- and between-imputation variance. Substitution needs the stated
/// probabilities to be calibrated given the covariates; imputation additionally conditions
/// each draw on the outcome (see [`soft_exposure_outcome_conditioned_draws`]).
///
/// # Errors
/// A shape or range violation, a rank-deficient design, fewer than two imputations or
/// cancellation refuse.
pub fn correct_soft_exposure_effect(
    data: &SoftExposureData<'_>,
    combination: &SoftExposureCombination<'_>,
    ctx: &ExecutionContext,
) -> Result<SoftExposureEffect, MeasurementError> {
    check_cancelled(ctx)?;
    let n = data.outcome.len();
    let identity = digest(&[
        ("outcome", data.outcome),
        ("probability", data.probability),
        ("covariates", data.covariates),
    ]);
    match combination {
        SoftExposureCombination::PosteriorMean => {
            let fit = soft_exposure_substitution(data).map_err(|e| stats_error(&e))?;
            let est = estimate(
                "soft_exposure_coefficient",
                fit.coefficients[1],
                None,
                fit.variance,
                vec![
                    component("sampling_given_exposure", fit.variance_sampling),
                    component("exposure_uncertainty", fit.variance_exposure_uncertainty),
                ],
                n as u64,
                identity,
                ParameterUncertainty::NotPropagated,
            );
            Ok(SoftExposureEffect {
                coefficients: fit.coefficients.clone(),
                covariance: fit.covariance.clone(),
                substitution: Some(fit),
                imputations: None,
                estimate: est,
            })
        }
        SoftExposureCombination::MultipleImputation { draw_count, seed } => {
            let draws = soft_exposure_outcome_conditioned_draws(data, *draw_count, *seed)
                .map_err(|e| stats_error(&e))?;
            check_cancelled(ctx)?;
            imputed(data, &draws.draws, *draw_count, identity)
        }
        SoftExposureCombination::SuppliedDraws { draws, draw_count } => {
            imputed(data, draws, *draw_count, identity)
        }
    }
}

fn imputed(
    data: &SoftExposureData<'_>,
    draws: &[f64],
    draw_count: usize,
    identity: SemanticDigest,
) -> Result<SoftExposureEffect, MeasurementError> {
    let combined = soft_exposure_imputation(
        data.outcome,
        draws,
        draw_count,
        data.covariates,
        data.covariate_count,
    )
    .map_err(|e| stats_error(&e))?;
    let d = combined.dim;
    let est = estimate(
        "soft_exposure_coefficient",
        combined.coefficients[1],
        None,
        combined.covariance[d + 1],
        vec![
            component("within_imputation", combined.within[d + 1]),
            component(
                "between_imputation",
                (1.0 + 1.0 / combined.draws as f64) * combined.between[d + 1],
            ),
        ],
        data.outcome.len() as u64,
        identity,
        ParameterUncertainty::Propagated,
    );
    Ok(SoftExposureEffect {
        coefficients: combined.coefficients.clone(),
        covariance: combined.covariance.clone(),
        substitution: None,
        imputations: Some(combined),
        estimate: est,
    })
}

/// Compare stated exposure probabilities with repeat codings in bins of the probability.
///
/// A report: nothing is gated on it.
///
/// # Errors
/// Mismatched lengths, codings outside `{0, 1}` or cancellation refuse.
pub fn soft_exposure_reliability_report(
    probability: &[f64],
    codings: &[u8],
    sizes: &[u32],
    bins: usize,
    reference: Option<&BinaryErrorRates>,
    ctx: &ExecutionContext,
) -> Result<SoftExposureReliability, MeasurementError> {
    check_cancelled(ctx)?;
    soft_exposure_reliability(probability, codings, sizes, bins, reference)
        .map_err(|e| stats_error(&e))
}

// ---------------------------------------------------------------------------
// Detection that depends on evidence density
// ---------------------------------------------------------------------------

/// A prevalence under covariate-dependent detection.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionDependentPrevalence {
    /// A point when the detection function was fully declared, else sharp bounds.
    pub adjusted: DetectionAdjusted,
    /// The point's estimate record; `None` for bounds.
    pub estimate: Option<DeclaredEstimate>,
}

/// Prevalence of a condition detected with a probability that depends on a covariate.
///
/// A fully declared function point-identifies the prevalence; a partially declared (monotone,
/// anchored) one gives sharp bounds. Mean detection net of false positives at or below
/// `min_net_detection` is not identified and refuses.
///
/// # Errors
/// A shape or range violation, a non-identifying detection, an inconsistent envelope or
/// cancellation refuse.
pub fn correct_detection_dependent_prevalence(
    data: &DetectionData<'_>,
    declaration: &DetectionDeclaration,
    false_positive: &FalsePositiveRate,
    min_net_detection: f64,
    ctx: &ExecutionContext,
) -> Result<DetectionDependentPrevalence, MeasurementError> {
    check_cancelled(ctx)?;
    let adjusted =
        detection_adjusted_prevalence(data, declaration, false_positive, min_net_detection)
            .map_err(|e| stats_error(&e))?;
    let estimate = match &adjusted {
        DetectionAdjusted::Point(point) => {
            let detected: Vec<f64> =
                data.detected.iter().map(|d| if *d { 1.0 } else { 0.0 }).collect();
            let identity =
                digest(&[("detected", detected.as_slice()), ("covariate", data.covariate)]);
            let exact = point.variance_parameters == 0.0;
            Some(self::estimate(
                "detection_adjusted_prevalence",
                point.prevalence,
                Some(point.observed_rate),
                point.variance,
                vec![
                    component("sampling_of_detections", point.variance_sampling),
                    component("detection_function_uncertainty", point.variance_parameters),
                ],
                point.units as u64,
                identity,
                if exact {
                    ParameterUncertainty::DeclaredExact
                } else {
                    ParameterUncertainty::Propagated
                },
            ))
        }
        DetectionAdjusted::Bounds(_) => None,
    };
    Ok(DetectionDependentPrevalence { adjusted, estimate })
}

// ---------------------------------------------------------------------------
// Latent-class repeat codings
// ---------------------------------------------------------------------------

/// Fit repeat codings of a binary or ordinal facet as replicates of one latent level.
///
/// Conditional independence of the codings given the latent level is an assumption of the
/// model. Three or more codings identify it (more for more levels); two need exchangeable
/// codings and a declared law of the lowest level; fewer refuse.
///
/// # Errors
/// A shape violation, too few codings for the declared levels and constraints, codings that
/// do not separate the levels, or cancellation refuse.
pub fn fit_repeat_codings(
    data: &LatentClassData<'_>,
    constraints: &LatentClassConstraints,
    settings: &LatentClassSettings,
    ctx: &ExecutionContext,
) -> Result<LatentClassFit, MeasurementError> {
    check_cancelled(ctx)?;
    fit_latent_class(data, constraints, settings).map_err(|e| stats_error(&e))
}
