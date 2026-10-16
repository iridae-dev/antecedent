//! Corrected functionals under a declared measurement model.
//!
//! Each entry point takes a [`MeasurementModel`], the binding of the data it
//! is applied to, and the data, and returns the corrected functional bound to
//! the model's identity. Four things are checked before any number is
//! computed, each as its own coded refusal:
//!
//! 1. the model is well formed and relates a non-latent indicator to a
//!    latent scientific construct (a proxy is never the construct);
//! 2. the data's population, regime, arm and representation are the ones the
//!    model was established for;
//! 3. the model's route is the route of the entry point;
//! 4. the declared error structure is the structure the entry point
//!    implements: a model declared differential is never run through the
//!    non-differential route, and the reverse.
//!
//! Variance is delta-method and includes the uncertainty of the estimated
//! measurement parameters (error variance, sensitivity and specificity,
//! confusion columns, posterior draws), not only downstream sampling
//! variance. No interval is produced: the whole measurement-plus-estimation
//! method has no measured coverage record, so the interval route is closed
//! (`cell_not_licensed`) and the result carries a point and an *uncalibrated*
//! standard error.
//!
//! The pure numerics are in [`antecedent_stats::measurement_error`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod binary;
pub mod categorical;
pub mod linear;
pub mod posterior;

use antecedent_core::ExecutionContext;
use antecedent_core::identity::SemanticDigest;
use antecedent_core::measurement_model::{
    ArmBinding, CorrectedFunctionalRecord, ErrorStructure, MeasurementInference, MeasurementModel,
    MeasurementModelError, MeasurementRoute, ParameterUncertainty, RepresentationVersion,
    VarianceComponent, measurement_data_digest,
};
use antecedent_stats::measurement_error::MeasurementErrorError;

pub use binary::{
    CorrectedBinaryPrevalence, CorrectedBinaryRiskDifference, RiskDifferenceInput,
    correct_binary_prevalence, correct_binary_risk_difference,
};
pub use categorical::{CategoricalFunctional, CorrectedCategoricalOutcome, correct_categorical};
pub use linear::{
    CorrectedLinearEffect, LinearMeasuredData, correct_classical_linear_effect,
    correct_repeated_linear_effect,
};
pub use posterior::{
    CorrectedPosteriorLinearEffect, PosteriorCombination, PosteriorLinearData,
    correct_latent_posterior_linear_effect,
};

/// A coded refusal from a measurement correction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail, `measurement_model.<slot>`.
    pub detail: &'static str,
    /// What was refused and what would license it.
    pub message: String,
}

impl MeasurementError {
    pub(crate) fn new(
        code: &'static str,
        detail: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self { code, detail, message: message.into() }
    }
}

impl std::fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for MeasurementError {}

/// The population, regime, arm and representation of the data a correction is
/// applied to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataBinding {
    /// Population the data describe.
    pub population_id: String,
    /// Regime the data were collected under.
    pub regime_id: String,
    /// Arm the data are from; `None` when the data span arms.
    pub arm_id: Option<String>,
    /// Representation the indicator is recorded in.
    pub representation: RepresentationVersion,
}

pub(crate) fn model_error(error: &MeasurementModelError) -> MeasurementError {
    match error {
        MeasurementModelError::ProxyOfferedAsConstruct { offered } => MeasurementError::new(
            antecedent_core::reason_code!("proxy_offered_as_construct"),
            "measurement_model.proxy_as_construct",
            format!(
                "a {} variable is offered as the latent scientific construct; relate it to the construct by a measurement model",
                offered.as_str()
            ),
        ),
        MeasurementModelError::ObservedIndicator => MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.indicator_observed",
            "an observed column has no error model; declare it noisy or misclassified",
        ),
        MeasurementModelError::StructureParameterMismatch => MeasurementError::new(
            antecedent_core::reason_code!("differential_error_not_pooled"),
            "measurement_model.structure_parameters",
            "the declared error structure disagrees with the parameter layout",
        ),
        other => MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.invalid_model",
            format!("{other:?}"),
        ),
    }
}

pub(crate) fn stats_error(error: &MeasurementErrorError) -> MeasurementError {
    match error {
        MeasurementErrorError::ReliabilityNotPositive { reliability, floor } => {
            MeasurementError::new(
                antecedent_core::reason_code!("measurement_not_identified"),
                "measurement_model.reliability_too_small",
                format!(
                    "implied reliability {reliability} is not above the declared floor {floor}; a more reliable measurement or averaged replicates is needed"
                ),
            )
        }
        MeasurementErrorError::YoudenTooSmall { youden, minimum } => MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.youden_too_small",
            format!(
                "Youden index {youden} is not above the declared minimum {minimum}; the indicator is not informative enough to invert"
            ),
        ),
        MeasurementErrorError::ConfusionIllConditioned { condition, maximum } => {
            MeasurementError::new(
                antecedent_core::reason_code!("measurement_not_identified"),
                "measurement_model.confusion_ill_conditioned",
                format!(
                    "confusion matrix condition {condition} exceeds the declared maximum {maximum}"
                ),
            )
        }
        MeasurementErrorError::EmptyLatentCategory => MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.confusion_ill_conditioned",
            "a latent category has zero corrected mass, so its outcome mean is not defined",
        ),
        MeasurementErrorError::InsufficientReplicates(why) => MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.insufficient_replicates",
            format!("replicates cannot identify the within-unit variance: {why}"),
        ),
        MeasurementErrorError::SingularDesign => MeasurementError::new(
            antecedent_core::reason_code!("design_rank_deficient"),
            "measurement_model.design_singular",
            "the observed design is rank deficient; no correction is formed",
        ),
        MeasurementErrorError::NonPositiveVariance(what) => MeasurementError::new(
            antecedent_core::reason_code!("measurement_not_identified"),
            "measurement_model.variance_not_positive",
            format!("the corrected variance of the {what} is not positive"),
        ),
        other => MeasurementError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "measurement_model.input_invalid",
            other.to_string(),
        ),
    }
}

/// Check the model is well formed and applies to this data.
pub(crate) fn check_binding(
    model: &MeasurementModel,
    data: &DataBinding,
) -> Result<(), MeasurementError> {
    model.validate().map_err(|error| model_error(&error))?;
    let mismatch = |detail: &'static str, expected: &str, supplied: &str| {
        MeasurementError::new(
            antecedent_core::reason_code!("measurement_binding_mismatch"),
            detail,
            format!(
                "the measurement model was established for `{expected}`, the data are `{supplied}`; establish the model for this binding or declare an invariance premise under a transport contract"
            ),
        )
    };
    if model.binding.population_id != data.population_id {
        return Err(mismatch(
            "measurement_model.population_mismatch",
            &model.binding.population_id,
            &data.population_id,
        ));
    }
    if model.binding.regime_id != data.regime_id {
        return Err(mismatch(
            "measurement_model.regime_mismatch",
            &model.binding.regime_id,
            &data.regime_id,
        ));
    }
    if let ArmBinding::Arm(bound) = &model.binding.arm {
        if data.arm_id.as_deref() != Some(bound.as_str()) {
            return Err(mismatch(
                "measurement_model.arm_mismatch",
                bound,
                data.arm_id.as_deref().unwrap_or("all_arms"),
            ));
        }
    }
    if model.representation != data.representation {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("representation_mapping_required"),
            "measurement_model.representation_unmapped",
            format!(
                "the model's indicator is {}@{}, the data are {}@{}; a changed representation needs an explicit mapping and a model established for the new coordinates",
                model.representation.coordinate_system_id,
                model.representation.version,
                data.representation.coordinate_system_id,
                data.representation.version
            ),
        ));
    }
    Ok(())
}

pub(crate) fn require_route(
    model: &MeasurementModel,
    route: MeasurementRoute,
) -> Result<(), MeasurementError> {
    if model.route != route {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("route_not_supported"),
            "measurement_model.route_mismatch",
            format!(
                "the model declares route `{}`; this entry point implements `{}`",
                model.route.as_str(),
                route.as_str()
            ),
        ));
    }
    Ok(())
}

/// Refuse a model whose declared structure is not the one the entry point
/// implements.
pub(crate) fn require_non_differential(model: &MeasurementModel) -> Result<(), MeasurementError> {
    match model.structure {
        ErrorStructure::NonDifferential => Ok(()),
        _ => Err(MeasurementError::new(
            antecedent_core::reason_code!("differential_error_not_pooled"),
            "measurement_model.differential_declared",
            "the model is declared differential; this route pools the error across arms, populations and time and is never run on it",
        )),
    }
}

pub(crate) fn check_cancelled(ctx: &ExecutionContext) -> Result<(), MeasurementError> {
    if ctx.cancellation.is_cancelled() {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "measurement_model.cancelled",
            "cancelled before a corrected functional was formed; this is not a verdict on the data",
        ));
    }
    Ok(())
}

pub(crate) struct RecordParts<'a> {
    pub model: &'a MeasurementModel,
    pub functional_id: &'a str,
    pub point: f64,
    pub naive_point: Option<f64>,
    pub variance: Option<f64>,
    pub components: Vec<VarianceComponent>,
    pub rows: u64,
    pub data_identity: SemanticDigest,
    pub parameter_uncertainty: ParameterUncertainty,
}

pub(crate) fn build_record(parts: RecordParts<'_>) -> CorrectedFunctionalRecord {
    let standard_error = parts.variance.map(f64::sqrt);
    CorrectedFunctionalRecord {
        model_identity: parts.model.identities().model,
        route: parts.model.route,
        functional_id: parts.functional_id.to_owned(),
        point: parts.point,
        naive_point: parts.naive_point,
        standard_error,
        variance_components: parts.components,
        rows: parts.rows,
        data_identity: parts.data_identity,
        inference: if standard_error.is_some() {
            MeasurementInference::UncalibratedStandardError
        } else {
            MeasurementInference::PointOnly
        },
        parameter_uncertainty: parts.parameter_uncertainty,
    }
}

pub(crate) fn component(name: &str, variance: f64) -> VarianceComponent {
    VarianceComponent { name: name.to_owned(), variance }
}

pub(crate) fn digest(parts: &[(&str, &[f64])]) -> SemanticDigest {
    measurement_data_digest(parts)
}

pub(crate) fn counts_as_reals(counts: &[u32]) -> Vec<f64> {
    counts.iter().map(|&c| f64::from(c)).collect()
}

/// The refusal for a requested interval: `cell_not_licensed`. No interval
/// route is open.
#[must_use]
pub fn interval_not_licensed() -> MeasurementError {
    MeasurementError::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        "measurement_model.interval_closed",
        "no coverage record measures the whole measurement-plus-estimation method; the point and an uncalibrated standard error are the result",
    )
}
