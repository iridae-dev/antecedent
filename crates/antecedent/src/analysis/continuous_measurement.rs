//! Continuous classical measurement error through a prepared measurement
//! model: regression calibration and SIMEX for linear and logistic outcome
//! models, with the error in the treatment, a covariate or the outcome.
//!
//! [`PreparedMeasurementModel::estimate_continuous`] re-runs the prepare
//! checks of the linear adjusted effect (construct, binding, structure,
//! representation) on the prepared model, then forms the correction of
//! `antecedent_estimate::measurement::continuous`. A logistic outcome model
//! enters through this entry point only: the nonlinear outcome model
//! operation stays refused by
//! [`PreparedMeasurementModel::check_operation`], and the route card names
//! this entry point as the next attempt.
//!
//! The result is uncalibrated: a point and a standard error (delta method for
//! linear calibration, a bootstrap of the whole procedure for logistic
//! calibration and SIMEX), no interval. There is no artifact layout for it:
//! [`ContinuousMeasurementEstimate::export`] refuses rather than write a
//! format no consumer can replay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::measurement::MeasurementOperation;
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, MeasurementInference, MeasurementModel, ParameterUncertainty,
};
pub use antecedent_estimate::measurement::continuous::{
    ContinuousCorrectionSpec, ContinuousCorrectionSummary, ContinuousMethodKind,
    CorrectedContinuousEffect, CorrectionMethod, DEFAULT_BOOTSTRAP_REPLICATES, ErrorLocation,
    MAX_BOOTSTRAP_REPLICATES, MIN_BOOTSTRAP_REPLICATES, OutcomeLink,
    SUPPLY_REPLICATES_OR_VALIDATION, SimexExtrapolant, SimexFit, USE_BINARY_MISCLASSIFICATION,
};
use antecedent_estimate::measurement::{
    ContinuousErrorData, DataBinding, MeasurementError, correct_continuous_error,
    interval_not_licensed,
};

use super::measurement_model::{
    MeasurementAnalysisBinding, MeasurementModelRefusal, PreparedMeasurementModel,
};

/// The data and the question of one continuous correction.
///
/// The design is `(1, regressor, covariates)`. With the error in the
/// treatment or a covariate, `regressor` is the error-prone column; with the
/// error in the outcome, `regressor` is the error-free treatment and `outcome`
/// is the error-prone column. When `replicate_values` is given, the
/// error-prone column is the row means of the replicates (a supplied column
/// may be left empty and must otherwise equal them).
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousErrorInput {
    /// Outcome.
    pub outcome: Vec<f64>,
    /// Regressor at coefficient index 1.
    pub regressor: Vec<f64>,
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: Vec<f64>,
    /// Number of covariate columns.
    pub covariate_count: usize,
    /// Every row's measurements of the error-prone column, back to back.
    pub replicate_values: Option<Vec<f64>>,
    /// Measurements averaged into each row of the error-prone column.
    pub replicate_counts: Option<Vec<u32>>,
    /// Location, link, method, bootstrap replicates and seed.
    pub spec: ContinuousCorrectionSpec,
}

/// A continuous correction bound to its model.
#[derive(Clone, Debug)]
pub struct ContinuousMeasurementEstimate {
    model: MeasurementModel,
    binding: MeasurementAnalysisBinding,
    effect: CorrectedContinuousEffect,
}

fn refusal_of(error: &MeasurementError) -> MeasurementModelRefusal {
    let remedy = match error.detail {
        "measurement_model.error_variance_not_identified"
        | "measurement_model.insufficient_replicates" => {
            Some(SUPPLY_REPLICATES_OR_VALIDATION.sentence().to_owned())
        }
        "measurement_model.binary_outcome_continuous_error" => {
            Some(USE_BINARY_MISCLASSIFICATION.sentence().to_owned())
        }
        _ => None,
    };
    MeasurementModelRefusal {
        code: error.code,
        detail: error.detail.to_owned(),
        message: error.message.clone(),
        remedy,
    }
}

impl PreparedMeasurementModel {
    /// Correct one coefficient for classical additive error in a continuous
    /// column of `input`.
    ///
    /// The prepare checks of the linear adjusted effect run again on this
    /// model and binding (for a logit link too: the corrected logistic
    /// coefficient enters through this entry point and nowhere else), then
    /// the estimator checks the binding, the declared structure, that the
    /// route identifies an error variance (classical or repeated) and the
    /// columns.
    ///
    /// # Errors
    /// The contract refusal of the linear adjusted effect; the estimator's
    /// coded refusals, among them `measurement_not_identified` when the error
    /// variance is not identified (remedy: supply replicates or a validation
    /// sample), `route_not_supported` for a logit outcome model with the
    /// error in the outcome (remedy: the binary misclassification route), an
    /// implied reliability at or below the floor, a bootstrap that cannot be
    /// funded or is unstable; or cancellation.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "preserve the established public signature that takes the input by value"
    )]
    pub fn estimate_continuous(
        &self,
        input: ContinuousErrorInput,
        ctx: &ExecutionContext,
    ) -> Result<ContinuousMeasurementEstimate, MeasurementModelRefusal> {
        if ctx.cancellation.is_cancelled() {
            return Err(refusal_of(&MeasurementError {
                code: antecedent_core::reason_code!("cancelled_no_claim"),
                detail: "measurement_model.cancelled",
                message: "cancelled before a corrected functional was formed; this is not a verdict on the data"
                    .to_owned(),
            }));
        }
        self.check_operation(MeasurementOperation::LinearAdjustedEffect)?;
        let binding = self.binding();
        // The data are in the model's coordinates, as for `estimate`.
        let data_binding = DataBinding {
            population_id: binding.population_id.clone(),
            regime_id: binding.regime_id.clone(),
            arm_id: binding.arm_id.clone(),
            representation: self.model().representation.clone(),
        };
        let data = ContinuousErrorData {
            outcome: &input.outcome,
            regressor: &input.regressor,
            covariates: &input.covariates,
            covariate_count: input.covariate_count,
            replicate_values: input.replicate_values.as_deref(),
            replicate_counts: input.replicate_counts.as_deref(),
        };
        let effect = correct_continuous_error(self.model(), &data_binding, &data, &input.spec, ctx)
            .map_err(|error| refusal_of(&error))?;
        Ok(ContinuousMeasurementEstimate {
            model: self.model().clone(),
            binding: binding.clone(),
            effect,
        })
    }
}

impl ContinuousMeasurementEstimate {
    /// The corrected coefficient, bound to the model identity.
    #[must_use]
    pub const fn record(&self) -> &CorrectedFunctionalRecord {
        &self.effect.record
    }

    /// The estimator's full result: coefficients, SIMEX path, bootstrap
    /// estimates and summary.
    #[must_use]
    pub const fn effect(&self) -> &CorrectedContinuousEffect {
        &self.effect
    }

    /// What the correction used: method, link, location, error variance,
    /// reliability and bootstrap replicates.
    #[must_use]
    pub const fn summary(&self) -> &ContinuousCorrectionSummary {
        &self.effect.summary
    }

    /// The model the result is bound to.
    #[must_use]
    pub const fn model(&self) -> &MeasurementModel {
        &self.model
    }

    /// The analysis binding the model was checked against.
    #[must_use]
    pub const fn binding(&self) -> &MeasurementAnalysisBinding {
        &self.binding
    }

    /// The corrected point value of the treatment coefficient.
    #[must_use]
    pub const fn point(&self) -> f64 {
        self.effect.record.point
    }

    /// The uncorrected value of the same coefficient.
    #[must_use]
    pub const fn naive_point(&self) -> Option<f64> {
        self.effect.record.naive_point
    }

    /// The uncalibrated standard error.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.effect.record.standard_error
    }

    /// What the uncertainty means.
    #[must_use]
    pub const fn inference(&self) -> MeasurementInference {
        self.effect.record.inference
    }

    /// Whether the measurement parameters' uncertainty is in the variance.
    #[must_use]
    pub const fn parameter_uncertainty(&self) -> ParameterUncertainty {
        self.effect.record.parameter_uncertainty
    }

    /// An interval for the corrected coefficient.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `measurement_model.interval_closed`.
    pub fn interval(&self) -> Result<std::convert::Infallible, MeasurementModelRefusal> {
        Err(refusal_of(&interval_not_licensed()))
    }

    /// Artifact bytes of the result.
    ///
    /// # Errors
    /// Always `route_not_supported`: the measurement-model artifact replays
    /// sufficient moments, and a bootstrap or SIMEX result has none a consumer
    /// could recompute; no layout is invented for it.
    pub fn export(&self, _artifact_id: &str) -> Result<Vec<u8>, MeasurementModelRefusal> {
        Err(MeasurementModelRefusal {
            code: antecedent_core::reason_code!("route_not_supported"),
            detail: "measurement_model.continuous_export".to_owned(),
            message: "a continuous measurement-error correction has no artifact layout: its \
                      bootstrap and SIMEX results carry no sufficient inputs an independent \
                      consumer could recompute"
                .to_owned(),
            remedy: None,
        })
    }
}
