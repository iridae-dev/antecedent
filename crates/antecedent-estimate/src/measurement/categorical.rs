//! Probabilistic categorical measurement: assigned labels with a declared
//! confusion model.
//!
//! The correction maps observed-label summaries to true-category proportions,
//! expected counts and outcome means. It consumes assigned labels and the
//! declared confusion matrix; per-unit calibrated probability vectors are not
//! consumed (a separate, closed route).

use antecedent_core::ExecutionContext;
use antecedent_core::identity::SemanticDigest;
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, MeasurementModel, MeasurementParameters, MeasurementRoute,
    ParameterUncertainty,
};
use antecedent_stats::measurement_error::{
    ConfusionMatrix, CorrectedCategorical, ObservedClassSummary, corrected_categories,
};

use super::{
    DataBinding, MeasurementError, RecordParts, build_record, check_binding, check_cancelled,
    component, digest, require_non_differential, require_route, stats_error,
};

/// A scalar functional of the corrected categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CategoricalFunctional<'a> {
    /// The proportion of one true category.
    Proportion(&'a str),
    /// The mean outcome among units in one true category.
    OutcomeMean(&'a str),
    /// The difference of outcome means `first - second`.
    OutcomeContrast {
        /// Minuend category.
        first: &'a str,
        /// Subtrahend category.
        second: &'a str,
    },
}

/// Corrected category proportions and outcome means, bound to the model.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedCategoricalOutcome {
    /// Corrected summaries with covariances.
    pub result: CorrectedCategorical,
    /// The observed-label summaries the correction consumed.
    pub classes: Vec<ObservedClassSummary>,
    /// True-category names, in axis order.
    pub levels: Vec<String>,
    /// Model identity the result is bound to.
    pub model_identity: SemanticDigest,
    /// Digest of the observed summaries.
    pub data_identity: SemanticDigest,
    /// Whether the confusion columns' uncertainty is in the variance.
    pub parameter_uncertainty: ParameterUncertainty,
    /// The model the result is bound to; records are built against it.
    model: MeasurementModel,
}

impl CorrectedCategoricalOutcome {
    fn index(&self, level: &str) -> Result<usize, MeasurementError> {
        self.levels.iter().position(|l| l == level).ok_or_else(|| {
            MeasurementError::new(
                antecedent_core::reason_code!("invalid_argument"),
                "measurement_model.unknown_level",
                format!("`{level}` is not a declared category of the confusion model"),
            )
        })
    }

    /// Value, uncorrected value and variance of one functional.
    ///
    /// # Errors
    /// An undeclared category refuses.
    pub fn functional(
        &self,
        functional: &CategoricalFunctional<'_>,
    ) -> Result<(f64, f64, f64), MeasurementError> {
        let k = self.levels.len();
        let r = &self.result;
        match *functional {
            CategoricalFunctional::Proportion(level) => {
                let i = self.index(level)?;
                Ok((
                    r.proportions[i],
                    r.observed_proportions[i],
                    r.proportion_covariance[i * k + i],
                ))
            }
            CategoricalFunctional::OutcomeMean(level) => {
                let i = self.index(level)?;
                Ok((r.outcome_means[i], self.classes[i].mean, r.outcome_mean_covariance[i * k + i]))
            }
            CategoricalFunctional::OutcomeContrast { first, second } => {
                let i = self.index(first)?;
                let j = self.index(second)?;
                let c = &r.outcome_mean_covariance;
                Ok((
                    r.outcome_means[i] - r.outcome_means[j],
                    self.classes[i].mean - self.classes[j].mean,
                    c[i * k + i] + c[j * k + j] - 2.0 * c[i * k + j],
                ))
            }
        }
    }

    /// The bound record of one functional.
    ///
    /// # Errors
    /// An undeclared category refuses.
    pub fn record(
        &self,
        functional: &CategoricalFunctional<'_>,
    ) -> Result<CorrectedFunctionalRecord, MeasurementError> {
        let (point, naive, variance) = self.functional(functional)?;
        let id = match functional {
            CategoricalFunctional::Proportion(_) => "category_proportion",
            CategoricalFunctional::OutcomeMean(_) => "category_outcome_mean",
            CategoricalFunctional::OutcomeContrast { .. } => "category_outcome_contrast",
        };
        let rows: u64 = self.classes.iter().map(|c| c.count).sum();
        Ok(build_record(RecordParts {
            model: &self.model,
            functional_id: id,
            point,
            naive_point: Some(naive),
            variance: Some(variance),
            components: vec![component("delta_method_total", variance)],
            rows,
            data_identity: self.data_identity,
            parameter_uncertainty: self.parameter_uncertainty,
        }))
    }
}

/// Correct observed-label summaries for a declared confusion model.
///
/// `classes[j]` summarizes the units assigned to category `j`, in the order of
/// the model's levels: how many, and the mean and sample variance of their
/// outcome.
///
/// # Errors
/// A malformed model, binding, route or structure mismatch, an observed class
/// with fewer than two units, a confusion matrix that is singular or worse
/// conditioned than the model's bound, or cancellation refuse.
pub fn correct_categorical(
    model: &MeasurementModel,
    binding: &DataBinding,
    classes: &[ObservedClassSummary],
    ctx: &ExecutionContext,
) -> Result<CorrectedCategoricalOutcome, MeasurementError> {
    check_cancelled(ctx)?;
    check_binding(model, binding)?;
    require_route(model, MeasurementRoute::ProbabilisticCategorical)?;
    require_non_differential(model)?;
    let MeasurementParameters::Categorical { levels, confusion, column_counts, max_condition } =
        &model.parameters
    else {
        return Err(MeasurementError::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            "measurement_model.invalid_model",
            "categorical route needs confusion-model parameters",
        ));
    };
    let matrix = ConfusionMatrix::new(levels.len(), confusion.clone(), column_counts.clone())
        .map_err(|e| stats_error(&e))?;
    let result =
        corrected_categories(classes, &matrix, *max_condition).map_err(|e| stats_error(&e))?;
    let flat: Vec<f64> =
        classes.iter().flat_map(|c| [c.count as f64, c.mean, c.variance]).collect();
    let uncertainty = if column_counts.is_some() {
        ParameterUncertainty::Propagated
    } else {
        ParameterUncertainty::DeclaredExact
    };
    Ok(CorrectedCategoricalOutcome {
        result,
        classes: classes.to_vec(),
        levels: levels.clone(),
        model_identity: model.identities().model,
        data_identity: digest(&[("observed_classes", &flat)]),
        parameter_uncertainty: uncertainty,
        model: model.clone(),
    })
}
