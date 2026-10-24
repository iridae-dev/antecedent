//! A logistic g-computation run through admitted recipes.
//!
//! The outcome model is a logistic regression on `[1 | T | Z...]`. It is fitted by the
//! generalized-linear recipe: the provider must declare a differentiable log density and
//! estimating equations, its score is checked against the gradient of its log density, and
//! the Newton solve is the recipe's. The standardized means at `T = 1` and `T = 0` are the
//! frequentist g-computation recipe's plug-in means, evaluated by a prediction-only
//! boundary provider at the fitted coefficients. The effect is their difference.
//!
//! A provider that lacks a capability, or whose body does not answer the log density, is
//! refused before any row is read.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    ExecutionContext, InferenceSelection, ProviderCapability, ProviderDeclaration, ProviderId,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_learn::recipes::execution::{
    EstimatingEquationModel, ExecutionError, LogDensityModel, PredictiveModel, ProviderCallError,
    RecipeProvider, SolveOptions,
};
use antecedent_learn::recipes::{CoherentComponent, Recipe, RecipeKind, StudyDefault};

use crate::error::CausalError;

/// Position of the treatment in the design `[1 | T | Z...]`.
const TREATMENT_COLUMN: usize = 1;
/// Relative disagreement allowed between the provider's score and the log-density gradient.
const GRADIENT_TOLERANCE: f64 = 1e-5;

/// A standardized logistic effect and the fit behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct RecipeAte {
    /// Standardized mean outcome at `T = 1` minus at `T = 0`.
    pub effect: f64,
    /// Standardized mean outcome at `T = 1`.
    pub treated_mean: f64,
    /// Standardized mean outcome at `T = 0`.
    pub control_mean: f64,
    /// Fitted coefficients on `[1 | T | Z...]`.
    pub coefficients: Vec<f64>,
    /// Row-major sandwich covariance of the coefficients.
    pub covariance: Vec<f64>,
    /// Newton iterations the solve took.
    pub iterations: usize,
    /// Rows the fit and the means averaged.
    pub rows: u64,
}

/// The provider declaration the logistic fit requires, with its log density answered.
///
/// # Errors
///
/// `name` is blank.
pub fn logistic_provider(id: u64, name: &str) -> Result<ProviderDeclaration, CausalError> {
    Ok(ProviderDeclaration::new(
        ProviderId::from_raw(id),
        name,
        [ProviderCapability::DifferentiableLogDensity, ProviderCapability::EstimatingEquations],
    )
    .map_err(|error| compile(error.to_string()))?
    .answering([ProviderCapability::DifferentiableLogDensity]))
}

struct Logistic<'a> {
    declaration: &'a ProviderDeclaration,
    design: &'a [f64],
    response: &'a [f64],
    width: usize,
}

impl Logistic<'_> {
    fn linear_predictor(&self, beta: &[f64], row: u64) -> f64 {
        let start = usize::try_from(row).unwrap_or(usize::MAX).saturating_mul(self.width);
        self.design[start..start + self.width].iter().zip(beta).map(|(x, b)| x * b).sum()
    }
}

impl RecipeProvider for Logistic<'_> {
    fn declaration(&self) -> &ProviderDeclaration {
        self.declaration
    }
}

impl LogDensityModel for Logistic<'_> {
    fn parameter_len(&self) -> usize {
        self.width
    }

    fn log_density(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError> {
        let eta = self.linear_predictor(parameter, row);
        let y = self.response[usize::try_from(row).unwrap_or(usize::MAX)];
        // y * eta - log(1 + exp(eta)), stable for large |eta|.
        Ok(y.mul_add(eta, -(eta.max(0.0) + (-eta.abs()).exp().ln_1p())))
    }
}

impl EstimatingEquationModel for Logistic<'_> {
    fn parameter_len(&self) -> usize {
        self.width
    }

    fn score(&self, parameter: &[f64], row: u64, out: &mut [f64]) -> Result<(), ProviderCallError> {
        let i = usize::try_from(row).unwrap_or(usize::MAX);
        let probability = sigmoid(self.linear_predictor(parameter, row));
        let residual = self.response[i] - probability;
        for (slot, x) in out.iter_mut().zip(&self.design[i * self.width..(i + 1) * self.width]) {
            *slot = residual * x;
        }
        Ok(())
    }
}

/// Prediction at a fixed treatment level: the boundary the plug-in mean reads.
struct Standardized<'a> {
    declaration: &'a ProviderDeclaration,
    design: &'a [f64],
    width: usize,
    level: f64,
}

impl RecipeProvider for Standardized<'_> {
    fn declaration(&self) -> &ProviderDeclaration {
        self.declaration
    }
}

impl PredictiveModel for Standardized<'_> {
    fn parameter_len(&self) -> usize {
        self.width
    }

    fn predict(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError> {
        let start = usize::try_from(row).unwrap_or(usize::MAX).saturating_mul(self.width);
        let eta: f64 = self.design[start..start + self.width]
            .iter()
            .enumerate()
            .map(|(column, x)| {
                parameter[column] * if column == TREATMENT_COLUMN { self.level } else { *x }
            })
            .sum();
        Ok(sigmoid(eta))
    }
}

fn sigmoid(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

fn compile(message: impl Into<String>) -> CausalError {
    CausalError::Compile { message: message.into() }
}

fn lower(error: ExecutionError) -> CausalError {
    match error {
        ExecutionError::Cancelled { .. } => CausalError::Cancelled { stage: "recipe" },
        other => compile(other.to_string()),
    }
}

fn column(data: &TabularData, id: VariableId) -> Result<Vec<f64>, CausalError> {
    let values = data.float64_values(id).map_err(|error| compile(error.to_string()))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(compile("the recipe fit reads complete numeric rows; a value is missing"));
    }
    Ok(values)
}

/// Fit the logistic outcome model with the generalized-linear recipe and average its
/// predictions at `T = 1` and `T = 0` with the g-computation recipe.
///
/// `fitter` is bound to both recipes, so it must declare a differentiable log density and
/// estimating equations and answer its log density.
///
/// # Errors
///
/// The provider lacks a declared or answered capability, a column is missing, not numeric or
/// has a missing value, the response is not 0/1, the score disagrees with the log-density
/// gradient, the solve fails to converge, or `ctx` is cancelled.
pub fn logistic_standardized_effect(
    data: &TabularData,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: &[VariableId],
    fitter: &ProviderDeclaration,
    ctx: &ExecutionContext,
) -> Result<RecipeAte, CausalError> {
    let response = column(data, outcome)?;
    if response.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(compile("a logistic recipe fit needs a 0/1 response"));
    }
    let rows = response.len();
    let width = adjustment.len() + 2;
    let mut design = vec![1.0; rows * width];
    for (offset, id) in std::iter::once(treatment).chain(adjustment.iter().copied()).enumerate() {
        for (row, value) in column(data, id)?.into_iter().enumerate() {
            design[row * width + 1 + offset] = value;
        }
    }
    let indices: Vec<u64> = (0..rows as u64).collect();
    let component = CoherentComponent::new("outcome", None, &indices)
        .map_err(|error| compile(error.to_string()))?;
    let study = StudyDefault::new(InferenceSelection::Frequentist)
        .map_err(|error| compile(error.to_string()))?;
    let glm = Recipe::new(RecipeKind::Glm, InferenceSelection::Frequentist)
        .and_then(|recipe| recipe.bind(&component, study, fitter))
        .map_err(|error| compile(error.to_string()))?;
    let plug_in = Recipe::new(RecipeKind::GComputation, InferenceSelection::Frequentist)
        .and_then(|recipe| recipe.bind(&component, study, fitter))
        .map_err(|error| compile(error.to_string()))?;

    let model = Logistic { declaration: fitter, design: &design, response: &response, width };
    let run = glm.executor().with_cancellation(&ctx.cancellation);
    run.verify_score_gradient(&model, &vec![0.0; width], GRADIENT_TOLERANCE).map_err(lower)?;
    let solution = run
        .solve_estimating_equations(&model, &vec![0.0; width], SolveOptions::default())
        .map_err(lower)?;

    let boundary = ProviderDeclaration::new(
        ProviderId::from_raw(fitter.id().raw().wrapping_add(1)),
        "standardized prediction",
        [ProviderCapability::Prediction],
    )
    .map_err(|error| compile(error.to_string()))?;
    let at = |level: f64| -> Result<f64, CausalError> {
        let predictor = Standardized { declaration: &boundary, design: &design, width, level };
        plug_in
            .executor()
            .with_cancellation(&ctx.cancellation)
            .plug_in_mean(&predictor, solution.estimate())
            .map_err(lower)
    };
    let treated_mean = at(1.0)?;
    let control_mean = at(0.0)?;
    Ok(RecipeAte {
        effect: treated_mean - control_mean,
        treated_mean,
        control_mean,
        coefficients: solution.estimate().to_vec(),
        covariance: solution.covariance().to_vec(),
        iterations: solution.iterations(),
        rows: solution.rows(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial() -> (TabularData, VariableId, VariableId) {
        // 100 control rows with 30 responders, 100 treated rows with 60.
        let treatment: Vec<f64> = (0..200).map(|i| f64::from(i >= 100)).collect();
        let response: Vec<f64> = (0..200)
            .map(|i| {
                let within = i % 100;
                let responders = if i < 100 { 30 } else { 60 };
                f64::from(within < responders)
            })
            .collect();
        let data =
            TabularData::from_f64_columns([("t", &treatment[..]), ("y", &response[..])]).unwrap();
        (data, VariableId::from_raw(0), VariableId::from_raw(1))
    }

    #[test]
    fn the_recipe_fit_recovers_the_two_by_two_log_odds_and_risk_difference() {
        let (data, t, y) = trial();
        let ctx = ExecutionContext::for_tests(1);
        let provider = logistic_provider(1, "logistic").unwrap();
        let fit = logistic_standardized_effect(&data, t, y, &[], &provider, &ctx).unwrap();
        let logit = |p: f64| (p / (1.0 - p)).ln();
        assert!((fit.coefficients[0] - logit(0.3)).abs() < 1e-8);
        assert!((fit.coefficients[1] - (logit(0.6) - logit(0.3))).abs() < 1e-8);
        assert!((fit.treated_mean - 0.6).abs() < 1e-9);
        assert!((fit.control_mean - 0.3).abs() < 1e-9);
        assert!((fit.effect - 0.3).abs() < 1e-9);
        assert_eq!(fit.rows, 200);
        // Var(beta_1) = 1/(n1 p1 (1-p1)) + 1/(n0 p0 (1-p0)) for the saturated model.
        let variance = 1.0 / (100.0 * 0.6 * 0.4) + 1.0 / (100.0 * 0.3 * 0.7);
        assert!((fit.covariance[3] - variance).abs() < 1e-6 * variance, "{}", fit.covariance[3]);
    }

    #[test]
    fn a_provider_that_lacks_a_capability_or_does_not_answer_its_density_is_refused() {
        let (data, t, y) = trial();
        let ctx = ExecutionContext::for_tests(1);
        let missing = ProviderDeclaration::new(
            ProviderId::from_raw(2),
            "density only",
            [ProviderCapability::DifferentiableLogDensity],
        )
        .unwrap()
        .answering([ProviderCapability::DifferentiableLogDensity]);
        assert!(logistic_standardized_effect(&data, t, y, &[], &missing, &ctx).is_err());
        let unanswered = ProviderDeclaration::new(
            ProviderId::from_raw(3),
            "declared only",
            [ProviderCapability::DifferentiableLogDensity, ProviderCapability::EstimatingEquations],
        )
        .unwrap();
        assert!(logistic_standardized_effect(&data, t, y, &[], &unanswered, &ctx).is_err());
    }

    #[test]
    fn cancellation_stops_the_fit() {
        let (data, t, y) = trial();
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let provider = logistic_provider(1, "logistic").unwrap();
        let error = logistic_standardized_effect(&data, t, y, &[], &provider, &ctx).unwrap_err();
        assert!(matches!(error, CausalError::Cancelled { .. }), "{error:?}");
    }
}
