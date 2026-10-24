// SPDX-License-Identifier: MIT OR Apache-2.0

//! The named recipes run: a bound recipe evaluates a likelihood, updates a
//! posterior, solves estimating equations, forms an orthogonal score and
//! integrates exactly. Each result is checked against an independent closed
//! form, and each refusal against the capability or provider it names.

use std::sync::atomic::{AtomicUsize, Ordering};

use antecedent_core::{
    CancellationToken, InferenceSelection, ProviderCapability as Cap, ProviderDeclaration,
    ProviderId,
};
use antecedent_learn::recipes::execution::{
    EquationSolution, EstimatingEquationModel, ExactIntegrationModel, ExecutionError,
    InfluenceModel, LogDensityModel, PredictiveModel, ProviderCallError, RecipeProvider,
    SolveOptions, UpdatingModel,
};
use antecedent_learn::recipes::{
    BoundRecipe, CoherentComponent, Recipe, RecipeError, RecipeKind, StudyDefault,
};

fn declare(id: u64, name: &str, capabilities: &[Cap], answers: &[Cap]) -> ProviderDeclaration {
    ProviderDeclaration::new(ProviderId::from_raw(id), name, capabilities.iter().copied())
        .unwrap()
        .answering(answers.iter().copied())
}

fn bind<'a>(
    kind: RecipeKind,
    inference: InferenceSelection,
    rows: &'a [u64],
    provider: &ProviderDeclaration,
) -> BoundRecipe<'a> {
    let component = Box::leak(Box::new(CoherentComponent::new("outcome", None, rows).unwrap()));
    let study = StudyDefault::new(inference).unwrap();
    Recipe::new(kind, inference).unwrap().bind(component, study, provider).unwrap()
}

fn rows(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

/// Gaussian mean with unit variance and a zero-mean normal prior, plus a covariate
/// for prediction.
struct GaussianMean {
    declaration: ProviderDeclaration,
    data: Vec<f64>,
    covariate: Vec<f64>,
    prior_sd: f64,
    calls: AtomicUsize,
}

impl GaussianMean {
    fn new(declaration: ProviderDeclaration, n: usize) -> Self {
        let data = (0..n).map(|i| 0.3 + 0.8 * (i as f64).sin()).collect();
        let covariate = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
        Self { declaration, data, covariate, prior_sd: 10.0, calls: AtomicUsize::new(0) }
    }
}

impl RecipeProvider for GaussianMean {
    fn declaration(&self) -> &ProviderDeclaration {
        &self.declaration
    }
}

impl LogDensityModel for GaussianMean {
    fn parameter_len(&self) -> usize {
        1
    }

    fn log_density(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let x = self.data[row as usize];
        Ok(-0.5 * (x - parameter[0]).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
    }
}

impl UpdatingModel for GaussianMean {
    fn log_prior(&self, parameter: &[f64]) -> Result<f64, ProviderCallError> {
        Ok(-0.5 * (parameter[0] / self.prior_sd).powi(2))
    }
}

impl PredictiveModel for GaussianMean {
    fn parameter_len(&self) -> usize {
        1
    }

    fn predict(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError> {
        Ok(parameter[0] + self.covariate[row as usize])
    }
}

impl EstimatingEquationModel for GaussianMean {
    fn parameter_len(&self) -> usize {
        1
    }

    fn score(&self, parameter: &[f64], row: u64, out: &mut [f64]) -> Result<(), ProviderCallError> {
        out[0] = self.data[row as usize] - parameter[0];
        Ok(())
    }
}

fn grid(low: f64, high: f64, step: f64) -> Vec<Vec<f64>> {
    let count = ((high - low) / step).round() as usize;
    (0..=count).map(|k| vec![low + step * k as f64]).collect()
}

#[test]
fn bayesian_update_matches_the_conjugate_posterior() {
    let n = 40;
    let provider =
        declare(1, "gaussian", &[Cap::Density, Cap::Updating], &[Cap::Density, Cap::Updating]);
    let model = GaussianMean::new(provider.clone(), n);
    let rows = rows(n);
    let bound = bind(RecipeKind::LinearGaussian, InferenceSelection::Bayesian, &rows, &provider);
    let run = bound.executor();

    let theta = 0.5;
    let log_likelihood = run.log_likelihood(&model, &[theta]).unwrap();
    let direct: f64 = model
        .data
        .iter()
        .map(|x| -0.5 * (x - theta).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
        .sum();
    assert!((log_likelihood - direct).abs() < 1e-9);

    let posterior = run.update(&model, &grid(-5.0, 5.0, 0.005)).unwrap();
    let precision = n as f64 + 1.0 / model.prior_sd.powi(2);
    let mean = model.data.iter().sum::<f64>() / precision;
    assert!((posterior.mean(0) - mean).abs() < 1e-6, "{} vs {mean}", posterior.mean(0));
    let variance = posterior.expectation(|p| (p[0] - mean).powi(2));
    assert!((variance - 1.0 / precision).abs() < 1e-6);
    assert!((posterior.weights().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(!posterior.exact(), "a quadrature grid is not an exact integral");
}

#[test]
fn a_body_that_is_not_the_bound_provider_or_lost_a_capability_is_refused() {
    let n = 10;
    let provider =
        declare(1, "gaussian", &[Cap::Density, Cap::Updating], &[Cap::Density, Cap::Updating]);
    let rows = rows(n);
    let run =
        bind(RecipeKind::LinearGaussian, InferenceSelection::Bayesian, &rows, &provider).executor();

    let other = GaussianMean::new(
        declare(2, "other", &[Cap::Density, Cap::Updating], &[Cap::Density, Cap::Updating]),
        n,
    );
    assert!(matches!(
        run.log_likelihood(&other, &[0.0]),
        Err(ExecutionError::ProviderMismatch { .. })
    ));

    let lost = GaussianMean::new(declare(1, "gaussian", &[Cap::Density], &[Cap::Density]), n);
    assert!(matches!(
        run.log_likelihood(&lost, &[0.0]),
        Err(ExecutionError::Recipe(RecipeError::MissingCapability { .. }))
    ));

    let unanswered = GaussianMean::new(
        declare(1, "gaussian", &[Cap::Density, Cap::Updating], &[Cap::Density]),
        n,
    );
    assert!(matches!(
        run.log_likelihood(&unanswered, &[0.0]),
        Err(ExecutionError::Recipe(RecipeError::UnansweredCall { .. }))
    ));

    let model = GaussianMean::new(provider.clone(), n);
    assert!(matches!(
        run.log_likelihood(&model, &[0.0, 1.0]),
        Err(ExecutionError::Dimension { .. })
    ));
    assert!(matches!(run.update(&model, &[]), Err(ExecutionError::EmptySupport)));
}

#[test]
fn cancellation_stops_the_row_loop_before_the_provider_is_called_again() {
    let n = 40;
    let provider =
        declare(1, "gaussian", &[Cap::Density, Cap::Updating], &[Cap::Density, Cap::Updating]);
    let model = GaussianMean::new(provider.clone(), n);
    let rows = rows(n);
    let bound = bind(RecipeKind::LinearGaussian, InferenceSelection::Bayesian, &rows, &provider);

    let token = CancellationToken::new();
    token.cancel();
    let refused = bound.executor().with_cancellation(&token).log_likelihood(&model, &[0.0]);
    assert!(matches!(refused, Err(ExecutionError::Cancelled { .. })));
    assert_eq!(model.calls.load(Ordering::Relaxed), 0);

    let late = CancellationToken::cancel_after_checks(5);
    let refused = bound.executor().with_cancellation(&late).log_likelihood(&model, &[0.0]);
    assert!(matches!(refused, Err(ExecutionError::Cancelled { .. })));
    let calls = model.calls.load(Ordering::Relaxed);
    assert!((1..n).contains(&calls), "stopped after {calls} of {n} rows");
}

/// Simple linear regression with the score `(y - a - b x) * (1, x)`.
struct Regression {
    declaration: ProviderDeclaration,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Regression {
    fn new(declaration: ProviderDeclaration, n: usize) -> Self {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 5.0).collect();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, x)| {
                1.0 + 2.0 * x + 0.3 * (3.0 * i as f64).sin() + 0.2 * (7.0 * i as f64).cos()
            })
            .collect();
        Self { declaration, x, y }
    }
}

impl RecipeProvider for Regression {
    fn declaration(&self) -> &ProviderDeclaration {
        &self.declaration
    }
}

impl EstimatingEquationModel for Regression {
    fn parameter_len(&self) -> usize {
        2
    }

    fn score(&self, parameter: &[f64], row: u64, out: &mut [f64]) -> Result<(), ProviderCallError> {
        let i = row as usize;
        let residual = self.y[i] - parameter[0] - parameter[1] * self.x[i];
        out[0] = residual;
        out[1] = residual * self.x[i];
        Ok(())
    }
}

fn ordinary_least_squares(model: &Regression) -> ([f64; 2], [f64; 4]) {
    let n = model.x.len() as f64;
    let sx: f64 = model.x.iter().sum();
    let sxx: f64 = model.x.iter().map(|x| x * x).sum();
    let sy: f64 = model.y.iter().sum();
    let sxy: f64 = model.x.iter().zip(&model.y).map(|(x, y)| x * y).sum();
    let det = n * sxx - sx * sx;
    let slope = (n * sxy - sx * sy) / det;
    let intercept = (sy - slope * sx) / n;
    let inverse = [sxx / det, -sx / det, -sx / det, n / det];
    let mut meat = [0.0; 4];
    for (x, y) in model.x.iter().zip(&model.y) {
        let e = y - intercept - slope * x;
        meat[0] += e * e;
        meat[1] += e * e * x;
        meat[2] += e * e * x;
        meat[3] += e * e * x * x;
    }
    let mul = |a: [f64; 4], b: [f64; 4]| {
        [
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ]
    };
    ([intercept, slope], mul(mul(inverse, meat), inverse))
}

#[test]
fn estimating_equations_recover_least_squares_and_the_sandwich_covariance() {
    let n = 30;
    let provider = declare(3, "regression", &[Cap::EstimatingEquations], &[]);
    let model = Regression::new(provider.clone(), n);
    let rows = rows(n);
    let run = bind(RecipeKind::LinearGaussian, InferenceSelection::Frequentist, &rows, &provider)
        .executor();

    let solution: EquationSolution =
        run.solve_estimating_equations(&model, &[0.0, 0.0], SolveOptions::default()).unwrap();
    let (estimate, covariance) = ordinary_least_squares(&model);
    for (found, expected) in solution.estimate().iter().zip(estimate) {
        assert!((found - expected).abs() < 1e-8, "{found} vs {expected}");
    }
    for (found, expected) in solution.covariance().iter().zip(covariance) {
        assert!(
            (found - expected).abs() < 1e-6 * expected.abs().max(1e-3),
            "{found} vs {expected}"
        );
    }
    assert!(solution.iterations() <= 3);
    assert!(solution.score_norm() <= 1e-10);
    assert_eq!(solution.rows(), n as u64);
    assert!(solution.standard_error(1).unwrap() > 0.0);
    assert!(solution.standard_error(2).is_none());

    // A Bayesian recipe does not run a frequentist solve.
    let density =
        declare(4, "density", &[Cap::Density, Cap::Updating], &[Cap::Density, Cap::Updating]);
    let bayes =
        bind(RecipeKind::LinearGaussian, InferenceSelection::Bayesian, &rows, &density).executor();
    assert!(matches!(
        bayes.solve_estimating_equations(&model, &[0.0, 0.0], SolveOptions::default()),
        Err(ExecutionError::NotApplicable { .. })
    ));
}

/// Mean equation whose second coordinate carries no information.
struct Degenerate {
    declaration: ProviderDeclaration,
}

impl RecipeProvider for Degenerate {
    fn declaration(&self) -> &ProviderDeclaration {
        &self.declaration
    }
}

impl EstimatingEquationModel for Degenerate {
    fn parameter_len(&self) -> usize {
        2
    }

    fn score(&self, parameter: &[f64], row: u64, out: &mut [f64]) -> Result<(), ProviderCallError> {
        out[0] = row as f64 - parameter[0];
        out[1] = 0.0;
        Ok(())
    }
}

#[test]
fn a_singular_jacobian_and_an_unconverged_solve_are_typed_refusals() {
    let provider = declare(3, "degenerate", &[Cap::EstimatingEquations], &[]);
    let rows = rows(6);
    let run = bind(RecipeKind::LinearGaussian, InferenceSelection::Frequentist, &rows, &provider)
        .executor();
    let model = Degenerate { declaration: provider };
    assert!(matches!(
        run.solve_estimating_equations(&model, &[0.0, 0.0], SolveOptions::default()),
        Err(ExecutionError::Singular { .. })
    ));
    let regression = Regression::new(model.declaration.clone(), 12);
    let zero_budget = SolveOptions { tolerance: 1e-30, max_iterations: 1 };
    assert!(matches!(
        run.solve_estimating_equations(&regression, &[0.0, 0.0], zero_budget),
        Err(ExecutionError::NotConverged { .. })
    ));
}

/// Logistic regression: a differentiable log density and its score.
struct Logistic {
    declaration: ProviderDeclaration,
    x: Vec<f64>,
    y: Vec<f64>,
    flip_score: bool,
}

impl Logistic {
    fn new(declaration: ProviderDeclaration, n: usize, flip_score: bool) -> Self {
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 5.0 - 1.0).collect();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let p = 1.0 / (1.0 + (-(-0.5 + 1.2 * x)).exp());
                let u = (i as f64 * 0.618_033_988_7).fract();
                f64::from(u8::from(u < p))
            })
            .collect();
        Self { declaration, x, y, flip_score }
    }

    fn probability(parameter: &[f64], x: f64) -> f64 {
        1.0 / (1.0 + (-(parameter[0] + parameter[1] * x)).exp())
    }
}

impl RecipeProvider for Logistic {
    fn declaration(&self) -> &ProviderDeclaration {
        &self.declaration
    }
}

impl LogDensityModel for Logistic {
    fn parameter_len(&self) -> usize {
        2
    }

    fn log_density(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError> {
        let i = row as usize;
        let p = Self::probability(parameter, self.x[i]);
        Ok(self.y[i] * p.ln() + (1.0 - self.y[i]) * (1.0 - p).ln())
    }
}

impl EstimatingEquationModel for Logistic {
    fn parameter_len(&self) -> usize {
        2
    }

    fn score(&self, parameter: &[f64], row: u64, out: &mut [f64]) -> Result<(), ProviderCallError> {
        let i = row as usize;
        let sign = if self.flip_score { -1.0 } else { 1.0 };
        let residual = sign * (self.y[i] - Self::probability(parameter, self.x[i]));
        out[0] = residual;
        out[1] = residual * self.x[i];
        Ok(())
    }
}

#[test]
fn glm_fit_solves_the_score_and_the_score_must_be_the_log_density_gradient() {
    let n = 200;
    let capabilities = [Cap::DifferentiableLogDensity, Cap::EstimatingEquations];
    let provider = declare(5, "logistic", &capabilities, &[Cap::DifferentiableLogDensity]);
    let rows = rows(n);
    let run = bind(RecipeKind::Glm, InferenceSelection::Frequentist, &rows, &provider).executor();
    let model = Logistic::new(provider.clone(), n, false);

    run.verify_score_gradient(&model, &[0.2, -0.4], 1e-5).unwrap();
    let solution =
        run.solve_estimating_equations(&model, &[0.0, 0.0], SolveOptions::default()).unwrap();
    assert!(solution.score_norm() <= 1e-10);
    // The root maximizes the summed log density: the gradient there is zero.
    let at_root = |shift: [f64; 2]| -> f64 {
        let theta = [solution.estimate()[0] + shift[0], solution.estimate()[1] + shift[1]];
        (0..n as u64).map(|row| model.log_density(&theta, row).unwrap()).sum()
    };
    let best = at_root([0.0, 0.0]);
    for shift in [[0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05]] {
        assert!(at_root(shift) < best);
    }

    let wrong = Logistic::new(provider, n, true);
    assert!(matches!(
        run.verify_score_gradient(&wrong, &[0.2, -0.4], 1e-5),
        Err(ExecutionError::ScoreIsNotGradient { .. })
    ));
}

/// Orthogonal-score contributions.
struct Contributions {
    declaration: ProviderDeclaration,
    values: Vec<f64>,
}

impl RecipeProvider for Contributions {
    fn declaration(&self) -> &ProviderDeclaration {
        &self.declaration
    }
}

impl InfluenceModel for Contributions {
    fn contribution(&self, row: u64) -> Result<f64, ProviderCallError> {
        Ok(self.values[row as usize])
    }
}

#[test]
fn orthogonal_score_estimates_the_mean_contribution_with_its_standard_error() {
    let n = 50;
    let provider =
        declare(6, "orthogonal", &[Cap::EstimatingEquations, Cap::InfluenceContributions], &[]);
    let values: Vec<f64> = (0..n).map(|i| 2.0 + (i as f64 * 0.9).sin()).collect();
    let model = Contributions { declaration: provider.clone(), values: values.clone() };
    let rows = rows(n);
    let run = bind(RecipeKind::OrthogonalScore, InferenceSelection::Frequentist, &rows, &provider)
        .executor();
    let estimate = run.orthogonal_estimate(&model).unwrap();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    assert!((estimate.estimate() - mean).abs() < 1e-12);
    assert!((estimate.standard_error() - (variance / n as f64).sqrt()).abs() < 1e-12);
    assert_eq!(estimate.rows(), n as u64);

    let poisoned = Contributions {
        declaration: provider,
        values: values
            .iter()
            .enumerate()
            .map(|(i, v)| if i == 7 { f64::NAN } else { *v })
            .collect(),
    };
    assert!(matches!(
        run.orthogonal_estimate(&poisoned),
        Err(ExecutionError::NonFinite { row: 7, .. })
    ));
}

/// A four-point law.
struct FourPoints {
    declaration: ProviderDeclaration,
    probabilities: [f64; 4],
}

impl RecipeProvider for FourPoints {
    fn declaration(&self) -> &ProviderDeclaration {
        &self.declaration
    }
}

impl ExactIntegrationModel for FourPoints {
    fn support_len(&self) -> u64 {
        4
    }

    fn probability(&self, index: u64) -> Result<f64, ProviderCallError> {
        Ok(self.probabilities[index as usize])
    }
}

#[test]
fn finite_discrete_expectation_is_an_exact_sum_over_a_normalized_law() {
    let provider = declare(7, "four points", &[Cap::ExactIntegration], &[]);
    let rows = rows(3);
    let run = bind(RecipeKind::FiniteDiscrete, InferenceSelection::Frequentist, &rows, &provider)
        .executor();
    let law = FourPoints { declaration: provider.clone(), probabilities: [0.1, 0.2, 0.3, 0.4] };
    let value = run.integrate_exactly(&law, |index| (index * index) as f64).unwrap();
    assert!((value - 5.0).abs() < 1e-12);

    let unnormalized = FourPoints { declaration: provider, probabilities: [0.1, 0.2, 0.3, 0.1] };
    assert!(matches!(
        run.integrate_exactly(&unnormalized, |index| index as f64),
        Err(ExecutionError::Argument { .. })
    ));
}

#[test]
fn generalized_update_concentrates_at_the_estimating_equation_root() {
    let n = 40;
    let provider = declare(8, "generalized", &[Cap::EstimatingEquations], &[]);
    let model = GaussianMean::new(provider.clone(), n);
    let rows = rows(n);
    let run =
        bind(RecipeKind::LinearGaussian, InferenceSelection::GeneralizedBayesian, &rows, &provider)
            .executor();
    let posterior = run.generalized_update(&model, &grid(-3.0, 3.0, 0.01), 1.0).unwrap();
    let mean = model.data.iter().sum::<f64>() / n as f64;
    assert!((posterior.mean(0) - mean).abs() < 1e-6);
    let variance = posterior.expectation(|p| (p[0] - mean).powi(2));
    assert!((variance - 1.0 / n as f64).abs() < 1e-5, "{variance}");

    assert!(matches!(
        run.generalized_update(&model, &grid(-3.0, 3.0, 0.01), 0.0),
        Err(ExecutionError::Argument { .. })
    ));
    // A frequentist recipe does not run a generalized update.
    let frequentist =
        bind(RecipeKind::LinearGaussian, InferenceSelection::Frequentist, &rows, &provider)
            .executor();
    assert!(matches!(
        frequentist.generalized_update(&model, &grid(-3.0, 3.0, 0.01), 1.0),
        Err(ExecutionError::NotApplicable { .. })
    ));
}

#[test]
fn bayesian_g_computation_averages_the_prediction_over_the_posterior() {
    let n = 40;
    let capabilities = [Cap::Prediction, Cap::Density, Cap::Updating];
    let provider = declare(9, "gcomp", &capabilities, &[Cap::Density, Cap::Updating]);
    let model = GaussianMean::new(provider.clone(), n);
    let rows = rows(n);
    let run =
        bind(RecipeKind::GComputation, InferenceSelection::Bayesian, &rows, &provider).executor();
    let posterior = run.update(&model, &grid(-4.0, 4.0, 0.01)).unwrap();
    let predicted = run.posterior_mean_prediction(&model, &posterior).unwrap();
    let covariate_mean = model.covariate.iter().sum::<f64>() / n as f64;
    assert!((predicted - (posterior.mean(0) + covariate_mean)).abs() < 1e-12);
    // The Bayesian recipe has no plug-in mean.
    assert!(matches!(run.plug_in_mean(&model, &[0.0]), Err(ExecutionError::NotApplicable { .. })));
}

#[test]
fn frequentist_g_computation_plugs_in_through_a_prediction_only_boundary() {
    let n = 25;
    let fitter = declare(10, "equations", &[Cap::EstimatingEquations], &[]);
    let rows = rows(n);
    let run =
        bind(RecipeKind::GComputation, InferenceSelection::Frequentist, &rows, &fitter).executor();

    let fit_model = GaussianMean::new(fitter.clone(), n);
    let solution =
        run.solve_estimating_equations(&fit_model, &[0.0], SolveOptions::default()).unwrap();
    let boundary = GaussianMean::new(declare(11, "predictor", &[Cap::Prediction], &[]), n);
    let mean = run.plug_in_mean(&boundary, solution.estimate()).unwrap();
    let covariate_mean = boundary.covariate.iter().sum::<f64>() / n as f64;
    let sample_mean = fit_model.data.iter().sum::<f64>() / n as f64;
    assert!((mean - (sample_mean + covariate_mean)).abs() < 1e-9);

    // A boundary that does not declare prediction is refused, and the boundary is
    // never accepted as the statistical model.
    let silent = GaussianMean::new(declare(12, "silent", &[Cap::Density], &[]), n);
    assert!(matches!(
        run.plug_in_mean(&silent, solution.estimate()),
        Err(ExecutionError::PredictionNotDeclared { .. })
    ));
    assert!(matches!(
        run.solve_estimating_equations(&boundary, &[0.0], SolveOptions::default()),
        Err(ExecutionError::ProviderMismatch { .. })
    ));
}
