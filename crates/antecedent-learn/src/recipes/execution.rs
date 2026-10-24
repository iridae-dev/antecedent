// SPDX-License-Identifier: MIT OR Apache-2.0

//! Executable interfaces behind the named recipes.
//!
//! A [`BoundRecipe`] says which inference a component names and which provider
//! it accepted. This module runs it. Each operation takes the provider body as
//! a trait object and re-checks, on every call, that the body belongs to the
//! bound provider and that its declaration still carries the capabilities the
//! recipe requires:
//!
//! * [`LogDensityModel`] and [`UpdatingModel`]: a Bayesian recipe's likelihood
//!   ([`RecipeRun::log_likelihood`]) and its posterior over a finite parameter
//!   support ([`RecipeRun::update`]), normalized exactly by log-sum-exp.
//! * [`EstimatingEquationModel`]: a frequentist recipe's fit
//!   ([`RecipeRun::solve_estimating_equations`], Newton on the mean score with a
//!   sandwich covariance) and a generalized-Bayesian recipe's loss-based update
//!   ([`RecipeRun::generalized_update`]).
//! * [`InfluenceModel`]: the orthogonal-score estimate and its standard error
//!   ([`RecipeRun::orthogonal_estimate`]).
//! * [`ExactIntegrationModel`]: the finite-discrete expectation by exact sum
//!   ([`RecipeRun::integrate_exactly`]).
//! * [`PredictiveModel`]: the g-computation mean, plug-in at a fitted parameter
//!   ([`RecipeRun::plug_in_mean`], through the prediction boundary) or averaged
//!   over a posterior ([`RecipeRun::posterior_mean_prediction`]).
//!
//! Every loop over rows reads physical `u64` indices and polls the cancellation
//! token before each provider call. A provider error, a non-finite value, a
//! singular Jacobian and an unconverged solve are typed refusals; none is
//! replaced by a default.

use std::fmt;

use antecedent_core::{
    CancellationToken, CapabilityRequest, InferenceSelection, ProviderCapability,
    ProviderDeclaration, ProviderId,
};

use super::{BoundRecipe, RecipeError, RecipeKind};

/// A provider body call failed. The message is the provider's own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCallError {
    message: String,
}

impl ProviderCallError {
    /// Failure carrying `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    /// The provider's message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ProviderCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ProviderCallError {}

/// Why a recipe operation was refused or failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutionError {
    /// The recipe refused the supplied provider declaration.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// The supplied body is not the provider the recipe was bound to.
    #[error("recipe is bound to provider {bound}; the body supplied is provider {supplied}")]
    ProviderMismatch {
        /// Provider the recipe accepted.
        bound: ProviderId,
        /// Provider whose body was supplied.
        supplied: ProviderId,
    },
    /// This operation is not one the bound recipe runs.
    #[error("{operation} is not run by the {kind} recipe under {inference:?} inference")]
    NotApplicable {
        /// Operation that was asked for.
        operation: &'static str,
        /// Recipe kind that was bound.
        kind: RecipeKind,
        /// Inference the recipe names.
        inference: InferenceSelection,
    },
    /// The prediction boundary provider does not declare prediction.
    #[error("provider {provider_name} does not declare prediction, so it cannot be the boundary")]
    PredictionNotDeclared {
        /// Name of the provider.
        provider_name: String,
    },
    /// Cancellation was requested.
    #[error("cancelled during {stage}")]
    Cancelled {
        /// Operation that observed the request.
        stage: &'static str,
    },
    /// A provider call failed.
    #[error("provider call failed: {0}")]
    Provider(#[from] ProviderCallError),
    /// A provider returned a value that is not a finite number.
    #[error("{what} is not finite at row {row}")]
    NonFinite {
        /// Quantity the provider returned.
        what: &'static str,
        /// Physical row (or support index) it was returned for.
        row: u64,
    },
    /// A vector had the wrong length.
    #[error("{what} has {found} entries; the model has {expected}")]
    Dimension {
        /// Which vector.
        what: &'static str,
        /// Length the model declares.
        expected: usize,
        /// Length supplied.
        found: usize,
    },
    /// The support the posterior ranges over was empty.
    #[error("the parameter support is empty")]
    EmptySupport,
    /// Every support point has zero posterior mass.
    #[error("every support point has zero likelihood or prior; there is no posterior")]
    ZeroEvidence,
    /// An argument was out of its legal range.
    #[error("{what} must be {requirement}")]
    Argument {
        /// Argument name.
        what: &'static str,
        /// What it must satisfy.
        requirement: &'static str,
    },
    /// The Jacobian of the mean score is singular at the iterate.
    #[error("the Jacobian of the mean score is singular after {iterations} iterations")]
    Singular {
        /// Newton iterations taken.
        iterations: usize,
    },
    /// The solve did not reach its tolerance.
    #[error("the estimating equations did not converge in {iterations} iterations")]
    NotConverged {
        /// Newton iterations taken.
        iterations: usize,
    },
    /// The rows are too few for the requested quantity.
    #[error("{what} needs at least {needed} rows; the component has {found}")]
    TooFewRows {
        /// Quantity.
        what: &'static str,
        /// Minimum rows.
        needed: u64,
        /// Rows in the component.
        found: u64,
    },
    /// The supplied score is not the gradient of the supplied log density.
    #[error(
        "the score disagrees with the log-density gradient in coordinate {coordinate}: \
         score {score}, gradient {gradient}"
    )]
    ScoreIsNotGradient {
        /// Coordinate that disagrees.
        coordinate: usize,
        /// Summed provider score.
        score: f64,
        /// Central-difference gradient of the summed log density.
        gradient: f64,
    },
}

/// Every provider body names its declaration. The recipe checks it on each call.
pub trait RecipeProvider {
    /// The declaration the recipe checks against its required capabilities.
    fn declaration(&self) -> &ProviderDeclaration;
}

/// Log density of one row given a parameter vector.
pub trait LogDensityModel: RecipeProvider {
    /// Number of parameters.
    fn parameter_len(&self) -> usize;

    /// Log density of physical `row` at `parameter`. `f64::NEG_INFINITY` is a
    /// legal zero density; `NaN` and `+inf` are refused.
    fn log_density(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError>;
}

/// A log-density model with a prior, so a posterior can be formed.
pub trait UpdatingModel: LogDensityModel {
    /// Log prior density (or log prior mass on a finite support) at `parameter`.
    fn log_prior(&self, parameter: &[f64]) -> Result<f64, ProviderCallError>;
}

/// Per-row estimating-equation contributions.
pub trait EstimatingEquationModel: RecipeProvider {
    /// Number of parameters, which is also the number of equations.
    fn parameter_len(&self) -> usize;

    /// Write the score of physical `row` at `parameter` into `out`, whose length
    /// is [`Self::parameter_len`].
    fn score(&self, parameter: &[f64], row: u64, out: &mut [f64]) -> Result<(), ProviderCallError>;
}

/// Per-row contributions of an orthogonal score.
pub trait InfluenceModel: RecipeProvider {
    /// The orthogonal-score contribution of physical `row` (a pseudo-outcome whose
    /// mean is the estimate).
    fn contribution(&self, row: u64) -> Result<f64, ProviderCallError>;
}

/// A finite law that can be integrated exactly.
pub trait ExactIntegrationModel: RecipeProvider {
    /// Number of support points.
    fn support_len(&self) -> u64;

    /// Probability of support point `index`.
    fn probability(&self, index: u64) -> Result<f64, ProviderCallError>;
}

/// A conditional-mean predictor.
pub trait PredictiveModel: RecipeProvider {
    /// Number of parameters.
    fn parameter_len(&self) -> usize;

    /// Predicted mean of physical `row` at `parameter`.
    fn predict(&self, parameter: &[f64], row: u64) -> Result<f64, ProviderCallError>;
}

/// Options of the Newton solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveOptions {
    /// Largest absolute coordinate of the mean score accepted as a root.
    pub tolerance: f64,
    /// Newton iterations allowed.
    pub max_iterations: usize,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self { tolerance: 1e-10, max_iterations: 50 }
    }
}

/// Root of the mean estimating equations and its sandwich covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct EquationSolution {
    estimate: Vec<f64>,
    covariance: Vec<f64>,
    iterations: usize,
    score_norm: f64,
    rows: u64,
}

impl EquationSolution {
    /// The root.
    #[must_use]
    pub fn estimate(&self) -> &[f64] {
        &self.estimate
    }

    /// Row-major `p x p` sandwich covariance of the estimate.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Standard error of coordinate `index`, or `None` when out of range.
    #[must_use]
    pub fn standard_error(&self, index: usize) -> Option<f64> {
        let p = self.estimate.len();
        (index < p).then(|| self.covariance[index * p + index].max(0.0).sqrt())
    }

    /// Newton iterations taken.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Largest absolute coordinate of the mean score at the root.
    #[must_use]
    pub const fn score_norm(&self) -> f64 {
        self.score_norm
    }

    /// Rows the equations averaged.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }
}

/// Orthogonal-score estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreEstimate {
    estimate: f64,
    standard_error: f64,
    rows: u64,
}

impl ScoreEstimate {
    /// Mean of the contributions.
    #[must_use]
    pub const fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Standard error from the sample variance of the contributions.
    #[must_use]
    pub const fn standard_error(&self) -> f64 {
        self.standard_error
    }

    /// Rows averaged.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }
}

/// Normalized weights over a finite parameter support.
#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    points: Vec<Vec<f64>>,
    log_weights: Vec<f64>,
    log_evidence: f64,
    exact: bool,
}

impl Posterior {
    /// Support points.
    #[must_use]
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Normalized weights, summing to one.
    #[must_use]
    pub fn weights(&self) -> Vec<f64> {
        self.log_weights.iter().map(|weight| weight.exp()).collect()
    }

    /// Log marginal likelihood of the support: log of the prior-weighted sum of the
    /// likelihood. For a generalized update this is the log normalizer of the Gibbs
    /// weights.
    #[must_use]
    pub const fn log_evidence(&self) -> f64 {
        self.log_evidence
    }

    /// Whether the provider declares exact integration, so the sum over the
    /// support is the integral and not a quadrature of it.
    #[must_use]
    pub const fn exact(&self) -> bool {
        self.exact
    }

    /// Posterior expectation of `functional`.
    pub fn expectation(&self, functional: impl Fn(&[f64]) -> f64) -> f64 {
        self.points
            .iter()
            .zip(&self.log_weights)
            .map(|(point, log_weight)| log_weight.exp() * functional(point))
            .sum()
    }

    /// Posterior mean of coordinate `index`.
    pub fn mean(&self, index: usize) -> f64 {
        self.expectation(|point| point[index])
    }
}

impl<'a> BoundRecipe<'a> {
    /// The operations this bound recipe runs.
    #[must_use]
    pub const fn executor(self) -> RecipeRun<'a> {
        RecipeRun { bound: self, cancellation: None }
    }
}

/// A bound recipe ready to run against provider bodies.
#[derive(Clone, Copy, Debug)]
pub struct RecipeRun<'a> {
    bound: BoundRecipe<'a>,
    cancellation: Option<&'a CancellationToken>,
}

impl<'a> RecipeRun<'a> {
    /// Poll `token` before every provider call.
    #[must_use]
    pub fn with_cancellation(self, token: &'a CancellationToken) -> Self {
        Self { cancellation: Some(token), ..self }
    }

    /// The recipe this run executes.
    #[must_use]
    pub const fn bound(&self) -> BoundRecipe<'a> {
        self.bound
    }

    fn poll(&self, stage: &'static str) -> Result<(), ExecutionError> {
        match self.cancellation {
            Some(token) if token.is_cancelled() => Err(ExecutionError::Cancelled { stage }),
            _ => Ok(()),
        }
    }

    fn not_applicable(&self, operation: &'static str) -> ExecutionError {
        ExecutionError::NotApplicable {
            operation,
            kind: self.bound.recipe().kind(),
            inference: self.bound.inference(),
        }
    }

    /// The body is the bound provider, and its declaration still carries what the recipe
    /// requires.
    fn gate(&self, declaration: &ProviderDeclaration) -> Result<(), ExecutionError> {
        if declaration.id() != self.bound.provider() {
            return Err(ExecutionError::ProviderMismatch {
                bound: self.bound.provider(),
                supplied: declaration.id(),
            });
        }
        self.bound.recipe().accept_provider(declaration)?;
        Ok(())
    }

    fn gate_likelihood(&self, declaration: &ProviderDeclaration) -> Result<(), ExecutionError> {
        if declaration.id() != self.bound.provider() {
            return Err(ExecutionError::ProviderMismatch {
                bound: self.bound.provider(),
                supplied: declaration.id(),
            });
        }
        self.bound.recipe().accept_as_likelihood(declaration)?;
        Ok(())
    }

    fn row_count(&self) -> u64 {
        self.bound.rows().len() as u64
    }

    fn sum_log_density<M: LogDensityModel + ?Sized>(
        &self,
        model: &M,
        parameter: &[f64],
    ) -> Result<f64, ExecutionError> {
        if parameter.len() != model.parameter_len() {
            return Err(ExecutionError::Dimension {
                what: "parameter",
                expected: model.parameter_len(),
                found: parameter.len(),
            });
        }
        let mut total = 0.0;
        for &row in self.bound.rows() {
            self.poll("log likelihood")?;
            let value = model.log_density(parameter, row)?;
            if value.is_nan() || value == f64::INFINITY {
                return Err(ExecutionError::NonFinite { what: "log density", row });
            }
            total += value;
        }
        Ok(total)
    }

    /// Log likelihood of the component's rows at `parameter`: the sum of the
    /// provider's per-row log densities.
    ///
    /// # Errors
    ///
    /// A recipe that is not an ordinary likelihood (frequentist or generalized
    /// Bayesian), a body that is not the bound provider or lost a required
    /// capability, a wrong-length parameter, a provider failure or cancellation.
    pub fn log_likelihood(
        &self,
        model: &dyn LogDensityModel,
        parameter: &[f64],
    ) -> Result<f64, ExecutionError> {
        self.gate_likelihood(model.declaration())?;
        self.sum_log_density(model, parameter)
    }

    /// Bayesian update over a finite parameter `support`: the posterior weight of
    /// each point is its prior mass times its likelihood, normalized by log-sum-exp.
    ///
    /// # Errors
    ///
    /// As [`Self::log_likelihood`], an empty support, or no point with positive mass.
    pub fn update(
        &self,
        model: &dyn UpdatingModel,
        support: &[Vec<f64>],
    ) -> Result<Posterior, ExecutionError> {
        self.gate_likelihood(model.declaration())?;
        if support.is_empty() {
            return Err(ExecutionError::EmptySupport);
        }
        let mut log_weights = Vec::with_capacity(support.len());
        for (index, point) in support.iter().enumerate() {
            self.poll("posterior update")?;
            let prior = model.log_prior(point)?;
            if prior.is_nan() || prior == f64::INFINITY {
                return Err(ExecutionError::NonFinite { what: "log prior", row: index as u64 });
            }
            let likelihood = if prior == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                self.sum_log_density(model, point)?
            };
            log_weights.push(prior + likelihood);
        }
        let exact = model.declaration().supports(ProviderCapability::ExactIntegration);
        normalize(support, log_weights, exact)
    }

    /// Generalized-Bayesian update over a finite `support`: weights proportional to
    /// `exp(-learning_rate * loss)` with `loss(theta) = n/2 * |mean score(theta)|^2`
    /// and a uniform prior. The learning rate is the caller's declared calibration.
    ///
    /// # Errors
    ///
    /// A recipe that is not generalized Bayesian, a non-positive learning rate, an empty
    /// support, a body that is not the bound provider or lost a capability, a provider
    /// failure or cancellation.
    pub fn generalized_update(
        &self,
        model: &dyn EstimatingEquationModel,
        support: &[Vec<f64>],
        learning_rate: f64,
    ) -> Result<Posterior, ExecutionError> {
        if self.bound.inference() != InferenceSelection::GeneralizedBayesian {
            return Err(self.not_applicable("generalized update"));
        }
        self.gate(model.declaration())?;
        if !(learning_rate.is_finite() && learning_rate > 0.0) {
            return Err(ExecutionError::Argument {
                what: "learning_rate",
                requirement: "finite and positive",
            });
        }
        if support.is_empty() {
            return Err(ExecutionError::EmptySupport);
        }
        let n = self.row_count() as f64;
        let mut log_weights = Vec::with_capacity(support.len());
        for point in support {
            let mean = self.mean_score(model, point)?;
            let loss = 0.5 * n * mean.iter().map(|value| value * value).sum::<f64>();
            log_weights.push(-learning_rate * loss);
        }
        normalize(support, log_weights, false)
    }

    fn mean_score<M: EstimatingEquationModel + ?Sized>(
        &self,
        model: &M,
        parameter: &[f64],
    ) -> Result<Vec<f64>, ExecutionError> {
        let p = model.parameter_len();
        if parameter.len() != p {
            return Err(ExecutionError::Dimension {
                what: "parameter",
                expected: p,
                found: parameter.len(),
            });
        }
        let mut total = vec![0.0; p];
        let mut buffer = vec![0.0; p];
        for &row in self.bound.rows() {
            self.poll("estimating equations")?;
            model.score(parameter, row, &mut buffer)?;
            for (sum, value) in total.iter_mut().zip(&buffer) {
                if !value.is_finite() {
                    return Err(ExecutionError::NonFinite { what: "score", row });
                }
                *sum += value;
            }
        }
        let n = self.row_count() as f64;
        for value in &mut total {
            *value /= n;
        }
        Ok(total)
    }

    /// Central-difference Jacobian of the mean score, row-major with entry
    /// `(r, c) = d mean_score_r / d theta_c`.
    fn jacobian<M: EstimatingEquationModel + ?Sized>(
        &self,
        model: &M,
        parameter: &[f64],
    ) -> Result<Vec<f64>, ExecutionError> {
        let p = parameter.len();
        let mut jacobian = vec![0.0; p * p];
        let mut shifted = parameter.to_vec();
        for c in 0..p {
            let step = 1e-6 * parameter[c].abs().max(1.0);
            shifted[c] = parameter[c] + step;
            let up = self.mean_score(model, &shifted)?;
            shifted[c] = parameter[c] - step;
            let down = self.mean_score(model, &shifted)?;
            shifted[c] = parameter[c];
            for r in 0..p {
                jacobian[r * p + c] = (up[r] - down[r]) / (2.0 * step);
            }
        }
        Ok(jacobian)
    }

    /// Solve the mean estimating equations from `start` by damped Newton iteration and
    /// report the sandwich covariance `A^-1 B A^-T / n`, with `A` the Jacobian of the
    /// mean score and `B` the mean outer product of the per-row scores.
    ///
    /// # Errors
    ///
    /// A recipe that is not frequentist, a body that is not the bound provider or lost a
    /// capability, a wrong-length start, fewer than two rows, a singular Jacobian,
    /// an unconverged solve, a provider failure or cancellation.
    pub fn solve_estimating_equations(
        &self,
        model: &dyn EstimatingEquationModel,
        start: &[f64],
        options: SolveOptions,
    ) -> Result<EquationSolution, ExecutionError> {
        if self.bound.inference() != InferenceSelection::Frequentist
            || !self
                .bound
                .recipe()
                .required_capabilities()
                .contains(&ProviderCapability::EstimatingEquations)
        {
            return Err(self.not_applicable("estimating-equation solve"));
        }
        self.gate(model.declaration())?;
        if !(options.tolerance.is_finite() && options.tolerance > 0.0)
            || options.max_iterations == 0
        {
            return Err(ExecutionError::Argument {
                what: "options",
                requirement: "a positive tolerance and at least one iteration",
            });
        }
        let rows = self.row_count();
        if rows < 2 {
            return Err(ExecutionError::TooFewRows {
                what: "a sandwich covariance",
                needed: 2,
                found: rows,
            });
        }
        let p = model.parameter_len();
        let mut theta = start.to_vec();
        let mut score = self.mean_score(model, &theta)?;
        let mut iterations = 0;
        while max_abs(&score) > options.tolerance {
            if iterations == options.max_iterations {
                return Err(ExecutionError::NotConverged { iterations });
            }
            iterations += 1;
            let inverse = invert(&self.jacobian(model, &theta)?, p)
                .ok_or(ExecutionError::Singular { iterations })?;
            let step = multiply_vector(&inverse, &score, p);
            let current = max_abs(&score);
            let mut scale = 1.0;
            let (next_theta, next_score) = loop {
                let candidate: Vec<f64> =
                    theta.iter().zip(&step).map(|(value, delta)| value - scale * delta).collect();
                let candidate_score = self.mean_score(model, &candidate)?;
                if max_abs(&candidate_score) < current || scale < 1e-10 {
                    break (candidate, candidate_score);
                }
                scale *= 0.5;
            };
            theta = next_theta;
            score = next_score;
        }
        let bread = invert(&self.jacobian(model, &theta)?, p)
            .ok_or(ExecutionError::Singular { iterations })?;
        let mut meat = vec![0.0; p * p];
        let mut buffer = vec![0.0; p];
        for &row in self.bound.rows() {
            self.poll("sandwich covariance")?;
            model.score(&theta, row, &mut buffer)?;
            for r in 0..p {
                for c in 0..p {
                    meat[r * p + c] += buffer[r] * buffer[c];
                }
            }
        }
        let n = rows as f64;
        for value in &mut meat {
            *value /= n;
        }
        let half = multiply_matrix(&bread, &meat, p);
        let mut covariance = multiply_matrix(&half, &transpose(&bread, p), p);
        for value in &mut covariance {
            *value /= n;
        }
        Ok(EquationSolution {
            estimate: theta,
            covariance,
            iterations,
            score_norm: max_abs(&score),
            rows,
        })
    }

    /// Check that the provider's score is the gradient of its summed log density at
    /// `parameter`, by central differences. A generalized linear recipe's fit is only
    /// the maximum-likelihood fit when this holds.
    ///
    /// # Errors
    ///
    /// A recipe whose required capabilities lack a differentiable log density, the
    /// refusals of [`Self::log_likelihood`] and [`Self::solve_estimating_equations`],
    /// or [`ExecutionError::ScoreIsNotGradient`].
    pub fn verify_score_gradient<M>(
        &self,
        model: &M,
        parameter: &[f64],
        tolerance: f64,
    ) -> Result<(), ExecutionError>
    where
        M: LogDensityModel + EstimatingEquationModel,
    {
        if !self
            .bound
            .recipe()
            .required_capabilities()
            .contains(&ProviderCapability::DifferentiableLogDensity)
        {
            return Err(self.not_applicable("score-gradient verification"));
        }
        self.gate(RecipeProvider::declaration(model))?;
        let p = EstimatingEquationModel::parameter_len(model);
        let rows = self.row_count() as f64;
        let mean = self.mean_score(model, parameter)?;
        let mut shifted = parameter.to_vec();
        for c in 0..p {
            let step = 1e-6 * parameter[c].abs().max(1.0);
            shifted[c] = parameter[c] + step;
            let up = self.sum_log_density(model, &shifted)?;
            shifted[c] = parameter[c] - step;
            let down = self.sum_log_density(model, &shifted)?;
            shifted[c] = parameter[c];
            let gradient = (up - down) / (2.0 * step);
            let score = mean[c] * rows;
            if (gradient - score).abs() > tolerance * gradient.abs().max(score.abs()).max(1.0) {
                return Err(ExecutionError::ScoreIsNotGradient { coordinate: c, score, gradient });
            }
        }
        Ok(())
    }

    /// Orthogonal-score estimate: the mean of the provider's per-row contributions and
    /// its standard error from their sample variance.
    ///
    /// # Errors
    ///
    /// A recipe that is not the orthogonal score, a body that is not the bound provider
    /// or lost a capability, fewer than two rows, a non-finite contribution, a provider
    /// failure or cancellation.
    pub fn orthogonal_estimate(
        &self,
        model: &dyn InfluenceModel,
    ) -> Result<ScoreEstimate, ExecutionError> {
        if self.bound.recipe().kind() != RecipeKind::OrthogonalScore {
            return Err(self.not_applicable("orthogonal-score estimate"));
        }
        self.gate(model.declaration())?;
        let rows = self.row_count();
        if rows < 2 {
            return Err(ExecutionError::TooFewRows {
                what: "an orthogonal-score standard error",
                needed: 2,
                found: rows,
            });
        }
        let mut values = Vec::with_capacity(self.bound.rows().len());
        for &row in self.bound.rows() {
            self.poll("orthogonal score")?;
            let value = model.contribution(row)?;
            if !value.is_finite() {
                return Err(ExecutionError::NonFinite { what: "score contribution", row });
            }
            values.push(value);
        }
        let n = rows as f64;
        let estimate = values.iter().sum::<f64>() / n;
        let variance =
            values.iter().map(|value| (value - estimate).powi(2)).sum::<f64>() / (n - 1.0);
        Ok(ScoreEstimate { estimate, standard_error: (variance / n).sqrt(), rows })
    }

    /// Expectation of `functional` (a function of the support index) under the
    /// provider's finite law, by an exact sum.
    ///
    /// # Errors
    ///
    /// A recipe that is not the frequentist finite-discrete recipe, a body that is not
    /// the bound provider or lost exact integration, a negative or non-finite probability,
    /// probabilities that do not sum to one, a provider failure or cancellation.
    pub fn integrate_exactly(
        &self,
        model: &dyn ExactIntegrationModel,
        functional: impl Fn(u64) -> f64,
    ) -> Result<f64, ExecutionError> {
        let recipe = self.bound.recipe();
        if recipe.kind() != RecipeKind::FiniteDiscrete
            || recipe.inference() != InferenceSelection::Frequentist
        {
            return Err(self.not_applicable("exact integration"));
        }
        self.gate(model.declaration())?;
        let support = model.support_len();
        if support == 0 {
            return Err(ExecutionError::EmptySupport);
        }
        let mut total = 0.0;
        let mut mass = 0.0;
        for index in 0..support {
            self.poll("exact integration")?;
            let probability = model.probability(index)?;
            if !(probability.is_finite() && probability >= 0.0) {
                return Err(ExecutionError::NonFinite { what: "probability", row: index });
            }
            mass += probability;
            total += probability * functional(index);
        }
        if (mass - 1.0).abs() > 1e-9 * (support as f64).max(1.0) {
            return Err(ExecutionError::Argument {
                what: "the probabilities",
                requirement: "summing to one",
            });
        }
        Ok(total)
    }

    /// Frequentist plug-in mean of the component's rows at a fitted `parameter`, through the
    /// prediction boundary. The predictor needs only the prediction capability; it is never
    /// the statistical model.
    ///
    /// # Errors
    ///
    /// A recipe that is not frequentist g-computation, a predictor that does not declare
    /// prediction, a wrong-length parameter, a non-finite prediction, a provider failure or
    /// cancellation.
    pub fn plug_in_mean(
        &self,
        predictor: &dyn PredictiveModel,
        parameter: &[f64],
    ) -> Result<f64, ExecutionError> {
        let recipe = self.bound.recipe();
        if recipe.kind() != RecipeKind::GComputation
            || recipe.inference() != InferenceSelection::Frequentist
        {
            return Err(self.not_applicable("plug-in mean"));
        }
        recipe.attach_prediction_boundary(self.bound.rows())?;
        let declaration = predictor.declaration();
        if declaration.compile(CapabilityRequest::new(ProviderCapability::Prediction)).is_err() {
            return Err(ExecutionError::PredictionNotDeclared {
                provider_name: declaration.name().to_owned(),
            });
        }
        self.mean_prediction(predictor, parameter)
    }

    /// Posterior mean of the component's mean prediction: the posterior-weighted sum of the
    /// plug-in mean at each support point.
    ///
    /// # Errors
    ///
    /// A recipe that is not Bayesian g-computation, a body that is not the bound provider
    /// or lost a capability, a non-finite prediction, a provider failure or cancellation.
    pub fn posterior_mean_prediction(
        &self,
        model: &dyn PredictiveModel,
        posterior: &Posterior,
    ) -> Result<f64, ExecutionError> {
        let recipe = self.bound.recipe();
        if recipe.kind() != RecipeKind::GComputation
            || recipe.inference() != InferenceSelection::Bayesian
        {
            return Err(self.not_applicable("posterior mean prediction"));
        }
        self.gate(model.declaration())?;
        let mut total = 0.0;
        for (point, log_weight) in posterior.points.iter().zip(&posterior.log_weights) {
            total += log_weight.exp() * self.mean_prediction(model, point)?;
        }
        Ok(total)
    }

    fn mean_prediction(
        &self,
        model: &dyn PredictiveModel,
        parameter: &[f64],
    ) -> Result<f64, ExecutionError> {
        if parameter.len() != model.parameter_len() {
            return Err(ExecutionError::Dimension {
                what: "parameter",
                expected: model.parameter_len(),
                found: parameter.len(),
            });
        }
        let mut total = 0.0;
        for &row in self.bound.rows() {
            self.poll("prediction")?;
            let value = model.predict(parameter, row)?;
            if !value.is_finite() {
                return Err(ExecutionError::NonFinite { what: "prediction", row });
            }
            total += value;
        }
        Ok(total / self.row_count() as f64)
    }
}

fn normalize(
    support: &[Vec<f64>],
    mut log_weights: Vec<f64>,
    exact: bool,
) -> Result<Posterior, ExecutionError> {
    let top = log_weights
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return Err(ExecutionError::ZeroEvidence);
    }
    let sum: f64 = log_weights.iter().map(|value| (value - top).exp()).sum();
    let log_evidence = top + sum.ln();
    for value in &mut log_weights {
        *value -= log_evidence;
    }
    Ok(Posterior { points: support.to_vec(), log_weights, log_evidence, exact })
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |top, value| top.max(value.abs()))
}

fn multiply_vector(matrix: &[f64], vector: &[f64], p: usize) -> Vec<f64> {
    (0..p).map(|r| (0..p).map(|c| matrix[r * p + c] * vector[c]).sum()).collect()
}

fn multiply_matrix(left: &[f64], right: &[f64], p: usize) -> Vec<f64> {
    let mut product = vec![0.0; p * p];
    for r in 0..p {
        for k in 0..p {
            let scale = left[r * p + k];
            for c in 0..p {
                product[r * p + c] += scale * right[k * p + c];
            }
        }
    }
    product
}

fn transpose(matrix: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for r in 0..p {
        for c in 0..p {
            out[c * p + r] = matrix[r * p + c];
        }
    }
    out
}

/// Gauss-Jordan inverse of a row-major `p x p` matrix with partial pivoting, or `None`
/// when a pivot is numerically zero.
fn invert(matrix: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut work = matrix.to_vec();
    let mut inverse = vec![0.0; p * p];
    for index in 0..p {
        inverse[index * p + index] = 1.0;
    }
    let scale = max_abs(matrix).max(f64::MIN_POSITIVE);
    for column in 0..p {
        let mut pivot = column;
        for row in column + 1..p {
            if work[row * p + column].abs() > work[pivot * p + column].abs() {
                pivot = row;
            }
        }
        if work[pivot * p + column].abs() <= 1e-12 * scale {
            return None;
        }
        if pivot != column {
            for k in 0..p {
                work.swap(pivot * p + k, column * p + k);
                inverse.swap(pivot * p + k, column * p + k);
            }
        }
        let diagonal = work[column * p + column];
        for k in 0..p {
            work[column * p + k] /= diagonal;
            inverse[column * p + k] /= diagonal;
        }
        for row in 0..p {
            if row == column {
                continue;
            }
            let factor = work[row * p + column];
            if factor == 0.0 {
                continue;
            }
            for k in 0..p {
                work[row * p + k] -= factor * work[column * p + k];
                inverse[row * p + k] -= factor * inverse[column * p + k];
            }
        }
    }
    Some(inverse)
}
