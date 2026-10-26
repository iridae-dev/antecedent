// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounded projected Adam for nonlinear feedback likelihoods and explicit
//! intervention-regime validation stopping. The smooth Armijo path is not
//! substituted at `ReLU` Jacobian jumps. Validation convergence, training-score
//! stationarity and epoch exhaustion are distinct reported facts.
use crate::cyclic_likelihood::{CyclicParameter, parameter_values, with_parameters};
use antecedent_core::ExecutionContext;
use antecedent_model::nonlinear_equilibrium::{
    Activation, EquilibriumError, EquilibriumModel, RegimeRow,
};
use serde::{Deserialize, Serialize};
/// All optimizer, feasibility and stopping policies retained in source identities.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicAdamPolicy {
    /// Initial Adam parameter step size.
    pub learning_rate: f64,
    /// Maximum rows in a shuffled gradient batch.
    pub batch_size: usize,
    /// Maximum complete data epochs.
    pub max_epochs: usize,
    /// Earliest epoch at which patience can stop fitting.
    pub minimum_epochs: usize,
    /// Complete epochs without the declared validation improvement before stopping.
    pub patience: usize,
    /// Required validation improvement per observation, relative to max(1,|loss|).
    pub relative_tolerance: f64,
    /// Maximum per-observation training KKT score accepted as score convergence.
    pub score_tolerance: f64,
    /// Per-observation L1 penalty on incoming legacy coefficients only.
    pub l1_penalty: f64,
    /// Domain-wide row-sum bound maintained after every update, strictly below one.
    pub contraction_margin: f64,
    /// Semantic shuffle seed, not an equilibrium-selection law.
    pub seed: u64,
}
impl Default for CyclicAdamPolicy {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 256,
            max_epochs: 200,
            minimum_epochs: 20,
            patience: 10,
            relative_tolerance: 1e-5,
            score_tolerance: 1e-3,
            l1_penalty: 0.001,
            contraction_margin: 0.95,
            seed: 0,
        }
    }
}
/// One complete validation evaluation; no test-regime outcomes choose a fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CyclicAdamEpoch {
    /// Completed epoch.
    pub epoch: usize,
    /// Actual unpenalized validation log likelihood per observation.
    pub validation_log_likelihood_per_row: f64,
    /// Update attempts retained or reduced by feasibility checks.
    pub rejected_updates: usize,
}
/// A selected feasible mechanism model and its distinct stopping diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CyclicAdamFit {
    /// Best validation model, conditional on declared noise and mechanism class.
    pub model: EquilibriumModel,
    /// Full unpenalized joint training log likelihood at that model.
    pub training_log_likelihood: f64,
    /// Full unpenalized joint validation log likelihood at that model.
    pub validation_log_likelihood: f64,
    /// Best model's epoch; zero denotes the original model.
    pub best_epoch: usize,
    /// Complete epochs actually processed.
    pub epochs: usize,
    /// Actual minibatch gradient steps attempted.
    pub gradient_steps: usize,
    /// Largest per-observation KKT score at the selected model.
    pub training_score_per_row: f64,
    /// Validation patience reached after the minimum epoch count.
    pub validation_converged: bool,
    /// Training score independently met its declared tolerance.
    pub training_score_converged: bool,
    /// `validation_patience`, `training_score`, or `epoch_budget`.
    pub stopping_reason: String,
    /// Every epoch's original validation result and feasibility refusals.
    pub trace: Vec<CyclicAdamEpoch>,
    /// Explicit piecewise-linear derivative convention when applicable.
    pub nonsmooth_convention: Option<String>,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn licensed(
    model: &EquilibriumModel,
    margin: f64,
    ctx: &ExecutionContext,
) -> Result<bool, EquilibriumError> {
    for equation in &model.equations {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cancelled"));
        }
        let bound: f64 = equation.derivative_bounds()?.iter().sum();
        if !bound.is_finite() || bound > margin {
            return Ok(false);
        }
    }
    Ok(true)
}
fn project(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    old: &[f64],
    proposed: &[f64],
    margin: f64,
    ctx: &ExecutionContext,
) -> Result<EquilibriumModel, EquilibriumError> {
    let mut model = with_parameters(template, parameters, proposed)?;
    // Row scaling changes only declared learnable coefficients. Fixed mechanisms
    // and intercepts are never silently reparameterized to earn a certificate.
    for (i, equation) in model.equations.iter_mut().enumerate() {
        if equation.expression.is_some() {
            continue;
        }
        let mut fixed = 0.0;
        let mut learned = 0.0;
        for (j, weight) in equation.weights.iter().enumerate() {
            if parameters.contains(&CyclicParameter::Weight { equation: i, parent: j }) {
                learned += weight.abs();
            } else {
                fixed += weight.abs();
            }
        }
        if fixed > margin {
            return Err(invalid("fixed coefficients exceed the declared Adam contraction margin"));
        }
        if fixed + learned > margin && learned > 0.0 {
            let scale = ((margin - fixed) * (1.0 - 8.0 * f64::EPSILON) / learned).max(0.0);
            for (j, weight) in equation.weights.iter_mut().enumerate() {
                if parameters.contains(&CyclicParameter::Weight { equation: i, parent: j }) {
                    *weight *= scale;
                }
            }
        }
    }
    if licensed(&model, margin, ctx)? {
        return Ok(model);
    }
    // Compositional parameters need not be linear coefficients. Bounded
    // interpolation refuses unlicensed proposals without rewriting their meaning.
    let mut scale = 0.5;
    for _ in 0..48 {
        let values: Vec<_> = old.iter().zip(proposed).map(|(a, b)| a + scale * (b - a)).collect();
        let candidate = with_parameters(template, parameters, &values)?;
        if licensed(&candidate, margin, ctx)? {
            return Ok(candidate);
        }
        scale *= 0.5;
    }
    Err(EquilibriumError::Uncertified { bound: f64::INFINITY })
}
/// Fit exact free-system Jacobian likelihoods using deterministic minibatch streams.
/// Validation is separately declared; it is never merged into training or used
/// as a counterfactual selection law. No global optimum or calibration is implied.
#[allow(
    clippy::too_many_lines,
    reason = "one bounded optimizer loop retains feasibility, objective and stopping diagnostics together"
)]
pub fn fit_cyclic_adam(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    training: &[RegimeRow],
    validation: &[RegimeRow],
    policy: CyclicAdamPolicy,
    ctx: &ExecutionContext,
) -> Result<CyclicAdamFit, EquilibriumError> {
    let mut values = parameter_values(template, parameters)?;
    if training.is_empty()
        || validation.is_empty()
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || policy.batch_size == 0
        || policy.batch_size > 16384
        || policy.max_epochs == 0
        || policy.max_epochs > 1000
        || policy.minimum_epochs > policy.max_epochs
        || policy.patience == 0
        || !policy.relative_tolerance.is_finite()
        || policy.relative_tolerance <= 0.0
        || !policy.score_tolerance.is_finite()
        || policy.score_tolerance <= 0.0
        || !policy.l1_penalty.is_finite()
        || policy.l1_penalty < 0.0
        || !policy.contraction_margin.is_finite()
        || policy.contraction_margin <= 0.0
        || policy.contraction_margin >= 1.0
        || training.len().div_ceil(policy.batch_size).saturating_mul(policy.max_epochs) > 100_000
    {
        return Err(invalid("Adam data/work/stopping policy"));
    }
    if !licensed(template, policy.contraction_margin, ctx)? {
        return Err(invalid("initial mechanisms do not meet the declared Adam contraction margin"));
    }
    let n = template.order();
    let reserve = training
        .len()
        .saturating_mul(16)
        .saturating_add(policy.batch_size.saturating_mul(n * 16 + 256))
        .saturating_add(parameters.len() * 128);
    let _lease = ctx
        .acquire_admission(0, reserve as u64)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let mut model = template.clone();
    let mut first = vec![0.0; values.len()];
    let mut second = vec![0.0; values.len()];
    let mut best_model = model.clone();
    let mut best_validation = model.log_likelihood(validation, ctx)?;
    let mut best_epoch = 0;
    let mut patience_reference = best_validation / validation.len() as f64;
    let mut stale = 0;
    let mut steps = 0usize;
    let mut trace = Vec::new();
    let mut stopped = false;
    let rng = antecedent_core::RngFactory::from_seed(policy.seed);
    for epoch in 1..=policy.max_epochs {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let mut order: Vec<_> = (0..training.len()).collect();
        let mut stream =
            rng.stream(antecedent_core::derive_stream(0x4359_434c_4144_414d, epoch as u64));
        antecedent_kernels::shuffle(&mut stream, &mut order);
        let mut rejected = 0;
        for indices in order.chunks(policy.batch_size) {
            let rows: Vec<_> = indices.iter().map(|&i| training[i].clone()).collect();
            let (_, scores) = model.likelihood_score(&rows, ctx)?;
            steps += 1;
            let mut proposed = values.clone();
            let correction1 = 1.0 - 0.9_f64.powf(steps as f64);
            let correction2 = 1.0 - 0.999_f64.powf(steps as f64);
            for (k, parameter) in parameters.iter().enumerate() {
                let score = match *parameter {
                    CyclicParameter::Intercept { equation } => scores[equation].intercept,
                    CyclicParameter::Weight { equation, parent } => {
                        scores[equation].weights[parent]
                    }
                    CyclicParameter::ExpressionParameter { equation, parameter } => {
                        scores[equation].parameters[parameter]
                    }
                } / rows.len() as f64;
                first[k] = 0.9 * first[k] + 0.1 * score;
                second[k] = 0.999 * second[k] + 0.001 * score * score;
                let scale = (second[k] / correction2).sqrt() + 1e-8;
                let next = values[k] + policy.learning_rate * (first[k] / correction1) / scale;
                proposed[k] = if matches!(parameter, CyclicParameter::Weight { .. }) {
                    next.signum()
                        * (next.abs() - policy.learning_rate * policy.l1_penalty / scale).max(0.0)
                } else {
                    next
                };
            }
            match project(template, parameters, &values, &proposed, policy.contraction_margin, ctx)
            {
                Ok(candidate) => {
                    model = candidate;
                    values = parameter_values(&model, parameters)?;
                }
                Err(EquilibriumError::Uncertified { .. }) => rejected += 1,
                Err(error) => return Err(error),
            }
        }
        let likelihood = model.log_likelihood(validation, ctx)?;
        let per_row = likelihood / validation.len() as f64;
        if likelihood > best_validation {
            best_validation = likelihood;
            best_model = model.clone();
            best_epoch = epoch;
        }
        if per_row
            > patience_reference + policy.relative_tolerance * patience_reference.abs().max(1.0)
        {
            patience_reference = per_row;
            stale = 0;
        } else {
            stale += 1;
        }
        trace.push(CyclicAdamEpoch {
            epoch,
            validation_log_likelihood_per_row: per_row,
            rejected_updates: rejected,
        });
        if epoch >= policy.minimum_epochs && stale >= policy.patience {
            stopped = true;
            break;
        }
    }
    let (training_log_likelihood, scores) = best_model.likelihood_score(training, ctx)?;
    let fitted_values = parameter_values(&best_model, parameters)?;
    let mut maximum = 0.0_f64;
    for (p, &value) in parameters.iter().zip(&fitted_values) {
        let raw = match *p {
            CyclicParameter::Intercept { equation } => scores[equation].intercept,
            CyclicParameter::Weight { equation, parent } => scores[equation].weights[parent],
            CyclicParameter::ExpressionParameter { equation, parameter } => {
                scores[equation].parameters[parameter]
            }
        } / training.len() as f64;
        let score = if matches!(p, CyclicParameter::Weight { .. }) {
            if value > 0.0 {
                raw - policy.l1_penalty
            } else if value < 0.0 {
                raw + policy.l1_penalty
            } else {
                raw.signum() * (raw.abs() - policy.l1_penalty).max(0.0)
            }
        } else {
            raw
        };
        maximum = maximum.max(score.abs());
    }
    let training_score_converged = maximum <= policy.score_tolerance;
    let nonsmooth = template.equations.iter().any(|e| e.activation == Activation::Relu);
    Ok(CyclicAdamFit{model:best_model,training_log_likelihood,validation_log_likelihood:best_validation,best_epoch,epochs:trace.len(),gradient_steps:steps,training_score_per_row:maximum,validation_converged:stopped,training_score_converged,stopping_reason:if stopped{"validation_patience"}else if training_score_converged{"training_score"}else{"epoch_budget"}.into(),trace,nonsmooth_convention:nonsmooth.then(||"Almost-everywhere ReLU Jacobians; zero derivative at the kink. Validation stopping and score stationarity are reported separately.".into())})
}
#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Clamp, Equation};
    #[test]
    fn projected_adam_fits_and_replays_a_piecewise_feedback_training_protocol() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.01],
                    activation: Activation::Relu,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.2, 0.0],
                    activation: Activation::Tanh,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        let rows: Vec<_> = [-2.0_f64, -1.0, 0.5, 1.0, 2.0]
            .into_iter()
            .flat_map(|x| {
                [-0.1, 0.1].map(|noise| RegimeRow {
                    values: vec![(0.7 * x).max(0.0) + noise, x],
                    clamps: vec![Clamp { variable: 1, value: x }],
                })
            })
            .collect();
        let ctx = ExecutionContext::for_tests(0);
        let policy = CyclicAdamPolicy {
            learning_rate: 0.05,
            batch_size: 10,
            max_epochs: 200,
            minimum_epochs: 50,
            patience: 20,
            l1_penalty: 0.0,
            ..CyclicAdamPolicy::default()
        };
        let fit = fit_cyclic_adam(
            &model,
            &[CyclicParameter::Weight { equation: 0, parent: 1 }],
            &rows,
            &rows,
            policy,
            &ctx,
        )
        .unwrap();
        assert!((fit.model.equations[0].weights[1] - 0.7).abs() < 0.01);
        assert!(fit.validation_converged);
        assert!(licensed(&fit.model, policy.contraction_margin, &ctx).unwrap());
        let replay = fit_cyclic_adam(
            &model,
            &[CyclicParameter::Weight { equation: 0, parent: 1 }],
            &rows,
            &rows,
            policy,
            &ctx,
        )
        .unwrap();
        assert_eq!(fit, replay);
    }
}
