// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared-mechanism stochastic feedback likelihoods with equation replacements.
//! The full stationary OU covariance, rather than fixed-noise SCM covariance,
//! defines the likelihood. Exact Lyapunov adjoints supply parameter scores.
//! Synthetic Bicycle-style interventions are a specialization with fixed
//! incoming coefficients, forcing and diffusion rows in the targeted equations.
use crate::cyclic_gradient::CyclicGradientPolicy;
use antecedent_core::ExecutionContext;
use antecedent_kernels::gauss_jordan_solve;
use antecedent_model::continuous_cyclic::OrnsteinUhlenbeck;
use antecedent_model::nonlinear_equilibrium::EquilibriumError;
use serde::{Deserialize, Serialize};
/// Affine stochastic mechanisms dX=(M X+b)dt+L dW, with stable equation identities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuMechanisms {
    /// Stable equation ids, in variable order.
    pub equations: Vec<String>,
    /// Row-major continuous-time drift matrix M.
    pub drift: Vec<f64>,
    /// Per-equation forcing b, not the stationary mean.
    pub forcing: Vec<f64>,
    /// Row-major joint diffusion factor L; its product LL^T defines the covariance rate.
    pub diffusion_factor: Vec<f64>,
}
/// A single equation replacement, leaving every untargeted mechanism unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuEquationReplacement {
    /// Original stable equation id.
    pub equation: String,
    /// Entire incoming drift row, including its self/decay coefficient.
    pub drift: Vec<f64>,
    /// Replacement forcing.
    pub forcing: f64,
    /// Replacement row of the same joint Brownian diffusion factor.
    pub diffusion_factor: Vec<f64>,
}
/// Independent stationary replicates in one prespecified equation-intervention regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuMechanismRegime {
    /// Stable design regime identity.
    pub id: String,
    /// Distinct targeted equation replacements; empty denotes observation.
    pub replacements: Vec<OuEquationReplacement>,
    /// IID stationary observations, in endogenous variable order.
    pub values: Vec<Vec<f64>>,
}
/// A baseline mechanism parameter shared across all untargeted regimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OuMechanismParameter {
    /// A drift coefficient, including optional self/decay changes.
    Drift {
        /// Equation coordinate.
        equation: usize,
        /// Endogenous input coordinate.
        parent: usize,
    },
    /// A forcing coefficient.
    Forcing {
        /// Equation coordinate.
        equation: usize,
    },
    /// An entry of the joint Brownian diffusion factor.
    Diffusion {
        /// Equation coordinate.
        equation: usize,
        /// Brownian source coordinate.
        source: usize,
    },
}
/// Actual stationary Gaussian law in a replaced-equation regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OuMechanismLaw {
    /// Stationary mean, solving M mean+b=0.
    pub mean: Vec<f64>,
    /// Row-major stationary covariance from the Lyapunov equation.
    pub covariance: Vec<f64>,
}
/// Bounded local shared-mechanism likelihood fit, with explicit numerical status.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OuMechanismFit {
    /// Estimated baseline mechanisms; target replacements remain fixed design.
    pub model: OuMechanisms,
    /// Complete observational/interventional stationary log likelihood.
    pub log_likelihood: f64,
    /// Maximum absolute final analytic score component.
    pub score_norm: f64,
    /// Actual accepted parameter steps.
    pub iterations: usize,
    /// Actual likelihood/score evaluations.
    pub evaluations: usize,
    /// Declared score tolerance reached, without a global optimum/identification claim.
    pub converged: bool,
    /// Backtracking exhausted its bounded attempts.
    pub line_search_exhausted: bool,
    /// Actual bounded optimizer, retaining the named replacement of steepest ascent.
    pub optimizer: String,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn invert(mut matrix: Vec<f64>, n: usize) -> Result<Vec<f64>, EquilibriumError> {
    let mut output = vec![0.0; n * n];
    for i in 0..n {
        output[i * n + i] = 1.0;
    }
    if !gauss_jordan_solve(&mut matrix, &mut output, n, n) {
        return Err(EquilibriumError::Singular);
    }
    Ok((0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| output[j * n + i]).collect())
}
fn chol(matrix: &[f64], n: usize) -> Result<Vec<f64>, EquilibriumError> {
    crate::linalg::SpdFactor::new(matrix, n)
        .map(crate::linalg::SpdFactor::into_lower)
        .ok_or(EquilibriumError::Singular)
}
impl OuMechanisms {
    /// Validate the finite specified mechanism shapes and ids. Stability is a separate law check.
    pub fn validate(&self) -> Result<usize, EquilibriumError> {
        let n = self.equations.len();
        let mut ids = std::collections::BTreeSet::new();
        if n == 0
            || self.drift.len() != n * n
            || self.forcing.len() != n
            || self.diffusion_factor.len() != n * n
            || self
                .drift
                .iter()
                .chain(&self.forcing)
                .chain(&self.diffusion_factor)
                .any(|v| !v.is_finite())
            || self
                .equations
                .iter()
                .any(|id| id.trim().is_empty() || id.len() > 256 || !ids.insert(id))
        {
            return Err(invalid("OU mechanism shape/identities"));
        }
        Ok(n)
    }
    pub(crate) fn replaced(
        &self,
        replacements: &[OuEquationReplacement],
    ) -> Result<(Self, Vec<bool>), EquilibriumError> {
        let n = self.validate()?;
        let mut model = self.clone();
        let mut replaced = vec![false; n];
        for r in replacements {
            let i = self
                .equations
                .iter()
                .position(|id| *id == r.equation)
                .ok_or_else(|| invalid("unknown OU equation"))?;
            if replaced[i]
                || r.drift.len() != n
                || r.diffusion_factor.len() != n
                || !r.forcing.is_finite()
                || r.drift.iter().chain(&r.diffusion_factor).any(|v| !v.is_finite())
            {
                return Err(invalid("OU replacement shape/duplicate"));
            }
            model.drift[i * n..(i + 1) * n].copy_from_slice(&r.drift);
            model.diffusion_factor[i * n..(i + 1) * n].copy_from_slice(&r.diffusion_factor);
            model.forcing[i] = r.forcing;
            replaced[i] = true;
        }
        Ok((model, replaced))
    }
    fn process(&self) -> Result<OrnsteinUhlenbeck, EquilibriumError> {
        let n = self.validate()?;
        let mut drift = self.drift.clone();
        let mut mean: Vec<_> = self.forcing.iter().map(|x| -x).collect();
        if !gauss_jordan_solve(&mut drift, &mut mean, n, 1) {
            return Err(EquilibriumError::Singular);
        }
        let mut diffusion = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                diffusion[i * n + j] = (0..n)
                    .map(|s| self.diffusion_factor[i * n + s] * self.diffusion_factor[j * n + s])
                    .sum();
            }
        }
        Ok(OrnsteinUhlenbeck { drift: self.drift.clone(), center: mean, diffusion })
    }
    /// Stationary law after named equation replacements, with a new stability check.
    pub fn stationary(
        &self,
        replacements: &[OuEquationReplacement],
        ctx: &ExecutionContext,
    ) -> Result<OuMechanismLaw, EquilibriumError> {
        let (model, _) = self.replaced(replacements)?;
        let process = model.process()?;
        let covariance = process.stationary_covariance(ctx)?;
        Ok(OuMechanismLaw { mean: process.center, covariance })
    }
}
pub(crate) struct Summary {
    mean: Vec<f64>,
    scatter: Vec<f64>,
    count: usize,
    replacements: Vec<OuEquationReplacement>,
}
pub(crate) fn summaries(
    regimes: &[OuMechanismRegime],
    n: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<Summary>, EquilibriumError> {
    if regimes.is_empty() {
        return Err(invalid("OU regime count"));
    }
    // Each regime keeps a mean and a scatter matrix: their size is leased against the context
    // rather than capped by a regime count.
    let _summaries =
        crate::util::admit_array::<f64>(ctx, regimes.len(), n.saturating_mul(n.saturating_add(1)))
            .map_err(|error| EquilibriumError::Resource(error.to_string()))?;
    let mut ids = std::collections::BTreeSet::new();
    let mut output = Vec::new();
    for regime in regimes {
        if regime.id.trim().is_empty() || !ids.insert(&regime.id) || regime.values.is_empty() {
            return Err(invalid("OU regime identity/observations"));
        }
        let mut mean = vec![0.0; n];
        let mut scatter = vec![0.0; n * n];
        for (k, row) in regime.values.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            if row.len() != n || row.iter().any(|x| !x.is_finite()) {
                return Err(invalid("OU observation shape"));
            }
            let delta: Vec<_> = row.iter().zip(&mean).map(|(x, m)| x - m).collect();
            for (i, m) in mean.iter_mut().enumerate() {
                *m += delta[i] / (k + 1) as f64;
            }
            for i in 0..n {
                for j in 0..n {
                    scatter[i * n + j] += delta[i] * (row[j] - mean[j]);
                }
            }
        }
        output.push(Summary {
            mean,
            scatter,
            count: regime.values.len(),
            replacements: regime.replacements.clone(),
        });
    }
    Ok(output)
}
pub(crate) fn scores(
    model: &OuMechanisms,
    data: &[Summary],
    parameters: &[OuMechanismParameter],
    ctx: &ExecutionContext,
) -> Result<(f64, Vec<f64>), EquilibriumError> {
    let n = model.validate()?;
    let mut likelihood = 0.0;
    let mut score = vec![0.0; parameters.len()];
    for regime in data {
        let (replaced, mask) = model.replaced(&regime.replacements)?;
        let process = replaced.process()?;
        let covariance = process.stationary_covariance(ctx)?;
        let lower = chol(&covariance, n)?;
        let precision = invert(covariance.clone(), n)?;
        let mean = &process.center;
        let count = regime.count as f64;
        let delta: Vec<_> = regime.mean.iter().zip(mean).map(|(x, m)| x - m).collect();
        let residual: Vec<_> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| regime.scatter[i * n + j] + count * delta[i] * delta[j])
            .collect();
        let quadratic: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| precision[i * n + j] * residual[j * n + i])
            .sum();
        likelihood -= 0.5
            * (quadratic
                + count
                    * (2.0 * (0..n).map(|i| lower[i * n + i].ln()).sum::<f64>()
                        + n as f64 * (2.0 * std::f64::consts::PI).ln()));
        let mean_score: Vec<_> = (0..n)
            .map(|i| count * (0..n).map(|j| precision[i * n + j] * delta[j]).sum::<f64>())
            .collect();
        let drift_inverse = invert(replaced.drift.clone(), n)?;
        let forcing_score: Vec<_> = (0..n)
            .map(|i| -(0..n).map(|j| drift_inverse[j * n + i] * mean_score[j]).sum::<f64>())
            .collect();
        let mut right = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                right[i * n + j] = (0..n)
                    .map(|a| {
                        (0..n)
                            .map(|b| {
                                precision[i * n + a] * residual[a * n + b] * precision[b * n + j]
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    * 0.5
                    - count * precision[i * n + j] * 0.5;
            }
        }
        // G is a symmetric covariance adjoint. Explicitly project rounding
        // asymmetry before the independently strict Lyapunov direction check.
        for i in 0..n {
            for j in 0..i {
                let value = 0.5 * right[i * n + j] + 0.5 * right[j * n + i];
                right[i * n + j] = value;
                right[j * n + i] = value;
            }
        }
        // The transpose process solves M^T Lambda + Lambda M = -G.
        let transpose: Vec<_> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| replaced.drift[j * n + i])
            .collect();
        let adjoint = OrnsteinUhlenbeck {
            drift: transpose,
            center: vec![0.0; n],
            diffusion: process.diffusion.clone(),
        }
        .lyapunov_direction(&right, ctx)?;
        for (k, parameter) in parameters.iter().enumerate() {
            let (equation, value) = match *parameter {
                OuMechanismParameter::Forcing { equation } => (equation, forcing_score[equation]),
                OuMechanismParameter::Drift { equation, parent } => (
                    equation,
                    forcing_score[equation] * mean[parent]
                        + 2.0
                            * (0..n)
                                .map(|j| adjoint[equation * n + j] * covariance[j * n + parent])
                                .sum::<f64>(),
                ),
                OuMechanismParameter::Diffusion { equation, source } => (
                    equation,
                    2.0 * (0..n)
                        .map(|j| {
                            adjoint[equation * n + j] * replaced.diffusion_factor[j * n + source]
                        })
                        .sum::<f64>(),
                ),
            };
            if !mask[equation] {
                score[k] += value;
            }
        }
    }
    if !likelihood.is_finite() || score.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite OU likelihood/score"));
    }
    Ok((likelihood, score))
}
pub(crate) fn parameters(
    model: &OuMechanisms,
    coordinates: &[OuMechanismParameter],
) -> Result<Vec<f64>, EquilibriumError> {
    let n = model.validate()?;
    if coordinates.is_empty() || coordinates.len() > 1024 {
        return Err(invalid("OU parameter count"));
    }
    let mut values = Vec::new();
    for (k, p) in coordinates.iter().enumerate() {
        if coordinates[..k].contains(p) {
            return Err(invalid("duplicate OU parameter"));
        }
        values.push(match *p {
            OuMechanismParameter::Forcing { equation } => {
                *model.forcing.get(equation).ok_or_else(|| invalid("OU forcing index"))?
            }
            OuMechanismParameter::Drift { equation, parent } => {
                if equation >= n || parent >= n {
                    return Err(invalid("OU drift index"));
                }
                model.drift[equation * n + parent]
            }
            OuMechanismParameter::Diffusion { equation, source } => {
                if equation >= n || source >= n {
                    return Err(invalid("OU diffusion index"));
                }
                model.diffusion_factor[equation * n + source]
            }
        });
    }
    Ok(values)
}
pub(crate) fn bind(
    template: &OuMechanisms,
    coordinates: &[OuMechanismParameter],
    values: &[f64],
) -> OuMechanisms {
    let n = template.equations.len();
    let mut model = template.clone();
    for (p, &v) in coordinates.iter().zip(values) {
        match *p {
            OuMechanismParameter::Forcing { equation } => model.forcing[equation] = v,
            OuMechanismParameter::Drift { equation, parent } => {
                model.drift[equation * n + parent] = v
            }
            OuMechanismParameter::Diffusion { equation, source } => {
                model.diffusion_factor[equation * n + source] = v
            }
        }
    }
    model
}
/// Full stationary IID regime likelihood and analytic Lyapunov/mean scores.
pub fn ou_mechanism_likelihood_score(
    model: &OuMechanisms,
    regimes: &[OuMechanismRegime],
    coordinates: &[OuMechanismParameter],
    ctx: &ExecutionContext,
) -> Result<(f64, Vec<f64>), EquilibriumError> {
    parameters(model, coordinates)?;
    let data = summaries(regimes, model.equations.len(), ctx)?;
    scores(model, &data, coordinates, ctx)
}
/// Fit shared baseline coefficients while preserving declared intervention-specific mechanisms.
/// Statistical and structural identifiability are not implied by a numerical local optimum.
pub fn fit_ou_mechanisms(
    template: &OuMechanisms,
    coordinates: &[OuMechanismParameter],
    regimes: &[OuMechanismRegime],
    policy: CyclicGradientPolicy,
    ctx: &ExecutionContext,
) -> Result<OuMechanismFit, EquilibriumError> {
    let mut values = parameters(template, coordinates)?;
    let data = summaries(regimes, template.equations.len(), ctx)?;
    if !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_tolerance.is_finite()
        || policy.score_tolerance <= 0.0
        || policy.l1_penalty != 0.0
        || policy.max_iterations == 0
        || policy.max_iterations > 100_000
    {
        return Err(invalid("OU mechanism fit policy"));
    }
    let mut model = template.clone();
    let (mut likelihood, mut score) = scores(&model, &data, coordinates, ctx)?;
    let mut iterations = 0;
    let mut evaluations = 1;
    let mut exhausted = false;
    let mut curvature = crate::cyclic_optim::InverseCurvature::default();
    while iterations < policy.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let norm = score.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if norm <= policy.score_tolerance {
            break;
        }
        let (mut direction, quasi_newton) = curvature.direction(&score);
        let maximum = direction.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1.0);
        for value in &mut direction {
            *value /= maximum;
        }
        let derivative: f64 = score.iter().zip(&direction).map(|(g, d)| g * d).sum();
        let mut step = if quasi_newton { 1.0 } else { policy.learning_rate };
        let mut accepted = None;
        for _ in 0..48 {
            let proposed: Vec<_> =
                values.iter().zip(&direction).map(|(v, d)| v + step * d).collect();
            let candidate = bind(template, coordinates, &proposed);
            evaluations += 1;
            match scores(&candidate, &data, coordinates, ctx) {
                Ok((next, next_score)) if next >= likelihood + 1e-4 * step * derivative => {
                    accepted = Some((proposed, candidate, next, next_score));
                    break;
                }
                Err(EquilibriumError::Resource(reason)) => {
                    return Err(EquilibriumError::Resource(reason));
                }
                _ => {}
            }
            step *= 0.5;
        }
        if let Some((proposed, candidate, next, next_score)) = accepted {
            curvature.update(&values, &proposed, &score, &next_score);
            values = proposed;
            model = candidate;
            likelihood = next;
            score = next_score;
            iterations += 1;
        } else {
            exhausted = true;
            break;
        }
    }
    let score_norm = score.iter().map(|v| v.abs()).fold(0.0, f64::max);
    Ok(OuMechanismFit {
        model,
        log_likelihood: likelihood,
        score_norm,
        iterations,
        evaluations,
        converged: score_norm <= policy.score_tolerance,
        line_search_exhausted: exhausted,
        optimizer: "limited_memory_bfgs_armijo_with_symmetric_lyapunov_adjoint".into(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lyapunov_adjoints_match_independent_parameter_differences_with_equation_interventions() {
        let ctx = ExecutionContext::for_tests(1);
        let model = OuMechanisms {
            equations: vec!["a".into(), "b".into()],
            drift: vec![-1.0, 0.3, -0.2, -1.0],
            forcing: vec![1.0, 0.7],
            diffusion_factor: vec![0.4, 0.1, 0.0, 0.3],
        };
        let data = vec![
            OuMechanismRegime {
                id: "observational".into(),
                replacements: vec![],
                values: vec![vec![0.3, 1.2], vec![1.0, 0.4]],
            },
            OuMechanismRegime {
                id: "a-equation".into(),
                replacements: vec![OuEquationReplacement {
                    equation: "a".into(),
                    drift: vec![-1.0, 0.0],
                    forcing: 0.1,
                    diffusion_factor: vec![0.001, 0.0],
                }],
                values: vec![vec![0.1, 0.5], vec![0.1001, 0.6]],
            },
        ];
        let coordinates = [
            OuMechanismParameter::Drift { equation: 0, parent: 1 },
            OuMechanismParameter::Drift { equation: 1, parent: 0 },
            OuMechanismParameter::Forcing { equation: 1 },
            OuMechanismParameter::Diffusion { equation: 0, source: 0 },
            OuMechanismParameter::Diffusion { equation: 0, source: 1 },
        ];
        let (_, score) = ou_mechanism_likelihood_score(&model, &data, &coordinates, &ctx).unwrap();
        let values = parameters(&model, &coordinates).unwrap();
        for i in 0..values.len() {
            let mut high = values.clone();
            let mut low = values.clone();
            high[i] += 1e-6;
            low[i] -= 1e-6;
            let a = ou_mechanism_likelihood_score(
                &bind(&model, &coordinates, &high),
                &data,
                &coordinates,
                &ctx,
            )
            .unwrap()
            .0;
            let b = ou_mechanism_likelihood_score(
                &bind(&model, &coordinates, &low),
                &data,
                &coordinates,
                &ctx,
            )
            .unwrap()
            .0;
            assert!(
                (score[i] - (a - b) / 2e-6).abs() < 1e-4,
                "{i}: {} vs {}",
                score[i],
                (a - b) / 2e-6
            );
        }
    }
}

/// Original-position stationary-law outcome. A failed numerical/stability check
/// is not a proof of nonexistence; it is never silently dropped or renormalized.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OuMechanismOutcome {
    /// Successfully evaluated stationary Gaussian law.
    Stationary(OuMechanismLaw),
    /// Stationarity or numerical evaluation was not established.
    Failed {
        /// Stable failure classification.
        failure: String,
        /// Original numerical/stability diagnostic.
        detail: String,
    },
}
impl OuMechanisms {
    /// Retain scientific/numerical refusal in its original intervention-world slot.
    /// Malformed replacements and resource/cancellation failures still abort.
    pub fn stationary_outcome(
        &self,
        replacements: &[OuEquationReplacement],
        ctx: &ExecutionContext,
    ) -> Result<OuMechanismOutcome, EquilibriumError> {
        self.replaced(replacements)?;
        match self.stationary(replacements, ctx) {
            Ok(law) => Ok(OuMechanismOutcome::Stationary(law)),
            Err(EquilibriumError::Resource(message)) => Err(EquilibriumError::Resource(message)),
            Err(error) => Ok(OuMechanismOutcome::Failed {
                failure: "stationarity_or_numerical_conditions_not_established".into(),
                detail: error.to_string(),
            }),
        }
    }
}
