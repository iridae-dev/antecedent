// SPDX-License-Identifier: MIT OR Apache-2.0
//! Analytic joint-equilibrium likelihood fitting, including mixed derivatives
//! of compositional mechanisms in the intervention-specific Jacobian term.
use crate::cyclic_likelihood::{CyclicParameter, parameter_values, with_parameters};
use antecedent_core::ExecutionContext;
use antecedent_model::nonlinear_equilibrium::{EquilibriumError, EquilibriumModel, RegimeRow};
use serde::{Deserialize, Serialize};

/// Bounded local gradient optimization policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicGradientPolicy {
    /// Initial Armijo backtracking step at each iteration.
    pub learning_rate: f64,
    /// Stop when the largest absolute score component reaches this value.
    pub score_tolerance: f64,
    /// Maximum accepted gradient steps.
    pub max_iterations: usize,
    /// L1 penalty on legacy incoming coefficients; zero denotes ordinary likelihood.
    #[serde(default)]
    pub l1_penalty: f64,
}
impl Default for CyclicGradientPolicy {
    fn default() -> Self {
        Self { learning_rate: 0.1, score_tolerance: 1e-6, max_iterations: 1000, l1_penalty: 0.0 }
    }
}
/// Local fit with explicit numerical termination information.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicGradientFit {
    /// Fitted specified graph and joint disturbance law.
    pub model: EquilibriumModel,
    /// Unpenalized likelihood, including free-system Jacobians.
    pub log_likelihood: f64,
    /// Actual maximized likelihood minus the declared L1 penalty.
    pub objective: f64,
    /// Actual accepted optimization steps.
    pub iterations: usize,
    /// Actual objective/score evaluations.
    pub evaluations: usize,
    /// Maximum absolute final score component.
    pub score_norm: f64,
    /// Score tolerance reached; no global optimality certificate.
    pub converged: bool,
    /// Backtracking found no acceptable finite, licensed improvement.
    pub line_search_exhausted: bool,
    /// Explicit almost-everywhere derivative convention for piecewise-linear mechanisms.
    pub nonsmooth_convention: Option<String>,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
/// Fit a differentiable additive-noise equilibrium family from known regimes.
/// Fixed covariance and specified graph are conditional assumptions. Every
/// proposed model must satisfy the likelihood's contraction or affine-bijection conditions.
/// L1 fitting uses proximal backtracking and an explicit KKT score tolerance.
#[allow(clippy::too_many_lines,reason="one bounded optimizer loop retains feasibility, objective and stopping diagnostics together")]
pub fn fit_cyclic_gradient(
    template: &EquilibriumModel,
    parameters: &[CyclicParameter],
    rows: &[RegimeRow],
    policy: CyclicGradientPolicy,
    ctx: &ExecutionContext,
) -> Result<CyclicGradientFit, EquilibriumError> {
    let mut values = parameter_values(template, parameters)?;
    if rows.is_empty()
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_tolerance.is_finite()
        || policy.score_tolerance <= 0.0
        || !policy.l1_penalty.is_finite()
        || policy.l1_penalty < 0.0
        || policy.max_iterations == 0
        || policy.max_iterations > 100_000
    {
        return Err(invalid("gradient fit policy/data"));
    }
    let linear = if template.equations.iter().all(|e| {
        e.expression.is_none()
            && e.activation == antecedent_model::nonlinear_equilibrium::Activation::Identity
    }) {
        Some(antecedent_model::linear_moments::LinearRegimeMoments::from_rows(
            rows,
            template.order(),
            ctx,
        )?)
    } else {
        None
    };
    let evaluate = |model: &EquilibriumModel| -> Result<(f64, Vec<f64>), EquilibriumError> {
        let (likelihood, scores) = if let Some(moments) = &linear {
            moments.likelihood_score(model, ctx)?
        } else {
            model.likelihood_score(rows, ctx)?
        };
        let selected = parameters
            .iter()
            .map(|p| match *p {
                CyclicParameter::Intercept { equation } => scores[equation].intercept,
                CyclicParameter::Weight { equation, parent } => scores[equation].weights[parent],
                CyclicParameter::ExpressionParameter { equation, parameter } => {
                    scores[equation].parameters[parameter]
                }
            })
            .collect();
        Ok((likelihood, selected))
    };
    let penalized: Vec<_> =
        parameters.iter().map(|p| matches!(p, CyclicParameter::Weight { .. })).collect();
    let objective = |likelihood: f64, values: &[f64]| {
        likelihood
            - policy.l1_penalty
                * values
                    .iter()
                    .zip(&penalized)
                    .filter_map(|(v, p)| p.then_some(v.abs()))
                    .sum::<f64>()
    };
    let kkt = |score: &[f64], values: &[f64]| {
        score
            .iter()
            .zip(values)
            .zip(&penalized)
            .map(|((&g, &v), &p)| {
                if !p {
                    g
                } else if v > 0.0 {
                    g - policy.l1_penalty
                } else if v < 0.0 {
                    g + policy.l1_penalty
                } else {
                    g.signum() * (g.abs() - policy.l1_penalty).max(0.0)
                }
            })
            .collect::<Vec<_>>()
    };
    let mut model = template.clone();
    let (mut likelihood, mut score) = evaluate(&model)?;
    let mut value = objective(likelihood, &values);
    let mut evaluations = 1;
    let mut iterations = 0;
    let mut exhausted = false;
    while iterations < policy.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let norm = kkt(&score, &values).iter().map(|s| s.abs()).fold(0.0, f64::max);
        if norm <= policy.score_tolerance {
            break;
        }
        let scale = norm.max(1.0);
        let mut step = policy.learning_rate;
        let mut accepted = None;
        for _ in 0..48 {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let proposed: Vec<_> = values
                .iter()
                .zip(&score)
                .zip(&penalized)
                .map(|((&v, &g), &p)| {
                    let next = v + step * g / scale;
                    if p {
                        next.signum() * (next.abs() - step * policy.l1_penalty / scale).max(0.0)
                    } else {
                        next
                    }
                })
                .collect();
            let change: f64 = proposed.iter().zip(&values).map(|(a, b)| (a - b).abs()).sum();
            if change == 0.0 {
                step *= 0.5;
                continue;
            }
            let predicted: f64 = proposed
                .iter()
                .zip(&values)
                .zip(&score)
                .map(|((next, old), g)| (next - old) * g)
                .sum::<f64>()
                - policy.l1_penalty
                    * proposed
                        .iter()
                        .zip(&values)
                        .zip(&penalized)
                        .filter_map(|((next, old), p)| p.then_some(next.abs() - old.abs()))
                        .sum::<f64>();
            let candidate = with_parameters(template, parameters, &proposed)?;
            evaluations += 1;
            match evaluate(&candidate) {
                Ok((next, next_score))
                    if objective(next, &proposed) >= value + 1e-4 * predicted.max(0.0) =>
                {
                    accepted = Some((proposed, candidate, next, next_score));
                    break;
                }
                Ok(_) | Err(EquilibriumError::Uncertified { .. } | EquilibriumError::Singular) => {}
                Err(error) => return Err(error),
            }
            step *= 0.5;
        }
        if let Some((next_values, next_model, next, next_score)) = accepted {
            values = next_values;
            model = next_model;
            likelihood = next;
            score = next_score;
            value = objective(likelihood, &values);
            iterations += 1;
        } else {
            exhausted = true;
            break;
        }
    }
    let score_norm = kkt(&score, &values).iter().map(|s| s.abs()).fold(0.0, f64::max);
    Ok(CyclicGradientFit {
        model,
        log_likelihood: likelihood,
        objective: value,
        iterations,
        evaluations,
        score_norm,
        converged: score_norm <= policy.score_tolerance,
        line_search_exhausted: exhausted,
        nonsmooth_convention: template.equations.iter().any(|e| e.activation == antecedent_model::nonlinear_equilibrium::Activation::Relu).then(|| "ReLU Jacobians and parameter scores are almost-everywhere derivatives; the zero derivative is used at a kink. Score convergence is not differentiability or global optimality.".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::{
        mechanism_expr::MechanismExpression as E,
        nonlinear_equilibrium::{Activation, Equation},
    };
    #[test]
    fn analytic_fit_recovers_expression_parameter_from_intervention_design() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "response".into(),
                    intercept: 0.0,
                    weights: vec![0.0; 2],
                    activation: Activation::Identity,
                    expression: Some(E::Multiply {
                        factors: vec![
                            E::Parameter { parameter: 0 },
                            E::Tanh { argument: Box::new(E::Variable { variable: 1 }) },
                        ],
                    }),
                    parameters: vec![0.0],
                },
                Equation {
                    id: "feedback".into(),
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
                    values: vec![0.7 * x.tanh() + noise, x],
                    clamps: vec![antecedent_model::nonlinear_equilibrium::Clamp {
                        variable: 1,
                        value: x,
                    }],
                })
            })
            .collect();
        let fit = fit_cyclic_gradient(
            &model,
            &[CyclicParameter::ExpressionParameter { equation: 0, parameter: 0 }],
            &rows,
            CyclicGradientPolicy::default(),
            &ExecutionContext::for_tests(42),
        )
        .unwrap();
        assert!(fit.converged, "{fit:?}");
        assert!((fit.model.equations[0].parameters[0] - 0.7).abs() < 1e-6);
    }
}

#[cfg(test)]
mod l1_tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Clamp, Equation};
    #[test]
    fn proximal_kkt_fit_preserves_the_declared_l1_objective() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        let rows = vec![
            RegimeRow {
                values: vec![-1.0, -0.7],
                clamps: vec![Clamp { variable: 0, value: -1.0 }],
            },
            RegimeRow { values: vec![1.0, 0.7], clamps: vec![Clamp { variable: 0, value: 1.0 }] },
        ];
        let fit = fit_cyclic_gradient(
            &model,
            &[CyclicParameter::Weight { equation: 1, parent: 0 }],
            &rows,
            CyclicGradientPolicy { l1_penalty: 0.4, ..CyclicGradientPolicy::default() },
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!(fit.converged, "{fit:?}");
        assert!((fit.model.equations[1].weights[0] - 0.5).abs() < 1e-6);
        assert!((fit.objective - (fit.log_likelihood - 0.2)).abs() < 1e-6);
    }
}
