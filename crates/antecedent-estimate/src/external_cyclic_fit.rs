// SPDX-License-Identifier: MIT OR Apache-2.0
//! Likelihood fitting of feedback systems that contain registered external mechanisms.
//!
//! The joint equilibrium likelihood needs `log |det(I - J)|`; the Jacobian comes from the
//! mechanism's declared analytic gradient or its declared finite-difference policy. The
//! parameter score of the Gaussian data term is analytic when the implementation supplies a
//! parameter gradient and otherwise a central difference. The score of the determinant term is
//! analytic, through the inverse of `I - J`, when the implementation declares analytic
//! derivatives and supplies the mixed derivatives of its input gradient; otherwise it is a
//! central difference in the fitted parameters with the policy's relative step. Fitting needs a
//! verified implementation, so it takes the same trusted resolver as replay. The result is
//! `verified_external` and model-conditional; local convergence is not a global optimum.

use antecedent_core::ExecutionContext;
use antecedent_model::external_mechanism::{
    ExternalFeedbackSystem, ExternalFitParameter, MechanismTrust, ResolvedFeedbackSystem,
    TrustedMechanismResolver,
};
use antecedent_model::nonlinear_equilibrium::{EquilibriumError, RegimeRow};
use serde::{Deserialize, Serialize};

/// Bounded local gradient-ascent policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalFitPolicy {
    /// Initial backtracking step.
    pub learning_rate: f64,
    /// Stop when the largest absolute score component reaches this value.
    pub score_tolerance: f64,
    /// Maximum accepted steps.
    pub max_iterations: usize,
    /// Relative central-difference step `h = step * max(1, |theta|)` used for the
    /// determinant term (and the data term when no analytic parameter gradient exists).
    pub difference_step: f64,
}
impl Default for ExternalFitPolicy {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            score_tolerance: 1e-6,
            max_iterations: 1000,
            difference_step: 1e-6,
        }
    }
}

/// A fit with explicit numerical termination information.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalFeedbackFit {
    /// The fitted portable system; identities unchanged, parameters replaced.
    pub system: ExternalFeedbackSystem,
    /// Joint equilibrium log-likelihood at the fit.
    pub log_likelihood: f64,
    /// Accepted steps.
    pub iterations: usize,
    /// Likelihood evaluations, including perturbations for the score.
    pub evaluations: usize,
    /// Largest absolute final score component.
    pub score_norm: f64,
    /// Score tolerance reached; no global optimality certificate.
    pub converged: bool,
    /// Backtracking found no acceptable finite improvement.
    pub line_search_exhausted: bool,
    /// How the score was obtained.
    pub score_basis: String,
    /// Always `verified_external` for a system with an external mechanism.
    pub trust: MechanismTrust,
}

fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}

struct Scored {
    value: f64,
    score: Vec<f64>,
    basis: &'static str,
    evaluations: usize,
}

fn evaluate(
    resolved: &ResolvedFeedbackSystem,
    targets: &[ExternalFitParameter],
    values: &[f64],
    rows: &[RegimeRow],
    step: f64,
    ctx: &ExecutionContext,
) -> Result<Scored, EquilibriumError> {
    let terms = resolved.likelihood_terms(rows, ctx)?;
    let analytic = resolved.data_term_score(rows, targets, ctx)?;
    let analytic_determinant = if analytic.is_some() {
        resolved.determinant_term_score(rows, targets, ctx)?
    } else {
        None
    };
    let mut evaluations = 1;
    let mut score = Vec::with_capacity(targets.len());
    for j in 0..targets.len() {
        if let (Some(data), Some(determinant)) = (&analytic, &analytic_determinant) {
            score.push(data[j] + determinant[j]);
            continue;
        }
        let h = step * values[j].abs().max(1.0);
        let mut plus = values.to_vec();
        let mut minus = values.to_vec();
        plus[j] += h;
        minus[j] -= h;
        let up = resolved.with_parameters(targets, &plus)?.likelihood_terms(rows, ctx)?;
        let down = resolved.with_parameters(targets, &minus)?.likelihood_terms(rows, ctx)?;
        evaluations += 2;
        score.push(match &analytic {
            Some(data) => data[j] + (up.log_determinant - down.log_determinant) / (2.0 * h),
            None => (up.total() - down.total()) / (2.0 * h),
        });
    }
    if score.iter().any(|s| !s.is_finite()) {
        return Err(invalid("nonfinite external likelihood score"));
    }
    Ok(Scored {
        value: terms.total(),
        score,
        basis: match (&analytic, &analytic_determinant) {
            (Some(_), Some(_)) => "analytic",
            (Some(_), None) => "analytic_data_term_central_difference_determinant",
            _ => "central_difference",
        },
        evaluations,
    })
}

/// Fit the declared scalars of an external-capable feedback system from known-regime rows.
/// Fixed covariance, graph and the declared mechanism family are conditional assumptions; every
/// proposed system must carry a declared contraction for the likelihood to be a density.
pub fn fit_external_feedback(
    template: &ExternalFeedbackSystem,
    resolver: Option<&dyn TrustedMechanismResolver>,
    parameters: &[ExternalFitParameter],
    rows: &[RegimeRow],
    policy: ExternalFitPolicy,
    ctx: &ExecutionContext,
) -> Result<ExternalFeedbackFit, EquilibriumError> {
    if rows.is_empty()
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_tolerance.is_finite()
        || policy.score_tolerance <= 0.0
        || !(policy.difference_step.is_finite()
            && policy.difference_step > 0.0
            && policy.difference_step <= 1e-2)
        || policy.max_iterations == 0
        || policy.max_iterations > 100_000
    {
        return Err(invalid("external fit policy/data"));
    }
    let mut resolved = template.resolve(resolver)?;
    let mut values = template.parameter_values(parameters)?;
    let mut current = evaluate(&resolved, parameters, &values, rows, policy.difference_step, ctx)?;
    let mut evaluations = current.evaluations;
    let mut iterations = 0;
    let mut exhausted = false;
    while iterations < policy.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let norm = current.score.iter().map(|s| s.abs()).fold(0.0, f64::max);
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
            let proposed: Vec<f64> =
                values.iter().zip(&current.score).map(|(v, g)| v + step * g / scale).collect();
            let predicted: f64 = proposed
                .iter()
                .zip(&values)
                .zip(&current.score)
                .map(|((next, old), g)| (next - old) * g)
                .sum();
            if predicted <= 0.0 {
                step *= 0.5;
                continue;
            }
            let candidate = resolved.with_parameters(parameters, &proposed)?;
            match candidate.likelihood_terms(rows, ctx) {
                Ok(terms) if terms.total() >= current.value + 1e-4 * predicted => {
                    let scored = evaluate(
                        &candidate,
                        parameters,
                        &proposed,
                        rows,
                        policy.difference_step,
                        ctx,
                    )?;
                    evaluations += 1 + scored.evaluations;
                    accepted = Some((proposed, candidate, scored));
                    break;
                }
                Ok(_) => evaluations += 1,
                Err(EquilibriumError::Uncertified { .. } | EquilibriumError::Singular) => {
                    evaluations += 1;
                }
                Err(error) => return Err(error),
            }
            step *= 0.5;
        }
        if let Some((next_values, next_resolved, scored)) = accepted {
            values = next_values;
            resolved = next_resolved;
            current = scored;
            iterations += 1;
        } else {
            exhausted = true;
            break;
        }
    }
    let score_norm = current.score.iter().map(|s| s.abs()).fold(0.0, f64::max);
    Ok(ExternalFeedbackFit {
        system: resolved.system().clone(),
        log_likelihood: current.value,
        iterations,
        evaluations,
        score_norm,
        converged: score_norm <= policy.score_tolerance,
        line_search_exhausted: exhausted,
        score_basis: current.basis.into(),
        trust: resolved.trust(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::external_mechanism::{
        DerivativeDeclaration, DifferenceScheme, ExternalMechanismImplementation,
        ExternalMechanismSpec, MechanismRegistry, SystemEquation,
    };
    use antecedent_model::nonlinear_equilibrium::{Activation, Clamp, Equation};
    use std::sync::Arc;

    /// `p0 * tanh(x)`: a saturating response with a global slope bound of `|p0|` <= 0.9.
    struct Saturating {
        spec: ExternalMechanismSpec,
    }
    const CONTENT: &[u8] = b"lab/saturating v1: p0 * tanh(x)";
    impl Saturating {
        fn new(derivative: DerivativeDeclaration) -> Arc<Self> {
            let mut spec = ExternalMechanismSpec::for_content(
                "lab/saturating",
                "1",
                CONTENT,
                vec![1],
                vec!["gain".into()],
                derivative,
            );
            spec.input_lipschitz = Some(vec![0.9]);
            Arc::new(Self { spec })
        }
    }
    impl ExternalMechanismImplementation for Saturating {
        fn spec(&self) -> ExternalMechanismSpec {
            self.spec.clone()
        }
        fn content(&self) -> &[u8] {
            CONTENT
        }
        fn value(&self, inputs: &[f64], parameters: &[f64]) -> Result<f64, EquilibriumError> {
            Ok(parameters[0] * inputs[0].tanh())
        }
        fn input_gradient(
            &self,
            inputs: &[f64],
            parameters: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![parameters[0] * (1.0 - inputs[0].tanh().powi(2))]))
        }
        fn parameter_gradient(
            &self,
            inputs: &[f64],
            _parameters: &[f64],
        ) -> Result<Option<Vec<f64>>, EquilibriumError> {
            Ok(Some(vec![inputs[0].tanh()]))
        }
        fn input_gradient_parameter_derivative(
            &self,
            inputs: &[f64],
            _parameters: &[f64],
        ) -> Result<Option<Vec<Vec<f64>>>, EquilibriumError> {
            Ok(Some(vec![vec![1.0 - inputs[0].tanh().powi(2)]]))
        }
    }

    fn template(spec: &ExternalMechanismSpec) -> ExternalFeedbackSystem {
        ExternalFeedbackSystem {
            equations: vec![
                SystemEquation::External {
                    id: "response".into(),
                    intercept: 0.0,
                    parameters: vec![0.0],
                    mechanism: spec.clone(),
                },
                SystemEquation::Builtin {
                    equation: Equation {
                        id: "feedback".into(),
                        intercept: 0.0,
                        weights: vec![0.2, 0.0],
                        activation: Activation::Tanh,
                        expression: None,
                        parameters: vec![],
                    },
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        }
    }

    fn rows() -> Vec<RegimeRow> {
        [-2.0_f64, -1.0, 0.5, 1.0, 2.0]
            .into_iter()
            .flat_map(|x| {
                [-0.1, 0.1].map(|noise| RegimeRow {
                    values: vec![0.7 * x.tanh() + noise, x],
                    clamps: vec![Clamp { variable: 1, value: x }],
                })
            })
            .collect()
    }

    #[test]
    fn analytic_and_finite_difference_fits_recover_the_external_parameter() {
        let ctx = ExecutionContext::for_tests(7);
        let targets = [ExternalFitParameter::Parameter { equation: 0, parameter: 0 }];
        let mut fitted = Vec::new();
        for derivative in [
            DerivativeDeclaration::Analytic,
            DerivativeDeclaration::FiniteDifference {
                step: 1e-6,
                scheme: DifferenceScheme::Central,
            },
        ] {
            let implementation = Saturating::new(derivative);
            let mut registry = MechanismRegistry::new();
            registry.register(implementation.clone()).unwrap();
            let fit = fit_external_feedback(
                &template(&implementation.spec),
                Some(&registry),
                &targets,
                &rows(),
                ExternalFitPolicy::default(),
                &ctx,
            )
            .unwrap();
            assert!(fit.converged, "{fit:?}");
            assert_eq!(fit.trust, MechanismTrust::VerifiedExternal);
            let SystemEquation::External { parameters, .. } = &fit.system.equations[0] else {
                panic!("external equation");
            };
            // Clamped regressor design: the maximum-likelihood gain is the least-squares slope.
            let xs: Vec<f64> = rows().iter().map(|r| r.values[1].tanh()).collect();
            let ys: Vec<f64> = rows().iter().map(|r| r.values[0]).collect();
            let closed: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum::<f64>()
                / xs.iter().map(|x| x * x).sum::<f64>();
            assert!((parameters[0] - closed).abs() < 1e-6);
            fitted.push(parameters[0]);
        }
        assert!((fitted[0] - fitted[1]).abs() < 1e-6);
    }

    #[test]
    fn fitting_without_a_resolver_is_refused() {
        let implementation = Saturating::new(DerivativeDeclaration::Analytic);
        let error = fit_external_feedback(
            &template(&implementation.spec),
            None,
            &[ExternalFitParameter::Parameter { equation: 0, parameter: 0 }],
            &rows(),
            ExternalFitPolicy::default(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap_err();
        assert!(error.to_string().contains("mechanism_resolver_missing"));
    }

    #[test]
    fn analytic_determinant_score_matches_a_central_difference_in_a_loop() {
        let ctx = ExecutionContext::for_tests(7);
        let implementation = Saturating::new(DerivativeDeclaration::Analytic);
        let mut registry = MechanismRegistry::new();
        registry.register(implementation.clone()).unwrap();
        let mut truth = template(&implementation.spec);
        let SystemEquation::External { parameters, .. } = &mut truth.equations[0] else {
            panic!("external equation");
        };
        parameters[0] = 0.7;
        let generator = truth.resolve(Some(&registry)).unwrap();
        // Unclamped equilibrium rows through the loop, so the determinant depends on the gain.
        let rows: Vec<RegimeRow> = [[0.3, -0.2], [-0.5, 0.4], [0.1, 0.6], [0.8, -0.7]]
            .iter()
            .map(|u| {
                let eq = generator
                    .solve(
                        u,
                        &[],
                        &[0.0, 0.0],
                        antecedent_model::nonlinear_equilibrium::SolvePolicy::default(),
                        &ctx,
                    )
                    .unwrap();
                RegimeRow { values: eq.values, clamps: vec![] }
            })
            .collect();
        let targets = [ExternalFitParameter::Parameter { equation: 0, parameter: 0 }];
        let at = template(&implementation.spec).with_parameters(&targets, &[0.5]).unwrap();
        let resolved = at.resolve(Some(&registry)).unwrap();
        let analytic = resolved.determinant_term_score(&rows, &targets, &ctx).unwrap().unwrap();
        let h = 1e-6;
        let det = |g: f64| {
            resolved
                .with_parameters(&targets, &[g])
                .unwrap()
                .likelihood_terms(&rows, &ctx)
                .unwrap()
                .log_determinant
        };
        let difference = (det(0.5 + h) - det(0.5 - h)) / (2.0 * h);
        assert!(difference.abs() > 1e-3);
        assert!((analytic[0] - difference).abs() < 1e-7, "{analytic:?} {difference}");
        let fit = fit_external_feedback(
            &template(&implementation.spec),
            Some(&registry),
            &targets,
            &rows,
            ExternalFitPolicy::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(fit.score_basis, "analytic");
        assert!(fit.converged);
    }
}
