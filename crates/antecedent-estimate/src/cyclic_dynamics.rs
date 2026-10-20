// SPDX-License-Identifier: MIT OR Apache-2.0
//! Likelihood fitting of deterministic nonlinear ODE feedback using analytic
//! forward sensitivities. Independent trajectories have explicit fixed initial
//! conditions and known independent Gaussian measurement variances. This is
//! not an SDE/process-noise estimator, nor nonparametric identification.

use antecedent_core::ExecutionContext;
use antecedent_model::{
    continuous_cyclic::{OdeParameter, OdePolicy, OdeSystem, TimedClamp},
    nonlinear_equilibrium::EquilibriumError,
};
use serde::{Deserialize, Serialize};

/// One independently observed trajectory with a fixed initial state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeObservation {
    /// State at the first observation time, not estimated from later data.
    pub initial: Vec<f64>,
    /// Strictly increasing times; at least one time after the initial time.
    pub times: Vec<f64>,
    /// Measurements at every time/state coordinate.
    pub values: Vec<Vec<f64>>,
    /// Known positive measurement variance per state coordinate.
    pub variances: Vec<f64>,
    /// Prespecified time-indexed mechanism replacements.
    pub schedule: Vec<TimedClamp>,
}

/// Local gradient-ascent policy; integration policy is specified separately.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeFitPolicy {
    /// Initial backtracking learning rate per iteration.
    pub learning_rate: f64,
    /// Largest absolute score component accepted at convergence.
    pub score_tolerance: f64,
    /// Maximum likelihood/gradient iterations.
    pub max_iterations: usize,
}
impl Default for OdeFitPolicy {
    fn default() -> Self {
        Self { learning_rate: 0.1, score_tolerance: 1e-6, max_iterations: 1000 }
    }
}

/// A retained local fit attempt with numerical score diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeFit {
    /// Estimated drift mechanisms.
    pub model: OdeSystem,
    /// Complete Gaussian measurement log likelihood.
    pub log_likelihood: f64,
    /// Actual gradient iterations.
    pub iterations: usize,
    /// Final maximum absolute score component.
    pub score_norm: f64,
    /// Score tolerance was reached; no global optimality/calibration is implied.
    pub converged: bool,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn bind(template: &OdeSystem, parameters: &[OdeParameter], values: &[f64]) -> OdeSystem {
    let mut model = template.clone();
    for (&parameter, &value) in parameters.iter().zip(values) {
        match parameter {
            OdeParameter::ExpressionParameter { equation, parameter } => {
                model.equations[equation].parameters[parameter] = value
            }
            OdeParameter::Intercept { equation } => model.equations[equation].intercept = value,
            OdeParameter::Weight { equation, parent } => {
                model.equations[equation].weights[parent] = value
            }
            OdeParameter::Decay { equation } => model.decay[equation] = value,
        }
    }
    model
}
fn likelihood_score(
    model: &OdeSystem,
    parameters: &[OdeParameter],
    data: &[OdeObservation],
    integration: OdePolicy,
    ctx: &ExecutionContext,
) -> Result<(f64, Vec<f64>), EquilibriumError> {
    let n = model.equations.len();
    let p = parameters.len();
    let mut likelihood = 0.0;
    let mut score = vec![0.0; p];
    for observed in data {
        if observed.times.len() < 2
            || observed.values.len() != observed.times.len()
            || observed.variances.len() != n
            || observed
                .values
                .iter()
                .any(|row| row.len() != n || row.iter().any(|v| !v.is_finite()))
            || observed.variances.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid("ODE observation model shape/variance"));
        }
        let path = model.sensitivities(
            &observed.initial,
            &observed.times,
            &observed.schedule,
            parameters,
            integration,
            ctx,
        )?;
        // The first state is supplied exactly by design, so it is not reused as
        // a noisy observation in the measurement likelihood.
        for t in 1..observed.times.len() {
            for i in 0..n {
                let residual = observed.values[t][i] - path.trajectory.states[t][i];
                let variance = observed.variances[i];
                likelihood -= 0.5
                    * (residual * residual / variance
                        + (2.0 * std::f64::consts::PI * variance).ln());
                for (j, component) in score.iter_mut().enumerate() {
                    *component += residual * path.sensitivities[t][i * p + j] / variance;
                }
            }
        }
    }
    if !likelihood.is_finite() || score.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite ODE likelihood/score"));
    }
    Ok((likelihood, score))
}

/// Fit specified ODE drift parameters jointly across independent trajectories
/// using forward sensitivities and Armijo backtracking. The drift structure,
/// initial conditions, action schedule and observation variances remain fixed.
#[allow(
    clippy::too_many_lines,
    reason = "bounded sensitivity optimizer retains validation, line search and convergence facts in one pass"
)]
pub fn fit_ode_likelihood(
    template: &OdeSystem,
    parameters: &[OdeParameter],
    data: &[OdeObservation],
    integration: OdePolicy,
    policy: OdeFitPolicy,
    ctx: &ExecutionContext,
) -> Result<OdeFit, EquilibriumError> {
    template.validate()?;
    if data.is_empty()
        || data.len() > 10_000
        || parameters.is_empty()
        || parameters.len() > 64
        || !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_tolerance.is_finite()
        || policy.score_tolerance <= 0.0
        || policy.max_iterations == 0
        || policy.max_iterations > 100_000
    {
        return Err(invalid("ODE fit policy/data"));
    }
    // Validate coordinates before indexing into template arrays.
    let n = template.equations.len();
    for (j, &parameter) in parameters.iter().enumerate() {
        if parameters[..j].contains(&parameter) {
            return Err(invalid("duplicate ODE parameter"));
        }
        match parameter {
            OdeParameter::ExpressionParameter { equation, parameter }
                if template
                    .equations
                    .get(equation)
                    .and_then(|e| e.parameters.get(parameter))
                    .is_none() =>
            {
                return Err(invalid("ODE expression parameter"));
            }
            OdeParameter::Weight { equation, .. }
                if template.equations.get(equation).is_some_and(|e| e.expression.is_some()) =>
            {
                return Err(invalid("compositional mechanisms use expression parameters"));
            }
            OdeParameter::Intercept { equation } | OdeParameter::Decay { equation }
                if equation >= n =>
            {
                return Err(invalid("ODE parameter equation"));
            }
            OdeParameter::Weight { equation, parent } if equation >= n || parent >= n => {
                return Err(invalid("ODE parameter coefficient"));
            }
            _ => {}
        }
    }
    let mut values: Vec<_> = parameters
        .iter()
        .map(|parameter| match *parameter {
            OdeParameter::ExpressionParameter { equation, parameter } => {
                template.equations[equation].parameters[parameter]
            }
            OdeParameter::Intercept { equation } => template.equations[equation].intercept,
            OdeParameter::Weight { equation, parent } => {
                template.equations[equation].weights[parent]
            }
            OdeParameter::Decay { equation } => template.decay[equation],
        })
        .collect();
    let mut model = template.clone();
    let (mut likelihood, mut score) = likelihood_score(&model, parameters, data, integration, ctx)?;
    let mut iterations = 0;
    while iterations < policy.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let norm = score.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if norm <= policy.score_tolerance {
            break;
        }
        iterations += 1;
        let mut rate = policy.learning_rate;
        let squared = score.iter().map(|v| v * v).sum::<f64>();
        let mut accepted = None;
        for _ in 0..40 {
            let next: Vec<_> = values.iter().zip(&score).map(|(v, g)| v + rate * g).collect();
            let candidate = bind(template, parameters, &next);
            match likelihood_score(&candidate, parameters, data, integration, ctx) {
                Ok((ll, gradient)) if ll >= likelihood + 1e-4 * rate * squared => {
                    accepted = Some((next, candidate, ll, gradient));
                    break;
                }
                Err(EquilibriumError::Resource(message)) => {
                    return Err(EquilibriumError::Resource(message));
                }
                _ => {}
            }
            rate *= 0.5;
        }
        let Some((next, candidate, ll, gradient)) = accepted else {
            break;
        };
        values = next;
        model = candidate;
        likelihood = ll;
        score = gradient;
    }
    let score_norm = score.iter().map(|v| v.abs()).fold(0.0, f64::max);
    Ok(OdeFit {
        model,
        log_likelihood: likelihood,
        iterations,
        score_norm,
        converged: score_norm <= policy.score_tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{Activation, Equation};
    #[test]
    fn ode_sensitivity_fit_recovers_decay_from_noisy_independent_trajectories() {
        let template = OdeSystem {
            equations: vec![Equation {
                id: "drift".into(),
                intercept: 0.0,
                weights: vec![0.0],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            }],
            decay: vec![1.0],
        };
        let data: Vec<_> = (0..8)
            .map(|i| {
                let initial = 1.0 + i as f64 / 4.0;
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                OdeObservation {
                    initial: vec![initial],
                    times: vec![0.0, 0.5, 1.0],
                    values: vec![
                        vec![initial],
                        vec![initial * (-1.0_f64).exp() + 0.005 * sign],
                        vec![initial * (-2.0_f64).exp() - 0.005 * sign],
                    ],
                    variances: vec![0.01],
                    schedule: vec![],
                }
            })
            .collect();
        let parameters = [OdeParameter::Decay { equation: 0 }];
        let ctx = ExecutionContext::for_tests(0);
        let fit = fit_ode_likelihood(
            &template,
            &parameters,
            &data,
            Default::default(),
            OdeFitPolicy { score_tolerance: 1e-5, ..Default::default() },
            &ctx,
        )
        .unwrap();
        assert!(fit.converged, "{fit:?}");
        assert!((fit.model.decay[0] - 2.0).abs() < 0.02);
        let (_, score) =
            likelihood_score(&template, &parameters, &data, Default::default(), &ctx).unwrap();
        let h = 1e-5;
        let high = bind(&template, &parameters, &[1.0 + h]);
        let low = bind(&template, &parameters, &[1.0 - h]);
        let finite_difference =
            (likelihood_score(&high, &parameters, &data, Default::default(), &ctx).unwrap().0
                - likelihood_score(&low, &parameters, &data, Default::default(), &ctx).unwrap().0)
                / (2.0 * h);
        assert!((score[0] - finite_difference).abs() < 1e-3);
    }
}

/// A stationary OU parameter coordinate; diffusion remains specified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OuParameter {
    /// Continuous-time drift matrix entry.
    Drift {
        /// Row coordinate.
        row: usize,
        /// Column coordinate.
        column: usize,
    },
    /// Unintervened equilibrium center.
    Center {
        /// State coordinate.
        variable: usize,
    },
}

/// Conditional stationary OU likelihood fit; no structure-identification claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuFit {
    /// Estimated continuous-time parameters with fixed diffusion.
    pub model: antecedent_model::continuous_cyclic::OrnsteinUhlenbeck,
    /// Joint likelihood under known sustained intervention regimes.
    pub log_likelihood: f64,
    /// Full coordinate sweeps used.
    pub sweeps: usize,
    /// Final coordinate displacement reached the declared tolerance.
    pub converged: bool,
}

/// Fit specified OU drift/center coordinates from independent stationary
/// replicates with known intervention regimes. Observational stationary data
/// need not identify every drift coordinate; numerical fit is not identification.
pub fn fit_ou_likelihood(
    template: &antecedent_model::continuous_cyclic::OrnsteinUhlenbeck,
    parameters: &[OuParameter],
    rows: &[antecedent_model::nonlinear_equilibrium::RegimeRow],
    options: crate::cyclic_likelihood::CyclicFitOptions,
    ctx: &ExecutionContext,
) -> Result<OuFit, EquilibriumError> {
    use crate::cyclic_likelihood::optimize_coordinates;
    template.stationary_log_likelihood(rows, ctx)?;
    let n = template.center.len();
    if rows.is_empty()
        || parameters.is_empty()
        || parameters.len() > 256
        || !options.initial_step.is_finite()
        || options.initial_step <= 0.0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.tolerance >= options.initial_step
        || options.max_sweeps == 0
        || options.max_sweeps > 100_000
        || options.l1_penalty != 0.0
    {
        return Err(invalid("OU fit policy/data"));
    }
    let mut initial = Vec::new();
    for (j, &parameter) in parameters.iter().enumerate() {
        if parameters[..j].contains(&parameter) {
            return Err(invalid("duplicate OU parameter"));
        }
        initial.push(match parameter {
            OuParameter::Drift { row, column } if row < n && column < n => {
                template.drift[row * n + column]
            }
            OuParameter::Center { variable } if variable < n => template.center[variable],
            _ => return Err(invalid("OU parameter coordinate")),
        });
    }
    let bind = |values: &[f64]| {
        let mut model = template.clone();
        for (&p, &v) in parameters.iter().zip(values) {
            match p {
                OuParameter::Drift { row, column } => model.drift[row * n + column] = v,
                OuParameter::Center { variable } => model.center[variable] = v,
            }
        }
        model
    };
    let target = |values: &[f64]| match bind(values).stationary_log_likelihood(rows, ctx) {
        Ok(ll) => Ok(ll),
        Err(e @ EquilibriumError::Resource(_)) => Err(e),
        Err(_) => Ok(f64::NEG_INFINITY),
    };
    let (values, _, sweeps, _, converged) = optimize_coordinates(initial, options, ctx, target)?;
    let model = bind(&values);
    let log_likelihood = model.stationary_log_likelihood(rows, ctx)?;
    Ok(OuFit { model, log_likelihood, sweeps, converged })
}

#[cfg(test)]
mod ou_fit_tests {
    use super::*;
    use antecedent_model::{
        continuous_cyclic::OrnsteinUhlenbeck, nonlinear_equilibrium::RegimeRow,
    };
    #[test]
    fn specified_diffusion_scalar_stationary_fit_recovers_drift_and_center() {
        let template =
            OrnsteinUhlenbeck { drift: vec![-1.0], center: vec![0.0], diffusion: vec![4.0] };
        let ctx = ExecutionContext::for_tests(77);
        let mut rng = ctx.rng.stream(783);
        let rows: Vec<_> = (0..800)
            .map(|_| RegimeRow {
                values: vec![0.5 + antecedent_kernels::standard_normal(&mut rng)],
                clamps: vec![],
            })
            .collect();
        let fit = fit_ou_likelihood(
            &template,
            &[OuParameter::Drift { row: 0, column: 0 }, OuParameter::Center { variable: 0 }],
            &rows,
            Default::default(),
            &ctx,
        )
        .unwrap();
        assert!(fit.converged);
        assert!((fit.model.drift[0] + 2.0).abs() < 0.3);
        assert!((fit.model.center[0] - 0.5).abs() < 0.12);
    }
}
