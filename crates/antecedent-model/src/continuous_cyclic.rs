// SPDX-License-Identifier: MIT OR Apache-2.0
//! Continuous-time feedback, distinct from static fixed-noise equilibria.
//! ODEs use adaptive Dormand-Prince 5(4). OU models use a Lyapunov solve and
//! finite-horizon moment ODEs. Convergence is numerical; a stationary law is
//! licensed only when a positive-definite Lyapunov witness verifies stability.

use crate::nonlinear_equilibrium::{
    Clamp, Equation, EquilibriumError, check_cancel, cholesky, dot,
};
use antecedent_core::ExecutionContext;
use antecedent_kernels::gauss_jordan_solve;
use serde::{Deserialize, Serialize};

fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}

/// Autonomous vector field `dx_i/dt = c_i + activation_i(w_i x) - decay_i*x_i`.
/// Self-dependence in `weights` is allowed for this continuous-time family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeSystem {
    /// Named drift mechanisms.
    pub equations: Vec<Equation>,
    /// Additional linear decay per variable.
    pub decay: Vec<f64>,
}

/// A sustained hard clamp starting at an explicit time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimedClamp {
    /// Absolute action time.
    pub time: f64,
    /// Mechanisms replaced from this time onward.
    pub clamps: Vec<Clamp>,
}

/// Adaptive integration and work policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdePolicy {
    /// Absolute local-error scale.
    pub absolute_tolerance: f64,
    /// Relative local-error scale.
    pub relative_tolerance: f64,
    /// Initial proposed time step.
    pub initial_step: f64,
    /// Maximum accepted plus rejected steps.
    pub max_steps: usize,
}
impl Default for OdePolicy {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-9,
            relative_tolerance: 1e-7,
            initial_step: 0.01,
            max_steps: 100_000,
        }
    }
}

/// Finite-horizon trajectory and actual step counts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeTrajectory {
    /// Requested observation times, including the initial time.
    pub times: Vec<f64>,
    /// States at those times, after actions at exactly that time.
    pub states: Vec<Vec<f64>>,
    /// Accepted integration steps.
    pub accepted_steps: usize,
    /// Rejected integration steps.
    pub rejected_steps: usize,
}

impl OdeSystem {
    /// Validate finite drift parameters, dimensions and mechanism identities.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        let n = self.equations.len();
        let mut ids = std::collections::BTreeSet::new();
        if n == 0 || n > 128 || self.decay.len() != n || self.decay.iter().any(|x| !x.is_finite()) {
            return Err(invalid("ODE shape/decay"));
        }
        for e in &self.equations {
            e.validate(n)?;
            if e.id.is_empty()
                || e.id.len() > 256
                || !ids.insert(&e.id)
                || !e.intercept.is_finite()
                || e.weights.len() != n
                || e.weights.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("ODE mechanism"));
            }
        }
        Ok(())
    }

    /// Evaluate the declared continuous-time drift.
    pub fn drift(&self, state: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        self.validate()?;
        if state.len() != self.equations.len() || state.iter().any(|v| !v.is_finite()) {
            return Err(invalid("ODE state"));
        }
        self.equations
            .iter()
            .enumerate()
            .map(|(i, e)| e.structural_value(state).map(|v| v - self.decay[i] * state[i]))
            .collect()
    }

    /// Integrate an initial-value system, landing exactly on observation/action
    /// boundaries. Replacing a mechanism clamps the state and its derivative.
    pub fn trajectory(
        &self,
        initial: &[f64],
        times: &[f64],
        schedule: &[TimedClamp],
        policy: OdePolicy,
        ctx: &ExecutionContext,
    ) -> Result<OdeTrajectory, EquilibriumError> {
        let plan = crate::compile::compile_ode(self)?;
        if initial.len() != plan.blocks[0].variables().len() {
            return Err(invalid("ODE initial-state dimension"));
        }
        integrate(initial, times, schedule, policy, ctx, |state| self.drift(state))
    }
}

fn validate_policy(policy: OdePolicy) -> Result<(), EquilibriumError> {
    if !policy.absolute_tolerance.is_finite()
        || policy.absolute_tolerance <= 0.0
        || !policy.relative_tolerance.is_finite()
        || policy.relative_tolerance <= 0.0
        || !policy.initial_step.is_finite()
        || policy.initial_step <= 0.0
        || policy.max_steps == 0
        || policy.max_steps > 10_000_000
    {
        return Err(invalid("ODE numerical policy"));
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "adaptive stepping, error control and exact action boundaries form one integration procedure"
)]
fn integrate<F>(
    initial: &[f64],
    times: &[f64],
    schedule: &[TimedClamp],
    policy: OdePolicy,
    ctx: &ExecutionContext,
    drift: F,
) -> Result<OdeTrajectory, EquilibriumError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, EquilibriumError>,
{
    validate_policy(policy)?;
    let n = initial.len();
    if n == 0
        || n > 1024
        || initial.iter().any(|v| !v.is_finite())
        || times.is_empty()
        || times.len() > 100_000
        || times.iter().any(|v| !v.is_finite())
        || times.windows(2).any(|p| p[1] <= p[0])
        || schedule.len() > 10_000
    {
        return Err(invalid("ODE initial/time dimensions"));
    }
    for (i, event) in schedule.iter().enumerate() {
        if !event.time.is_finite()
            || event.time < times[0]
            || event.time > times[times.len() - 1]
            || (i > 0 && schedule[i - 1].time > event.time)
        {
            return Err(invalid("action time/order"));
        }
        let mut seen = std::collections::BTreeSet::new();
        if event
            .clamps
            .iter()
            .any(|c| c.variable >= n || !c.value.is_finite() || !seen.insert(c.variable))
        {
            return Err(invalid("scheduled clamp"));
        }
    }
    let bytes = n
        .checked_mul(times.len() + 32)
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| invalid("ODE retention overflow"))?;
    let _lease = ctx
        .acquire_admission(1, u64::try_from(bytes).map_err(|_| invalid("ODE bytes"))?)
        .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    let mut state = initial.to_vec();
    let mut clamped = vec![None; n];
    let mut t = times[0];
    let mut h = policy.initial_step;
    let mut event_index = 0;
    let mut accepted = 0;
    let mut rejected = 0;
    let mut states = Vec::with_capacity(times.len());
    for &target in times {
        loop {
            while event_index < schedule.len() && schedule[event_index].time <= t {
                for c in &schedule[event_index].clamps {
                    clamped[c.variable] = Some(c.value);
                    state[c.variable] = c.value;
                }
                event_index += 1;
            }
            if t >= target {
                break;
            }
            check_cancel(ctx)?;
            if accepted + rejected >= policy.max_steps {
                return Err(EquilibriumError::Resource(
                    "ODE step budget exhausted (possible stiffness)".into(),
                ));
            }
            let boundary = if event_index < schedule.len() {
                target.min(schedule[event_index].time)
            } else {
                target
            };
            h = h.min(boundary - t);
            if t + h <= t {
                return Err(EquilibriumError::Nonconvergence { residual: h });
            }
            let eval = |values: &[f64]| -> Result<Vec<f64>, EquilibriumError> {
                let mut values = values.to_vec();
                for (i, c) in clamped.iter().enumerate() {
                    if let Some(v) = c {
                        values[i] = *v;
                    }
                }
                let mut d = drift(&values)?;
                if d.len() != n || d.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("ODE nonfinite derivative"));
                }
                for (i, c) in clamped.iter().enumerate() {
                    if c.is_some() {
                        d[i] = 0.0;
                    }
                }
                Ok(d)
            };
            let (next, error) = dormand_prince(&state, h, &eval)?;
            let ratio = state
                .iter()
                .zip(&next)
                .zip(error)
                .map(|((&a, &b), e)| {
                    e.abs()
                        / (policy.absolute_tolerance
                            + policy.relative_tolerance * a.abs().max(b.abs()))
                })
                .fold(0.0, f64::max);
            if !ratio.is_finite() || next.iter().any(|v| !v.is_finite()) {
                return Err(invalid("ODE nonfinite step"));
            }
            if ratio <= 1.0 {
                state = next;
                // Exact boundary assignment avoids accumulating action-time drift.
                t = if h >= boundary - t { boundary } else { t + h };
                accepted += 1;
            } else {
                rejected += 1;
            }
            h *= if ratio == 0.0 { 5.0 } else { (0.9 * ratio.powf(-0.2)).clamp(0.1, 5.0) };
        }
        states.push(state.clone());
    }
    Ok(OdeTrajectory {
        times: times.to_vec(),
        states,
        accepted_steps: accepted,
        rejected_steps: rejected,
    })
}

fn dormand_prince<F>(y: &[f64], h: f64, eval: &F) -> Result<(Vec<f64>, Vec<f64>), EquilibriumError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, EquilibriumError>,
{
    let a: [&[f64]; 6] = [
        &[1.0 / 5.0],
        &[3.0 / 40.0, 9.0 / 40.0],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
        &[9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
        &[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
    ];
    let b = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0];
    let low = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339_200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ];
    let mut k = vec![eval(y)?];
    for coefficients in a {
        let next: Vec<_> = (0..y.len())
            .map(|i| {
                y[i] + h * coefficients.iter().enumerate().map(|(j, c)| c * k[j][i]).sum::<f64>()
            })
            .collect();
        k.push(eval(&next)?);
    }
    let next = (0..y.len())
        .map(|i| y[i] + h * b.iter().enumerate().map(|(j, c)| c * k[j][i]).sum::<f64>())
        .collect();
    let error = (0..y.len())
        .map(|i| {
            h * b.iter().zip(low).enumerate().map(|(j, (c, d))| (c - d) * k[j][i]).sum::<f64>()
        })
        .collect();
    Ok((next, error))
}

/// Linear SDE `dX = M(X - center)dt + L dW`, with supplied `diffusion = LL^T`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrnsteinUhlenbeck {
    /// Row-major continuous-time drift matrix, not static SCM coefficients.
    pub drift: Vec<f64>,
    /// Equilibrium center.
    pub center: Vec<f64>,
    /// Row-major positive-definite diffusion covariance.
    pub diffusion: Vec<f64>,
}

/// Gaussian moment trajectory for the declared initial law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuMoments {
    /// Observation times.
    pub times: Vec<f64>,
    /// Mean vector per time.
    pub means: Vec<Vec<f64>>,
    /// Row-major covariance per time.
    pub covariances: Vec<Vec<f64>>,
}

impl OrnsteinUhlenbeck {
    pub(crate) fn validate(&self) -> Result<usize, EquilibriumError> {
        let n = self.center.len();
        if n == 0
            || n > 16
            || self.drift.len() != n * n
            || self.diffusion.len() != n * n
            || self.drift.iter().chain(&self.center).any(|v| !v.is_finite())
        {
            return Err(invalid("OU dimensions"));
        }
        cholesky(&self.diffusion, n)?;
        Ok(n)
    }

    fn lyapunov(&self, q: &[f64], n: usize) -> Result<Vec<f64>, EquilibriumError> {
        let k = n * n;
        let mut operator = vec![0.0; k * k];
        for i in 0..n {
            for j in 0..n {
                for a in 0..n {
                    operator[(i * n + j) * k + a * n + j] += self.drift[i * n + a];
                    operator[(i * n + j) * k + i * n + a] += self.drift[j * n + a];
                }
            }
        }
        let mut rhs: Vec<_> = q.iter().map(|x| -x).collect();
        if !gauss_jordan_solve(&mut operator, &mut rhs, k, 1) {
            return Err(EquilibriumError::Singular);
        }
        for i in 0..n {
            for j in 0..i {
                let mean = 0.5 * (rhs[i * n + j] + rhs[j * n + i]);
                rhs[i * n + j] = mean;
                rhs[j * n + i] = mean;
            }
        }
        let residual = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| {
                q[i * n + j]
                    + (0..n)
                        .map(|a| {
                            self.drift[i * n + a] * rhs[a * n + j]
                                + rhs[i * n + a] * self.drift[j * n + a]
                        })
                        .sum::<f64>()
            })
            .map(f64::abs)
            .fold(0.0, f64::max);
        if !residual.is_finite()
            || residual > 1e-8 * (1.0 + q.iter().map(|v| v.abs()).fold(0.0, f64::max))
        {
            return Err(EquilibriumError::Nonconvergence { residual });
        }
        Ok(rhs)
    }

    /// Stationary covariance, after a positive-definite Lyapunov stability witness.
    pub fn stationary_covariance(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, EquilibriumError> {
        let plan = crate::compile::compile_ou(self)?;
        let n = plan.blocks[0].variables().len();
        let bytes = 32 * n * n * n * n + 128 * n * n;
        let _lease = ctx
            .acquire_admission(1, bytes as u64)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        check_cancel(ctx)?;
        let mut identity = vec![0.0; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let witness = self.lyapunov(&identity, n)?;
        cholesky(&witness, n)
            .map_err(|_| invalid("OU drift has no positive-definite stability witness"))?;
        let covariance = self.lyapunov(&self.diffusion, n)?;
        cholesky(&covariance, n)?;
        Ok(covariance)
    }

    /// Solve M X + X M^T = -Q for a finite symmetric direction, including
    /// indefinite likelihood/sensitivity adjoints. This does not certify that
    /// M is stable; stationary-law callers separately establish that condition.
    pub fn lyapunov_direction(
        &self,
        direction: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, EquilibriumError> {
        let n = self.validate()?;
        if direction.len() != n * n
            || direction.iter().any(|v| !v.is_finite())
            || (0..n).any(|i| {
                (0..n).any(|j| {
                    (direction[i * n + j] - direction[j * n + i]).abs()
                        > 1e-12 * (1.0 + direction[i * n + j].abs().max(direction[j * n + i].abs()))
                })
            })
        {
            return Err(invalid("Lyapunov direction shape/symmetry"));
        }
        let _lease = ctx
            .acquire_admission(1, (32 * n * n * n * n + 128 * n * n) as u64)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        check_cancel(ctx)?;
        self.lyapunov(direction, n)
    }

    /// Finite-horizon mean/covariance, without assuming a stationary initial law.
    pub fn moments(
        &self,
        initial_mean: &[f64],
        initial_covariance: &[f64],
        times: &[f64],
        policy: OdePolicy,
        ctx: &ExecutionContext,
    ) -> Result<OuMoments, EquilibriumError> {
        let n = self.validate()?;
        if initial_mean.len() != n || initial_covariance.len() != n * n {
            return Err(invalid("OU initial law"));
        }
        // Positive semidefinite deterministic zero covariance is a valid initial law.
        if initial_covariance.iter().any(|v| *v != 0.0) {
            cholesky(initial_covariance, n)?;
        }
        let mut initial = initial_mean.to_vec();
        initial.extend_from_slice(initial_covariance);
        let trajectory = integrate(&initial, times, &[], policy, ctx, |state| {
            let mean = &state[..n];
            let cov = &state[n..];
            let mut derivative: Vec<_> = (0..n)
                .map(|i| (0..n).map(|j| self.drift[i * n + j] * (mean[j] - self.center[j])).sum())
                .collect();
            for i in 0..n {
                for j in 0..n {
                    derivative.push(
                        self.diffusion[i * n + j]
                            + (0..n)
                                .map(|a| {
                                    self.drift[i * n + a] * cov[a * n + j]
                                        + cov[i * n + a] * self.drift[j * n + a]
                                })
                                .sum::<f64>(),
                    );
                }
            }
            Ok(derivative)
        })?;
        Ok(OuMoments {
            times: trajectory.times,
            means: trajectory.states.iter().map(|x| x[..n].to_vec()).collect(),
            covariances: trajectory.states.iter().map(|x| x[n..].to_vec()).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_equilibrium::Activation;
    #[test]
    fn ode_feedback_matches_analytic_rotation_and_exact_action_boundaries() {
        let ode = OdeSystem {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, -1.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![1.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            decay: vec![0.0; 2],
        };
        let ctx = ExecutionContext::for_tests(2);
        let path =
            ode.trajectory(&[1.0, 0.0], &[0.0, 1.0], &[], OdePolicy::default(), &ctx).unwrap();
        assert!((path.states[1][0] - 1.0_f64.cos()).abs() < 1e-7);
        assert!((path.states[1][1] - 1.0_f64.sin()).abs() < 1e-7);
        let event = TimedClamp { time: 0.5, clamps: vec![Clamp { variable: 0, value: 2.0 }] };
        let path = ode
            .trajectory(&[1.0, 0.0], &[0.0, 0.5, 1.0], &[event], OdePolicy::default(), &ctx)
            .unwrap();
        assert_eq!(path.states[1][0], 2.0);
        assert_eq!(path.states[2][0], 2.0);
        assert!((path.states[2][1] - (0.5_f64.sin() + 1.0)).abs() < 1e-7);
        let p = OdePolicy { max_steps: 1, ..OdePolicy::default() };
        assert!(ode.trajectory(&[1.0, 0.0], &[0.0, 10.0], &[], p, &ctx).is_err());
    }
    #[test]
    fn ou_stationary_is_not_static_scm_covariance_and_transients_keep_initial_law() {
        let ou = OrnsteinUhlenbeck { drift: vec![-2.0], center: vec![3.0], diffusion: vec![4.0] };
        let ctx = ExecutionContext::for_tests(4);
        assert!((ou.stationary_covariance(&ctx).unwrap()[0] - 1.0).abs() < 1e-12);
        let path =
            ou.moments(&[0.0], &[0.0], &[0.0, 0.5, 1.0], OdePolicy::default(), &ctx).unwrap();
        assert!((path.means[2][0] - 3.0 * (1.0 - (-2.0_f64).exp())).abs() < 1e-6);
        assert!((path.covariances[2][0] - (1.0 - (-4.0_f64).exp())).abs() < 1e-6);
        let unstable = OrnsteinUhlenbeck { drift: vec![2.0], ..ou };
        assert!(unstable.stationary_covariance(&ctx).is_err());
    }
}

/// A drift parameter with an analytic forward-sensitivity equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OdeParameter {
    /// Additive drift intercept.
    Intercept {
        /// Equation index.
        equation: usize,
    },
    /// Coefficient inside the equation's scalar activation.
    Weight {
        /// Equation index.
        equation: usize,
        /// Parent variable index (self-dependence is allowed for ODEs).
        parent: usize,
    },
    /// Local parameter of a compositional drift mechanism.
    ExpressionParameter {
        /// Equation index.
        equation: usize,
        /// Local parameter index.
        parameter: usize,
    },
    /// Linear decay coefficient.
    Decay {
        /// Equation/variable index.
        equation: usize,
    },
}

/// State path and analytic parameter sensitivities, jointly integrated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeSensitivityTrajectory {
    /// State trajectory and actual work counts.
    pub trajectory: OdeTrajectory,
    /// At each observation time, row-major `(state,parameter)` derivatives.
    pub sensitivities: Vec<Vec<f64>>,
}

impl OdeSystem {
    /// Integrate forward sensitivities for fixed initial conditions and
    /// parameter-independent scheduled clamp values. Clamping also sets that
    /// state's parameter sensitivities to zero from the action time onward.
    pub fn sensitivities(
        &self,
        initial: &[f64],
        times: &[f64],
        schedule: &[TimedClamp],
        parameters: &[OdeParameter],
        policy: OdePolicy,
        ctx: &ExecutionContext,
    ) -> Result<OdeSensitivityTrajectory, EquilibriumError> {
        self.validate()?;
        let n = self.equations.len();
        let p = parameters.len();
        if initial.len() != n || p == 0 || p > 64 || n * (p + 1) > 1024 {
            return Err(invalid("ODE sensitivity dimensions"));
        }
        for (j, &parameter) in parameters.iter().enumerate() {
            if parameters[..j].contains(&parameter) {
                return Err(invalid("duplicate ODE sensitivity parameter"));
            }
            match parameter {
                OdeParameter::ExpressionParameter { equation, parameter }
                    if self
                        .equations
                        .get(equation)
                        .and_then(|e| e.parameters.get(parameter))
                        .is_none() =>
                {
                    return Err(invalid("ODE expression parameter"));
                }
                OdeParameter::Weight { equation, .. }
                    if self.equations.get(equation).is_some_and(|e| e.expression.is_some()) =>
                {
                    return Err(invalid("compositional mechanisms use expression parameters"));
                }
                OdeParameter::Intercept { equation } | OdeParameter::Decay { equation }
                    if equation >= n =>
                {
                    return Err(invalid("ODE sensitivity equation"));
                }
                OdeParameter::Weight { equation, parent } if equation >= n || parent >= n => {
                    return Err(invalid("ODE sensitivity coefficient"));
                }
                _ => {}
            }
        }
        let mut augmented = initial.to_vec();
        augmented.resize(n * (p + 1), 0.0);
        let mut actions = Vec::with_capacity(schedule.len());
        for action in schedule {
            let mut clamps = action.clamps.clone();
            for clamp in &action.clamps {
                if clamp.variable >= n {
                    return Err(invalid("ODE sensitivity clamp"));
                }
                for j in 0..p {
                    clamps.push(Clamp { variable: n + clamp.variable * p + j, value: 0.0 });
                }
            }
            actions.push(TimedClamp { time: action.time, clamps });
        }
        let path = integrate(&augmented, times, &actions, policy, ctx, |state| {
            let x = &state[..n];
            let s = &state[n..];
            let mut derivative = self.drift(x)?;
            for i in 0..n {
                let equation = &self.equations[i];
                let activation_derivative =
                    equation.activation.derivative(dot(&equation.weights, x));
                let input_derivatives = equation.input_derivatives(x)?;
                for (j, parameter) in parameters.iter().enumerate() {
                    let mut ds = (0..n)
                        .map(|k| {
                            (input_derivatives[k] - if i == k { self.decay[i] } else { 0.0 })
                                * s[k * p + j]
                        })
                        .sum::<f64>();
                    ds += match *parameter {
                        OdeParameter::ExpressionParameter { equation: eq, parameter }
                            if eq == i =>
                        {
                            let tape = equation
                                .expression
                                .as_ref()
                                .ok_or_else(|| invalid("missing expression"))?
                                .compile(n, equation.parameters.len())?;
                            let evaluation = tape.evaluate(x, &equation.parameters)?;
                            tape.parameter_score(&evaluation, 1.0, &vec![0.0; n])?[parameter]
                        }
                        OdeParameter::Intercept { equation } if equation == i => 1.0,
                        OdeParameter::Decay { equation } if equation == i => -x[i],
                        OdeParameter::Weight { equation, parent } if equation == i => {
                            activation_derivative * x[parent]
                        }
                        _ => 0.0,
                    };
                    derivative.push(ds);
                }
            }
            Ok(derivative)
        })?;
        let sensitivities = path.states.iter().map(|state| state[n..].to_vec()).collect();
        let trajectory = OdeTrajectory {
            times: path.times,
            states: path.states.iter().map(|state| state[..n].to_vec()).collect(),
            accepted_steps: path.accepted_steps,
            rejected_steps: path.rejected_steps,
        };
        Ok(OdeSensitivityTrajectory { trajectory, sensitivities })
    }
}

#[cfg(test)]
mod sensitivity_tests {
    use super::*;
    use crate::nonlinear_equilibrium::Activation;
    #[test]
    fn analytic_forward_sensitivity_matches_closed_form_and_clamps_reset_it() {
        let model = OdeSystem {
            equations: vec![Equation {
                id: "decay".into(),
                intercept: 0.0,
                weights: vec![0.0],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            }],
            decay: vec![2.0],
        };
        let ctx = ExecutionContext::for_tests(1);
        let parameters = [OdeParameter::Decay { equation: 0 }];
        let path = model
            .sensitivities(&[3.0], &[0.0, 1.0], &[], &parameters, Default::default(), &ctx)
            .unwrap();
        assert!((path.sensitivities[1][0] + 3.0 * (-2.0_f64).exp()).abs() < 1e-7);
        let actions = [TimedClamp { time: 0.5, clamps: vec![Clamp { variable: 0, value: 1.0 }] }];
        let path = model
            .sensitivities(
                &[3.0],
                &[0.0, 0.5, 1.0],
                &actions,
                &parameters,
                Default::default(),
                &ctx,
            )
            .unwrap();
        assert_eq!(path.sensitivities[1][0], 0.0);
        assert_eq!(path.sensitivities[2][0], 0.0);
    }
}

/// Stationary law after a sustained hard intervention on an OU system.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuInterventionalLaw {
    /// Full mean, including fixed clamp coordinates.
    pub mean: Vec<f64>,
    /// Full row-major covariance, zero on clamped rows and columns.
    pub covariance: Vec<f64>,
    /// Coordinates retaining stochastic dynamics.
    pub free: Vec<usize>,
}

impl OrnsteinUhlenbeck {
    /// Replace clamped drift/noise equations and compute the surviving stationary
    /// Gaussian law. Rechecks stability of the free drift; a stable original
    /// system does not imply stability after arbitrary interventions.
    pub fn stationary_intervention(
        &self,
        clamps: &[Clamp],
        ctx: &ExecutionContext,
    ) -> Result<OuInterventionalLaw, EquilibriumError> {
        let n = self.validate()?;
        let mut fixed = vec![None; n];
        for c in clamps {
            if c.variable >= n || !c.value.is_finite() || fixed[c.variable].is_some() {
                return Err(invalid("OU clamp target/value/duplicate"));
            }
            fixed[c.variable] = Some(c.value);
        }
        let free: Vec<_> = (0..n).filter(|&i| fixed[i].is_none()).collect();
        let k = free.len();
        let mut mean = self.center.clone();
        for c in clamps {
            mean[c.variable] = c.value;
        }
        let mut covariance = vec![0.0; n * n];
        if k == 0 {
            return Ok(OuInterventionalLaw { mean, covariance, free });
        }
        let drift: Vec<_> =
            free.iter().flat_map(|&i| free.iter().map(move |&j| self.drift[i * n + j])).collect();
        let diffusion: Vec<_> = free
            .iter()
            .flat_map(|&i| free.iter().map(move |&j| self.diffusion[i * n + j]))
            .collect();
        let mut matrix = drift.clone();
        let mut delta: Vec<_> = free
            .iter()
            .map(|&i| {
                -clamps
                    .iter()
                    .map(|c| self.drift[i * n + c.variable] * (c.value - self.center[c.variable]))
                    .sum::<f64>()
            })
            .collect();
        if !gauss_jordan_solve(&mut matrix, &mut delta, k, 1) {
            return Err(EquilibriumError::Singular);
        }
        for (j, &i) in free.iter().enumerate() {
            mean[i] += delta[j];
        }
        let reduced = Self { drift, center: free.iter().map(|&i| mean[i]).collect(), diffusion };
        let marginal = reduced.stationary_covariance(ctx)?;
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                covariance[i * n + j] = marginal[a * k + b];
            }
        }
        Ok(OuInterventionalLaw { mean, covariance, free })
    }

    /// Joint stationary Gaussian likelihood of independent equilibrium replicates
    /// under known sustained intervention regimes. This is not a transition
    /// likelihood for dependent time-series rows.
    #[allow(
        clippy::float_cmp,
        reason = "observed hard clamps must equal the declared structural value exactly"
    )]
    pub fn stationary_log_likelihood(
        &self,
        rows: &[crate::nonlinear_equilibrium::RegimeRow],
        ctx: &ExecutionContext,
    ) -> Result<f64, EquilibriumError> {
        let n = self.validate()?;
        let mut total = 0.0;
        for row in rows {
            check_cancel(ctx)?;
            if row.values.len() != n || row.values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("OU observed state"));
            }
            for clamp in &row.clamps {
                if clamp.variable >= n || row.values[clamp.variable] != clamp.value {
                    return Err(invalid("OU observed clamp disagreement"));
                }
            }
            let law = self.stationary_intervention(&row.clamps, ctx)?;
            let free = &law.free;
            let k = free.len();
            if k == 0 {
                continue;
            }
            let full_covariance = &law.covariance;
            let covariance: Vec<_> = free
                .iter()
                .flat_map(|&i| free.iter().map(move |&j| full_covariance[i * n + j]))
                .collect();
            let l = cholesky(&covariance, k)?;
            let mut residual: Vec<_> = free.iter().map(|&i| row.values[i] - law.mean[i]).collect();
            for i in 0..k {
                for j in 0..i {
                    residual[i] -= l[i * k + j] * residual[j];
                }
                residual[i] /= l[i * k + i];
            }
            total -= 0.5
                * (dot(&residual, &residual) + k as f64 * (2.0 * std::f64::consts::PI).ln())
                + (0..k).map(|i| l[i * k + i].ln()).sum::<f64>();
        }
        if !total.is_finite() {
            return Err(invalid("nonfinite OU likelihood"));
        }
        Ok(total)
    }
}

#[cfg(test)]
mod ou_intervention_tests {
    use super::*;
    #[test]
    fn ou_clamp_keeps_marginal_diffusion_and_rechecks_free_stability() {
        let model = OrnsteinUhlenbeck {
            drift: vec![-1.0, 0.5, 0.2, -2.0],
            center: vec![0.0, 0.0],
            diffusion: vec![2.0, 0.7, 0.7, 4.0],
        };
        let ctx = ExecutionContext::for_tests(0);
        let law =
            model.stationary_intervention(&[Clamp { variable: 0, value: 3.0 }], &ctx).unwrap();
        assert!((law.mean[1] - 0.3).abs() < 1e-12);
        assert!((law.covariance[3] - 1.0).abs() < 1e-12);
        let ll = model
            .stationary_log_likelihood(
                &[crate::nonlinear_equilibrium::RegimeRow {
                    values: vec![3.0, 0.3],
                    clamps: vec![Clamp { variable: 0, value: 3.0 }],
                }],
                &ctx,
            )
            .unwrap();
        assert!((ll + 0.5 * (2.0 * std::f64::consts::PI).ln()).abs() < 1e-12);
        // Full drift trace=-1, determinant=1 is stable; clamping node 0
        // leaves positive drift +1 in node 1 and must withhold a stationary law.
        let unstable_after_clamp = OrnsteinUhlenbeck { drift: vec![-2.0, 3.0, -1.0, 1.0], ..model };
        assert!(unstable_after_clamp.stationary_covariance(&ctx).is_ok());
        assert!(
            unstable_after_clamp
                .stationary_intervention(&[Clamp { variable: 0, value: 0.0 }], &ctx)
                .is_err()
        );
    }
}

#[cfg(test)]
mod compositional_drift_tests {
    use super::*;
    use crate::{mechanism_expr::MechanismExpression as E, nonlinear_equilibrium::Activation};
    #[test]
    fn compositional_drift_parameter_sensitivity_matches_exponential_truth() {
        let model = OdeSystem {
            equations: vec![Equation {
                id: "growth".into(),
                intercept: 0.0,
                weights: vec![0.0],
                activation: Activation::Identity,
                parameters: vec![0.3],
                expression: Some(E::Multiply {
                    factors: vec![E::Parameter { parameter: 0 }, E::Variable { variable: 0 }],
                }),
            }],
            decay: vec![0.0],
        };
        let result = model
            .sensitivities(
                &[1.0],
                &[0.0, 0.5, 1.0],
                &[],
                &[OdeParameter::ExpressionParameter { equation: 0, parameter: 0 }],
                OdePolicy::default(),
                &ExecutionContext::for_tests(5),
            )
            .unwrap();
        for (i, &time) in result.trajectory.times.iter().enumerate() {
            let expected = (0.3 * time).exp();
            assert!((result.trajectory.states[i][0] - expected).abs() < 1e-7);
            assert!((result.sensitivities[i][0] - time * expected).abs() < 1e-7);
        }
    }
}
