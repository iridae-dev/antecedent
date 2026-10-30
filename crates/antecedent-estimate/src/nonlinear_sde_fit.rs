// SPDX-License-Identifier: MIT OR Apache-2.0
//! Quasi-likelihood fitting of nonlinear diagonal-noise SDEs from dense trajectories.
//!
//! For a fully observed trajectory the Euler transition density
//! `x' | x ~ N(x + f(x,u,theta) dt, diag(g(x,u,theta)^2 dt))` is a pseudo-likelihood whose
//! maximizer is consistent as the observation interval shrinks (the classical Euler
//! quasi-maximum-likelihood estimator of a discretely observed diffusion). It is not the
//! exact transition law, so the fit carries a separate *discretization* record: the
//! interval-times-rate product `dt * ||J_f||` of every transition measures how far each
//! interval is from the regime in which the Euler density is accurate. Intervals above a
//! declared flag threshold make the fit a model-conditional approximation; intervals
//! above the refusal threshold are refused, because no justified procedure exists there
//! without latent-path integration.
//!
//! With a declared linear Gaussian observation model (noise, partial observation) the
//! likelihood is the extended-Kalman-filter innovation likelihood of the Euler-discretized
//! model. That filter linearizes the nonlinear dynamics, so such a fit is always a
//! model-conditional approximation and is labelled as one.
//!
//! Fitting (optimizer termination), identification (observed information) and
//! discretization (interval-rate evidence) are separate records; none implies another.
use crate::cyclic_gradient::CyclicGradientPolicy;
use crate::cyclic_optim::InverseCurvature;
use antecedent_core::ExecutionContext;
use antecedent_model::nonlinear_sde::{
    CompiledSde, SdeContrast, SdeEnsembleReport, SdeError, SdeSimulationPolicy, SdeWorld,
    WorldPlan, counter_normal, simulate_sde,
};
use serde::{Deserialize, Serialize};

const MAX_FIT_PARAMETERS: usize = 16;
const MAX_TRAJECTORIES: usize = 4096;
const MAX_OBSERVATIONS: usize = 1 << 20;
const MAX_OBSERVED: usize = 16;
const MAX_PARAMETER_DRAWS: usize = 512;
const GRADIENT_STEP: f64 = 1e-5;
const HESSIAN_STEP: f64 = 1e-3;
const INFORMATION_FLOOR: f64 = 1e-6;
const FLAT_DIAGONAL: f64 = 1e-9;
const BOUNDARY_FRACTION_FLAG: f64 = 0.05;

/// Why a trajectory design cannot identify the requested parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdeIdentificationFailure {
    /// No trajectory has two observation times, so no transition is observed.
    NoTransitionInformation,
    /// The observed information of the requested coordinates is singular.
    SingularInformation,
}

impl SdeIdentificationFailure {
    /// Stable machine-readable detail code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::NoTransitionInformation => "sde_fit.no_transition_information",
            Self::SingularInformation => "sde_fit.singular_information",
        }
    }
}

/// Typed failure of fitting or reporting.
#[derive(Debug, thiserror::Error)]
pub enum SdeFitError {
    /// Malformed shape, non-finite number, unordered times or invalid policy.
    #[error("invalid SDE fit input: {0}")]
    Invalid(String),
    /// A supported scientific refusal with a stable detail code.
    #[error("unsupported SDE fit ({code}): {detail}")]
    Unsupported {
        /// Stable detail code.
        code: &'static str,
        /// Evidence behind the refusal.
        detail: String,
    },
    /// The design does not identify the requested coordinates.
    #[error("SDE parameters are not identified ({failure:?}): {detail}")]
    Unidentified {
        /// Stable classification.
        failure: SdeIdentificationFailure,
        /// The evidence that led to the refusal.
        detail: String,
    },
    /// A numerical condition failed.
    #[error("SDE fit numerical failure: {0}")]
    Numerical(String),
    /// Resource admission or cancellation failed.
    #[error("SDE fit resource refusal: {0}")]
    Resource(String),
}

impl From<SdeError> for SdeFitError {
    fn from(error: SdeError) -> Self {
        match error {
            SdeError::Invalid(r) => Self::Invalid(r),
            SdeError::Unsupported { code, detail } => Self::Unsupported { code, detail },
            SdeError::Numerical(r) => Self::Numerical(r),
            SdeError::Resource(r) => Self::Resource(r),
        }
    }
}

fn bad(reason: &str) -> SdeFitError {
    SdeFitError::Invalid(reason.into())
}
fn numerical(reason: &str) -> SdeFitError {
    SdeFitError::Numerical(reason.into())
}

/// Linear Gaussian observation model `y = H x + offset + noise`, noise independent per row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeObservationModel {
    /// Row-major `m x d` observation matrix.
    pub matrix: Vec<f64>,
    /// Observation offset; empty denotes zero.
    #[serde(default)]
    pub offset: Vec<f64>,
    /// Known observation-noise standard deviation per observed row.
    pub noise_standard_deviation: Vec<f64>,
}

impl SdeObservationModel {
    fn validate(&self, d: usize) -> Result<usize, SdeFitError> {
        let m = self.noise_standard_deviation.len();
        if m == 0
            || m > MAX_OBSERVED
            || self.matrix.len() != m * d
            || !(self.offset.is_empty() || self.offset.len() == m)
            || self.matrix.iter().chain(&self.offset).any(|v| !v.is_finite())
            || self.noise_standard_deviation.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.matrix.iter().all(|v| *v == 0.0)
        {
            return Err(bad("SDE observation model shape/finiteness"));
        }
        Ok(m)
    }
}

/// Gaussian prior of the hidden state at a trajectory's first observation time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeFilterPrior {
    /// Prior mean.
    pub mean: Vec<f64>,
    /// Row-major prior covariance.
    pub covariance: Vec<f64>,
}

/// One observation of a trajectory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeObservation {
    /// Observation time.
    pub time: f64,
    /// Observed vector: the states themselves without an observation model, otherwise one
    /// entry per observation-matrix row.
    pub values: Vec<f64>,
}

/// One unit observed at strictly increasing times under a known intervention regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeTrajectory {
    /// Stable trajectory identity.
    pub id: String,
    /// Known holds and replaced inputs in force; the default is the baseline.
    #[serde(default)]
    pub intervention: SdeWorld,
    /// Observations at strictly increasing times.
    pub observations: Vec<SdeObservation>,
}

fn default_flag() -> f64 {
    0.1
}
fn default_refusal() -> f64 {
    0.5
}
fn default_floor() -> f64 {
    1e-12
}

/// Optimizer and discretization-admission policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeFitPolicy {
    /// Optimizer settings shared with the other trajectory fits (no L1 penalty).
    pub optimizer: CyclicGradientPolicy,
    /// Intervals with `dt * rate` above this make the fit a model-conditional approximation.
    #[serde(default = "default_flag")]
    pub coarse_flag: f64,
    /// Intervals with `dt * rate` above this are refused.
    #[serde(default = "default_refusal")]
    pub coarse_refusal: f64,
    /// Smallest admissible Euler transition variance.
    #[serde(default = "default_floor")]
    pub variance_floor: f64,
    /// Largest filter substep for the extended Kalman likelihood; zero uses one step per interval.
    #[serde(default)]
    pub filter_step: f64,
}

impl Default for SdeFitPolicy {
    fn default() -> Self {
        Self {
            optimizer: CyclicGradientPolicy::default(),
            coarse_flag: default_flag(),
            coarse_refusal: default_refusal(),
            variance_floor: default_floor(),
            filter_step: 0.0,
        }
    }
}

impl SdeFitPolicy {
    fn validate(&self) -> Result<(), SdeFitError> {
        let o = &self.optimizer;
        if !o.learning_rate.is_finite()
            || o.learning_rate <= 0.0
            || !o.score_tolerance.is_finite()
            || o.score_tolerance <= 0.0
            || o.l1_penalty != 0.0
            || o.max_iterations == 0
            || o.max_iterations > 100_000
            || !self.coarse_flag.is_finite()
            || !self.coarse_refusal.is_finite()
            || self.coarse_flag <= 0.0
            || self.coarse_flag > self.coarse_refusal
            || self.coarse_refusal > 2.0
            || !self.variance_floor.is_finite()
            || self.variance_floor <= 0.0
            || !self.filter_step.is_finite()
            || self.filter_step < 0.0
        {
            return Err(bad("SDE fit policy"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Small dense linear algebra (row-major, symmetric positive definite).
// ---------------------------------------------------------------------------

fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum > 0.0) || !sum.is_finite() {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

fn spd_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky(a, n)?;
    let mut inverse = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let column = cholesky_solve(&l, n, &e);
        for i in 0..n {
            inverse[i * n + j] = column[i];
        }
    }
    Some(inverse)
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations.
fn symmetric_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..i {
                off += m[i * n + j] * m[i * n + j];
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in 0..p {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| m[i * n + i]).collect()
}

// ---------------------------------------------------------------------------
// Likelihood.
// ---------------------------------------------------------------------------

struct Prepared<'a> {
    trajectory: &'a SdeTrajectory,
    plan: WorldPlan,
}

#[derive(Default)]
struct Probe {
    transitions: usize,
    minimum_interval: Option<f64>,
    maximum_interval: f64,
    maximum_rate_interval: f64,
    rate_interval_sum: f64,
    above_flag: usize,
    boundary_transitions: usize,
    observations: usize,
}

struct Problem<'a> {
    compiled: &'a CompiledSde,
    coordinates: &'a [usize],
    base: Vec<f64>,
    observation: Option<&'a SdeObservationModel>,
    prior: Option<&'a SdeFilterPrior>,
    prepared: Vec<Prepared<'a>>,
    policy: &'a SdeFitPolicy,
    d: usize,
    m: usize,
}

fn prepare<'a>(
    compiled: &CompiledSde,
    trajectories: &'a [SdeTrajectory],
) -> Result<Vec<Prepared<'a>>, SdeFitError> {
    let mut prepared = Vec::with_capacity(trajectories.len());
    for trajectory in trajectories {
        let first = trajectory.observations[0].time;
        let mut world = trajectory.intervention.clone();
        if world.id.trim().is_empty() {
            world.id = trajectory.id.clone();
        }
        let plan = compiled.system().plan_world_continuous(&world, first)?;
        prepared.push(Prepared { trajectory, plan });
    }
    Ok(prepared)
}

fn checked_filter_substeps(dt: f64, step: f64) -> Result<usize, SdeFitError> {
    if step == 0.0 {
        return Ok(1);
    }
    let count = (dt / step).ceil().max(1.0);
    if !count.is_finite() || count >= usize::MAX as f64 {
        return Err(bad("SDE filter substep count is not representable"));
    }
    Ok(count as usize)
}

const LN_2PI: f64 = 1.837_877_066_409_345_3;

impl Problem<'_> {
    fn theta(&self, values: &[f64]) -> Vec<f64> {
        let mut theta = self.base.clone();
        for (slot, value) in self.coordinates.iter().zip(values) {
            theta[*slot] = *value;
        }
        theta
    }

    fn log_likelihood(
        &self,
        values: &[f64],
        ctx: &ExecutionContext,
        mut probe: Option<&mut Probe>,
    ) -> Result<f64, SdeFitError> {
        let theta = self.theta(values);
        let mut total = 0.0;
        for item in &self.prepared {
            if ctx.cancellation.is_cancelled() {
                return Err(SdeFitError::Resource("cancelled".into()));
            }
            total += if let Some(observation) = self.observation {
                self.filter_trajectory(item, observation, &theta, ctx, probe.as_deref_mut())?
            } else {
                self.direct_trajectory(item, &theta, probe.as_deref_mut())?
            };
        }
        if total.is_finite() {
            Ok(total)
        } else {
            Err(numerical("non-finite quasi log likelihood"))
        }
    }

    /// `product` is the effective step times the local rate: the full interval for the
    /// direct Euler density, the filter substep for the extended Kalman likelihood.
    fn record(&self, probe: &mut Probe, dt: f64, product: f64) {
        probe.transitions += 1;
        probe.minimum_interval = Some(probe.minimum_interval.map_or(dt, |m| m.min(dt)));
        probe.maximum_interval = probe.maximum_interval.max(dt);
        probe.maximum_rate_interval = probe.maximum_rate_interval.max(product);
        probe.rate_interval_sum += product;
        if product > self.policy.coarse_flag {
            probe.above_flag += 1;
        }
    }

    /// Euler Gaussian transition log density summed over one fully observed trajectory.
    fn direct_trajectory(
        &self,
        item: &Prepared<'_>,
        theta: &[f64],
        mut probe: Option<&mut Probe>,
    ) -> Result<f64, SdeFitError> {
        let d = self.d;
        let mut total = 0.0;
        let observations = &item.trajectory.observations;
        if let Some(p) = probe.as_deref_mut() {
            p.observations += observations.len();
        }
        for pair in observations.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let dt = b.time - a.time;
            let local =
                self.compiled.local_coefficients(&a.values, &item.plan.inputs, a.time, theta)?;
            let mut near_boundary = false;
            for i in 0..d {
                if item.plan.held_value(i, a.time).is_some()
                    || item.plan.held_value(i, b.time).is_some()
                {
                    continue;
                }
                let variance = local.diffusion[i] * local.diffusion[i] * dt;
                if !(variance > self.policy.variance_floor) {
                    return Err(SdeFitError::Unsupported {
                        code: "sde_fit.degenerate_transition",
                        detail: format!(
                            "the Euler transition variance of '{}' is {variance:e} at an observed \
                             state (floor {:e}); the Gaussian transition density is degenerate \
                             there. Observe with noise and fit with the extended Kalman likelihood, \
                             or hold the coordinate with an intervention",
                            self.compiled.system().variables[i],
                            self.policy.variance_floor
                        ),
                    });
                }
                let residual = b.values[i] - a.values[i] - local.drift[i] * dt;
                total -= 0.5 * (LN_2PI + variance.ln() + residual * residual / variance);
                let boundary = self.compiled.system().boundary(i);
                let width = 3.0 * variance.sqrt();
                if boundary.lower.is_some_and(|l| a.values[i] - l < width)
                    || boundary.upper.is_some_and(|u| u - a.values[i] < width)
                {
                    near_boundary = true;
                }
            }
            if let Some(p) = probe.as_deref_mut() {
                self.record(p, dt, dt * local.rate);
                if near_boundary {
                    p.boundary_transitions += 1;
                }
            }
        }
        Ok(total)
    }

    /// Extended-Kalman innovation log likelihood of the Euler-discretized model.
    fn filter_trajectory(
        &self,
        item: &Prepared<'_>,
        observation: &SdeObservationModel,
        theta: &[f64],
        ctx: &ExecutionContext,
        mut probe: Option<&mut Probe>,
    ) -> Result<f64, SdeFitError> {
        let d = self.d;
        let prior = self.prior.ok_or_else(|| bad("SDE filter prior"))?;
        let mut mean = prior.mean.clone();
        let mut cov = prior.covariance.clone();
        let observations = &item.trajectory.observations;
        if let Some(p) = probe.as_deref_mut() {
            p.observations += observations.len();
        }
        let mut total = self.update(&mut mean, &mut cov, observation, &observations[0].values)?;
        for pair in observations.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let dt = b.time - a.time;
            let substeps = checked_filter_substeps(dt, self.policy.filter_step)?;
            let h = dt / substeps as f64;
            let mut worst_rate = 0.0f64;
            for s in 0..substeps {
                if ctx.cancellation.is_cancelled() {
                    return Err(SdeFitError::Resource("cancelled".into()));
                }
                let t = a.time + s as f64 * h;
                let local = self.compiled.local_coefficients(&mean, &item.plan.inputs, t, theta)?;
                worst_rate = worst_rate.max(local.rate);
                // F = I + J h; P <- F P F^T + diag(g^2) h; m <- m + f h.
                let mut f = vec![0.0; d * d];
                for i in 0..d {
                    for j in 0..d {
                        f[i * d + j] =
                            if i == j { 1.0 } else { 0.0 } + local.jacobian[i * d + j] * h;
                    }
                }
                let mut fp = vec![0.0; d * d];
                for i in 0..d {
                    for k in 0..d {
                        let x = f[i * d + k];
                        if x != 0.0 {
                            for j in 0..d {
                                fp[i * d + j] += x * cov[k * d + j];
                            }
                        }
                    }
                }
                let mut next = vec![0.0; d * d];
                for i in 0..d {
                    for j in 0..d {
                        let mut sum = 0.0;
                        for k in 0..d {
                            sum += fp[i * d + k] * f[j * d + k];
                        }
                        next[i * d + j] = sum;
                    }
                    next[i * d + i] += local.diffusion[i] * local.diffusion[i] * h;
                }
                for i in 0..d {
                    mean[i] += local.drift[i] * h;
                }
                cov = next;
                for i in 0..d {
                    if let Some(value) = item.plan.held_value(i, t + h) {
                        mean[i] = value;
                        for j in 0..d {
                            cov[i * d + j] = 0.0;
                            cov[j * d + i] = 0.0;
                        }
                    }
                }
            }
            if let Some(p) = probe.as_deref_mut() {
                self.record(p, dt, h * worst_rate);
            }
            total += self.update(&mut mean, &mut cov, observation, &b.values)?;
        }
        Ok(total)
    }

    fn update(
        &self,
        mean: &mut [f64],
        cov: &mut [f64],
        observation: &SdeObservationModel,
        y: &[f64],
    ) -> Result<f64, SdeFitError> {
        let (d, m) = (self.d, self.m);
        let mut z = vec![0.0; m];
        for r in 0..m {
            let predicted: f64 =
                (0..d).map(|i| observation.matrix[r * d + i] * mean[i]).sum::<f64>()
                    + observation.offset.get(r).copied().unwrap_or(0.0);
            z[r] = y[r] - predicted;
        }
        // hp = H P (m x d); s = hp H^T + R.
        let mut hp = vec![0.0; m * d];
        for r in 0..m {
            for k in 0..d {
                let x = observation.matrix[r * d + k];
                if x != 0.0 {
                    for j in 0..d {
                        hp[r * d + j] += x * cov[k * d + j];
                    }
                }
            }
        }
        let mut s = vec![0.0; m * m];
        for r in 0..m {
            for c in 0..m {
                let mut sum = 0.0;
                for k in 0..d {
                    sum += hp[r * d + k] * observation.matrix[c * d + k];
                }
                s[r * m + c] = sum;
            }
            s[r * m + r] += observation.noise_standard_deviation[r].powi(2);
        }
        let l = cholesky(&s, m)
            .ok_or_else(|| numerical("innovation covariance is not positive definite"))?;
        let solved = cholesky_solve(&l, m, &z);
        let log_det: f64 = 2.0 * (0..m).map(|r| l[r * m + r].ln()).sum::<f64>();
        let quadratic: f64 = z.iter().zip(&solved).map(|(a, b)| a * b).sum();
        let log_density = -0.5 * (m as f64 * LN_2PI + log_det + quadratic);
        // X = S^{-1} hp column by column; gain K = X^T (d x m).
        let mut x = vec![0.0; m * d];
        for j in 0..d {
            let column: Vec<f64> = (0..m).map(|r| hp[r * d + j]).collect();
            let sol = cholesky_solve(&l, m, &column);
            for r in 0..m {
                x[r * d + j] = sol[r];
            }
        }
        for i in 0..d {
            mean[i] += (0..m).map(|r| x[r * d + i] * z[r]).sum::<f64>();
        }
        let mut updated = cov.to_vec();
        for i in 0..d {
            for j in 0..d {
                let correction: f64 = (0..m).map(|r| x[r * d + i] * hp[r * d + j]).sum();
                updated[i * d + j] -= correction;
            }
        }
        for i in 0..d {
            for j in 0..i {
                let avg = 0.5 * (updated[i * d + j] + updated[j * d + i]);
                updated[i * d + j] = avg;
                updated[j * d + i] = avg;
            }
            if updated[i * d + i] < 0.0 {
                updated[i * d + i] = 0.0;
            }
        }
        cov.copy_from_slice(&updated);
        Ok(log_density)
    }

    fn gradient(
        &self,
        values: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, usize), SdeFitError> {
        let p = values.len();
        let mut score = vec![0.0; p];
        let mut evaluations = 0;
        for j in 0..p {
            let h = GRADIENT_STEP * (1.0 + values[j].abs());
            let mut high = values.to_vec();
            let mut low = values.to_vec();
            high[j] += h;
            low[j] -= h;
            let up = self.log_likelihood(&high, ctx, None)?;
            let down = self.log_likelihood(&low, ctx, None)?;
            evaluations += 2;
            score[j] = (up - down) / (2.0 * h);
        }
        Ok((score, evaluations))
    }

    /// Observed information: the negative central-difference Hessian of the log quasi likelihood.
    fn information(
        &self,
        values: &[f64],
        center: f64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, SdeFitError> {
        let p = values.len();
        let steps: Vec<f64> = values.iter().map(|v| HESSIAN_STEP * (1.0 + v.abs())).collect();
        let shifted = |a: usize, sa: f64, b: Option<(usize, f64)>| {
            let mut v = values.to_vec();
            v[a] += sa * steps[a];
            if let Some((c, sc)) = b {
                v[c] += sc * steps[c];
            }
            v
        };
        let mut info = vec![0.0; p * p];
        for a in 0..p {
            let up = self.log_likelihood(&shifted(a, 1.0, None), ctx, None)?;
            let down = self.log_likelihood(&shifted(a, -1.0, None), ctx, None)?;
            info[a * p + a] = -(up - 2.0 * center + down) / (steps[a] * steps[a]);
            for b in 0..a {
                let pp = self.log_likelihood(&shifted(a, 1.0, Some((b, 1.0))), ctx, None)?;
                let pm = self.log_likelihood(&shifted(a, 1.0, Some((b, -1.0))), ctx, None)?;
                let mp = self.log_likelihood(&shifted(a, -1.0, Some((b, 1.0))), ctx, None)?;
                let mm = self.log_likelihood(&shifted(a, -1.0, Some((b, -1.0))), ctx, None)?;
                let value = -(pp - pm - mp + mm) / (4.0 * steps[a] * steps[b]);
                info[a * p + b] = value;
                info[b * p + a] = value;
            }
        }
        Ok(info)
    }
}

// ---------------------------------------------------------------------------
// Evidence.
// ---------------------------------------------------------------------------

/// Identification status of the fitted coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdeIdentificationStatus {
    /// Locally identified by the observed information with no stated condition.
    Identified,
    /// The observed information is not positive definite.
    NotIdentified,
    /// Locally identified, but only under a stated condition of the declared model.
    ModelConditional,
}

/// Optimizer termination, distinct from identification and discretization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeFitting {
    /// Declared score tolerance reached; no global optimum or identification claim.
    pub converged: bool,
    /// Maximum absolute final score component.
    pub score_norm: f64,
    /// Accepted parameter steps.
    pub iterations: usize,
    /// Likelihood evaluations, including difference stencils.
    pub evaluations: usize,
    /// Backtracking exhausted its bounded attempts.
    pub line_search_exhausted: bool,
    /// Named optimizer.
    pub optimizer: String,
    /// How the score and information were obtained.
    pub derivative_method: String,
}

/// Identification evidence, separate from optimizer convergence and discretization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeIdentification {
    /// Identified, not identified or model-conditional.
    pub status: SdeIdentificationStatus,
    /// Number of fitted coordinates.
    pub parameter_count: usize,
    /// Numerical rank of the correlation-scaled observed information.
    pub information_rank: usize,
    /// Smallest eigenvalue of the correlation-scaled observed information.
    pub information_min_eigenvalue: Option<f64>,
    /// Condition number of the correlation-scaled observed information.
    pub information_condition: Option<f64>,
    /// State coordinates no observation-matrix row reads (always empty for direct observation).
    pub unobserved_components: Vec<String>,
    /// Conditions that make an otherwise locally identified result model-conditional.
    pub conditions: Vec<String>,
    /// Declared assumptions the identification is relative to.
    pub assumptions: Vec<String>,
}

/// Whether the observation intervals support the Euler transition approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdeDiscretizationStatus {
    /// Every interval-rate product is below the flag threshold.
    Adequate,
    /// Some interval-rate product exceeds the flag threshold: the fit is a
    /// model-conditional approximation whose bias is not quantified.
    ModelConditionalApproximation,
}

/// Discretization evidence of the quasi likelihood.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeDiscretization {
    /// Which likelihood was evaluated.
    pub likelihood: String,
    /// Adequate or model-conditional.
    pub status: SdeDiscretizationStatus,
    /// Transitions used.
    pub transitions: usize,
    /// Smallest observation interval.
    pub minimum_interval: Option<f64>,
    /// Largest observation interval.
    pub maximum_interval: f64,
    /// Largest `interval * ||J_f||_inf` over transitions at the reported fit.
    pub maximum_rate_interval_product: f64,
    /// Mean of that product.
    pub mean_rate_interval_product: f64,
    /// Transitions above the flag threshold.
    pub transitions_above_flag: usize,
    /// Flag threshold of the policy.
    pub flag_threshold: f64,
    /// Refusal threshold of the policy.
    pub refusal_threshold: f64,
    /// Fraction of transitions within three standard deviations of a declared bound, where
    /// the Gaussian density ignores the reflection.
    pub boundary_proximate_fraction: f64,
    /// What the product measures and what it does not.
    pub note: String,
}

/// Quasi-likelihood fit with three separate evidence blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeFit {
    /// Fitted parameter coordinates (indices into the system's parameters).
    pub parameters: Vec<usize>,
    /// Names of the fitted coordinates.
    pub names: Vec<String>,
    /// Fitted values, aligned with `parameters`.
    pub values: Vec<f64>,
    /// The full parameter vector at the fit (unfitted entries keep their declared values).
    pub theta: Vec<f64>,
    /// Quasi log likelihood at the fit (a pseudo-likelihood, not the exact log likelihood).
    pub log_quasi_likelihood: f64,
    /// Observation vectors used.
    pub observation_count: usize,
    /// Row-major inverse observed information (model-based sampling covariance of the
    /// quasi-maximum-likelihood estimate); `None` unless the information is positive definite.
    pub parameter_covariance: Option<Vec<f64>>,
    /// Square roots of the diagonal of `parameter_covariance`.
    pub standard_errors: Option<Vec<f64>>,
    /// Optimizer termination.
    pub fitting: SdeFitting,
    /// Identification evidence.
    pub identification: SdeIdentification,
    /// Discretization evidence.
    pub discretization: SdeDiscretization,
}

fn validate_design(
    compiled: &CompiledSde,
    coordinates: &[usize],
    observation: Option<&SdeObservationModel>,
    prior: Option<&SdeFilterPrior>,
    trajectories: &[SdeTrajectory],
) -> Result<(usize, usize), SdeFitError> {
    let d = compiled.dimension();
    let p = compiled.system().parameters.len();
    if coordinates.is_empty() || coordinates.len() > MAX_FIT_PARAMETERS {
        return Err(bad("SDE fit parameter count"));
    }
    let mut seen = std::collections::BTreeSet::new();
    if coordinates.iter().any(|c| *c >= p || !seen.insert(*c)) {
        return Err(bad("SDE fit parameter coordinates must be distinct declared parameters"));
    }
    let m = match observation {
        Some(model) => {
            let m = model.validate(d)?;
            let Some(prior) = prior else {
                return Err(bad("an SDE observation model needs a filter prior"));
            };
            if prior.mean.len() != d
                || prior.covariance.len() != d * d
                || prior.mean.iter().chain(&prior.covariance).any(|v| !v.is_finite())
                || cholesky_or_psd(&prior.covariance, d).is_none()
            {
                return Err(bad("SDE filter prior shape or covariance"));
            }
            m
        }
        None => {
            if prior.is_some() {
                return Err(bad("an SDE filter prior needs an observation model"));
            }
            d
        }
    };
    if trajectories.is_empty() || trajectories.len() > MAX_TRAJECTORIES {
        return Err(bad("SDE trajectory count"));
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut total = 0;
    for t in trajectories {
        if t.id.trim().is_empty() || !ids.insert(&t.id) || t.observations.is_empty() {
            return Err(bad("SDE trajectory identity/observations"));
        }
        total += t.observations.len();
        if total > MAX_OBSERVATIONS {
            return Err(bad("SDE observation count"));
        }
        let mut previous: Option<f64> = None;
        for o in &t.observations {
            if !o.time.is_finite()
                || previous.is_some_and(|q| o.time <= q)
                || o.values.len() != m
                || o.values.iter().any(|v| !v.is_finite())
            {
                return Err(bad("SDE observation time/values"));
            }
            previous = Some(o.time);
        }
        if observation.is_none() {
            for o in &t.observations {
                for i in 0..d {
                    if !compiled.system().boundary(i).contains_value(o.values[i]) {
                        return Err(bad("SDE observed state outside its declared interval"));
                    }
                }
            }
        }
    }
    Ok((d, m))
}

/// Symmetric positive semi-definiteness by an eigenvalue test (a prior may be singular).
fn cholesky_or_psd(a: &[f64], n: usize) -> Option<()> {
    let sym = (0..n).all(|i| {
        (0..i).all(|j| (a[i * n + j] - a[j * n + i]).abs() <= 1e-9 * (1.0 + a[i * n + j].abs()))
    });
    let min = symmetric_eigenvalues(a, n).into_iter().fold(f64::INFINITY, f64::min);
    (sym && min >= -1e-9).then_some(())
}

struct InformationSummary {
    covariance: Option<Vec<f64>>,
    rank: usize,
    min_eigenvalue: Option<f64>,
    condition: Option<f64>,
}

fn summarize_information(info: &[f64], p: usize) -> InformationSummary {
    let mut info = info.to_vec();
    for i in 0..p {
        for j in 0..i {
            let avg = 0.5 * (info[i * p + j] + info[j * p + i]);
            info[i * p + j] = avg;
            info[j * p + i] = avg;
        }
    }
    let max_diagonal = (0..p).map(|i| info[i * p + i]).fold(0.0, f64::max);
    let flat = (0..p).filter(|i| !(info[i * p + i] > FLAT_DIAGONAL * max_diagonal)).count();
    if !(max_diagonal > 0.0) || flat > 0 {
        return InformationSummary {
            covariance: None,
            rank: p - flat.min(p),
            min_eigenvalue: Some(0.0),
            condition: None,
        };
    }
    let mut scaled = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            scaled[i * p + j] = info[i * p + j] / (info[i * p + i] * info[j * p + j]).sqrt();
        }
    }
    let spectrum = symmetric_eigenvalues(&scaled, p);
    let minimum = spectrum.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = spectrum.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rank = spectrum.iter().filter(|v| **v > INFORMATION_FLOOR).count();
    let positive = rank == p;
    let covariance = if positive { spd_inverse(&info, p) } else { None };
    let rank = if positive && covariance.is_none() { rank.saturating_sub(1) } else { rank };
    InformationSummary {
        covariance,
        rank,
        min_eigenvalue: Some(minimum),
        condition: (minimum > 0.0).then(|| maximum / minimum),
    }
}

/// Fit the named parameter coordinates by the Euler quasi likelihood (or, with a declared
/// observation model, the extended-Kalman innovation likelihood).
///
/// Typed refusals: `Unidentified` when no transition is observed or the converged
/// information is singular; `Unsupported` (`sde_fit.observation_interval_too_coarse`) when an
/// interval-rate product exceeds the refusal threshold, and
/// (`sde_fit.degenerate_transition`) when a transition variance vanishes.
#[allow(
    clippy::too_many_lines,
    reason = "one fit: validation, optimization, then three evidence blocks"
)]
pub fn fit_sde_quasi_likelihood(
    compiled: &CompiledSde,
    coordinates: &[usize],
    observation: Option<&SdeObservationModel>,
    prior: Option<&SdeFilterPrior>,
    trajectories: &[SdeTrajectory],
    policy: SdeFitPolicy,
    ctx: &ExecutionContext,
) -> Result<SdeFit, SdeFitError> {
    policy.validate()?;
    let (d, m) = validate_design(compiled, coordinates, observation, prior, trajectories)?;
    let p = coordinates.len();
    let observation_total: usize = trajectories.iter().map(|t| t.observations.len()).sum();
    let _lease = ctx
        .acquire_admission(0, (64 * (d * d + m * m + p * p) + 16 * observation_total) as u64 + 4096)
        .map_err(|e| SdeFitError::Resource(e.to_string()))?;
    let prepared = prepare(compiled, trajectories)?;
    if !trajectories.iter().any(|t| t.observations.len() >= 2) {
        return Err(SdeFitError::Unidentified {
            failure: SdeIdentificationFailure::NoTransitionInformation,
            detail: "no trajectory has two observation times, so no state transition is observed; \
                     snapshots identify at most a stationary law"
                .into(),
        });
    }
    let base = compiled.system().parameter_values();
    let problem = Problem {
        compiled,
        coordinates,
        base: base.clone(),
        observation,
        prior,
        prepared,
        policy: &policy,
        d,
        m,
    };
    let mut values: Vec<f64> = coordinates.iter().map(|c| base[*c]).collect();
    // The starting values must be admissible, and the design must not be refused as too coarse.
    let mut probe = Probe::default();
    let mut likelihood = problem.log_likelihood(&values, ctx, Some(&mut probe))?;
    refuse_if_coarse(&probe, &policy)?;
    let mut evaluations = 1;
    let (mut score, used) = problem.gradient(&values, ctx)?;
    evaluations += used;
    let mut iterations = 0;
    let mut exhausted = false;
    let mut curvature = InverseCurvature::default();
    let optimizer = policy.optimizer;
    while iterations < optimizer.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(SdeFitError::Resource("cancelled".into()));
        }
        let norm = score.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if norm <= optimizer.score_tolerance {
            break;
        }
        let (mut direction, quasi_newton) = curvature.direction(&score);
        let maximum = direction.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1.0);
        for value in &mut direction {
            *value /= maximum;
        }
        let derivative: f64 = score.iter().zip(&direction).map(|(g, dd)| g * dd).sum();
        let mut step = if quasi_newton { 1.0 } else { optimizer.learning_rate };
        let mut accepted = None;
        for _ in 0..48 {
            let proposed: Vec<f64> =
                values.iter().zip(&direction).map(|(v, dd)| v + step * dd).collect();
            evaluations += 1;
            match problem.log_likelihood(&proposed, ctx, None) {
                Ok(next) if next >= likelihood + 1e-4 * step * derivative => {
                    match problem.gradient(&proposed, ctx) {
                        Ok((next_score, used)) => {
                            evaluations += used;
                            accepted = Some((proposed, next, next_score));
                            break;
                        }
                        Err(SdeFitError::Resource(reason)) => {
                            return Err(SdeFitError::Resource(reason));
                        }
                        Err(_) => {}
                    }
                }
                Err(SdeFitError::Resource(reason)) => return Err(SdeFitError::Resource(reason)),
                _ => {}
            }
            step *= 0.5;
        }
        if let Some((proposed, next, next_score)) = accepted {
            curvature.update(&values, &proposed, &score, &next_score);
            values = proposed;
            likelihood = next;
            score = next_score;
            iterations += 1;
        } else {
            exhausted = true;
            break;
        }
    }
    let score_norm = score.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let converged = score_norm <= optimizer.score_tolerance;

    // Evidence is evaluated once at the reported point.
    let mut probe = Probe::default();
    let final_likelihood = problem.log_likelihood(&values, ctx, Some(&mut probe))?;
    refuse_if_coarse(&probe, &policy)?;
    let info = problem.information(&values, final_likelihood, ctx)?;
    evaluations += 1 + 2 * p + 4 * p * p.saturating_sub(1) / 2;
    let summary = summarize_information(&info, p);
    let system = compiled.system();
    let unobserved_components: Vec<String> = match observation {
        Some(model) => (0..d)
            .filter(|i| (0..m).all(|r| model.matrix[r * d + i] == 0.0))
            .map(|i| system.variables[i].clone())
            .collect(),
        None => Vec::new(),
    };
    let information_positive = summary.covariance.is_some();
    if converged && !information_positive {
        return Err(SdeFitError::Unidentified {
            failure: SdeIdentificationFailure::SingularInformation,
            detail: format!(
                "the observed information of the {p} fitted coordinates has numerical rank {} at the \
                 converged optimum (smallest scaled eigenvalue {:?}); components without a direct \
                 observation: {unobserved_components:?}",
                summary.rank, summary.min_eigenvalue
            ),
        });
    }
    let transitions = probe.transitions.max(1);
    let boundary_fraction = probe.boundary_transitions as f64 / transitions as f64;
    let mut conditions = Vec::new();
    if observation.is_some() {
        conditions.push("extended_kalman_linearization".to_owned());
    }
    if probe.above_flag > 0 {
        conditions.push("observation_interval_coarse_for_euler".to_owned());
    }
    if boundary_fraction > BOUNDARY_FRACTION_FLAG {
        conditions.push("reflecting_boundary_not_in_likelihood".to_owned());
    }
    if !unobserved_components.is_empty() {
        conditions.push("unobserved_state_filtered".to_owned());
    }
    let status = if !information_positive {
        SdeIdentificationStatus::NotIdentified
    } else if conditions.is_empty() {
        SdeIdentificationStatus::Identified
    } else {
        SdeIdentificationStatus::ModelConditional
    };
    let standard_errors = summary
        .covariance
        .as_ref()
        .map(|c| (0..p).map(|i| c[i * p + i].max(0.0).sqrt()).collect::<Vec<f64>>());
    let theta = problem.theta(&values);
    Ok(SdeFit {
        parameters: coordinates.to_vec(),
        names: coordinates.iter().map(|c| system.parameters[*c].name.clone()).collect(),
        values,
        theta,
        log_quasi_likelihood: final_likelihood,
        observation_count: observation_total,
        parameter_covariance: summary.covariance,
        standard_errors,
        fitting: SdeFitting {
            converged,
            score_norm,
            iterations,
            evaluations,
            line_search_exhausted: exhausted,
            optimizer: "limited_memory_bfgs_armijo_with_central_difference_score".into(),
            derivative_method: "central_difference_of_quasi_log_likelihood".into(),
        },
        identification: SdeIdentification {
            status,
            parameter_count: p,
            information_rank: summary.rank,
            information_min_eigenvalue: summary.min_eigenvalue,
            information_condition: summary.condition,
            unobserved_components,
            conditions,
            assumptions: vec![
                "the declared drift and diagonal diffusion are the data-generating structure".into(),
                "known intervention regime per trajectory; held coordinates carry no transition density".into(),
                "inputs are read at the left end of each observation interval".into(),
                "standard errors are model-based (inverse observed information of the pseudo-likelihood), not sandwich or bootstrap".into(),
                "local identification from the observed information at the reported optimum".into(),
            ],
        },
        discretization: SdeDiscretization {
            likelihood: if observation.is_some() {
                "extended_kalman_euler_innovation".into()
            } else {
                "euler_gaussian_transition".into()
            },
            status: if probe.above_flag > 0 {
                SdeDiscretizationStatus::ModelConditionalApproximation
            } else {
                SdeDiscretizationStatus::Adequate
            },
            transitions: probe.transitions,
            minimum_interval: probe.minimum_interval,
            maximum_interval: probe.maximum_interval,
            maximum_rate_interval_product: probe.maximum_rate_interval,
            mean_rate_interval_product: probe.rate_interval_sum / transitions as f64,
            transitions_above_flag: probe.above_flag,
            flag_threshold: policy.coarse_flag,
            refusal_threshold: policy.coarse_refusal,
            boundary_proximate_fraction: boundary_fraction,
            note: "interval x local relaxation rate of the drift; the Euler transition density \
                   is accurate when it is small. The product bounds nothing about the size of \
                   the bias, which is not estimated."
                .into(),
        },
    })
}

fn refuse_if_coarse(probe: &Probe, policy: &SdeFitPolicy) -> Result<(), SdeFitError> {
    if probe.maximum_rate_interval > policy.coarse_refusal {
        return Err(SdeFitError::Unsupported {
            code: "sde_fit.observation_interval_too_coarse",
            detail: format!(
                "an observation interval of {:.4} times the local drift rate gives {:.4}, above \
                 the refusal threshold {}; the Euler transition density is not a justified \
                 approximation there. Observe more densely, or fit a model whose transition law \
                 is exact (a linear Gaussian OU)",
                probe.maximum_interval, probe.maximum_rate_interval, policy.coarse_refusal
            ),
        });
    }
    Ok(())
}

/// Evaluate the quasi log likelihood at a full parameter vector.
///
/// # Errors
/// As [`fit_sde_quasi_likelihood`].
pub fn sde_quasi_log_likelihood(
    compiled: &CompiledSde,
    theta: &[f64],
    observation: Option<&SdeObservationModel>,
    prior: Option<&SdeFilterPrior>,
    trajectories: &[SdeTrajectory],
    policy: SdeFitPolicy,
    ctx: &ExecutionContext,
) -> Result<f64, SdeFitError> {
    policy.validate()?;
    let all: Vec<usize> = (0..compiled.system().parameters.len()).collect();
    if all.is_empty() {
        return Err(bad("SDE has no parameters"));
    }
    let (d, m) = validate_design(compiled, &[0], observation, prior, trajectories)?;
    if theta.len() != all.len() || theta.iter().any(|v| !v.is_finite()) {
        return Err(bad("SDE parameter vector"));
    }
    let prepared = prepare(compiled, trajectories)?;
    let problem = Problem {
        compiled,
        coordinates: &all,
        base: theta.to_vec(),
        observation,
        prior,
        prepared,
        policy: &policy,
        d,
        m,
    };
    problem.log_likelihood(theta, ctx, None)
}

// ---------------------------------------------------------------------------
// Report: fit, ensembles at the fit, and parameter uncertainty kept separate.
// ---------------------------------------------------------------------------

/// Between-draw spread of one world's ensemble mean over fitted-parameter draws.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeWorldParameterSpread {
    /// World identity.
    pub id: String,
    /// Sample variance, across parameter draws, of the ensemble mean per time and coordinate.
    pub mean_between_draw_variance: Vec<Vec<f64>>,
}

/// Between-draw spread of one paired contrast.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeContrastParameterSpread {
    /// Contrast identity.
    pub id: String,
    /// Sample variance, across parameter draws, of the paired mean difference.
    pub mean_difference_between_draw_variance: Vec<Vec<f64>>,
}

/// Statistical (parameter) uncertainty of the simulated responses, by asymptotic normal draws.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeParameterUncertainty {
    /// Requested draws.
    pub draws: usize,
    /// Draws whose simulation completed.
    pub completed_draws: usize,
    /// Seed of the parameter draws.
    pub seed: u64,
    /// Spread per world.
    pub worlds: Vec<SdeWorldParameterSpread>,
    /// Spread per contrast.
    pub contrasts: Vec<SdeContrastParameterSpread>,
    /// What the variances mean and what they exclude.
    pub meaning: String,
}

/// A fit (when requested) and the ensembles simulated at it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeReport {
    /// The quasi-likelihood fit, when trajectories were supplied.
    pub fit: Option<SdeFit>,
    /// Ensembles and paired contrasts at the fitted (or declared) parameters, with Monte
    /// Carlo and numerical-approximation error.
    pub simulation: SdeEnsembleReport,
    /// Parameter-draw spread, when requested; separate from the simulation's own error.
    pub parameter_uncertainty: Option<SdeParameterUncertainty>,
}

fn sample_variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
}

/// Simulate worlds at the fit (or declared parameters), and propagate the fit's sampling
/// covariance by common-random-number ensembles at asymptotic-normal parameter draws.
///
/// # Errors
/// `Unsupported` (`sde_fit.parameter_uncertainty_unavailable`) when draws are requested
/// without a positive-definite parameter covariance; otherwise as `simulate_sde`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines, reason = "one report assembly")]
pub fn sde_report(
    compiled: &CompiledSde,
    fit: Option<SdeFit>,
    initial: &[f64],
    policy: &SdeSimulationPolicy,
    worlds: &[SdeWorld],
    contrasts: &[SdeContrast],
    parameter_draws: usize,
    parameter_seed: u64,
    ctx: &ExecutionContext,
) -> Result<SdeReport, SdeFitError> {
    if parameter_draws > MAX_PARAMETER_DRAWS || parameter_draws == 1 {
        return Err(bad("SDE parameter draws must be 0 or 2..=512"));
    }
    let theta: Option<&[f64]> = fit.as_ref().map(|f| f.theta.as_slice());
    let simulation = simulate_sde(compiled, theta, initial, policy, worlds, contrasts, ctx)?;
    let mut uncertainty = None;
    if parameter_draws > 0 {
        let Some(fitted) = fit.as_ref() else {
            return Err(SdeFitError::Unsupported {
                code: "sde_fit.parameter_uncertainty_unavailable",
                detail: "parameter uncertainty needs a fit; supply trajectories".into(),
            });
        };
        let Some(covariance) = fitted.parameter_covariance.as_ref() else {
            return Err(SdeFitError::Unsupported {
                code: "sde_fit.parameter_uncertainty_unavailable",
                detail: "the observed information is not positive definite, so there is no \
                         sampling covariance to propagate"
                    .into(),
            });
        };
        let p = fitted.values.len();
        let factor = cholesky(covariance, p)
            .ok_or_else(|| numerical("parameter covariance is not positive definite"))?;
        let mut draw_policy = policy.clone();
        draw_policy.check_step = false;
        draw_policy.retain_final_samples = false;
        let mut world_means: Vec<Vec<Vec<Vec<f64>>>> = vec![Vec::new(); simulation.worlds.len()];
        let mut contrast_means: Vec<Vec<Vec<Vec<f64>>>> =
            vec![Vec::new(); simulation.contrasts.len()];
        let mut completed = 0;
        for b in 0..parameter_draws {
            if ctx.cancellation.is_cancelled() {
                return Err(SdeFitError::Resource("cancelled".into()));
            }
            let z: Vec<f64> = (0..p)
                .map(|j| counter_normal(parameter_seed, b as u64, u64::MAX, j as u64))
                .collect();
            let mut theta_b = fitted.theta.clone();
            for (a, coordinate) in fitted.parameters.iter().enumerate() {
                let shift: f64 = (0..=a).map(|c| factor[a * p + c] * z[c]).sum();
                theta_b[*coordinate] = fitted.values[a] + shift;
            }
            match simulate_sde(
                compiled,
                Some(&theta_b),
                initial,
                &draw_policy,
                worlds,
                contrasts,
                ctx,
            ) {
                Ok(draw) => {
                    completed += 1;
                    for (slot, w) in draw.worlds.into_iter().enumerate() {
                        world_means[slot].push(w.mean);
                    }
                    for (slot, c) in draw.contrasts.into_iter().enumerate() {
                        contrast_means[slot].push(c.mean_difference);
                    }
                }
                Err(SdeError::Resource(reason)) => return Err(SdeFitError::Resource(reason)),
                Err(_) => {}
            }
        }
        if completed < 2 {
            return Err(SdeFitError::Numerical(
                "fewer than two parameter draws produced an admissible simulation".into(),
            ));
        }
        let spread = |runs: &Vec<Vec<Vec<f64>>>| -> Vec<Vec<f64>> {
            let (t, dim) = (runs[0].len(), runs[0][0].len());
            (0..t)
                .map(|ti| {
                    (0..dim)
                        .map(|i| {
                            sample_variance(&runs.iter().map(|r| r[ti][i]).collect::<Vec<f64>>())
                        })
                        .collect()
                })
                .collect()
        };
        uncertainty = Some(SdeParameterUncertainty {
            draws: parameter_draws,
            completed_draws: completed,
            seed: parameter_seed,
            worlds: simulation
                .worlds
                .iter()
                .zip(&world_means)
                .map(|(w, runs)| SdeWorldParameterSpread {
                    id: w.id.clone(),
                    mean_between_draw_variance: spread(runs),
                })
                .collect(),
            contrasts: simulation
                .contrasts
                .iter()
                .zip(&contrast_means)
                .map(|(c, runs)| SdeContrastParameterSpread {
                    id: c.id.clone(),
                    mean_difference_between_draw_variance: spread(runs),
                })
                .collect(),
            meaning: "variance across asymptotic-normal draws of the fitted parameters of the \
                      ensemble mean, each draw simulated with the same random numbers so that \
                      Monte Carlo noise largely cancels; it is conditional on the declared model, \
                      excludes Monte Carlo and discretization error, and is not summed with them"
                .into(),
        });
    }
    Ok(SdeReport { fit, simulation, parameter_uncertainty: uncertainty })
}

#[cfg(test)]
mod tests;
