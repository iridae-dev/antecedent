// SPDX-License-Identifier: MIT OR Apache-2.0
//! Continuous-time Ornstein–Uhlenbeck fitting from continuous trajectories.
//!
//! The hidden state follows `dX = (M X + b) dt + L dW` with piecewise-constant,
//! known equation replacements. Observations at arbitrary, irregular times follow
//! `y = H x + d + e` with a declared linear Gaussian observation model. Between
//! observations the transition law is exact: the matrix exponential gives the mean
//! map and the van Loan block exponential gives the process covariance, so no time
//! discretization error enters the likelihood. A Kalman filter evaluates the
//! Gaussian likelihood, which covers full and partial observation.
//!
//! Fitting (optimizer termination), identification (observed information and
//! observability, sampling-interval aliasing) and numerical stability (matrix
//! exponential accuracy, conditioning, stability of the drift) are separate
//! evidence fields. The score is analytic (Fréchet derivatives of the matrix exponential
//! carried through the Kalman recursion); the information is the Gauss–Newton form.
use crate::cyclic_gradient::CyclicGradientPolicy;
use crate::cyclic_optim::InverseCurvature;
use crate::linalg::{symmetrize_average, SpdFactor};
use crate::ou_mechanisms::{
    bind, parameters, OuEquationReplacement, OuMechanismParameter, OuMechanisms,
};
use antecedent_core::ExecutionContext;
use antecedent_model::nonlinear_equilibrium::EquilibriumError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_PARAMETERS: usize = 64;
const MAX_TRAJECTORIES: usize = 4096;
const MAX_SEGMENTS: usize = 4096;
const MAX_TIMES: usize = 1 << 20;
const MAX_OBSERVED: usize = 64;
#[cfg(test)]
const GRADIENT_STEP: f64 = 1e-5;
#[cfg(test)]
const HESSIAN_STEP: f64 = 1e-3;
const PREDICTION_STEP: f64 = 1e-5;
// The information is analytic (no difference-stencil noise), so only rounding-level scaled
// eigenvalues count as singular; weak but positive identification is reported through the
// condition number instead of being refused.
const INFORMATION_FLOOR: f64 = 1e-10;
const ILL_CONDITIONED: f64 = 1e8;
const FLAT_DIAGONAL: f64 = 1e-10;

/// Why a trajectory design cannot identify the requested dynamics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OuIdentificationFailure {
    /// No trajectory has two observation times, so no transition is observed.
    NoTransitionInformation,
    /// The observed information of the requested coordinates is singular.
    SingularInformation,
}

impl OuIdentificationFailure {
    /// Stable machine-readable detail code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::NoTransitionInformation => "ou_trajectory.no_transition_information",
            Self::SingularInformation => "ou_trajectory.singular_information",
        }
    }
}

impl std::fmt::Display for OuIdentificationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// Typed failure of trajectory fitting or prediction.
#[derive(Debug, thiserror::Error)]
pub enum OuTrajectoryError {
    /// Malformed shape, non-finite number, unordered times or invalid policy.
    #[error("invalid OU trajectory input: {0}")]
    Invalid(String),
    /// The design does not identify the requested coordinates.
    #[error("OU trajectory dynamics are not identified ({failure}): {detail}")]
    Unidentified {
        /// Stable classification.
        failure: OuIdentificationFailure,
        /// The evidence that led to the refusal.
        detail: String,
    },
    /// A numerical condition (stability, positive definiteness, finiteness) failed.
    #[error("OU trajectory numerical failure: {0}")]
    Numerical(String),
    /// Resource admission or cancellation failed.
    #[error("OU trajectory resource refusal: {0}")]
    Resource(String),
}

impl From<EquilibriumError> for OuTrajectoryError {
    fn from(error: EquilibriumError) -> Self {
        match error {
            EquilibriumError::Resource(reason) => Self::Resource(reason),
            EquilibriumError::Invalid(reason) => Self::Invalid(reason),
            other => Self::Numerical(other.to_string()),
        }
    }
}

fn bad(reason: &str) -> OuTrajectoryError {
    OuTrajectoryError::Invalid(reason.into())
}
fn numerical(reason: &str) -> OuTrajectoryError {
    OuTrajectoryError::Numerical(reason.into())
}

/// Linear Gaussian observation model `y = H x + offset + noise`, noise independent per row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuObservationModel {
    /// Row-major `m x n` observation matrix H.
    pub matrix: Vec<f64>,
    /// Observation offset d; empty denotes zero.
    #[serde(default)]
    pub offset: Vec<f64>,
    /// Known observation-noise standard deviation per observed row.
    pub noise_standard_deviation: Vec<f64>,
}

impl OuObservationModel {
    /// Every state component observed directly with the given noise standard deviation.
    #[must_use]
    pub fn direct(n: usize, noise_standard_deviation: f64) -> Self {
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            matrix[i * n + i] = 1.0;
        }
        Self {
            matrix,
            offset: Vec::new(),
            noise_standard_deviation: vec![noise_standard_deviation; n],
        }
    }

    /// Validate against the state dimension and return the observed dimension.
    pub fn validate(&self, n: usize) -> Result<usize, OuTrajectoryError> {
        let m = self.noise_standard_deviation.len();
        if m == 0
            || m > MAX_OBSERVED
            || self.matrix.len() != m * n
            || !(self.offset.is_empty() || self.offset.len() == m)
            || self.matrix.iter().chain(&self.offset).any(|v| !v.is_finite())
            || self.noise_standard_deviation.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.matrix.iter().all(|v| *v == 0.0)
        {
            return Err(bad("OU observation model shape/finiteness"));
        }
        Ok(m)
    }

    fn offset_at(&self, i: usize) -> f64 {
        self.offset.get(i).copied().unwrap_or(0.0)
    }
}

/// One piece of a known piecewise-constant intervention schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuScheduleSegment {
    /// Time from which these replacements hold, until the next segment starts.
    pub start: f64,
    /// Equation replacements in force; empty denotes the baseline mechanism.
    pub replacements: Vec<OuEquationReplacement>,
}

/// Law of the hidden state at the first schedule time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OuInitialLaw {
    /// The stationary law of the first segment's mechanism (requires a stable drift).
    Stationary,
    /// A declared Gaussian initial law; zero covariance is a known initial state.
    Gaussian {
        /// Initial mean.
        mean: Vec<f64>,
        /// Row-major initial covariance.
        covariance: Vec<f64>,
    },
}

/// One observation of the process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuObservation {
    /// Observation time.
    pub time: f64,
    /// Observed vector, one entry per observation-matrix row.
    pub values: Vec<f64>,
}

/// One continuous trajectory of a single unit under a known intervention schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectory {
    /// Stable trajectory identity.
    pub id: String,
    /// Known piecewise-constant intervention schedule; starts strictly increase.
    pub schedule: Vec<OuScheduleSegment>,
    /// Hidden-state law at the first schedule time.
    pub initial: OuInitialLaw,
    /// Observations at strictly increasing, possibly irregular times.
    pub observations: Vec<OuObservation>,
}

/// A held-out intervention trajectory whose state law is predicted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuPredictionDesign {
    /// Stable identity.
    pub id: String,
    /// Known piecewise-constant intervention schedule.
    pub schedule: Vec<OuScheduleSegment>,
    /// Hidden-state law at the first schedule time.
    pub initial: OuInitialLaw,
    /// Strictly increasing prediction times, at or after the first schedule time.
    pub times: Vec<f64>,
}

/// A sustained intervention world whose stationary law is predicted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuStationaryDesign {
    /// Stable identity.
    pub id: String,
    /// Equation replacements in force; empty denotes the baseline.
    pub replacements: Vec<OuEquationReplacement>,
}

// ---------------------------------------------------------------------------
// Dense helpers (row-major).
// ---------------------------------------------------------------------------

fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

fn matmul(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for l in 0..k {
            let x = a[i * k + l];
            if x == 0.0 {
                continue;
            }
            for j in 0..c {
                out[i * c + j] += x * b[l * c + j];
            }
        }
    }
    out
}

fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// Scaling-and-squaring Taylor matrix exponential of a row-major `d x d` matrix.
fn expm(a: &[f64], d: usize) -> Result<Vec<f64>, OuTrajectoryError> {
    let norm = (0..d).map(|j| (0..d).map(|i| a[i * d + j].abs()).sum::<f64>()).fold(0.0, f64::max);
    if !norm.is_finite() || norm > 1e6 {
        return Err(numerical("matrix exponential argument is not finite or is too large"));
    }
    let mut squarings = 0_i32;
    let mut scaled_norm = norm;
    while scaled_norm > 0.25 {
        scaled_norm *= 0.5;
        squarings += 1;
    }
    let factor = 0.5_f64.powi(squarings);
    let scaled: Vec<f64> = a.iter().map(|v| v * factor).collect();
    let mut result = identity(d);
    let mut term = identity(d);
    for k in 1..=40_u32 {
        term = matmul(&term, &scaled, d, d, d);
        let divisor = f64::from(k);
        for value in &mut term {
            *value /= divisor;
        }
        for (r, t) in result.iter_mut().zip(&term) {
            *r += t;
        }
        if term.iter().all(|v| v.abs() < 1e-19) {
            break;
        }
    }
    for _ in 0..squarings {
        result = matmul(&result, &result, d, d, d);
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(numerical("matrix exponential overflowed"));
    }
    Ok(result)
}

/// `max |exp(A) exp(-A) - I|`: an independent accuracy witness of the exponential.
fn expm_inverse_residual(a: &[f64], d: usize) -> Result<f64, OuTrajectoryError> {
    let forward = expm(a, d)?;
    let negated: Vec<f64> = a.iter().map(|v| -v).collect();
    let backward = expm(&negated, d)?;
    let product = matmul(&forward, &backward, d, d, d);
    let eye = identity(d);
    Ok(product.iter().zip(&eye).map(|(p, e)| (p - e).abs()).fold(0.0, f64::max))
}

// ---------------------------------------------------------------------------
// Exact transitions and propagation.
// ---------------------------------------------------------------------------

struct Transition {
    phi: Vec<f64>,
    shift: Vec<f64>,
    q: Vec<f64>,
}

/// Exact transition over `h` of `dX = (M X + b) dt + L dW`.
fn transition(
    model: &OuMechanisms,
    h: f64,
    residual: Option<&mut f64>,
) -> Result<Transition, OuTrajectoryError> {
    let n = model.equations.len();
    let width = n + 1;
    let mut augmented = vec![0.0; width * width];
    for i in 0..n {
        for j in 0..n {
            augmented[i * width + j] = model.drift[i * n + j] * h;
        }
        augmented[i * width + n] = model.forcing[i] * h;
    }
    let mean_map = expm(&augmented, width)?;
    let mut phi = vec![0.0; n * n];
    let mut shift = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            phi[i * n + j] = mean_map[i * width + j];
        }
        shift[i] = mean_map[i * width + n];
    }
    // Van Loan: exp([[-M, LL^T], [0, M^T]] h) has Phi^T in its lower-right block and
    // Phi^{-1} Q in its upper-right block.
    let size = 2 * n;
    let factor = &model.diffusion_factor;
    let mut loan = vec![0.0; size * size];
    for i in 0..n {
        for j in 0..n {
            loan[i * size + j] = -model.drift[i * n + j] * h;
            loan[(n + i) * size + n + j] = model.drift[j * n + i] * h;
            loan[i * size + n + j] =
                (0..n).map(|s| factor[i * n + s] * factor[j * n + s]).sum::<f64>() * h;
        }
    }
    let block = expm(&loan, size)?;
    if let Some(slot) = residual {
        let worst =
            expm_inverse_residual(&augmented, width)?.max(expm_inverse_residual(&loan, size)?);
        *slot = slot.max(worst);
    }
    let mut upper = vec![0.0; n * n];
    let mut lower_right = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            upper[i * n + j] = block[i * size + n + j];
            lower_right[i * n + j] = block[(n + i) * size + n + j];
        }
    }
    let mut q = matmul(&transpose(&lower_right, n, n), &upper, n, n, n);
    symmetrize_average(&mut q, n);
    Ok(Transition { phi, shift, q })
}

/// Segment-wise exact propagation of the hidden-state law.
struct Propagator {
    n: usize,
    starts: Vec<f64>,
    models: Vec<OuMechanisms>,
    cache: BTreeMap<(usize, u64), Transition>,
    check: bool,
    residual: f64,
}

impl Propagator {
    fn new(
        template: &OuMechanisms,
        schedule: &[OuScheduleSegment],
        check: bool,
    ) -> Result<Self, OuTrajectoryError> {
        if schedule.is_empty() || schedule.len() > MAX_SEGMENTS {
            return Err(bad("OU intervention schedule length"));
        }
        let mut starts = Vec::with_capacity(schedule.len());
        let mut models = Vec::with_capacity(schedule.len());
        for (k, segment) in schedule.iter().enumerate() {
            if !segment.start.is_finite() || (k > 0 && segment.start <= schedule[k - 1].start) {
                return Err(bad("OU schedule starts must be finite and strictly increasing"));
            }
            starts.push(segment.start);
            models.push(template.replaced(&segment.replacements)?.0);
        }
        Ok(Self {
            n: template.equations.len(),
            starts,
            models,
            cache: BTreeMap::new(),
            check,
            residual: 0.0,
        })
    }

    fn segment_of(&self, time: f64) -> usize {
        self.starts.partition_point(|s| *s <= time).saturating_sub(1)
    }

    fn advance(
        &mut self,
        from: f64,
        to: f64,
        mean: &mut Vec<f64>,
        covariance: &mut Vec<f64>,
    ) -> Result<(), OuTrajectoryError> {
        let n = self.n;
        let mut time = from;
        while time < to {
            let k = self.segment_of(time);
            let end = self.starts.get(k + 1).map_or(to, |next| next.min(to));
            let h = end - time;
            let key = (k, h.to_bits());
            if !self.cache.contains_key(&key) {
                let mut residual = 0.0;
                let computed = transition(
                    &self.models[k],
                    h,
                    if self.check { Some(&mut residual) } else { None },
                )?;
                self.residual = self.residual.max(residual);
                self.cache.insert(key, computed);
            }
            let step = &self.cache[&key];
            let moved = matmul(&step.phi, mean, n, n, 1);
            for i in 0..n {
                mean[i] = moved[i] + step.shift[i];
            }
            let left = matmul(&step.phi, covariance, n, n, n);
            let mut next = matmul(&left, &transpose(&step.phi, n, n), n, n, n);
            for (value, q) in next.iter_mut().zip(&step.q) {
                *value += q;
            }
            symmetrize_average(&mut next, n);
            *covariance = next;
            time = end;
        }
        Ok(())
    }
}

fn initial_law(
    model: &OuMechanisms,
    initial: &OuInitialLaw,
    schedule: &[OuScheduleSegment],
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>), OuTrajectoryError> {
    match initial {
        OuInitialLaw::Stationary => {
            let law = model.stationary(&schedule[0].replacements, ctx).map_err(|error| {
                match OuTrajectoryError::from(error) {
                    OuTrajectoryError::Resource(reason) => OuTrajectoryError::Resource(reason),
                    other => OuTrajectoryError::Numerical(format!(
                        "the stationary initial law is not established: {other}"
                    )),
                }
            })?;
            Ok((law.mean, law.covariance))
        }
        OuInitialLaw::Gaussian { mean, covariance } => Ok((mean.clone(), covariance.clone())),
    }
}

fn validate_initial(initial: &OuInitialLaw, n: usize) -> Result<(), OuTrajectoryError> {
    if let OuInitialLaw::Gaussian { mean, covariance } = initial {
        let symmetric = (0..n).all(|i| {
            (0..n).all(|j| {
                (covariance.get(i * n + j).copied().unwrap_or(f64::NAN)
                    - covariance.get(j * n + i).copied().unwrap_or(f64::NAN))
                .abs()
                    <= 1e-9 * (1.0 + covariance.get(i * n + j).copied().unwrap_or(0.0).abs())
            })
        });
        if mean.len() != n
            || covariance.len() != n * n
            || mean.iter().chain(covariance).any(|v| !v.is_finite())
            || !symmetric
            || (0..n).any(|i| covariance[i * n + i] < 0.0)
        {
            return Err(bad("OU initial Gaussian law shape/finiteness"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Kalman likelihood.
// ---------------------------------------------------------------------------

/// Filtered posterior of the hidden state at a trajectory's last observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuFilteredState {
    /// Trajectory identity.
    pub trajectory: String,
    /// Last observation time.
    pub time: f64,
    /// Filtered mean of every state component, observed or not.
    pub mean: Vec<f64>,
    /// Filtered row-major covariance.
    pub covariance: Vec<f64>,
    /// Filtered standard deviation per state component; large values mark poorly
    /// determined (for example unobserved) components.
    pub standard_deviation: Vec<f64>,
}

#[derive(Default)]
struct Probe {
    expm_residual: f64,
    innovation_condition: f64,
    filtered: Vec<OuFilteredState>,
}

struct Problem<'a> {
    template: &'a OuMechanisms,
    observation: &'a OuObservationModel,
    coordinates: &'a [OuMechanismParameter],
    trajectories: &'a [OuTrajectory],
    n: usize,
    m: usize,
}

impl Problem<'_> {
    fn trajectory_log_likelihood(
        &self,
        model: &OuMechanisms,
        trajectory: &OuTrajectory,
        ctx: &ExecutionContext,
        mut probe: Option<&mut Probe>,
    ) -> Result<f64, OuTrajectoryError> {
        let (n, m) = (self.n, self.m);
        let observation = self.observation;
        let mut propagator = Propagator::new(model, &trajectory.schedule, probe.is_some())?;
        let (mut mean, mut covariance) =
            initial_law(model, &trajectory.initial, &trajectory.schedule, ctx)?;
        let mut time = trajectory.schedule[0].start;
        let noise: Vec<f64> = observation.noise_standard_deviation.iter().map(|s| s * s).collect();
        let h_transpose = transpose(&observation.matrix, m, n);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut total = 0.0;
        for point in &trajectory.observations {
            if ctx.cancellation.is_cancelled() {
                return Err(OuTrajectoryError::Resource("cancelled".into()));
            }
            propagator.advance(time, point.time, &mut mean, &mut covariance)?;
            time = point.time;
            let hp = matmul(&observation.matrix, &covariance, m, n, n);
            let mut innovation = matmul(&hp, &h_transpose, m, n, m);
            for i in 0..m {
                innovation[i * m + i] += noise[i];
            }
            symmetrize_average(&mut innovation, m);
            let factor = SpdFactor::new(&innovation, m)
                .ok_or_else(|| numerical("the innovation covariance is not positive definite"))?;
            let predicted = matmul(&observation.matrix, &mean, m, n, 1);
            let residual: Vec<f64> =
                (0..m).map(|i| point.values[i] - predicted[i] - observation.offset_at(i)).collect();
            let solved = factor
                .solve(&residual)
                .ok_or_else(|| numerical("the innovation solve is not finite"))?;
            let quadratic: f64 = residual.iter().zip(&solved).map(|(a, b)| a * b).sum();
            total -= 0.5 * (m as f64 * log_two_pi + factor.log_det() + quadratic);
            let inverse = factor.inverse();
            let gain = matmul(&transpose(&hp, m, n), &inverse, n, m, m);
            let correction = matmul(&gain, &residual, n, m, 1);
            for i in 0..n {
                mean[i] += correction[i];
            }
            let gain_h = matmul(&gain, &observation.matrix, n, m, n);
            let mut complement = identity(n);
            for (value, g) in complement.iter_mut().zip(&gain_h) {
                *value -= g;
            }
            let left = matmul(&complement, &covariance, n, n, n);
            let mut updated = matmul(&left, &transpose(&complement, n, n), n, n, n);
            for i in 0..n {
                for j in 0..n {
                    updated[i * n + j] +=
                        (0..m).map(|l| gain[i * m + l] * noise[l] * gain[j * m + l]).sum::<f64>();
                }
            }
            symmetrize_average(&mut updated, n);
            covariance = updated;
            if let Some(probe) = probe.as_deref_mut() {
                let lower = factor.clone().into_lower();
                let diagonal: Vec<f64> = (0..m).map(|i| lower[i * m + i]).collect();
                let largest = diagonal.iter().copied().fold(0.0, f64::max);
                let smallest = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
                probe.innovation_condition =
                    probe.innovation_condition.max((largest / smallest).powi(2));
            }
        }
        if let Some(probe) = probe {
            probe.expm_residual = probe.expm_residual.max(propagator.residual);
            probe.filtered.push(OuFilteredState {
                trajectory: trajectory.id.clone(),
                time,
                standard_deviation: (0..n).map(|i| covariance[i * n + i].max(0.0).sqrt()).collect(),
                mean,
                covariance,
            });
        }
        Ok(total)
    }

    fn log_likelihood(
        &self,
        theta: &[f64],
        ctx: &ExecutionContext,
        mut probe: Option<&mut Probe>,
    ) -> Result<f64, OuTrajectoryError> {
        let model = bind(self.template, self.coordinates, theta);
        let mut total = 0.0;
        for trajectory in self.trajectories {
            total +=
                self.trajectory_log_likelihood(&model, trajectory, ctx, probe.as_deref_mut())?;
        }
        if total.is_finite() {
            Ok(total)
        } else {
            Err(numerical("non-finite OU trajectory likelihood"))
        }
    }

    /// Central-difference score of the exact likelihood: the test cross-check of the analytic score.
    #[cfg(test)]
    fn gradient(
        &self,
        theta: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, usize), OuTrajectoryError> {
        let mut score = Vec::with_capacity(theta.len());
        for i in 0..theta.len() {
            let h = GRADIENT_STEP * (1.0 + theta[i].abs());
            let mut high = theta.to_vec();
            let mut low = theta.to_vec();
            high[i] += h;
            low[i] -= h;
            let up = self.log_likelihood(&high, ctx, None)?;
            let down = self.log_likelihood(&low, ctx, None)?;
            score.push((up - down) / (2.0 * h));
        }
        Ok((score, 2 * theta.len()))
    }

    /// Negative central-difference Hessian of the likelihood: the test cross-check of the
    /// analytic information.
    #[cfg(test)]
    fn information(
        &self,
        theta: &[f64],
        center: f64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, OuTrajectoryError> {
        let p = theta.len();
        let steps: Vec<f64> = theta.iter().map(|v| HESSIAN_STEP * (1.0 + v.abs())).collect();
        let shifted = |moves: &[(usize, f64)]| -> Result<f64, OuTrajectoryError> {
            let mut point = theta.to_vec();
            for (i, sign) in moves {
                point[*i] += sign * steps[*i];
            }
            self.log_likelihood(&point, ctx, None)
        };
        let mut info = vec![0.0; p * p];
        for i in 0..p {
            let up = shifted(&[(i, 1.0)])?;
            let down = shifted(&[(i, -1.0)])?;
            info[i * p + i] = -(up - 2.0 * center + down) / (steps[i] * steps[i]);
            for j in 0..i {
                let pp = shifted(&[(i, 1.0), (j, 1.0)])?;
                let pm = shifted(&[(i, 1.0), (j, -1.0)])?;
                let mp = shifted(&[(i, -1.0), (j, 1.0)])?;
                let mm = shifted(&[(i, -1.0), (j, -1.0)])?;
                let value = -(pp - pm - mp + mm) / (4.0 * steps[i] * steps[j]);
                info[i * p + j] = value;
                info[j * p + i] = value;
            }
        }
        Ok(info)
    }
}

// ---------------------------------------------------------------------------
// Eigenvalues (shifted complex QR) for stability, aliasing and information rank.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn plus(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn minus(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    fn times(self, o: Self) -> Self {
        Self::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn sqrt(self) -> Self {
        let r = self.abs();
        let re = ((r + self.re) * 0.5).max(0.0).sqrt();
        let im = ((r - self.re) * 0.5).max(0.0).sqrt();
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
}

/// Eigenvalues `(re, im)` of a real row-major `n x n` matrix; `None` when the
/// bounded iteration does not converge.
fn eigenvalues(a: &[f64], n: usize) -> Option<Vec<(f64, f64)>> {
    if a.len() != n * n || a.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let fro = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if fro == 0.0 {
        return Some(vec![(0.0, 0.0); n]);
    }
    let tolerance = 1e-14 * fro;
    let mut h: Vec<Cx> = a.iter().map(|v| Cx::new(*v, 0.0)).collect();
    let mut out: Vec<Cx> = Vec::with_capacity(n);
    let mut active = n;
    let mut iterations = 0_usize;
    while active > 0 {
        if active == 1 {
            out.push(h[0]);
            break;
        }
        let last = (active - 1) * n;
        let off: f64 = (0..active - 1).map(|j| h[last + j].abs()).sum();
        if off <= tolerance {
            out.push(h[last + active - 1]);
            active -= 1;
            iterations = 0;
            continue;
        }
        iterations += 1;
        if iterations > 400 {
            return None;
        }
        let a11 = h[(active - 2) * n + active - 2];
        let a12 = h[(active - 2) * n + active - 1];
        let a21 = h[last + active - 2];
        let a22 = h[last + active - 1];
        let half = a11.minus(a22).scale(0.5);
        let root = half.times(half).plus(a12.times(a21)).sqrt();
        let e1 = a22.plus(half).plus(root);
        let e2 = a22.plus(half).minus(root);
        let mut shift = if e1.minus(a22).abs() <= e2.minus(a22).abs() { e1 } else { e2 };
        if iterations % 11 == 10 {
            shift = a22.plus(Cx::new(off, 0.5 * off));
        }
        for i in 0..active {
            h[i * n + i] = h[i * n + i].minus(shift);
        }
        let mut rotations = Vec::new();
        for j in 0..active - 1 {
            for i in j + 1..active {
                let x = h[j * n + j];
                let y = h[i * n + j];
                let r = (x.re * x.re + x.im * x.im + y.re * y.re + y.im * y.im).sqrt();
                if r == 0.0 {
                    continue;
                }
                let c = x.scale(1.0 / r);
                let s = y.scale(1.0 / r);
                for k in 0..active {
                    let hj = h[j * n + k];
                    let hi = h[i * n + k];
                    h[j * n + k] = c.conj().times(hj).plus(s.conj().times(hi));
                    h[i * n + k] = c.times(hi).minus(s.times(hj));
                }
                rotations.push((j, i, c, s));
            }
        }
        for (j, i, c, s) in rotations {
            for k in 0..active {
                let rj = h[k * n + j];
                let ri = h[k * n + i];
                h[k * n + j] = rj.times(c).plus(ri.times(s));
                h[k * n + i] = ri.times(c.conj()).minus(rj.times(s.conj()));
            }
        }
        for i in 0..active {
            h[i * n + i] = h[i * n + i].plus(shift);
        }
    }
    Some(out.iter().map(|z| (z.re, z.im)).collect())
}

// ---------------------------------------------------------------------------
// Identification and numerical evidence.
// ---------------------------------------------------------------------------

fn numerical_rank(mut a: Vec<f64>, rows: usize, cols: usize, tolerance: f64) -> usize {
    let mut rank = 0;
    while rank < rows.min(cols) {
        let mut best = 0.0;
        let (mut bi, mut bj) = (rank, rank);
        for i in rank..rows {
            for j in rank..cols {
                if a[i * cols + j].abs() > best {
                    best = a[i * cols + j].abs();
                    bi = i;
                    bj = j;
                }
            }
        }
        if best <= tolerance {
            break;
        }
        for j in 0..cols {
            a.swap(rank * cols + j, bi * cols + j);
        }
        for i in 0..rows {
            a.swap(i * cols + rank, i * cols + bj);
        }
        for i in rank + 1..rows {
            let factor = a[i * cols + rank] / a[rank * cols + rank];
            for j in rank..cols {
                a[i * cols + j] -= factor * a[rank * cols + j];
            }
        }
        rank += 1;
    }
    rank
}

/// Rank of the observability matrix `[H; HA; ...; HA^(n-1)]` of the pair (H, A).
///
/// A rank below `n` means a state subspace never influences any observation.
#[must_use]
pub fn ou_observability_rank(drift: &[f64], n: usize, matrix: &[f64], m: usize) -> usize {
    if drift.len() != n * n || matrix.len() != m * n || n == 0 {
        return 0;
    }
    let fro = drift.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
    let scaled: Vec<f64> = drift.iter().map(|v| v / fro).collect();
    let normalize = |block: &mut [f64]| {
        for i in 0..m {
            let norm = (0..n).map(|j| block[i * n + j].powi(2)).sum::<f64>().sqrt();
            for j in 0..n {
                block[i * n + j] = if norm > 1e-10 { block[i * n + j] / norm } else { 0.0 };
            }
        }
    };
    let mut block = matrix.to_vec();
    normalize(&mut block);
    let mut stacked = block.clone();
    for _ in 1..n {
        block = matmul(&block, &scaled, m, n, n);
        normalize(&mut block);
        stacked.extend_from_slice(&block);
    }
    numerical_rank(stacked, n * m, n, 1e-8)
}

/// Sampling-interval aliasing check of a drift matrix.
///
/// Two drifts whose eigenvalues differ by `2 pi i k / dt` produce the same
/// transition at sampling interval `dt`. The principal branch is the only one
/// consistent with the data when `max |Im lambda| dt < pi` at the smallest
/// observed interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuAliasingCheck {
    /// Smallest positive spacing between consecutive observations.
    pub minimum_interval: Option<f64>,
    /// Largest oscillation frequency `max |Im lambda|` of the drift.
    pub largest_oscillation_frequency: Option<f64>,
    /// `largest_oscillation_frequency * minimum_interval`; aliasing needs at least pi.
    pub phase_at_minimum_interval: Option<f64>,
    /// Whether another drift branch reproduces the transitions; `None` when the
    /// eigenvalue iteration did not converge.
    pub aliasing_possible: Option<bool>,
}

/// Evaluate [`OuAliasingCheck`] for a drift at the smallest observation interval.
#[must_use]
pub fn ou_aliasing_check(
    drift: &[f64],
    n: usize,
    minimum_interval: Option<f64>,
) -> OuAliasingCheck {
    let frequency = eigenvalues(drift, n)
        .map(|values| values.iter().map(|(_, im)| im.abs()).fold(0.0, f64::max));
    let phase = frequency.zip(minimum_interval).map(|(f, dt)| f * dt);
    OuAliasingCheck {
        minimum_interval,
        largest_oscillation_frequency: frequency,
        phase_at_minimum_interval: phase,
        aliasing_possible: phase.map(|v| v >= std::f64::consts::PI),
    }
}

/// Identification status of the fitted coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OuIdentificationStatus {
    /// Locally identified by the observed information, with every state subspace
    /// observable and no sampling-interval aliasing.
    Identified,
    /// The observed information is not positive definite at the reported point.
    NotIdentified,
    /// Locally identified, but only under a stated condition of the declared model.
    ModelConditional,
}

/// Identification evidence, separate from optimizer convergence and numerics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectoryIdentification {
    /// Identified, not identified or model-conditional.
    pub status: OuIdentificationStatus,
    /// Number of fitted coordinates.
    pub parameter_count: usize,
    /// Numerical rank of the correlation-scaled observed information.
    pub information_rank: usize,
    /// Smallest eigenvalue of the correlation-scaled observed information.
    pub information_min_eigenvalue: Option<f64>,
    /// State dimension.
    pub state_dimension: usize,
    /// Rank of the observability matrix of the fitted drift and the observation matrix.
    pub observability_rank: usize,
    /// Equations with no direct observation (zero column of the observation matrix).
    pub unobserved_components: Vec<String>,
    /// State dimensions that no observation ever depends on.
    pub unobservable_dimension: usize,
    /// Sampling-interval aliasing check.
    pub aliasing: OuAliasingCheck,
    /// Conditions that make an otherwise locally identified result model-conditional.
    pub conditions: Vec<String>,
    /// Declared assumptions the identification is relative to.
    pub assumptions: Vec<String>,
}

/// Optimizer termination, distinct from identification and numerical stability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectoryFitting {
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

/// Numerical-stability evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectoryNumerics {
    /// Largest real part of a fitted drift eigenvalue; `None` when not determined.
    pub spectral_abscissa: Option<f64>,
    /// Whether the fitted baseline drift is asymptotically stable.
    pub drift_stable: Option<bool>,
    /// Worst `|exp(A) exp(-A) - I|` over the transitions the likelihood used.
    pub expm_inverse_residual: f64,
    /// Whether the exponential residual is below the declared accuracy bound.
    pub expm_reliable: bool,
    /// Worst squared ratio of the largest to smallest innovation Cholesky pivot.
    pub innovation_condition: f64,
    /// Condition number of the correlation-scaled observed information.
    pub information_condition: Option<f64>,
}

/// Maximum-likelihood fit of an OU model from trajectories, with evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectoryFit {
    /// Fitted mechanisms (template with the coordinates replaced).
    pub model: OuMechanisms,
    /// Fitted coordinates.
    pub parameters: Vec<OuMechanismParameter>,
    /// Fitted values, aligned with `parameters`.
    pub values: Vec<f64>,
    /// Exact-transition Kalman log likelihood at the fit.
    pub log_likelihood: f64,
    /// Number of observation vectors used.
    pub observation_count: usize,
    /// Row-major inverse observed information (sampling covariance); `None` unless
    /// the information is positive definite.
    pub parameter_covariance: Option<Vec<f64>>,
    /// Square roots of the diagonal of `parameter_covariance`.
    pub standard_errors: Option<Vec<f64>>,
    /// Optimizer termination.
    pub fitting: OuTrajectoryFitting,
    /// Identification evidence.
    pub identification: OuTrajectoryIdentification,
    /// Numerical-stability evidence.
    pub numerics: OuTrajectoryNumerics,
    /// Filtered hidden-state posterior at each trajectory's end, observed or not.
    pub filtered_states: Vec<OuFilteredState>,
}

struct DesignFacts {
    n: usize,
    m: usize,
    minimum_interval: Option<f64>,
    transitions: usize,
    observations: usize,
}

fn validate_design(
    template: &OuMechanisms,
    observation: &OuObservationModel,
    coordinates: &[OuMechanismParameter],
    trajectories: &[OuTrajectory],
) -> Result<DesignFacts, OuTrajectoryError> {
    let n = template.validate()?;
    let m = observation.validate(n)?;
    if coordinates.is_empty() || coordinates.len() > MAX_PARAMETERS {
        return Err(bad("OU trajectory parameter count"));
    }
    parameters(template, coordinates)?;
    if trajectories.is_empty() || trajectories.len() > MAX_TRAJECTORIES {
        return Err(bad("OU trajectory count"));
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut minimum: Option<f64> = None;
    let mut transitions = 0;
    let mut observations = 0;
    for trajectory in trajectories {
        if trajectory.id.trim().is_empty() || !ids.insert(&trajectory.id) {
            return Err(bad("OU trajectory identity"));
        }
        // Constructing the propagator validates the schedule and every replacement.
        Propagator::new(template, &trajectory.schedule, false)?;
        validate_initial(&trajectory.initial, n)?;
        let first = trajectory.schedule[0].start;
        if trajectory.observations.is_empty()
            || observations + trajectory.observations.len() > MAX_TIMES
        {
            return Err(bad("OU trajectory observation count"));
        }
        let mut previous: Option<f64> = None;
        for point in &trajectory.observations {
            if !point.time.is_finite()
                || point.time < first
                || previous.is_some_and(|p| point.time <= p)
                || point.values.len() != m
                || point.values.iter().any(|v| !v.is_finite())
            {
                return Err(bad("OU observation time/values"));
            }
            if let Some(p) = previous {
                transitions += 1;
                minimum = Some(minimum.map_or(point.time - p, |q| q.min(point.time - p)));
            }
            previous = Some(point.time);
        }
        observations += trajectory.observations.len();
    }
    Ok(DesignFacts { n, m, minimum_interval: minimum, transitions, observations })
}

struct InformationSummary {
    covariance: Option<Vec<f64>>,
    rank: usize,
    min_eigenvalue: Option<f64>,
    condition: Option<f64>,
}

fn summarize_information(info: &[f64], p: usize) -> InformationSummary {
    let mut info = info.to_vec();
    symmetrize_average(&mut info, p);
    let max_diagonal = (0..p).map(|i| info[i * p + i]).fold(0.0, f64::max);
    let flat = (0..p).filter(|i| info[i * p + i] <= FLAT_DIAGONAL * max_diagonal).count();
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
    let spectrum = eigenvalues(&scaled, p);
    let (rank, min_eigenvalue, condition) = match spectrum {
        Some(values) => {
            let reals: Vec<f64> = values.iter().map(|(re, _)| *re).collect();
            let minimum = reals.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = reals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let rank = reals.iter().filter(|v| **v > INFORMATION_FLOOR).count();
            (rank, Some(minimum), (minimum > 0.0).then(|| maximum / minimum))
        }
        None => (p, None, None),
    };
    let positive = rank == p && min_eigenvalue.map_or(true, |v| v > INFORMATION_FLOOR);
    let covariance = if positive { SpdFactor::new(&info, p).map(|f| f.inverse()) } else { None };
    let rank = if positive && covariance.is_none() { rank.saturating_sub(1) } else { rank };
    InformationSummary { covariance, rank, min_eigenvalue, condition }
}

fn validate_policy(policy: &CyclicGradientPolicy) -> Result<(), OuTrajectoryError> {
    if !policy.learning_rate.is_finite()
        || policy.learning_rate <= 0.0
        || !policy.score_tolerance.is_finite()
        || policy.score_tolerance <= 0.0
        || policy.l1_penalty != 0.0
        || policy.max_iterations == 0
        || policy.max_iterations > 100_000
    {
        return Err(bad("OU trajectory fit policy"));
    }
    Ok(())
}

/// Fit the shared mechanism coordinates by exact-transition Kalman maximum likelihood.
///
/// Typed refusals: `Unidentified` when no transition is observed, or when the
/// optimizer converged and the observed information is singular at the optimum.
#[allow(
    clippy::too_many_lines,
    reason = "one fit: validation, optimization, then three evidence blocks"
)]
pub fn fit_ou_trajectories(
    template: &OuMechanisms,
    observation: &OuObservationModel,
    coordinates: &[OuMechanismParameter],
    trajectories: &[OuTrajectory],
    policy: CyclicGradientPolicy,
    ctx: &ExecutionContext,
) -> Result<OuTrajectoryFit, OuTrajectoryError> {
    let facts = validate_design(template, observation, coordinates, trajectories)?;
    validate_policy(&policy)?;
    let (n, m, p) = (facts.n, facts.m, coordinates.len());
    let _lease = crate::util::admit_array::<f64>(
        ctx,
        1,
        64 * (2 * n + 1) * (2 * n + 1) + 16 * m * n + 64 * p * p,
    )
    .map_err(|error| OuTrajectoryError::Resource(error.to_string()))?;
    if facts.transitions == 0 {
        return Err(OuTrajectoryError::Unidentified {
            failure: OuIdentificationFailure::NoTransitionInformation,
            detail: "no trajectory has two observation times, so no state transition is observed; \
                     snapshot data identify only a stationary law"
                .into(),
        });
    }
    let problem = Problem { template, observation, coordinates, trajectories, n, m };
    let mut values = parameters(template, coordinates)?;
    let mut model = template.clone();
    let mut likelihood = problem.log_likelihood(&values, ctx, None)?;
    let mut evaluations = 1;
    let (_, mut score, _) = problem.score_and_information(&values, ctx)?;
    evaluations += 1;
    let mut iterations = 0;
    let mut exhausted = false;
    let mut curvature = InverseCurvature::default();
    while iterations < policy.max_iterations {
        if ctx.cancellation.is_cancelled() {
            return Err(OuTrajectoryError::Resource("cancelled".into()));
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
            let proposed: Vec<f64> =
                values.iter().zip(&direction).map(|(v, d)| v + step * d).collect();
            evaluations += 1;
            match problem.log_likelihood(&proposed, ctx, None) {
                Ok(next) if next >= likelihood + 1e-4 * step * derivative => {
                    match problem.score_and_information(&proposed, ctx) {
                        Ok((_, next_score, _)) => {
                            evaluations += 1;
                            accepted = Some((proposed, next, next_score));
                            break;
                        }
                        Err(OuTrajectoryError::Resource(reason)) => {
                            return Err(OuTrajectoryError::Resource(reason));
                        }
                        Err(_) => {}
                    }
                }
                Err(OuTrajectoryError::Resource(reason)) => {
                    return Err(OuTrajectoryError::Resource(reason));
                }
                _ => {}
            }
            step *= 0.5;
        }
        if let Some((proposed, next, next_score)) = accepted {
            curvature.update(&values, &proposed, &score, &next_score);
            model = bind(template, coordinates, &proposed);
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
    let converged = score_norm <= policy.score_tolerance;

    // Evidence is evaluated once at the reported point.
    let mut probe = Probe::default();
    let final_likelihood = problem.log_likelihood(&values, ctx, Some(&mut probe))?;
    let (_, _, info) = problem.score_and_information(&values, ctx)?;
    evaluations += 2;
    let summary = summarize_information(&info, p);
    let observability_rank = ou_observability_rank(&model.drift, n, &observation.matrix, m);
    let unobserved_components: Vec<String> = (0..n)
        .filter(|i| (0..m).all(|r| observation.matrix[r * n + i] == 0.0))
        .map(|i| model.equations[i].clone())
        .collect();
    let aliasing = ou_aliasing_check(&model.drift, n, facts.minimum_interval);
    let information_positive = summary.covariance.is_some();
    if converged && !information_positive {
        return Err(OuTrajectoryError::Unidentified {
            failure: OuIdentificationFailure::SingularInformation,
            detail: format!(
                "the observed information of the {p} fitted coordinates has numerical rank {} at the \
                 converged optimum (smallest scaled eigenvalue {:?}); observability rank is {observability_rank} \
                 of {n} and the equations {unobserved_components:?} have no direct observation",
                summary.rank, summary.min_eigenvalue
            ),
        });
    }
    let mut conditions = Vec::new();
    if observability_rank < n {
        conditions.push("unobservable_state_subspace".to_owned());
    }
    match aliasing.aliasing_possible {
        Some(true) => conditions.push("sampling_interval_aliasing".to_owned()),
        None => conditions.push("oscillation_frequency_not_determined".to_owned()),
        Some(false) => {}
    }
    if summary.condition.is_some_and(|c| c > ILL_CONDITIONED) {
        conditions.push("ill_conditioned_information".to_owned());
    }
    let status = if !information_positive {
        OuIdentificationStatus::NotIdentified
    } else if conditions.is_empty() {
        OuIdentificationStatus::Identified
    } else {
        OuIdentificationStatus::ModelConditional
    };
    let spectrum = eigenvalues(&model.drift, n);
    let abscissa =
        spectrum.map(|values| values.iter().map(|(re, _)| *re).fold(f64::NEG_INFINITY, f64::max));
    let standard_errors = summary
        .covariance
        .as_ref()
        .map(|c| (0..p).map(|i| c[i * p + i].max(0.0).sqrt()).collect::<Vec<f64>>());
    Ok(OuTrajectoryFit {
        model,
        parameters: coordinates.to_vec(),
        values,
        log_likelihood: final_likelihood,
        observation_count: facts.observations,
        parameter_covariance: summary.covariance,
        standard_errors,
        fitting: OuTrajectoryFitting {
            converged,
            score_norm,
            iterations,
            evaluations,
            line_search_exhausted: exhausted,
            optimizer: "limited_memory_bfgs_armijo_with_analytic_score".into(),
            derivative_method: "analytic_frechet_sensitivities_through_the_kalman_filter; information is the Gauss-Newton form (innovation-score outer product with the zero-expectation terms removed)".into(),
        },
        identification: OuTrajectoryIdentification {
            status,
            parameter_count: p,
            information_rank: summary.rank,
            information_min_eigenvalue: summary.min_eigenvalue,
            state_dimension: n,
            observability_rank,
            unobserved_components,
            unobservable_dimension: n - observability_rank,
            aliasing,
            conditions,
            assumptions: vec![
                "linear Gaussian observation model with known matrix, offset and noise".into(),
                "known piecewise-constant intervention schedule per trajectory".into(),
                "declared initial hidden-state law per trajectory".into(),
                "baseline mechanism structure and diffusion parameterization as specified".into(),
                "local identification from the observed information at the reported optimum".into(),
            ],
        },
        numerics: OuTrajectoryNumerics {
            spectral_abscissa: abscissa,
            drift_stable: abscissa.map(|v| v < 0.0),
            expm_inverse_residual: probe.expm_residual,
            expm_reliable: probe.expm_residual < 1e-6,
            innovation_condition: probe.innovation_condition,
            information_condition: summary.condition,
        },
        filtered_states: probe.filtered,
    })
}

// ---------------------------------------------------------------------------
// Prediction with propagated sampling uncertainty.
// ---------------------------------------------------------------------------

/// Predicted hidden-state and observation laws of a held-out intervention trajectory.
///
/// `state_covariance` and `observation_variance` are process (aleatory) variability
/// at the fitted parameters; the `*_parameter_*` fields are sampling uncertainty of
/// the fitted parameters propagated by the delta method. They are never summed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectoryPrediction {
    /// Design identity.
    pub id: String,
    /// Prediction times.
    pub times: Vec<f64>,
    /// Hidden-state mean per time.
    pub state_mean: Vec<Vec<f64>>,
    /// Hidden-state process covariance per time, row-major.
    pub state_covariance: Vec<Vec<f64>>,
    /// Sampling covariance of the state mean per time (`n x n`); `None` without a
    /// positive-definite parameter covariance.
    pub state_mean_parameter_covariance: Option<Vec<Vec<f64>>>,
    /// Sampling variance of each state-covariance entry per time.
    pub state_covariance_parameter_variance: Option<Vec<Vec<f64>>>,
    /// Observation mean `H mean + d` per time.
    pub observation_mean: Vec<Vec<f64>>,
    /// Observation variance `diag(H P H^T) + noise^2` per time.
    pub observation_variance: Vec<Vec<f64>>,
    /// Sampling variance of the observation mean per time.
    pub observation_mean_parameter_variance: Option<Vec<Vec<f64>>>,
}

/// Stationary law of a sustained intervention world with propagated sampling uncertainty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuStationaryPrediction {
    /// Design identity.
    pub id: String,
    /// Why no stationary law exists or could be propagated; `None` when established.
    pub failure: Option<String>,
    /// Stationary mean.
    pub mean: Vec<f64>,
    /// Stationary row-major covariance (process variability).
    pub covariance: Vec<f64>,
    /// Sampling covariance of the stationary mean (`n x n`).
    pub mean_parameter_covariance: Option<Vec<f64>>,
    /// Sampling variance of each stationary-covariance entry.
    pub covariance_parameter_variance: Option<Vec<f64>>,
}

/// Predictions for a fit, retained together.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuTrajectoryReport {
    /// The fit and its three evidence blocks.
    pub fit: OuTrajectoryFit,
    /// Held-out intervention trajectories.
    pub predictions: Vec<OuTrajectoryPrediction>,
    /// Sustained-intervention stationary worlds.
    pub stationary: Vec<OuStationaryPrediction>,
}

fn design_output(
    model: &OuMechanisms,
    design: &OuPredictionDesign,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, OuTrajectoryError> {
    let mut propagator = Propagator::new(model, &design.schedule, false)?;
    let (mut mean, mut covariance) = initial_law(model, &design.initial, &design.schedule, ctx)?;
    let mut time = design.schedule[0].start;
    let mut means = Vec::new();
    let mut covariances = Vec::new();
    for &t in &design.times {
        if ctx.cancellation.is_cancelled() {
            return Err(OuTrajectoryError::Resource("cancelled".into()));
        }
        propagator.advance(time, t, &mut mean, &mut covariance)?;
        time = t;
        means.extend_from_slice(&mean);
        covariances.extend_from_slice(&covariance);
    }
    means.extend(covariances);
    Ok(means)
}

/// Central-difference Jacobian columns (one per fitted coordinate) of `f` in the parameters.
fn jacobian_columns(
    fit: &OuTrajectoryFit,
    f: &dyn Fn(&OuMechanisms) -> Result<Vec<f64>, OuTrajectoryError>,
) -> Result<Vec<Vec<f64>>, OuTrajectoryError> {
    let mut columns = Vec::with_capacity(fit.values.len());
    for j in 0..fit.values.len() {
        let h = PREDICTION_STEP * (1.0 + fit.values[j].abs());
        let mut high = fit.values.clone();
        let mut low = fit.values.clone();
        high[j] += h;
        low[j] -= h;
        let up = f(&bind(&fit.model, &fit.parameters, &high))?;
        let down = f(&bind(&fit.model, &fit.parameters, &low))?;
        columns.push(up.iter().zip(&down).map(|(a, b)| (a - b) / (2.0 * h)).collect());
    }
    Ok(columns)
}

/// `J Sigma J^T` entry `(k, l)` from Jacobian columns and the parameter covariance.
fn propagated(columns: &[Vec<f64>], sigma: &[f64], k: usize, l: usize) -> f64 {
    let p = columns.len();
    (0..p)
        .map(|a| (0..p).map(|b| columns[a][k] * sigma[a * p + b] * columns[b][l]).sum::<f64>())
        .sum()
}

fn validate_prediction_design(
    model: &OuMechanisms,
    design: &OuPredictionDesign,
    n: usize,
) -> Result<(), OuTrajectoryError> {
    if design.id.trim().is_empty() || design.times.is_empty() || design.times.len() > MAX_TIMES {
        return Err(bad("OU prediction design identity/times"));
    }
    Propagator::new(model, &design.schedule, false)?;
    validate_initial(&design.initial, n)?;
    let first = design.schedule[0].start;
    let mut previous: Option<f64> = None;
    for &t in &design.times {
        if !t.is_finite() || t < first || previous.is_some_and(|p| t <= p) {
            return Err(bad(
                "OU prediction times must be finite, ordered and after the first schedule time",
            ));
        }
        previous = Some(t);
    }
    Ok(())
}

/// Predict held-out intervention trajectories with delta-method sampling uncertainty.
#[allow(
    clippy::too_many_lines,
    reason = "one prediction: center, Jacobian, then the observation projection"
)]
pub fn predict_ou_trajectories(
    fit: &OuTrajectoryFit,
    observation: &OuObservationModel,
    designs: &[OuPredictionDesign],
    ctx: &ExecutionContext,
) -> Result<Vec<OuTrajectoryPrediction>, OuTrajectoryError> {
    let n = fit.model.validate()?;
    let m = observation.validate(n)?;
    let mut ids = std::collections::BTreeSet::new();
    let mut output = Vec::with_capacity(designs.len());
    for design in designs {
        if !ids.insert(&design.id) {
            return Err(bad("duplicate OU prediction design identity"));
        }
        validate_prediction_design(&fit.model, design, n)?;
        let steps = design.times.len();
        let center = design_output(&fit.model, design, ctx)?;
        let columns = match &fit.parameter_covariance {
            Some(_) => Some(jacobian_columns(fit, &|model| design_output(model, design, ctx))?),
            None => None,
        };
        let sigma = fit.parameter_covariance.as_deref();
        let state_mean: Vec<Vec<f64>> =
            (0..steps).map(|t| center[t * n..(t + 1) * n].to_vec()).collect();
        let cov_base = steps * n;
        let state_covariance: Vec<Vec<f64>> = (0..steps)
            .map(|t| center[cov_base + t * n * n..cov_base + (t + 1) * n * n].to_vec())
            .collect();
        let mean_parameter: Option<Vec<Vec<f64>>> = columns.as_ref().zip(sigma).map(|(c, s)| {
            (0..steps)
                .map(|t| {
                    (0..n * n).map(|e| propagated(c, s, t * n + e / n, t * n + e % n)).collect()
                })
                .collect()
        });
        let covariance_parameter: Option<Vec<Vec<f64>>> =
            columns.as_ref().zip(sigma).map(|(c, s)| {
                (0..steps)
                    .map(|t| {
                        (0..n * n)
                            .map(|e| {
                                let k = cov_base + t * n * n + e;
                                propagated(c, s, k, k).max(0.0)
                            })
                            .collect()
                    })
                    .collect()
            });
        let h_transpose = transpose(&observation.matrix, m, n);
        let observation_mean: Vec<Vec<f64>> = state_mean
            .iter()
            .map(|mean| {
                let projected = matmul(&observation.matrix, mean, m, n, 1);
                (0..m).map(|i| projected[i] + observation.offset_at(i)).collect()
            })
            .collect();
        let observation_variance: Vec<Vec<f64>> = state_covariance
            .iter()
            .map(|cov| {
                let hp = matmul(&observation.matrix, cov, m, n, n);
                let hph = matmul(&hp, &h_transpose, m, n, m);
                (0..m)
                    .map(|i| hph[i * m + i] + observation.noise_standard_deviation[i].powi(2))
                    .collect()
            })
            .collect();
        let observation_parameter: Option<Vec<Vec<f64>>> = mean_parameter.as_ref().map(|blocks| {
            blocks
                .iter()
                .map(|block| {
                    let hc = matmul(&observation.matrix, block, m, n, n);
                    let hch = matmul(&hc, &h_transpose, m, n, m);
                    (0..m).map(|i| hch[i * m + i].max(0.0)).collect()
                })
                .collect()
        });
        output.push(OuTrajectoryPrediction {
            id: design.id.clone(),
            times: design.times.clone(),
            state_mean,
            state_covariance,
            state_mean_parameter_covariance: mean_parameter,
            state_covariance_parameter_variance: covariance_parameter,
            observation_mean,
            observation_variance,
            observation_mean_parameter_variance: observation_parameter,
        });
    }
    Ok(output)
}

fn stationary_output(
    model: &OuMechanisms,
    replacements: &[OuEquationReplacement],
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, OuTrajectoryError> {
    let law = model.stationary(replacements, ctx)?;
    let mut out = law.mean;
    out.extend(law.covariance);
    Ok(out)
}

/// Predict sustained-intervention stationary laws with delta-method sampling uncertainty.
/// A world whose stationary law is not established keeps its slot with the failure.
pub fn predict_ou_stationary(
    fit: &OuTrajectoryFit,
    designs: &[OuStationaryDesign],
    ctx: &ExecutionContext,
) -> Result<Vec<OuStationaryPrediction>, OuTrajectoryError> {
    let n = fit.model.validate()?;
    let mut ids = std::collections::BTreeSet::new();
    let mut output = Vec::with_capacity(designs.len());
    for design in designs {
        if design.id.trim().is_empty() || !ids.insert(&design.id) {
            return Err(bad("OU stationary design identity"));
        }
        fit.model.replaced(&design.replacements)?;
        let failed = |reason: String| OuStationaryPrediction {
            id: design.id.clone(),
            failure: Some(reason),
            mean: Vec::new(),
            covariance: Vec::new(),
            mean_parameter_covariance: None,
            covariance_parameter_variance: None,
        };
        let center = match stationary_output(&fit.model, &design.replacements, ctx) {
            Ok(center) => center,
            Err(OuTrajectoryError::Resource(reason)) => {
                return Err(OuTrajectoryError::Resource(reason))
            }
            Err(error) => {
                output.push(failed(error.to_string()));
                continue;
            }
        };
        let propagate = fit.parameter_covariance.as_deref().map(|sigma| {
            jacobian_columns(fit, &|model| stationary_output(model, &design.replacements, ctx))
                .map(|columns| (columns, sigma))
        });
        let (mean_parameter, covariance_parameter) = match propagate {
            None => (None, None),
            Some(Err(OuTrajectoryError::Resource(reason))) => {
                return Err(OuTrajectoryError::Resource(reason));
            }
            Some(Err(error)) => {
                let mut slot = failed(format!(
                    "the stationary law exists at the fit but its sampling uncertainty could not be propagated: {error}"
                ));
                slot.mean = center[..n].to_vec();
                slot.covariance = center[n..].to_vec();
                output.push(slot);
                continue;
            }
            Some(Ok((columns, sigma))) => (
                Some((0..n * n).map(|e| propagated(&columns, sigma, e / n, e % n)).collect()),
                Some(
                    (0..n * n)
                        .map(|e| propagated(&columns, sigma, n + e, n + e).max(0.0))
                        .collect(),
                ),
            ),
        };
        output.push(OuStationaryPrediction {
            id: design.id.clone(),
            failure: None,
            mean: center[..n].to_vec(),
            covariance: center[n..].to_vec(),
            mean_parameter_covariance: mean_parameter,
            covariance_parameter_variance: covariance_parameter,
        });
    }
    Ok(output)
}

/// Predictions of held-out trajectories and stationary worlds for an existing fit.
pub fn ou_trajectory_report(
    fit: OuTrajectoryFit,
    observation: &OuObservationModel,
    designs: &[OuPredictionDesign],
    stationary: &[OuStationaryDesign],
    ctx: &ExecutionContext,
) -> Result<OuTrajectoryReport, OuTrajectoryError> {
    let predictions = predict_ou_trajectories(&fit, observation, designs, ctx)?;
    let stationary = predict_ou_stationary(&fit, stationary, ctx)?;
    Ok(OuTrajectoryReport { fit, predictions, stationary })
}

mod sensitivity;

#[cfg(test)]
mod tests;
