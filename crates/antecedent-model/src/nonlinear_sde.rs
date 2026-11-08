// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounded nonlinear stochastic differential equations with diagonal noise.
//!
//! The model is `dX = f(X, u(t), theta) dt + g(X, u(t), theta) dW` with `g` diagonal and
//! both `f` and `g` written in the portable [`MechanismExpression`] language. External
//! inputs `u(t)` are declared piecewise-constant or linearly interpolated schedules; a
//! diffusion may be wrapped in a square root so that sqrt-type (demographic) noise needs
//! no new expression operator. State coordinates may carry reflecting or clamping bounds,
//! which is how continuous count-like states stay nonnegative.
//!
//! # Validation
//!
//! A system is admitted only when interval derivative arithmetic over the declared state
//! box and input envelopes certifies finite Lipschitz constants of the drift and of the
//! diffusion. A nonlinearity that multiplies unbounded states is refused with a typed
//! reason rather than simulated. A square-root diffusion whose argument reaches zero is
//! admitted but certified only as Holder-1/2: the certificate records that fact and the
//! strong-convergence guarantees of the schemes are then withheld.
//!
//! # Simulation
//!
//! Euler-Maruyama and diagonal Milstein run on a declared fixed step with a counter-based
//! Gaussian generator, so a (seed, path, step, component) quadruple always yields the same
//! increment: runs replay bit for bit, paired worlds share every random number, and the
//! half-step refinement used to estimate numerical error re-uses the same Brownian path
//! (coarse increments are sums of fine ones). Monte Carlo sampling error, the observed
//! discretization effect and (in the estimation layer) parameter uncertainty are reported
//! as separate quantities and are never summed.
use crate::mechanism_expr::{MechanismExpression, MechanismTape};
use crate::nonlinear_equilibrium::EquilibriumError;
use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

mod interval;
mod simulation;
mod world;
use interval::{Iv, enclose, imax_abs};
#[cfg(test)]
use simulation::{admit, mean_and_se, numerical_error, run_path, sum_polled};
pub use simulation::{sample_sde_paths, simulate_sde};
use world::action_cell;
pub use world::{BASELINE_WORLD, SdeAction, SdeContrast, SdeWorld, WorldPlan};

/// Largest state dimension.
pub const MAX_SDE_STATE: usize = 16;
/// Largest number of external inputs.
pub const MAX_SDE_INPUTS: usize = 8;
/// Largest number of named parameters.
pub const MAX_SDE_PARAMETERS: usize = 64;
/// Largest number of knots of one input or action schedule.
pub const MAX_SDE_KNOTS: usize = 4096;
/// Largest number of Monte Carlo paths.
pub const MAX_SDE_PATHS: usize = 1 << 20;
/// Largest number of declared steps.
pub const MAX_SDE_STEPS: usize = 1 << 20;
/// Largest number of retained per-path state cells, summed over worlds and both step levels.
pub const MAX_SDE_RETAINED_CELLS: usize = 1 << 25;
/// Largest total number of scheme steps (paths x worlds x step levels).
pub const MAX_SDE_STEP_EVALUATIONS: u64 = 150_000_000;
const MAX_PATH_SAMPLES: usize = 1 << 16;
/// Iterations of a long loop between cancellation polls.
const CANCEL_STRIDE: usize = 4096;
const MAX_WORLDS: usize = 64;
const ROUNDING_MARGIN: f64 = 1e-9;

/// Typed failure of validation, simulation or estimation.
#[derive(Debug, thiserror::Error)]
pub enum SdeError {
    /// Malformed shape, non-finite number, unordered times or invalid policy.
    #[error("invalid SDE input: {0}")]
    Invalid(String),
    /// A supported scientific refusal, with a stable machine-readable detail code.
    #[error("unsupported SDE ({code}): {detail}")]
    Unsupported {
        /// Stable detail code.
        code: &'static str,
        /// Evidence behind the refusal.
        detail: String,
    },
    /// A numerical condition (finiteness, positive definiteness) failed.
    #[error("SDE numerical failure: {0}")]
    Numerical(String),
    /// Resource admission or cancellation failed.
    #[error("SDE resource refusal: {0}")]
    Resource(String),
}

impl From<EquilibriumError> for SdeError {
    fn from(error: EquilibriumError) -> Self {
        match error {
            EquilibriumError::Resource(reason) => Self::Resource(reason),
            EquilibriumError::Invalid(reason) => Self::Invalid(reason),
            other => Self::Numerical(other.to_string()),
        }
    }
}

fn bad(reason: &str) -> SdeError {
    SdeError::Invalid(reason.into())
}
fn unsupported(code: &'static str, detail: impl Into<String>) -> SdeError {
    SdeError::Unsupported { code, detail: detail.into() }
}
fn check_cancel(ctx: &ExecutionContext) -> Result<(), SdeError> {
    if ctx.cancellation.is_cancelled() {
        Err(SdeError::Resource("cancelled".into()))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Declared model.
// ---------------------------------------------------------------------------

/// A named scalar parameter shared by the drift and diffusion expressions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeParameter {
    /// Stable parameter name.
    pub name: String,
    /// Value used unless an estimate or draw overrides it.
    pub value: f64,
}

/// How an input schedule is read between knots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdeInterpolation {
    /// The value at the latest knot not after `t` holds until the next knot.
    #[default]
    PiecewiseConstant,
    /// Linear between knots; constant outside the first and last knot.
    Linear,
}

/// A time-dependent external input `u(t)`, a declared schedule and not a state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeInput {
    /// Stable input name.
    pub name: String,
    /// Interpolation between knots.
    #[serde(default)]
    pub interpolation: SdeInterpolation,
    /// Strictly increasing knot times.
    pub times: Vec<f64>,
    /// Value at each knot.
    pub values: Vec<f64>,
    /// Closed envelope `[lower, upper]` the certificate assumes for any schedule of this
    /// input, including world replacements; defaults to the range of `values`.
    #[serde(default)]
    pub envelope: Option<[f64; 2]>,
}

impl SdeInput {
    fn validate(&self) -> Result<(), SdeError> {
        let n = self.times.len();
        if self.name.trim().is_empty()
            || n == 0
            || n > MAX_SDE_KNOTS
            || self.values.len() != n
            || self.times.iter().chain(&self.values).any(|v| !v.is_finite())
            || self.times.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(bad("SDE input schedule"));
        }
        if let Some([lo, hi]) = self.envelope {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(bad("SDE input envelope"));
            }
            if self.values.iter().any(|&value| value < lo || value > hi) {
                return Err(unsupported(
                    antecedent_core::reason_details::sde::INPUT_OUTSIDE_CERTIFIED_ENVELOPE,
                    format!(
                        "input '{}' has a schedule value outside its declared envelope [{lo}, {hi}]",
                        self.name
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The input value at time `t` (constant before the first and after the last knot).
    #[must_use]
    pub fn value_at(&self, t: f64) -> f64 {
        let n = self.times.len();
        if t <= self.times[0] {
            return self.values[0];
        }
        if t >= self.times[n - 1] {
            return self.values[n - 1];
        }
        let index = self.times.partition_point(|&x| x <= t) - 1;
        match self.interpolation {
            SdeInterpolation::PiecewiseConstant => self.values[index],
            SdeInterpolation::Linear => {
                let (t0, t1) = (self.times[index], self.times[index + 1]);
                let w = (t - t0) / (t1 - t0);
                self.values[index] * (1.0 - w) + self.values[index + 1] * w
            }
        }
    }

    /// The closed range of values the certificate assumes.
    #[must_use]
    pub fn range(&self) -> (f64, f64) {
        if let Some([lo, hi]) = self.envelope {
            return (lo, hi);
        }
        let lo = self.values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = self.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    }
}

/// Diagonal diffusion of one state: `g = expression`, or `sqrt(max(expression, 0))`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeDiffusion {
    /// Mechanism expression over the state, then input, coordinates.
    pub expression: MechanismExpression,
    /// Take the (nonnegative part's) square root, for sqrt-type state-dependent noise.
    #[serde(default)]
    pub square_root: bool,
}

/// What happens when a step leaves a coordinate's declared interval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdeBoundaryRule {
    /// Mirror the excess back into the interval (the standard reflected scheme).
    #[default]
    Reflect,
    /// Project onto the nearest bound.
    Clamp,
}

/// Optional interval for one state coordinate; both ends absent means unbounded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeBoundary {
    /// Lower bound, e.g. zero for a count-like coordinate.
    #[serde(default)]
    pub lower: Option<f64>,
    /// Upper bound, e.g. the total population.
    #[serde(default)]
    pub upper: Option<f64>,
    /// Reflection or clamping.
    #[serde(default)]
    pub rule: SdeBoundaryRule,
}

impl SdeBoundary {
    fn apply(&self, x: f64) -> f64 {
        match (self.lower, self.upper) {
            (None, None) => x,
            (lower, upper) => match self.rule {
                SdeBoundaryRule::Clamp => {
                    let mut y = x;
                    if let Some(l) = lower {
                        y = y.max(l);
                    }
                    if let Some(u) = upper {
                        y = y.min(u);
                    }
                    y
                }
                SdeBoundaryRule::Reflect => match (lower, upper) {
                    (Some(l), None) => l + (x - l).abs(),
                    (None, Some(u)) => u - (u - x).abs(),
                    (Some(l), Some(u)) => {
                        let w = u - l;
                        if w <= 0.0 {
                            l
                        } else {
                            let mut r = (x - l).rem_euclid(2.0 * w);
                            if r > w {
                                r = 2.0 * w - r;
                            }
                            l + r
                        }
                    }
                    (None, None) => x,
                },
            },
        }
    }
    fn contains(&self, x: f64) -> bool {
        self.lower.is_none_or(|l| x >= l) && self.upper.is_none_or(|u| x <= u)
    }
}

/// The declared nonlinear SDE.
///
/// Expression coordinates `0..d` are the states, `d..d+k` the external inputs, and
/// expression parameters index [`Self::parameters`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeSystem {
    /// Stable state names, in coordinate order.
    pub variables: Vec<String>,
    /// External inputs (time-dependent forcing).
    #[serde(default)]
    pub inputs: Vec<SdeInput>,
    /// Named parameters.
    #[serde(default)]
    pub parameters: Vec<SdeParameter>,
    /// Drift expression per state.
    pub drift: Vec<MechanismExpression>,
    /// Diagonal diffusion per state.
    pub diffusion: Vec<SdeDiffusion>,
    /// Optional interval per state; empty means every state is unbounded.
    #[serde(default)]
    pub boundaries: Vec<SdeBoundary>,
}

/// Which discretization advances the system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SdeScheme {
    /// Euler-Maruyama: strong order 1/2, weak order 1.
    EulerMaruyama,
    /// Diagonal Milstein: strong order 1 when each diffusion depends only on its own state.
    Milstein,
}

/// Certified regularity of a compiled system on its declared box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeCertificate {
    /// Bound on the drift Jacobian's induced infinity norm over the declared box.
    pub drift_lipschitz: f64,
    /// The same bound for the diffusion; `None` when a square-root diffusion reaches zero
    /// (then the diffusion is only Holder-1/2 and strong rates are not certified).
    pub diffusion_lipschitz: Option<f64>,
    /// States whose square-root diffusion argument can reach zero on the box.
    pub square_root_degenerate: Vec<usize>,
    /// Each diffusion depends on no other state, so the diagonal Milstein term is exact.
    pub diagonal_milstein_admissible: bool,
    /// Lower and upper bound of every state (absent where unbounded).
    pub state_bounds: Vec<[Option<f64>; 2]>,
    /// Statement of what the certificate does and does not establish.
    pub scope: String,
}

/// A validated, compiled system.
#[derive(Clone, Debug)]
pub struct CompiledSde {
    system: SdeSystem,
    drift: Vec<MechanismTape>,
    diffusion: Vec<MechanismTape>,
    certificate: SdeCertificate,
}

impl SdeSystem {
    /// Default parameter values in declaration order.
    #[must_use]
    pub fn parameter_values(&self) -> Vec<f64> {
        self.parameters.iter().map(|p| p.value).collect()
    }

    /// The declared interval of state `i` (unbounded when none is declared).
    #[must_use]
    pub fn boundary(&self, i: usize) -> SdeBoundary {
        self.boundaries.get(i).copied().unwrap_or_default()
    }

    /// Validate shapes, compile every expression and certify regularity on the declared box.
    ///
    /// # Errors
    /// `Invalid` for malformed declarations; `Unsupported` (`sde.unbounded_growth`) when a
    /// Lipschitz constant over the declared box is not finite.
    #[allow(
        clippy::too_many_lines,
        reason = "the scientific calculation and its checks form one bounded procedure"
    )]
    pub fn compile(&self) -> Result<CompiledSde, SdeError> {
        let d = self.variables.len();
        let k = self.inputs.len();
        let p = self.parameters.len();
        if d == 0
            || d > MAX_SDE_STATE
            || k > MAX_SDE_INPUTS
            || p > MAX_SDE_PARAMETERS
            || self.drift.len() != d
            || self.diffusion.len() != d
            || !(self.boundaries.is_empty() || self.boundaries.len() == d)
        {
            return Err(bad("SDE shape"));
        }
        let mut names = std::collections::BTreeSet::new();
        for name in self
            .variables
            .iter()
            .chain(self.inputs.iter().map(|i| &i.name))
            .chain(self.parameters.iter().map(|q| &q.name))
        {
            if name.trim().is_empty() || name.len() > 256 || !names.insert(name.as_str()) {
                return Err(bad("SDE names must be nonempty and distinct"));
            }
        }
        if self.parameters.iter().any(|q| !q.value.is_finite()) {
            return Err(bad("SDE parameter value"));
        }
        for input in &self.inputs {
            input.validate()?;
        }
        for (i, b) in self.boundaries.iter().enumerate() {
            let ok = b.lower.is_none_or(f64::is_finite)
                && b.upper.is_none_or(f64::is_finite)
                && match (b.lower, b.upper) {
                    (Some(l), Some(u)) => l < u,
                    _ => true,
                };
            if !ok {
                return Err(SdeError::Invalid(format!("SDE boundary of state {i}")));
            }
        }
        let mut drift = Vec::with_capacity(d);
        let mut diffusion = Vec::with_capacity(d);
        for i in 0..d {
            drift.push(self.drift[i].compile(d + k, p)?);
            diffusion.push(self.diffusion[i].expression.compile(d + k, p)?);
        }
        let theta = self.parameter_values();
        let mut boxes = Vec::with_capacity(d + k);
        for i in 0..d {
            let b = self.boundary(i);
            boxes.push(Iv {
                lo: b.lower.unwrap_or(f64::NEG_INFINITY),
                hi: b.upper.unwrap_or(f64::INFINITY),
            });
        }
        for input in &self.inputs {
            let (lo, hi) = input.range();
            boxes.push(Iv { lo, hi });
        }
        let mut drift_lipschitz = 0.0f64;
        let mut diffusion_lipschitz = Some(0.0f64);
        let mut degenerate = Vec::new();
        for i in 0..d {
            let (_, g) = enclose(&self.drift[i], &boxes, &theta, 0)?;
            let row: f64 = g.iter().take(d).map(|x| imax_abs(*x)).sum();
            if !row.is_finite() {
                return Err(unsupported(
                    antecedent_core::reason_details::sde::UNBOUNDED_GROWTH,
                    format!(
                        "the drift of '{}' has an unbounded derivative over the declared state box; \
                         declare finite lower and upper bounds on every state the nonlinearity multiplies",
                        self.variables[i]
                    ),
                ));
            }
            drift_lipschitz = drift_lipschitz.max(row * (1.0 + ROUNDING_MARGIN));
            let (value, g) = enclose(&self.diffusion[i].expression, &boxes, &theta, 0)?;
            let row: f64 = g.iter().take(d).map(|x| imax_abs(*x)).sum();
            if !row.is_finite() {
                return Err(unsupported(
                    antecedent_core::reason_details::sde::UNBOUNDED_GROWTH,
                    format!(
                        "the diffusion of '{}' is unbounded over the declared state box; declare \
                         finite bounds, or use an expression with a bounded derivative",
                        self.variables[i]
                    ),
                ));
            }
            if self.diffusion[i].square_root {
                if value.lo > 0.0 {
                    let bound = row / (2.0 * value.lo.sqrt());
                    diffusion_lipschitz = diffusion_lipschitz.map(|c| c.max(bound));
                } else {
                    degenerate.push(i);
                    diffusion_lipschitz = None;
                }
            } else {
                diffusion_lipschitz =
                    diffusion_lipschitz.map(|c| c.max(row * (1.0 + ROUNDING_MARGIN)));
            }
        }
        let diagonal_milstein_admissible =
            (0..d).all(|i| diffusion[i].reads().iter().all(|&r| r == i || r >= d));
        let certificate = SdeCertificate {
            drift_lipschitz,
            diffusion_lipschitz,
            square_root_degenerate: degenerate,
            diagonal_milstein_admissible,
            state_bounds: (0..d)
                .map(|i| {
                    let b = self.boundary(i);
                    [b.lower, b.upper]
                })
                .collect(),
            scope: "Interval derivative bounds over the declared state box and input envelopes \
                    (outward margin 1e-9, not directed rounding); they establish local-in-box \
                    regularity of the specified coefficients, which the reflected scheme keeps \
                    the state inside. They are not a proof of strong convergence for \
                    square-root diffusions that reach zero."
                .into(),
        };
        Ok(CompiledSde { system: self.clone(), drift, diffusion, certificate })
    }
}

impl CompiledSde {
    /// The validated declaration.
    #[must_use]
    pub fn system(&self) -> &SdeSystem {
        &self.system
    }
    /// The regularity certificate.
    #[must_use]
    pub fn certificate(&self) -> &SdeCertificate {
        &self.certificate
    }
    /// State dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.system.variables.len()
    }
    pub(crate) fn input_values(&self, inputs: &[SdeInput], t: f64, vars: &mut [f64]) {
        let d = self.dimension();
        for (j, input) in inputs.iter().enumerate() {
            vars[d + j] = input.value_at(t);
        }
    }

    /// Drift vector at `state` and time `t` under the given input schedules.
    ///
    /// # Errors
    /// Non-finite state or evaluation.
    pub fn drift_at(
        &self,
        state: &[f64],
        inputs: &[SdeInput],
        t: f64,
        theta: &[f64],
    ) -> Result<Vec<f64>, SdeError> {
        let d = self.dimension();
        let mut vars = vec![0.0; d + inputs.len()];
        vars[..d].copy_from_slice(state);
        self.input_values(inputs, t, &mut vars);
        (0..d).map(|i| Ok(self.drift[i].evaluate(&vars, theta)?.value())).collect()
    }

    /// Drift, diagonal diffusion, drift Jacobian and its local relaxation rate at one state.
    ///
    /// # Errors
    /// Non-finite state or evaluation.
    pub fn local_coefficients(
        &self,
        state: &[f64],
        inputs: &[SdeInput],
        t: f64,
        theta: &[f64],
    ) -> Result<LocalCoefficients, SdeError> {
        let d = self.dimension();
        let mut vars = vec![0.0; d + inputs.len()];
        vars[..d].copy_from_slice(state);
        self.input_values(inputs, t, &mut vars);
        let mut out = LocalCoefficients {
            drift: vec![0.0; d],
            diffusion: vec![0.0; d],
            jacobian: vec![0.0; d * d],
            rate: 0.0,
        };
        for i in 0..d {
            let e = self.drift[i].evaluate(&vars, theta)?;
            out.drift[i] = e.value();
            let mut row = 0.0;
            for j in 0..d {
                out.jacobian[i * d + j] = e.gradient()[j];
                row += e.gradient()[j].abs();
            }
            out.rate = out.rate.max(row);
            let g = self.diffusion[i].evaluate(&vars, theta)?;
            out.diffusion[i] = if self.system.diffusion[i].square_root {
                g.value().max(0.0).sqrt()
            } else {
                g.value()
            };
        }
        Ok(out)
    }
}

/// Coefficients of the SDE at one state and time.
#[derive(Clone, Debug)]
pub struct LocalCoefficients {
    /// Drift `f`.
    pub drift: Vec<f64>,
    /// Diagonal diffusion `g`.
    pub diffusion: Vec<f64>,
    /// Row-major drift Jacobian with respect to the states.
    pub jacobian: Vec<f64>,
    /// Largest absolute row sum of the Jacobian (the local relaxation rate).
    pub rate: f64,
}

// ---------------------------------------------------------------------------
// Interventions.
// ---------------------------------------------------------------------------

impl SdeBoundary {
    /// Whether `x` lies in the declared interval.
    #[must_use]
    pub fn contains_value(&self, x: f64) -> bool {
        self.contains(x)
    }
    /// Apply the declared reflection or clamp to `x`.
    #[must_use]
    pub fn project(&self, x: f64) -> f64 {
        self.apply(x)
    }
    fn contains_all(&self, values: &[f64]) -> bool {
        values.iter().all(|&v| self.contains(v))
    }
}

// ---------------------------------------------------------------------------
// Random numbers.
// ---------------------------------------------------------------------------

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` finalizer of `x + golden`; a bijective 64-bit mixer.
#[must_use]
pub fn splitmix_mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Standard normal deviate addressed by `(seed, path, step, component)`.
///
/// `key = mix(mix(mix(mix(seed) ^ path) ^ step) ^ component)`; the two uniforms are
/// `((mix(key) >> 11) + 0.5) / 2^53` and `((mix(key ^ 1) >> 11) + 0.5) / 2^53`, combined by
/// the cosine branch of Box-Muller. The definition is intentionally reproducible in any
/// language with 64-bit integers and IEEE doubles.
#[must_use]
pub fn counter_normal(seed: u64, path: u64, step: u64, component: u64) -> f64 {
    let key =
        splitmix_mix(splitmix_mix(splitmix_mix(splitmix_mix(seed) ^ path) ^ step) ^ component);
    let scale = 9_007_199_254_740_992.0;
    let u1 = ((splitmix_mix(key) >> 11) as f64 + 0.5) / scale;
    let u2 = ((splitmix_mix(key ^ 1) >> 11) as f64 + 0.5) / scale;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Simulation.
// ---------------------------------------------------------------------------

fn default_one() -> usize {
    1
}
fn default_weak_tolerance() -> f64 {
    0.02
}
fn default_true() -> bool {
    true
}

/// Declared step, horizon, scheme, ensemble and seed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeSimulationPolicy {
    /// Time of the initial state.
    pub start: f64,
    /// Declared fixed step.
    pub step: f64,
    /// Number of declared steps; the horizon is `start + steps * step`.
    pub steps: usize,
    /// Record every this-many steps (plus the initial state).
    #[serde(default = "default_one")]
    pub output_every: usize,
    /// Discretization.
    pub scheme: SdeScheme,
    /// Monte Carlo paths per world.
    pub paths: usize,
    /// Seed of the counter-based generator.
    pub seed: u64,
    /// Relative tolerance of the step-halving convergence flag.
    #[serde(default = "default_weak_tolerance")]
    pub weak_tolerance: f64,
    /// Also run the half-step refinement with shared Brownian paths and report the
    /// numerical-approximation error.
    #[serde(default = "default_true")]
    pub check_step: bool,
    /// Retain the declared-step final state of every path (capped), keeping the joint
    /// response distribution available beyond its moments.
    #[serde(default)]
    pub retain_final_samples: bool,
}

impl SdeSimulationPolicy {
    /// Validate finiteness and bounds.
    ///
    /// # Errors
    /// `Invalid` for non-finite or out-of-range fields.
    #[allow(
        clippy::float_cmp,
        reason = "exact equality is required for discrete support or floating point progress"
    )]
    pub fn validate(&self) -> Result<(), SdeError> {
        if !self.start.is_finite()
            || !self.step.is_finite()
            || self.step <= 0.0
            || self.steps == 0
            || self.steps > MAX_SDE_STEPS
            || self.output_every == 0
            || self.output_every > self.steps
            || !self.steps.is_multiple_of(self.output_every)
            || self.paths < 2
            || self.paths > MAX_SDE_PATHS
            || !self.weak_tolerance.is_finite()
            || self.weak_tolerance < 0.0
            || (self.retain_final_samples && self.paths > MAX_PATH_SAMPLES)
        {
            return Err(bad(
                "SDE simulation policy (step > 0, steps divisible by output_every, 2..=2^20 paths)",
            ));
        }
        let dt = if self.check_step { self.step / 2.0 } else { self.step };
        let end = self.start + self.steps as f64 * self.step;
        if !end.is_finite() || self.start + dt == self.start || end - dt == end {
            return Err(bad("SDE simulation grid is not representable"));
        }
        Ok(())
    }
    /// Recorded times, including the initial time.
    #[must_use]
    pub fn output_times(&self) -> Vec<f64> {
        (0..=self.steps / self.output_every)
            .map(|j| self.start + (j * self.output_every) as f64 * self.step)
            .collect()
    }
}

/// Numerical-approximation evidence of one mean functional, from step halving.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeNumericalError {
    /// The refined step, half the declared step.
    pub refined_step: f64,
    /// `mean(declared step) - mean(refined step)` per time and coordinate, paired by path.
    pub weak_difference: Vec<Vec<f64>>,
    /// Paired Monte Carlo standard error of that difference.
    pub weak_difference_standard_error: Vec<Vec<f64>>,
    /// Root-mean-square pathwise difference between the two levels (strong-error proxy).
    pub strong_rms_difference: Vec<Vec<f64>>,
    /// `2 * mean(refined) - mean(declared)`, the weak-order-one Richardson estimate.
    pub richardson_mean: Vec<Vec<f64>>,
    /// No difference exceeds the declared relative tolerance beyond two paired standard errors.
    pub step_converged: bool,
}

/// Moments of one world's ensemble at the declared step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeWorldEnsemble {
    /// World identity.
    pub id: String,
    /// Ensemble mean per time and coordinate.
    pub mean: Vec<Vec<f64>>,
    /// Sample variance (n - 1) per time and coordinate.
    pub variance: Vec<Vec<f64>>,
    /// Row-major sample covariance across coordinates per time (the joint response law's
    /// second moment).
    pub covariance: Vec<Vec<f64>>,
    /// Monte Carlo standard error of the mean: `sqrt(variance / paths)`.
    pub monte_carlo_standard_error: Vec<Vec<f64>>,
    /// Step-halving evidence; absent when the check was not requested.
    pub numerical: Option<SdeNumericalError>,
    /// Final declared-step state of each path, when retained.
    pub final_samples: Option<Vec<Vec<f64>>>,
}

/// Paired contrast of two worlds sharing every random number.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeContrastEnsemble {
    /// Contrast identity.
    pub id: String,
    /// Treated world.
    pub treated: String,
    /// Control world.
    pub control: String,
    /// Mean of `treated - control` per time and coordinate.
    pub mean_difference: Vec<Vec<f64>>,
    /// Paired Monte Carlo standard error of that mean.
    pub monte_carlo_standard_error: Vec<Vec<f64>>,
    /// Step-halving evidence of the contrast.
    pub numerical: Option<SdeNumericalError>,
}

/// Replayable ensemble report with the three error sources kept apart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeEnsembleReport {
    /// Scheme that produced the reported moments.
    pub scheme: SdeScheme,
    /// Declared step.
    pub step: f64,
    /// Seed of the generator.
    pub seed: u64,
    /// Paths per world.
    pub paths: usize,
    /// Recorded times.
    pub times: Vec<f64>,
    /// The parameter vector the ensemble was simulated at.
    pub parameters: Vec<f64>,
    /// Worlds, baseline first.
    pub worlds: Vec<SdeWorldEnsemble>,
    /// Paired contrasts.
    pub contrasts: Vec<SdeContrastEnsemble>,
    /// Regularity certificate of the system.
    pub certificate: SdeCertificate,
    /// Statement of what each error field means.
    pub error_semantics: Vec<String>,
    #[serde(skip)]
    retention: SdeRetention,
}

impl SdeEnsembleReport {
    /// Bytes this report holds in the shared admission pool until its last clone drops; zero
    /// when it was not produced under a pool.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.bytes()
    }
}

/// Raw per-path states at the declared step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdePaths {
    /// Recorded times.
    pub times: Vec<f64>,
    /// Number of paths.
    pub paths: usize,
    /// State dimension.
    pub dimension: usize,
    /// `states[(path * times.len() + t) * dimension + i]`.
    pub states: Vec<f64>,
    #[serde(skip)]
    retention: SdeRetention,
}

impl SdePaths {
    /// Bytes these paths hold in the shared admission pool until their last clone drops; zero
    /// when they were not produced under a pool.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.bytes()
    }
}

/// Lease funding a returned report or path array. Equality ignores it: two outputs are equal
/// when their numbers are.
#[derive(Clone, Debug, Default)]
struct SdeRetention(Option<Arc<AdmissionLease>>);

impl SdeRetention {
    fn bytes(&self) -> u64 {
        self.0.as_ref().map_or(0, |lease| lease.remainder().bytes())
    }
}

impl PartialEq for SdeRetention {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests;
