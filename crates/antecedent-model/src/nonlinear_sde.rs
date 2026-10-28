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
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Copy, Debug)]
struct Iv {
    lo: f64,
    hi: f64,
}
const ZERO: Iv = Iv { lo: 0.0, hi: 0.0 };

fn fmul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}
fn sane(v: Iv) -> Iv {
    if v.lo.is_nan() || v.hi.is_nan() {
        Iv { lo: f64::NEG_INFINITY, hi: f64::INFINITY }
    } else {
        v
    }
}
fn iadd(a: Iv, b: Iv) -> Iv {
    sane(Iv { lo: a.lo + b.lo, hi: a.hi + b.hi })
}
fn imul(a: Iv, b: Iv) -> Iv {
    let p = [fmul(a.lo, b.lo), fmul(a.lo, b.hi), fmul(a.hi, b.lo), fmul(a.hi, b.hi)];
    sane(Iv {
        lo: p.iter().copied().fold(f64::INFINITY, f64::min),
        hi: p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}
fn imax_abs(a: Iv) -> f64 {
    a.lo.abs().max(a.hi.abs())
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

/// Interval enclosure of an expression and of its gradient over a coordinate box.
fn enclose(
    expr: &MechanismExpression,
    boxes: &[Iv],
    theta: &[f64],
    depth: usize,
) -> Result<(Iv, Vec<Iv>), SdeError> {
    if depth > 48 {
        return Err(bad("SDE expression depth"));
    }
    let n = boxes.len();
    match expr {
        MechanismExpression::Constant { value } => {
            Ok((Iv { lo: *value, hi: *value }, vec![ZERO; n]))
        }
        MechanismExpression::Parameter { parameter } => {
            let v = *theta.get(*parameter).ok_or_else(|| bad("SDE parameter binding"))?;
            Ok((Iv { lo: v, hi: v }, vec![ZERO; n]))
        }
        MechanismExpression::Variable { variable } => {
            let b = *boxes.get(*variable).ok_or_else(|| bad("SDE variable binding"))?;
            let mut g = vec![ZERO; n];
            g[*variable] = Iv { lo: 1.0, hi: 1.0 };
            Ok((b, g))
        }
        MechanismExpression::Add { terms } => {
            let mut acc = (ZERO, vec![ZERO; n]);
            for term in terms {
                let (v, g) = enclose(term, boxes, theta, depth + 1)?;
                acc.0 = iadd(acc.0, v);
                for (a, b) in acc.1.iter_mut().zip(g) {
                    *a = iadd(*a, b);
                }
            }
            Ok(acc)
        }
        MechanismExpression::Multiply { factors } => {
            let mut acc = (Iv { lo: 1.0, hi: 1.0 }, vec![ZERO; n]);
            for factor in factors {
                let (v, g) = enclose(factor, boxes, theta, depth + 1)?;
                let mut next = vec![ZERO; n];
                for j in 0..n {
                    next[j] = iadd(imul(acc.1[j], v), imul(acc.0, g[j]));
                }
                acc = (imul(acc.0, v), next);
            }
            Ok(acc)
        }
        MechanismExpression::Tanh { argument } => {
            let (v, g) = enclose(argument, boxes, theta, depth + 1)?;
            let t = Iv { lo: v.lo.tanh(), hi: v.hi.tanh() };
            let (tt_lo, tt_hi) = if t.lo >= 0.0 {
                (t.lo * t.lo, t.hi * t.hi)
            } else if t.hi <= 0.0 {
                (t.hi * t.hi, t.lo * t.lo)
            } else {
                (0.0, (t.lo * t.lo).max(t.hi * t.hi))
            };
            let d = Iv { lo: 1.0 - tt_hi, hi: 1.0 - tt_lo };
            Ok((t, g.into_iter().map(|x| imul(d, x)).collect()))
        }
        MechanismExpression::Sin { argument } => {
            let (_, g) = enclose(argument, boxes, theta, depth + 1)?;
            let unit = Iv { lo: -1.0, hi: 1.0 };
            Ok((unit, g.into_iter().map(|x| imul(unit, x)).collect()))
        }
        MechanismExpression::Softplus { argument } => {
            let (v, g) = enclose(argument, boxes, theta, depth + 1)?;
            let value = Iv { lo: softplus(v.lo), hi: softplus(v.hi) };
            let d = Iv { lo: logistic(v.lo), hi: logistic(v.hi) };
            Ok((value, g.into_iter().map(|x| imul(d, x)).collect()))
        }
    }
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
                    "sde.unbounded_growth",
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
                    "sde.unbounded_growth",
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

/// A sustained `do(X_i = value)` that follows a piecewise-constant schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeAction {
    /// State coordinate held.
    pub variable: usize,
    /// Strictly increasing action times, on the simulation grid.
    pub times: Vec<f64>,
    /// From `times[j]` until the next time the coordinate is held at `values[j]`.
    pub values: Vec<f64>,
}

impl SdeAction {
    fn value_at(&self, t: f64, tol: f64) -> Option<f64> {
        if t + tol < self.times[0] {
            return None;
        }
        let index = self.times.partition_point(|&x| x <= t + tol) - 1;
        Some(self.values[index])
    }
}

/// A named intervention regime: coordinate holds and replaced input schedules.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeWorld {
    /// Stable world identity.
    pub id: String,
    /// Component holds (`do` on state coordinates).
    #[serde(default)]
    pub actions: Vec<SdeAction>,
    /// Replacement schedules, by input name; must lie inside the declared envelope.
    #[serde(default)]
    pub inputs: Vec<SdeInput>,
}

/// A paired contrast of two worlds under shared random numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeContrast {
    /// Stable contrast identity.
    pub id: String,
    /// World id of the treated arm.
    pub treated: String,
    /// World id of the control arm.
    pub control: String,
}

/// Reserved identity of the world with no intervention.
pub const BASELINE_WORLD: &str = "baseline";

/// A world with its input schedules fully resolved against the system.
#[derive(Clone, Debug)]
pub struct WorldPlan {
    /// World identity.
    pub id: String,
    /// Component holds.
    pub actions: Vec<SdeAction>,
    /// One schedule per system input, in system order.
    pub inputs: Vec<SdeInput>,
}

impl SdeSystem {
    /// Resolve a world against this system and a simulation grid.
    ///
    /// # Errors
    /// Invalid actions, unknown or out-of-envelope replacement inputs, or action times off
    /// the grid (`sde.action_time_off_grid`).
    pub fn plan_world(
        &self,
        world: &SdeWorld,
        start: f64,
        step: f64,
    ) -> Result<WorldPlan, SdeError> {
        let d = self.variables.len();
        if world.id.trim().is_empty() {
            return Err(bad("SDE world identity"));
        }
        let mut held = std::collections::BTreeSet::new();
        for action in &world.actions {
            let n = action.times.len();
            if action.variable >= d
                || n == 0
                || n > MAX_SDE_KNOTS
                || action.values.len() != n
                || action.times.iter().chain(&action.values).any(|v| !v.is_finite())
                || action.times.windows(2).any(|w| w[1] <= w[0])
                || !held.insert(action.variable)
            {
                return Err(bad("SDE action"));
            }
            if !self.boundary(action.variable).contains_all(&action.values) {
                return Err(bad("SDE action value outside the declared state interval"));
            }
            for &time in &action.times {
                if time > start {
                    let cells = (time - start) / step;
                    if (cells - cells.round()).abs() > 1e-9 {
                        return Err(unsupported(
                            "sde.action_time_off_grid",
                            format!(
                                "action time {time} is not a multiple of the declared step {step} \
                                 from the start {start}; an off-grid action would silently shift"
                            ),
                        ));
                    }
                }
            }
        }
        let mut inputs = self.inputs.clone();
        for replacement in &world.inputs {
            replacement.validate()?;
            let slot = inputs
                .iter()
                .position(|i| i.name == replacement.name)
                .ok_or_else(|| bad("SDE replacement input is not declared by the system"))?;
            let (lo, hi) = self.inputs[slot].range();
            let (rlo, rhi) = replacement.range();
            if rlo < lo || rhi > hi {
                return Err(unsupported(
                    "sde.input_outside_certified_envelope",
                    format!(
                        "replacement schedule of '{}' has range [{rlo}, {rhi}] outside the \
                         certified envelope [{lo}, {hi}]; widen the declared envelope so the \
                         regularity certificate covers it",
                        replacement.name
                    ),
                ));
            }
            inputs[slot] = replacement.clone();
            inputs[slot].envelope = self.inputs[slot].envelope;
        }
        Ok(WorldPlan { id: world.id.clone(), actions: world.actions.clone(), inputs })
    }
}

impl WorldPlan {
    /// The value coordinate `variable` is held at under this world's actions at time `t`,
    /// if an action is in force (relative time tolerance `1e-9`).
    #[must_use]
    pub fn held_value(&self, variable: usize, t: f64) -> Option<f64> {
        let tol = 1e-9 * (1.0 + t.abs());
        self.actions.iter().find(|a| a.variable == variable).and_then(|a| a.value_at(t, tol))
    }
}

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

/// The SplitMix64 finalizer of `x + golden`; a bijective 64-bit mixer.
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
    pub fn validate(&self) -> Result<(), SdeError> {
        if !self.start.is_finite()
            || !self.step.is_finite()
            || self.step <= 0.0
            || self.steps == 0
            || self.steps > MAX_SDE_STEPS
            || self.output_every == 0
            || self.output_every > self.steps
            || self.steps % self.output_every != 0
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
}

struct Levels {
    coarse: Vec<f64>,
    fine: Option<Vec<f64>>,
}

/// One path at one step level, writing the recorded states into `out`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines, reason = "one scheme step loop")]
fn run_path(
    compiled: &CompiledSde,
    theta: &[f64],
    world: &WorldPlan,
    policy: &SdeSimulationPolicy,
    initial: &[f64],
    path: u64,
    fine: bool,
    out: &mut [f64],
) -> Result<(), SdeError> {
    let d = compiled.dimension();
    let k = world.inputs.len();
    let m = if fine { 2usize } else { 1usize };
    let dt = policy.step / m as f64;
    let tol = 1e-9 * policy.step;
    let record_every = policy.output_every * m;
    let total = policy.steps * m;
    let mut x = initial.to_vec();
    let mut vars = vec![0.0; d + k];
    let mut next = vec![0.0; d];
    let apply_actions = |x: &mut [f64], t: f64| {
        for action in &world.actions {
            if let Some(value) = action.value_at(t, tol) {
                x[action.variable] = value;
            }
        }
    };
    apply_actions(&mut x, policy.start);
    out[..d].copy_from_slice(&x);
    let mut slot = 1usize;
    let sqrt_half = (policy.step / 2.0).sqrt();
    for n in 0..total {
        let t = policy.start + n as f64 * dt;
        vars[..d].copy_from_slice(&x);
        for (j, input) in world.inputs.iter().enumerate() {
            vars[d + j] = input.value_at(t);
        }
        for i in 0..d {
            let held =
                world.actions.iter().any(|a| a.variable == i && a.value_at(t, tol).is_some());
            if held {
                next[i] = x[i];
                continue;
            }
            let f = compiled.drift[i].evaluate(&vars, theta)?.value();
            let ge = compiled.diffusion[i].evaluate(&vars, theta)?;
            let sqrt = compiled.system.diffusion[i].square_root;
            let h = ge.value();
            let g = if sqrt { h.max(0.0).sqrt() } else { h };
            let dw = if fine {
                sqrt_half * counter_normal(policy.seed, path, n as u64, i as u64)
            } else {
                sqrt_half
                    * (counter_normal(policy.seed, path, 2 * n as u64, i as u64)
                        + counter_normal(policy.seed, path, 2 * n as u64 + 1, i as u64))
            };
            let mut value = x[i] + f * dt + g * dw;
            if matches!(policy.scheme, SdeScheme::Milstein) {
                let dh = ge.gradient()[i];
                let correction = if sqrt {
                    if h > 0.0 {
                        0.25 * dh
                    } else {
                        0.0
                    }
                } else {
                    0.5 * g * dh
                };
                value += correction * (dw * dw - dt);
            }
            next[i] = value;
        }
        for i in 0..d {
            x[i] = compiled.system.boundary(i).apply(next[i]);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(SdeError::Numerical(format!(
                "state became non-finite at step {n} of path {path}"
            )));
        }
        apply_actions(&mut x, t + dt);
        if (n + 1) % record_every == 0 {
            out[slot * d..(slot + 1) * d].copy_from_slice(&x);
            slot += 1;
        }
    }
    Ok(())
}

fn run_world(
    compiled: &CompiledSde,
    theta: &[f64],
    world: &WorldPlan,
    policy: &SdeSimulationPolicy,
    initial: &[f64],
    ctx: &ExecutionContext,
) -> Result<Levels, SdeError> {
    let d = compiled.dimension();
    let n_out = policy.steps / policy.output_every + 1;
    let cells = policy.paths * n_out * d;
    let mut coarse = vec![0.0; cells];
    let mut fine = policy.check_step.then(|| vec![0.0; cells]);
    for p in 0..policy.paths {
        check_cancel(ctx)?;
        let range = p * n_out * d..(p + 1) * n_out * d;
        run_path(
            compiled,
            theta,
            world,
            policy,
            initial,
            p as u64,
            false,
            &mut coarse[range.clone()],
        )?;
        if let Some(fine) = fine.as_mut() {
            run_path(compiled, theta, world, policy, initial, p as u64, true, &mut fine[range])?;
        }
    }
    Ok(Levels { coarse, fine })
}

fn validate_initial(compiled: &CompiledSde, initial: &[f64]) -> Result<(), SdeError> {
    let d = compiled.dimension();
    if initial.len() != d
        || initial.iter().any(|v| !v.is_finite())
        || !(0..d).all(|i| compiled.system.boundary(i).contains(initial[i]))
    {
        return Err(bad("SDE initial state (dimension, finiteness, declared intervals)"));
    }
    Ok(())
}

fn resolve_theta(compiled: &CompiledSde, theta: Option<&[f64]>) -> Result<Vec<f64>, SdeError> {
    let theta = theta.map_or_else(|| compiled.system.parameter_values(), <[f64]>::to_vec);
    if theta.len() != compiled.system.parameters.len() || theta.iter().any(|v| !v.is_finite()) {
        return Err(bad("SDE parameter vector"));
    }
    Ok(theta)
}

fn admit(
    compiled: &CompiledSde,
    policy: &SdeSimulationPolicy,
    worlds: usize,
    ctx: &ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, SdeError> {
    let d = compiled.dimension();
    let n_out = policy.steps / policy.output_every + 1;
    let levels = if policy.check_step { 2 } else { 1 };
    let cells = worlds
        .checked_mul(policy.paths)
        .and_then(|c| c.checked_mul(n_out))
        .and_then(|c| c.checked_mul(d))
        .and_then(|c| c.checked_mul(levels))
        .ok_or_else(|| bad("SDE retained size overflow"))?;
    if cells > MAX_SDE_RETAINED_CELLS {
        return Err(SdeError::Resource(format!(
            "SDE ensemble retains {cells} state cells, above the limit {MAX_SDE_RETAINED_CELLS}"
        )));
    }
    let work = (worlds as u64)
        .saturating_mul(policy.paths as u64)
        .saturating_mul(policy.steps as u64)
        .saturating_mul(if policy.check_step { 3 } else { 1 });
    if work > MAX_SDE_STEP_EVALUATIONS {
        return Err(SdeError::Resource(format!(
            "SDE ensemble needs {work} scheme steps, above the limit {MAX_SDE_STEP_EVALUATIONS}"
        )));
    }
    ctx.acquire_admission(0, (cells as u64).saturating_mul(8).saturating_add(4096))
        .map_err(|e| SdeError::Resource(e.to_string()))
}

fn mean_and_se(values: impl Iterator<Item = f64>, n: usize) -> (f64, f64, f64) {
    // Two-pass would need storage; Welford keeps the single pass stable.
    let (mut mean, mut m2, mut count) = (0.0, 0.0, 0.0);
    for v in values {
        count += 1.0;
        let delta = v - mean;
        mean += delta / count;
        m2 += delta * (v - mean);
    }
    let variance = if n > 1 { m2 / (count - 1.0) } else { 0.0 };
    (mean, variance, (variance / count).sqrt())
}

fn numerical_error(
    coarse: impl Fn(usize, usize, usize) -> f64,
    fine: impl Fn(usize, usize, usize) -> f64,
    paths: usize,
    n_out: usize,
    d: usize,
    step: f64,
    tolerance: f64,
) -> SdeNumericalError {
    let mut weak = vec![vec![0.0; d]; n_out];
    let mut weak_se = vec![vec![0.0; d]; n_out];
    let mut strong = vec![vec![0.0; d]; n_out];
    let mut richardson = vec![vec![0.0; d]; n_out];
    let mut converged = true;
    for t in 0..n_out {
        for i in 0..d {
            let (mean_diff, var_diff, se) =
                mean_and_se((0..paths).map(|p| coarse(p, t, i) - fine(p, t, i)), paths);
            let _ = var_diff;
            let (mean_fine, var_fine, _) = mean_and_se((0..paths).map(|p| fine(p, t, i)), paths);
            let rms = ((0..paths)
                .map(|p| {
                    let e = coarse(p, t, i) - fine(p, t, i);
                    e * e
                })
                .sum::<f64>()
                / paths as f64)
                .sqrt();
            let scale = mean_fine.abs() + var_fine.sqrt() + 1e-12;
            if mean_diff.abs() - 2.0 * se > tolerance * scale {
                converged = false;
            }
            weak[t][i] = mean_diff;
            weak_se[t][i] = se;
            strong[t][i] = rms;
            richardson[t][i] = 2.0 * mean_fine - (mean_fine + mean_diff);
        }
    }
    SdeNumericalError {
        refined_step: step / 2.0,
        weak_difference: weak,
        weak_difference_standard_error: weak_se,
        strong_rms_difference: strong,
        richardson_mean: richardson,
        step_converged: converged,
    }
}

/// Simulate every world at the declared step, with paired contrasts and step halving.
///
/// `worlds` need not contain the baseline: it is always simulated first, as
/// [`BASELINE_WORLD`], and a user world may not reuse that identity.
///
/// # Errors
/// `Invalid`, `Unsupported` (Milstein with state-coupled diffusion, off-grid actions,
/// out-of-envelope inputs), `Numerical` for non-finite states and `Resource` for budget or
/// cancellation.
#[allow(clippy::too_many_lines, reason = "one ensemble: plan, run, then three kinds of moments")]
pub fn simulate_sde(
    compiled: &CompiledSde,
    theta: Option<&[f64]>,
    initial: &[f64],
    policy: &SdeSimulationPolicy,
    worlds: &[SdeWorld],
    contrasts: &[SdeContrast],
    ctx: &ExecutionContext,
) -> Result<SdeEnsembleReport, SdeError> {
    policy.validate()?;
    validate_initial(compiled, initial)?;
    let theta = resolve_theta(compiled, theta)?;
    if matches!(policy.scheme, SdeScheme::Milstein)
        && !compiled.certificate.diagonal_milstein_admissible
    {
        return Err(unsupported(
            "sde.milstein_noncommutative",
            "a diffusion depends on another state, so the diagonal Milstein correction omits \
             Levy-area terms; use Euler-Maruyama or make each diffusion depend only on its own state",
        ));
    }
    if worlds.len() + 1 > MAX_WORLDS || contrasts.len() > MAX_WORLDS * MAX_WORLDS {
        return Err(bad("SDE world/contrast count"));
    }
    let mut plans = vec![compiled.system.plan_world(
        &SdeWorld { id: BASELINE_WORLD.into(), ..SdeWorld::default() },
        policy.start,
        policy.step,
    )?];
    let mut ids = std::collections::BTreeSet::new();
    ids.insert(BASELINE_WORLD.to_owned());
    for world in worlds {
        if !ids.insert(world.id.clone()) {
            return Err(bad("SDE world identities must be distinct and not 'baseline'"));
        }
        plans.push(compiled.system.plan_world(world, policy.start, policy.step)?);
    }
    let mut contrast_slots = Vec::with_capacity(contrasts.len());
    let mut contrast_ids = std::collections::BTreeSet::new();
    for c in contrasts {
        let find = |id: &str| plans.iter().position(|p| p.id == id);
        let (Some(t), Some(r)) = (find(&c.treated), find(&c.control)) else {
            return Err(bad("SDE contrast references an undeclared world"));
        };
        if c.id.trim().is_empty() || !contrast_ids.insert(c.id.clone()) {
            return Err(bad("SDE contrast identity"));
        }
        contrast_slots.push((t, r));
    }
    let _lease = admit(compiled, policy, plans.len(), ctx)?;
    let d = compiled.dimension();
    let n_out = policy.steps / policy.output_every + 1;
    let paths = policy.paths;
    let times = policy.output_times();
    let mut runs = Vec::with_capacity(plans.len());
    for plan in &plans {
        runs.push(run_world(compiled, &theta, plan, policy, initial, ctx)?);
    }
    let at = |levels: &[f64], p: usize, t: usize, i: usize| levels[(p * n_out + t) * d + i];
    let mut world_reports = Vec::with_capacity(plans.len());
    for (plan, run) in plans.iter().zip(&runs) {
        let mut mean = vec![vec![0.0; d]; n_out];
        let mut variance = vec![vec![0.0; d]; n_out];
        let mut mc = vec![vec![0.0; d]; n_out];
        let mut covariance = vec![vec![0.0; d * d]; n_out];
        for t in 0..n_out {
            for i in 0..d {
                let (m, v, se) = mean_and_se((0..paths).map(|p| at(&run.coarse, p, t, i)), paths);
                mean[t][i] = m;
                variance[t][i] = v;
                mc[t][i] = se;
            }
            for i in 0..d {
                for j in 0..=i {
                    let c = (0..paths)
                        .map(|p| {
                            (at(&run.coarse, p, t, i) - mean[t][i])
                                * (at(&run.coarse, p, t, j) - mean[t][j])
                        })
                        .sum::<f64>()
                        / (paths as f64 - 1.0);
                    covariance[t][i * d + j] = c;
                    covariance[t][j * d + i] = c;
                }
            }
        }
        let numerical = run.fine.as_ref().map(|fine| {
            numerical_error(
                |p, t, i| at(&run.coarse, p, t, i),
                |p, t, i| at(fine, p, t, i),
                paths,
                n_out,
                d,
                policy.step,
                policy.weak_tolerance,
            )
        });
        let final_samples = policy.retain_final_samples.then(|| {
            (0..paths).map(|p| (0..d).map(|i| at(&run.coarse, p, n_out - 1, i)).collect()).collect()
        });
        world_reports.push(SdeWorldEnsemble {
            id: plan.id.clone(),
            mean,
            variance,
            covariance,
            monte_carlo_standard_error: mc,
            numerical,
            final_samples,
        });
    }
    let mut contrast_reports = Vec::with_capacity(contrasts.len());
    for (c, &(ti, ci)) in contrasts.iter().zip(&contrast_slots) {
        let (treated, control) = (&runs[ti], &runs[ci]);
        let mut mean_difference = vec![vec![0.0; d]; n_out];
        let mut se = vec![vec![0.0; d]; n_out];
        for t in 0..n_out {
            for i in 0..d {
                let (m, _, s) = mean_and_se(
                    (0..paths).map(|p| at(&treated.coarse, p, t, i) - at(&control.coarse, p, t, i)),
                    paths,
                );
                mean_difference[t][i] = m;
                se[t][i] = s;
            }
        }
        let numerical = match (&treated.fine, &control.fine) {
            (Some(tf), Some(cf)) => Some(numerical_error(
                |p, t, i| at(&treated.coarse, p, t, i) - at(&control.coarse, p, t, i),
                |p, t, i| at(tf, p, t, i) - at(cf, p, t, i),
                paths,
                n_out,
                d,
                policy.step,
                policy.weak_tolerance,
            )),
            _ => None,
        };
        contrast_reports.push(SdeContrastEnsemble {
            id: c.id.clone(),
            treated: c.treated.clone(),
            control: c.control.clone(),
            mean_difference,
            monte_carlo_standard_error: se,
            numerical,
        });
    }
    Ok(SdeEnsembleReport {
        scheme: policy.scheme,
        step: policy.step,
        seed: policy.seed,
        paths,
        times,
        parameters: theta,
        worlds: world_reports,
        contrasts: contrast_reports,
        certificate: compiled.certificate.clone(),
        error_semantics: vec![
            "monte_carlo_standard_error is sampling error of a finite ensemble at the stated parameters; it shrinks with paths".into(),
            "numerical.weak_difference is the observed effect of halving the step on the mean, paired by path; it estimates discretization bias and shrinks with the step".into(),
            "neither field is statistical uncertainty of fitted parameters, which the estimation layer reports separately".into(),
            "paired contrasts share every Brownian increment across worlds, so their Monte Carlo standard error is that of the paired difference".into(),
        ],
    })
}

/// Raw declared-step paths of one world, for data generation and downstream functionals.
///
/// # Errors
/// As [`simulate_sde`].
pub fn sample_sde_paths(
    compiled: &CompiledSde,
    theta: Option<&[f64]>,
    initial: &[f64],
    policy: &SdeSimulationPolicy,
    world: &SdeWorld,
    ctx: &ExecutionContext,
) -> Result<SdePaths, SdeError> {
    policy.validate()?;
    validate_initial(compiled, initial)?;
    let theta = resolve_theta(compiled, theta)?;
    if matches!(policy.scheme, SdeScheme::Milstein)
        && !compiled.certificate.diagonal_milstein_admissible
    {
        return Err(unsupported(
            "sde.milstein_noncommutative",
            "a diffusion depends on another state; the diagonal Milstein correction is not exact",
        ));
    }
    let plan = compiled.system.plan_world(world, policy.start, policy.step)?;
    let mut single = policy.clone();
    single.check_step = false;
    let _lease = admit(compiled, &single, 1, ctx)?;
    let run = run_world(compiled, &theta, &plan, &single, initial, ctx)?;
    Ok(SdePaths {
        times: policy.output_times(),
        paths: policy.paths,
        dimension: compiled.dimension(),
        states: run.coarse,
    })
}

#[cfg(test)]
mod tests;
