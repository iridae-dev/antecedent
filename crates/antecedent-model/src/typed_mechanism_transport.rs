//! Transport of fitted nonlinear and non-Gaussian mechanisms into a target
//! population's composed model, with target-specific refits.
//!
//! [`crate::mechanism_transport`] carries additive-noise linear-Gaussian
//! mechanisms. This module is the same selection-diagram route for typed
//! families: a conditional mean that is a polynomial or interaction basis of the
//! parents, or a compiled [`MechanismExpression`], under one of
//!
//! - Gaussian noise (constant variance),
//! - heteroscedastic Gaussian noise (log-variance linear in the standardized
//!   parents),
//! - log-normal noise (Gaussian on the log scale),
//! - Bernoulli (logit link), Poisson (log link) and gamma (log link,
//!   estimated dispersion) responses.
//!
//! # Boundary
//!
//! - **Refit**: [`fit_mechanism`] fits one mechanism of one family from one
//!   population's data by Fisher scoring (Levenberg damping for expression
//!   means) and returns the parameter covariance. A target mechanism is refitted
//!   with the *source mechanism's* design (parent centres and scales), so its
//!   parameters are comparable to the source's by construction.
//! - **Admission**: the declared selection diagram states, per variable, whether
//!   the mechanism is invariant (no selection node into it) or shifted (a
//!   selection node into it). An invariant mechanism is carried from the source.
//!   A shifted mechanism is used only from a target fit; a shifted mechanism
//!   with no target fit is refused with the typed finding
//!   [`DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE`]. Nothing is ever treated as
//!   invariant because evidence is missing.
//! - **Declared invariance is tested where it can be**: when a target fit is
//!   also supplied for a mechanism declared invariant, a Wald test compares the
//!   full parameter vectors of the source and target fits, whose covariance is
//!   the sum of the two fit covariances (independent samples). A rejection at the
//!   request's level refuses the transport
//!   ([`DETAIL_INVARIANCE_CONTRADICTED`]); the carried source mechanism is kept
//!   when the test does not reject, and the disposition records the statistic.
//!   Absence of rejection is not evidence of invariance; with no target fit the
//!   declaration is an untested premise and the disposition says so.
//! - **Precedence**: a shifted mechanism's target fit is the one used; it is never
//!   averaged with the source's.
//! - **Execution**: interventional means are Monte Carlo over the structural
//!   model (truncated factorization), with per-node, per-unit random streams so
//!   every arm of a contrast shares its noise (paired draws) and the outcome
//!   mechanism enters through its exact conditional mean. The Monte Carlo
//!   standard error is a numerical-approximation uncertainty and is reported
//!   apart from the parameter uncertainty.
//! - **Uncertainty**: parameter uncertainty of a contrast is first order,
//!   `g' Sigma g`, where `g` is the score-function gradient of the interventional
//!   mean with respect to every parameter of every mechanism involved, and `Sigma`
//!   is the source-fit covariance for a carried mechanism and the target-fit
//!   covariance for a refit one, joined by the declared dependence. Unknown
//!   dependence between more than one involved mechanism is a refusal, never an
//!   independence default. The transport assumption adds no variance and the
//!   result says so ([`UNCERTAINTY_SCOPE`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::identity::CanonicalWriter;
use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, mix64};
use antecedent_kernels::standard_normal;
use antecedent_stats::gram::invert_square;
use antecedent_stats::special::{chi_square_sf, ln_gamma};

use crate::compose::coherence::{CoherenceFinding, valid_covariance};
use crate::compose::{
    CoherenceKind, CompositionError, ContextCoordinate, InvarianceDeclaration, MeanContrast,
    MechanismContext, MechanismOrigin, MechanismProvenance, SelectionDiagramDeclaration,
    noise_stream_index,
};
use crate::mechanism_expr::{MechanismExpression, MechanismTape};
pub use crate::mechanism_transport::{MechanismRole, UNCERTAINTY_SCOPE, selection_targets};

/// Version of this route's identity encoding.
pub const TYPED_TRANSPORT_VERSION: u32 = 1;

/// Most parents of one mechanism.
pub const MAX_TYPED_PARENTS: usize = 32;
/// Highest polynomial degree of a mean basis.
pub const MAX_POLYNOMIAL_DEGREE: u8 = 6;
/// Most free mean parameters of an expression mean.
pub const MAX_EXPRESSION_PARAMETERS: usize = 64;
/// Largest number of Monte Carlo units of one contrast arm.
pub const MAX_TYPED_DRAWS: usize = 1 << 24;
/// Default level of the declared-invariance consistency test.
pub const DEFAULT_INVARIANCE_LEVEL: f64 = 0.01;

/// Stream domain of the typed route's Monte Carlo units.
pub const TYPED_TRANSPORT_STREAM_DOMAIN: StreamDomain =
    StreamDomain::external(StreamDomain::EXTERNAL_BASE + 0x5454);

/// A shifted mechanism (a selection node points into it) has no target fit.
pub const DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE: &str =
    "typed_mechanism_transport.shifted_without_target_evidence";
/// The target fit of a mechanism declared invariant rejects the declaration.
pub const DETAIL_INVARIANCE_CONTRADICTED: &str =
    "typed_mechanism_transport.invariance_contradicted";
/// A target fit has another family, mean or parent set than the source mechanism it checks.
pub const DETAIL_TARGET_SPEC_DIFFERS: &str =
    "typed_mechanism_transport.target_specification_differs";
/// A target fit was not fitted on the source mechanism's design, so the parameters are not comparable.
pub const DETAIL_DESIGN_NOT_SHARED: &str = "typed_mechanism_transport.design_not_shared";
/// A shifted mechanism's target fit changes the parent set of the source mechanism.
pub const DETAIL_PARENTS_CHANGED: &str = "typed_mechanism_transport.parents_changed";
/// A mechanism names a parent that has no mechanism in the transported model.
pub const DETAIL_PARENT_WITHOUT_MECHANISM: &str =
    "typed_mechanism_transport.parent_without_mechanism";
/// A selection node points into a variable no supplied mechanism produces.
pub const DETAIL_SELECTION_TARGET_UNKNOWN: &str =
    "typed_mechanism_transport.selection_target_unknown_variable";
/// A carried mechanism was fitted in another regime without an explicit invariance declaration.
pub const DETAIL_REGIME_NOT_DECLARED: &str =
    "typed_mechanism_transport.regime_not_declared_invariant";
/// The consistency-test level is outside `(0, 0.5)`.
pub const DETAIL_INVARIANCE_LEVEL: &str = "typed_mechanism_transport.invariance_level_invalid";
/// The declared parameter dependence is malformed.
pub const DETAIL_DEPENDENCE_INVALID: &str = "typed_mechanism_transport.dependence_invalid";
/// The structure has a directed cycle.
pub const DETAIL_CYCLE: &str = "typed_mechanism_transport.cycle";
/// A claim over more than one mechanism's parameters has no declared dependence.
pub const DETAIL_DEPENDENCE_UNKNOWN: &str =
    "typed_mechanism_transport.parameter_dependence_unknown";
/// The declared dependence omits a pair of involved mechanisms.
pub const DETAIL_DEPENDENCE_INCOMPLETE: &str =
    "typed_mechanism_transport.parameter_dependence_incomplete";
/// A request names a variable the transported model does not have.
pub const DETAIL_UNKNOWN_VARIABLE: &str = "typed_mechanism_transport.unknown_variable";

const TAG_IDENTITY: &str = "antecedent.model.typed_mechanism_transport";

/// The conditional-mean (linear predictor) form of a mechanism.
#[derive(Clone, Debug, PartialEq)]
pub enum MeanSpec {
    /// `intercept + sum_j sum_{d=1..degree} b_{jd} z_j^d`, plus every pairwise
    /// product `z_j z_k` when `interactions`. `z` are the standardized parents.
    Polynomial {
        /// Highest power of each parent.
        degree: u8,
        /// Whether pairwise products of the standardized parents are included.
        interactions: bool,
    },
    /// A compiled expression over the raw parents (variable `j` is parent `j`)
    /// and `start.len()` free parameters, fitted from `start`.
    Expression {
        /// The mean function.
        expression: MechanismExpression,
        /// Starting parameter values (also fixes the parameter count).
        start: Vec<f64>,
    },
}

/// The response distribution of a mechanism given its linear predictor.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResponseFamily {
    /// Normal with identity link and constant variance.
    Gaussian,
    /// Normal with identity link and `log variance = g0 + sum_j g_j z_j`.
    HeteroscedasticGaussian,
    /// `log y` is Gaussian with constant variance (`y > 0`).
    LogNormal,
    /// Bernoulli with logit link (`y` in {0, 1}).
    Bernoulli,
    /// Poisson with log link (`y` a non-negative integer).
    Poisson,
    /// Gamma with log link and estimated dispersion (`y > 0`).
    Gamma,
}

impl ResponseFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::HeteroscedasticGaussian => "heteroscedastic_gaussian",
            Self::LogNormal => "log_normal",
            Self::Bernoulli => "bernoulli",
            Self::Poisson => "poisson",
            Self::Gamma => "gamma",
        }
    }

    /// The family a stable name spells.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "gaussian" => Self::Gaussian,
            "heteroscedastic_gaussian" => Self::HeteroscedasticGaussian,
            "log_normal" => Self::LogNormal,
            "bernoulli" => Self::Bernoulli,
            "poisson" => Self::Poisson,
            "gamma" => Self::Gamma,
            _ => return None,
        })
    }

    const fn code(self) -> u64 {
        match self {
            Self::Gaussian => 1,
            Self::HeteroscedasticGaussian => 2,
            Self::LogNormal => 3,
            Self::Bernoulli => 4,
            Self::Poisson => 5,
            Self::Gamma => 6,
        }
    }
}

/// What a mechanism is: its variable, parents, mean form and response family.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismSpec {
    /// The variable the mechanism produces.
    pub variable: String,
    /// Parent variables, in the order the mean reads them.
    pub parents: Vec<String>,
    /// Conditional-mean form.
    pub mean: MeanSpec,
    /// Response family.
    pub family: ResponseFamily,
}

/// Why a mechanism could not be fitted.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TypedFitError {
    /// The specification is malformed or outside the bounded family.
    #[error("invalid mechanism specification: {0}")]
    Spec(String),
    /// A required column is absent.
    #[error("missing column `{0}`")]
    MissingColumn(String),
    /// The data are malformed (ragged, non-finite, too few rows).
    #[error("invalid data: {0}")]
    Data(String),
    /// The response is outside the family's support.
    #[error("`{variable}` is outside the support of the {family} family: {reason}")]
    Support {
        /// The variable.
        variable: String,
        /// The family name.
        family: &'static str,
        /// What is violated.
        reason: String,
    },
    /// The iteration did not converge (including complete separation).
    #[error("the fit of `{0}` did not converge")]
    NotConverged(String),
    /// The design is singular.
    #[error("the design of `{0}` is singular")]
    Singular(String),
}

impl TypedFitError {
    /// The registered reason code this failure is refused under.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::NotConverged(_) | Self::Singular(_) => {
                antecedent_core::reason_code!("mechanism_fit_not_converged")
            }
            _ => antecedent_core::reason_code!("mechanism_composition_incoherent"),
        }
    }
}

/// A mechanism fitted from one population's data.
#[derive(Clone, Debug)]
pub struct FittedMechanism {
    spec: MechanismSpec,
    scaling: Vec<(f64, f64)>,
    theta: Vec<f64>,
    mean_dim: usize,
    covariance: Vec<f64>,
    n_obs: usize,
    context: MechanismContext,
    provenance: MechanismProvenance,
    tape: Option<MechanismTape>,
}

fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}

fn polynomial_width(parents: usize, degree: u8, interactions: bool) -> usize {
    1 + parents * usize::from(degree)
        + if interactions { parents * parents.saturating_sub(1) / 2 } else { 0 }
}

fn scale_dim(family: ResponseFamily, parents: usize) -> usize {
    match family {
        ResponseFamily::Gaussian | ResponseFamily::LogNormal | ResponseFamily::Gamma => 1,
        ResponseFamily::HeteroscedasticGaussian => 1 + parents,
        ResponseFamily::Bernoulli | ResponseFamily::Poisson => 0,
    }
}

impl FittedMechanism {
    /// The specification fitted.
    #[must_use]
    pub fn spec(&self) -> &MechanismSpec {
        &self.spec
    }

    /// The variable produced.
    #[must_use]
    pub fn variable(&self) -> &str {
        &self.spec.variable
    }

    /// Population and regime the data came from.
    #[must_use]
    pub fn context(&self) -> &MechanismContext {
        &self.context
    }

    /// Provenance of the fit.
    #[must_use]
    pub fn provenance(&self) -> &MechanismProvenance {
        &self.provenance
    }

    /// Rows the fit used.
    #[must_use]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Fitted parameters: mean parameters, then scale parameters
    /// (`ln sigma`, the log-variance coefficients, or `ln dispersion`).
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.theta
    }

    /// Number of mean parameters (the leading block of [`Self::parameters`]).
    #[must_use]
    pub fn mean_dimension(&self) -> usize {
        self.mean_dim
    }

    /// Row-major parameter covariance of the fit.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Per-parent `(centre, scale)` of the standardization the polynomial basis uses.
    #[must_use]
    pub fn scaling(&self) -> &[(f64, f64)] {
        &self.scaling
    }

    fn standardize(&self, raw: &[f64]) -> Vec<f64> {
        raw.iter().zip(&self.scaling).map(|(x, (c, s))| (x - c) / s).collect()
    }

    fn polynomial_features(&self, z: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.push(1.0);
        if let MeanSpec::Polynomial { degree, interactions } = &self.spec.mean {
            for zj in z {
                let mut power = 1.0;
                for _ in 0..*degree {
                    power *= zj;
                    out.push(power);
                }
            }
            if *interactions {
                for j in 0..z.len() {
                    for k in (j + 1)..z.len() {
                        out.push(z[j] * z[k]);
                    }
                }
            }
        }
    }

    /// The linear predictor at raw parent values under parameters `theta`.
    fn eta_with(&self, theta: &[f64], raw: &[f64]) -> f64 {
        match &self.spec.mean {
            MeanSpec::Polynomial { .. } => {
                let z = self.standardize(raw);
                let mut f = Vec::with_capacity(self.mean_dim);
                self.polynomial_features(&z, &mut f);
                f.iter().zip(&theta[..self.mean_dim]).map(|(a, b)| a * b).sum()
            }
            MeanSpec::Expression { .. } => match &self.tape {
                Some(tape) => {
                    tape.evaluate(raw, &theta[..self.mean_dim]).map_or(f64::NAN, |e| e.value())
                }
                None => f64::NAN,
            },
        }
    }

    fn sigma_with(&self, theta: &[f64]) -> f64 {
        theta[self.mean_dim].exp()
    }

    fn log_variance_with(&self, theta: &[f64], z: &[f64]) -> f64 {
        let g = &theta[self.mean_dim..];
        g[0] + g[1..].iter().zip(z).map(|(a, b)| a * b).sum::<f64>()
    }

    /// `E[y | parents]` under parameters `theta`.
    fn mean_with(&self, theta: &[f64], raw: &[f64]) -> f64 {
        let eta = self.eta_with(theta, raw);
        match self.spec.family {
            ResponseFamily::Gaussian | ResponseFamily::HeteroscedasticGaussian => eta,
            ResponseFamily::LogNormal => {
                let s = self.sigma_with(theta);
                (eta + 0.5 * s * s).exp()
            }
            ResponseFamily::Bernoulli => logistic(eta),
            ResponseFamily::Poisson | ResponseFamily::Gamma => eta.exp(),
        }
    }

    /// `log p(x | parents)` under parameters `theta`.
    fn log_density_with(&self, theta: &[f64], raw: &[f64], x: f64) -> f64 {
        let eta = self.eta_with(theta, raw);
        let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        match self.spec.family {
            ResponseFamily::Gaussian => {
                let ls = theta[self.mean_dim];
                let r = (x - eta) / ls.exp();
                -half_ln_2pi - ls - 0.5 * r * r
            }
            ResponseFamily::HeteroscedasticGaussian => {
                let z = self.standardize(raw);
                let lv = self.log_variance_with(theta, &z);
                let r = x - eta;
                -half_ln_2pi - 0.5 * lv - 0.5 * r * r / lv.exp()
            }
            ResponseFamily::LogNormal => {
                let ls = theta[self.mean_dim];
                let lx = x.ln();
                let r = (lx - eta) / ls.exp();
                -half_ln_2pi - ls - 0.5 * r * r - lx
            }
            ResponseFamily::Bernoulli => x * eta - softplus(eta),
            ResponseFamily::Poisson => x * eta - eta.exp() - ln_gamma(x + 1.0),
            ResponseFamily::Gamma => {
                let k = (-theta[self.mean_dim]).exp();
                let mu = eta.exp();
                (k - 1.0) * x.ln() - x * k / mu - k * (mu / k).ln() - ln_gamma(k)
            }
        }
    }

    /// One draw of the response given raw parent values.
    fn sample_with(&self, theta: &[f64], raw: &[f64], rng: &mut CausalRng) -> f64 {
        let eta = self.eta_with(theta, raw);
        match self.spec.family {
            ResponseFamily::Gaussian => eta + self.sigma_with(theta) * standard_normal(rng),
            ResponseFamily::HeteroscedasticGaussian => {
                let z = self.standardize(raw);
                eta + (0.5 * self.log_variance_with(theta, &z)).exp() * standard_normal(rng)
            }
            ResponseFamily::LogNormal => {
                (eta + self.sigma_with(theta) * standard_normal(rng)).exp()
            }
            ResponseFamily::Bernoulli => f64::from(u8::from(rng.next_f64() < logistic(eta))),
            ResponseFamily::Poisson => sample_poisson(eta.exp(), rng),
            ResponseFamily::Gamma => {
                let phi = theta[self.mean_dim].exp();
                sample_gamma(1.0 / phi, eta.exp() * phi, rng)
            }
        }
    }

    /// Labels of [`Self::parameters`], aligned.
    #[must_use]
    pub fn parameter_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(self.theta.len());
        match &self.spec.mean {
            MeanSpec::Polynomial { degree, interactions } => {
                labels.push("intercept".to_owned());
                for parent in &self.spec.parents {
                    for d in 1..=*degree {
                        labels.push(format!("{parent}^{d}"));
                    }
                }
                if *interactions {
                    for j in 0..self.spec.parents.len() {
                        for k in (j + 1)..self.spec.parents.len() {
                            labels
                                .push(format!("{}*{}", self.spec.parents[j], self.spec.parents[k]));
                        }
                    }
                }
            }
            MeanSpec::Expression { .. } => {
                for i in 0..self.mean_dim {
                    labels.push(format!("expression[{i}]"));
                }
            }
        }
        match self.spec.family {
            ResponseFamily::Gaussian | ResponseFamily::LogNormal => labels.push("log_sigma".into()),
            ResponseFamily::Gamma => labels.push("log_dispersion".into()),
            ResponseFamily::HeteroscedasticGaussian => {
                labels.push("log_variance[intercept]".into());
                for parent in &self.spec.parents {
                    labels.push(format!("log_variance[{parent}]"));
                }
            }
            ResponseFamily::Bernoulli | ResponseFamily::Poisson => {}
        }
        labels
    }

    /// `E[variable | parents]` at the given raw parent values, in `spec().parents` order.
    ///
    /// # Errors
    /// [`TypedFitError::Data`] when the arity is wrong or the value is not finite.
    pub fn conditional_mean(&self, parents: &[f64]) -> Result<f64, TypedFitError> {
        if parents.len() != self.spec.parents.len() {
            return Err(TypedFitError::Data("parent arity".into()));
        }
        let m = self.mean_with(&self.theta, parents);
        if m.is_finite() {
            Ok(m)
        } else {
            Err(TypedFitError::Data("non-finite conditional mean".into()))
        }
    }
}

fn sample_poisson(lambda: f64, rng: &mut CausalRng) -> f64 {
    if !lambda.is_finite() || lambda < 0.0 || lambda > 1.0e5 {
        return f64::NAN;
    }
    let mut remaining = lambda;
    let mut total = 0.0_f64;
    while remaining > 0.0 {
        let chunk = remaining.min(200.0);
        remaining -= chunk;
        let limit = (-chunk).exp();
        let mut p = 1.0;
        loop {
            p *= rng.next_f64();
            if p <= limit {
                break;
            }
            total += 1.0;
        }
    }
    total
}

fn sample_gamma(shape: f64, scale: f64, rng: &mut CausalRng) -> f64 {
    if !shape.is_finite() || !scale.is_finite() || shape <= 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    if shape < 1.0 {
        let u = rng.next_f64().max(1e-300);
        return sample_gamma(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..10_000 {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3 * scale;
        }
    }
    f64::NAN
}

// ------------------------------------------------------------------ fitting

#[derive(Clone, Copy)]
enum Link {
    Gaussian,
    Bernoulli,
    Poisson,
    Gamma,
}

impl Link {
    fn mean(self, eta: f64) -> f64 {
        match self {
            Self::Gaussian => eta,
            Self::Bernoulli => logistic(eta),
            Self::Poisson | Self::Gamma => eta.exp(),
        }
    }

    /// Unit-dispersion log-likelihood up to a constant.
    fn objective(self, y: f64, eta: f64) -> f64 {
        match self {
            Self::Gaussian => -0.5 * (y - eta) * (y - eta),
            Self::Bernoulli => y * eta - softplus(eta),
            Self::Poisson => y * eta - eta.exp(),
            Self::Gamma => -y * (-eta).exp() - eta,
        }
    }

    /// `(score weight, Fisher weight)` per observation.
    fn score_and_weight(self, y: f64, eta: f64) -> (f64, f64) {
        let mu = self.mean(eta);
        match self {
            Self::Gaussian => (y - mu, 1.0),
            Self::Bernoulli => (y - mu, mu * (1.0 - mu)),
            Self::Poisson => (y - mu, mu),
            Self::Gamma => ((y - mu) / mu, 1.0),
        }
    }
}

struct MeanProblem<'a> {
    base: &'a FittedMechanism,
    rows: &'a [Vec<f64>],
    y: &'a [f64],
    link: Link,
    features: Option<Vec<f64>>,
}

impl<'a> MeanProblem<'a> {
    fn new(base: &'a FittedMechanism, rows: &'a [Vec<f64>], y: &'a [f64], link: Link) -> Self {
        let features = match base.spec.mean {
            MeanSpec::Polynomial { .. } => {
                let mut all = Vec::with_capacity(rows.len() * base.mean_dim);
                let mut f = Vec::new();
                for row in rows {
                    let z = base.standardize(row);
                    base.polynomial_features(&z, &mut f);
                    all.extend_from_slice(&f);
                }
                Some(all)
            }
            MeanSpec::Expression { .. } => None,
        };
        Self { base, rows, y, link, features }
    }

    fn eta(&self, theta: &[f64]) -> Vec<f64> {
        let k = self.base.mean_dim;
        match &self.features {
            Some(x) => (0..self.rows.len())
                .map(|i| x[i * k..(i + 1) * k].iter().zip(theta).map(|(a, b)| a * b).sum())
                .collect(),
            None => self.rows.iter().map(|r| self.base.eta_with(theta, r)).collect(),
        }
    }

    fn jacobian(&self, theta: &[f64]) -> Vec<f64> {
        let k = self.base.mean_dim;
        if let Some(x) = &self.features {
            return x.clone();
        }
        let n = self.rows.len();
        let mut jac = vec![0.0; n * k];
        let mut shifted = theta.to_vec();
        for j in 0..k {
            let h = 1e-6 * (1.0 + theta[j].abs());
            shifted[j] = theta[j] + h;
            let up = self.eta(&shifted);
            shifted[j] = theta[j] - h;
            let down = self.eta(&shifted);
            shifted[j] = theta[j];
            for i in 0..n {
                jac[i * k + j] = (up[i] - down[i]) / (2.0 * h);
            }
        }
        jac
    }

    fn objective(&self, eta: &[f64], obs_weights: Option<&[f64]>) -> f64 {
        let mut total = 0.0;
        for (i, e) in eta.iter().enumerate() {
            let w = obs_weights.map_or(1.0, |w| w[i]);
            total += w * self.link.objective(self.y[i], *e);
        }
        total
    }
}

fn matvec(a: &[f64], k: usize, v: &[f64]) -> Vec<f64> {
    (0..k).map(|i| (0..k).map(|j| a[i * k + j] * v[j]).sum()).collect()
}

/// Fisher scoring with Levenberg damping for the mean parameters. Returns the
/// parameters and the Fisher information at them.
fn fit_mean(
    problem: &MeanProblem<'_>,
    start: &[f64],
    obs_weights: Option<&[f64]>,
    variable: &str,
) -> Result<(Vec<f64>, Vec<f64>), TypedFitError> {
    let k = problem.base.mean_dim;
    let n = problem.rows.len();
    let max_iter = if problem.features.is_some() { 200 } else { 500 };
    let mut theta = start.to_vec();
    let mut eta = problem.eta(&theta);
    if eta.iter().any(|e| !e.is_finite()) {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    let mut objective = problem.objective(&eta, obs_weights);
    let mut lambda = 0.0_f64;
    let mut converged = false;
    for _ in 0..max_iter {
        let jac = problem.jacobian(&theta);
        let mut score = vec![0.0; k];
        let mut info = vec![0.0; k * k];
        for i in 0..n {
            let ow = obs_weights.map_or(1.0, |w| w[i]);
            let (s, w) = problem.link.score_and_weight(problem.y[i], eta[i]);
            let row = &jac[i * k..(i + 1) * k];
            for a in 0..k {
                score[a] += ow * s * row[a];
                let wa = ow * w * row[a];
                for b in 0..=a {
                    info[a * k + b] += wa * row[b];
                }
            }
        }
        for a in 0..k {
            for b in 0..a {
                info[b * k + a] = info[a * k + b];
            }
        }
        let mut accepted = false;
        for _ in 0..40 {
            let mut damped = info.clone();
            for a in 0..k {
                damped[a * k + a] += lambda * damped[a * k + a].abs().max(1e-12);
            }
            let Some(inverse) = invert_square(&damped, k) else {
                lambda = (lambda * 10.0).max(1e-6);
                continue;
            };
            let delta = matvec(&inverse, k, &score);
            let candidate: Vec<f64> = theta.iter().zip(&delta).map(|(t, d)| t + d).collect();
            let candidate_eta = problem.eta(&candidate);
            let candidate_objective = problem.objective(&candidate_eta, obs_weights);
            if candidate_objective.is_finite()
                && candidate_objective >= objective - 1e-12 * objective.abs().max(1.0)
            {
                let step = delta
                    .iter()
                    .zip(&theta)
                    .map(|(d, t)| d.abs() / (1.0 + t.abs()))
                    .fold(0.0_f64, f64::max);
                theta = candidate;
                eta = candidate_eta;
                objective = candidate_objective;
                lambda = if lambda < 1e-12 { 0.0 } else { lambda / 10.0 };
                accepted = true;
                converged = step < 1e-10;
                break;
            }
            lambda = (lambda * 10.0).max(1e-6);
        }
        if !accepted {
            // Exhausted damping attempts do not establish stationarity or convergence.
            break;
        }
        if converged {
            break;
        }
    }
    if !converged {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    // A linear predictor this large is a divergent (separated) fit, not an estimate.
    let bound = if matches!(problem.link, Link::Bernoulli) { 25.0 } else { 40.0 };
    if eta.iter().any(|e| !e.is_finite() || e.abs() > bound)
        && !matches!(problem.link, Link::Gaussian)
    {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    // Information at the final parameters.
    let jac = problem.jacobian(&theta);
    let mut info = vec![0.0; k * k];
    for i in 0..n {
        let ow = obs_weights.map_or(1.0, |w| w[i]);
        let (_, w) = problem.link.score_and_weight(problem.y[i], eta[i]);
        let row = &jac[i * k..(i + 1) * k];
        for a in 0..k {
            for b in 0..k {
                info[a * k + b] += ow * w * row[a] * row[b];
            }
        }
    }
    Ok((theta, info))
}

fn validate_spec(spec: &MechanismSpec) -> Result<Option<MechanismTape>, TypedFitError> {
    if spec.variable.trim().is_empty() {
        return Err(TypedFitError::Spec("blank variable".into()));
    }
    let distinct: BTreeSet<&String> = spec.parents.iter().collect();
    if distinct.len() != spec.parents.len()
        || spec.parents.iter().any(|p| p.trim().is_empty() || *p == spec.variable)
    {
        return Err(TypedFitError::Spec(
            "parents must be distinct, non-blank and not the variable itself".into(),
        ));
    }
    if spec.parents.len() > MAX_TYPED_PARENTS {
        return Err(TypedFitError::Spec(format!("more than {MAX_TYPED_PARENTS} parents")));
    }
    match &spec.mean {
        MeanSpec::Polynomial { degree, .. } => {
            if *degree == 0 || *degree > MAX_POLYNOMIAL_DEGREE {
                return Err(TypedFitError::Spec(format!(
                    "polynomial degree outside 1..={MAX_POLYNOMIAL_DEGREE}"
                )));
            }
            Ok(None)
        }
        MeanSpec::Expression { expression, start } => {
            if spec.parents.is_empty() {
                return Err(TypedFitError::Spec(
                    "an expression mean needs at least one parent".into(),
                ));
            }
            if start.is_empty()
                || start.len() > MAX_EXPRESSION_PARAMETERS
                || start.iter().any(|v| !v.is_finite())
            {
                return Err(TypedFitError::Spec(
                    "expression start must be finite with 1..=64 parameters".into(),
                ));
            }
            expression
                .compile(spec.parents.len(), start.len())
                .map(Some)
                .map_err(|e| TypedFitError::Spec(e.to_string()))
        }
    }
}

fn column<'a>(
    columns: &'a BTreeMap<String, Vec<f64>>,
    name: &str,
) -> Result<&'a Vec<f64>, TypedFitError> {
    columns.get(name).ok_or_else(|| TypedFitError::MissingColumn(name.to_owned()))
}

fn check_support(spec: &MechanismSpec, y: &[f64]) -> Result<(), TypedFitError> {
    let bad = |reason: &str| TypedFitError::Support {
        variable: spec.variable.clone(),
        family: spec.family.as_str(),
        reason: reason.to_owned(),
    };
    match spec.family {
        ResponseFamily::Gaussian | ResponseFamily::HeteroscedasticGaussian => Ok(()),
        ResponseFamily::LogNormal | ResponseFamily::Gamma => {
            if y.iter().any(|v| *v <= 0.0) {
                Err(bad("values must be strictly positive"))
            } else {
                Ok(())
            }
        }
        ResponseFamily::Bernoulli => {
            if y.iter().any(|v| *v != 0.0 && *v != 1.0) {
                Err(bad("values must be 0 or 1"))
            } else {
                Ok(())
            }
        }
        ResponseFamily::Poisson => {
            if y.iter().any(|v| *v < 0.0 || v.fract() != 0.0) {
                Err(bad("values must be non-negative integers"))
            } else {
                Ok(())
            }
        }
    }
}

/// Fit one mechanism from one population's data.
///
/// `design_from` fixes the parent standardization (so the parameters are
/// comparable to that fit's); the specification must then equal that fit's
/// except for nothing — pass it exactly when the fit is a target refit of that
/// source mechanism.
///
/// # Errors
/// [`TypedFitError`]: a malformed specification or data, a response outside
/// the family's support, a singular design, or an iteration that did not
/// converge.
pub fn fit_mechanism(
    spec: &MechanismSpec,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
    design_from: Option<&FittedMechanism>,
) -> Result<FittedMechanism, TypedFitError> {
    let tape = validate_spec(spec)?;
    let y_raw = column(columns, &spec.variable)?;
    let n = y_raw.len();
    let parent_columns: Vec<&Vec<f64>> =
        spec.parents.iter().map(|p| column(columns, p)).collect::<Result<_, _>>()?;
    if parent_columns.iter().any(|c| c.len() != n) {
        return Err(TypedFitError::Data("columns differ in length".into()));
    }
    if y_raw.iter().chain(parent_columns.iter().flat_map(|c| c.iter())).any(|v| !v.is_finite()) {
        return Err(TypedFitError::Data("non-finite value".into()));
    }
    check_support(spec, y_raw)?;
    let p = spec.parents.len();
    let mean_dim = match &spec.mean {
        MeanSpec::Polynomial { degree, interactions } => {
            polynomial_width(p, *degree, *interactions)
        }
        MeanSpec::Expression { start, .. } => start.len(),
    };
    let total_dim = mean_dim + scale_dim(spec.family, p);
    if n < total_dim + 2 {
        return Err(TypedFitError::Data(format!(
            "{n} rows cannot identify {total_dim} parameters"
        )));
    }
    let scaling: Vec<(f64, f64)> = match design_from {
        Some(source) => {
            if source.spec.parents != spec.parents {
                return Err(TypedFitError::Spec(
                    "the design source has other parents than the refit".into(),
                ));
            }
            source.scaling.clone()
        }
        None => {
            let mut out = Vec::with_capacity(p);
            for (name, col) in spec.parents.iter().zip(&parent_columns) {
                let mean = col.iter().sum::<f64>() / n as f64;
                let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
                if var <= 0.0 || !var.is_finite() {
                    return Err(TypedFitError::Singular(format!("{}:{name}", spec.variable)));
                }
                out.push((mean, var.sqrt()));
            }
            out
        }
    };
    let rows: Vec<Vec<f64>> =
        (0..n).map(|i| parent_columns.iter().map(|c| c[i]).collect()).collect();

    let mut base = FittedMechanism {
        spec: spec.clone(),
        scaling,
        theta: vec![0.0; total_dim],
        mean_dim,
        covariance: vec![0.0; total_dim * total_dim],
        n_obs: n,
        context,
        provenance,
        tape,
    };
    if let MeanSpec::Expression { start, .. } = &spec.mean {
        base.theta[..mean_dim].copy_from_slice(start);
    }

    let nf = n as f64;
    let mut start = base.theta[..mean_dim].to_vec();
    let variable = spec.variable.as_str();
    match spec.family {
        ResponseFamily::Gaussian | ResponseFamily::LogNormal => {
            let y: Vec<f64> = if spec.family == ResponseFamily::LogNormal {
                y_raw.iter().map(|v| v.ln()).collect()
            } else {
                y_raw.clone()
            };
            let problem = MeanProblem::new(&base, &rows, &y, Link::Gaussian);
            let (theta, info) = fit_mean(&problem, &start, None, variable)?;
            let eta = problem.eta(&theta);
            let rss: f64 = y.iter().zip(&eta).map(|(a, b)| (a - b) * (a - b)).sum();
            let dof = nf - mean_dim as f64;
            let sigma2 = rss / dof;
            if !(sigma2.is_finite() && sigma2 > 0.0) {
                return Err(TypedFitError::Singular(variable.to_owned()));
            }
            let inverse = invert_square(&info, mean_dim)
                .ok_or_else(|| TypedFitError::Singular(variable.to_owned()))?;
            base.theta[..mean_dim].copy_from_slice(&theta);
            base.theta[mean_dim] = 0.5 * sigma2.ln();
            for a in 0..mean_dim {
                for b in 0..mean_dim {
                    base.covariance[a * total_dim + b] = sigma2 * inverse[a * mean_dim + b];
                }
            }
            base.covariance[mean_dim * total_dim + mean_dim] = 1.0 / (2.0 * dof);
        }
        ResponseFamily::HeteroscedasticGaussian => {
            fit_heteroscedastic(&mut base, &rows, y_raw, &start)?;
        }
        ResponseFamily::Bernoulli | ResponseFamily::Poisson | ResponseFamily::Gamma => {
            let link = match spec.family {
                ResponseFamily::Bernoulli => Link::Bernoulli,
                ResponseFamily::Poisson => Link::Poisson,
                _ => Link::Gamma,
            };
            if matches!(spec.mean, MeanSpec::Polynomial { .. })
                && matches!(link, Link::Poisson | Link::Gamma)
            {
                start[0] = (y_raw.iter().sum::<f64>() / nf).max(1e-12).ln();
            }
            let problem = MeanProblem::new(&base, &rows, y_raw, link);
            let (theta, info) = fit_mean(&problem, &start, None, variable)?;
            let inverse = invert_square(&info, mean_dim)
                .ok_or_else(|| TypedFitError::Singular(variable.to_owned()))?;
            let final_eta = problem.eta(&theta);
            base.theta[..mean_dim].copy_from_slice(&theta);
            if matches!(link, Link::Gamma) {
                let eta = final_eta;
                let r2: Vec<f64> = y_raw
                    .iter()
                    .zip(&eta)
                    .map(|(y, e)| {
                        let mu = e.exp();
                        ((y - mu) / mu) * ((y - mu) / mu)
                    })
                    .collect();
                let dof = nf - mean_dim as f64;
                let phi = r2.iter().sum::<f64>() / dof;
                if !(phi.is_finite() && phi > 0.0) {
                    return Err(TypedFitError::Singular(variable.to_owned()));
                }
                let mean_r2 = r2.iter().sum::<f64>() / nf;
                let var_r2 =
                    r2.iter().map(|v| (v - mean_r2) * (v - mean_r2)).sum::<f64>() / (nf - 1.0);
                base.theta[mean_dim] = phi.ln();
                for a in 0..mean_dim {
                    for b in 0..mean_dim {
                        base.covariance[a * total_dim + b] = phi * inverse[a * mean_dim + b];
                    }
                }
                base.covariance[mean_dim * total_dim + mean_dim] =
                    var_r2 / (nf * mean_r2 * mean_r2);
            } else {
                base.covariance.copy_from_slice(&inverse);
            }
        }
    }
    if base.theta.iter().any(|v| !v.is_finite()) || base.covariance.iter().any(|v| !v.is_finite()) {
        return Err(TypedFitError::NotConverged(variable.to_owned()));
    }
    Ok(base)
}

fn fit_heteroscedastic(
    base: &mut FittedMechanism,
    rows: &[Vec<f64>],
    y: &[f64],
    start: &[f64],
) -> Result<(), TypedFitError> {
    let variable = base.spec.variable.clone();
    let n = rows.len();
    let k = base.mean_dim;
    let p = base.spec.parents.len();
    let q = p + 1;
    let total = k + q;
    // Design of the log-variance: intercept and standardized parents.
    let mut z_all = Vec::with_capacity(n * q);
    for row in rows {
        z_all.push(1.0);
        z_all.extend(base.standardize(row));
    }
    let mut ztz = vec![0.0; q * q];
    for i in 0..n {
        for a in 0..q {
            for b in 0..q {
                ztz[a * q + b] += z_all[i * q + a] * z_all[i * q + b];
            }
        }
    }
    let ztz_inverse =
        invert_square(&ztz, q).ok_or_else(|| TypedFitError::Singular(variable.clone()))?;

    let problem_frozen = base.clone();
    let problem = MeanProblem::new(&problem_frozen, rows, y, Link::Gaussian);
    let (mut theta, mut info) = fit_mean(&problem, start, None, &variable)?;
    let mut eta = problem.eta(&theta);
    let mean_r2 = y.iter().zip(&eta).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / n as f64;
    let mut gamma = vec![0.0; q];
    gamma[0] = mean_r2.max(1e-300).ln();
    let mut converged = false;
    for _ in 0..500 {
        let variances: Vec<f64> = (0..n)
            .map(|i| (0..q).map(|a| gamma[a] * z_all[i * q + a]).sum::<f64>().exp())
            .collect();
        if variances.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(TypedFitError::NotConverged(variable));
        }
        let weights: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
        let (next_theta, next_info) = fit_mean(&problem, &theta, Some(&weights), &variable)?;
        let theta_step = next_theta
            .iter()
            .zip(&theta)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0_f64, f64::max);
        theta = next_theta;
        info = next_info;
        eta = problem.eta(&theta);
        let u: Vec<f64> =
            (0..n).map(|i| (y[i] - eta[i]) * (y[i] - eta[i]) / variances[i] - 1.0).collect();
        let mut zu = vec![0.0; q];
        for i in 0..n {
            for a in 0..q {
                zu[a] += z_all[i * q + a] * u[i];
            }
        }
        let mut delta = matvec(&ztz_inverse, q, &zu);
        let largest = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
        if largest > 2.0 {
            for d in &mut delta {
                *d *= 2.0 / largest;
            }
        }
        for (g, d) in gamma.iter_mut().zip(&delta) {
            *g += d;
        }
        if largest < 1e-9 && theta_step < 1e-9 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TypedFitError::NotConverged(variable));
    }
    let inverse =
        invert_square(&info, k).ok_or_else(|| TypedFitError::Singular(variable.clone()))?;
    base.theta[..k].copy_from_slice(&theta);
    base.theta[k..].copy_from_slice(&gamma);
    for a in 0..k {
        for b in 0..k {
            base.covariance[a * total + b] = inverse[a * k + b];
        }
    }
    for a in 0..q {
        for b in 0..q {
            base.covariance[(k + a) * total + (k + b)] = 2.0 * ztz_inverse[a * q + b];
        }
    }
    Ok(())
}

/// Refit a source mechanism's specification on target data, on the source's
/// design, so the two fits' parameters are comparable.
///
/// # Errors
/// As [`fit_mechanism`].
pub fn refit_in_target(
    source: &FittedMechanism,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
) -> Result<FittedMechanism, TypedFitError> {
    fit_mechanism(&source.spec, columns, context, provenance, Some(source))
}

// --------------------------------------------------------------- transport

/// Declared dependence between the parameter estimates of the mechanisms.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterDependence {
    /// Nothing declared: a claim over more than one mechanism's parameters is refused.
    Unknown,
    /// The fits used disjoint, independently generated evidence (analyst's declaration).
    IndependentByDesign {
        /// Why independence holds.
        justification: String,
    },
    /// One aligned joint covariance of the listed mechanisms' full parameter
    /// vectors, concatenated in `variables` order.
    Joint {
        /// Mechanism variables, in covariance order.
        variables: Vec<String>,
        /// Row-major joint covariance.
        covariance: Vec<f64>,
    },
    /// Cross-covariance blocks between pairs of mechanisms; the diagonal blocks
    /// are the fits' own covariances.
    CrossCovariance {
        /// One block per declared pair.
        blocks: Vec<CrossBlock>,
    },
}

/// One declared cross-covariance block between two mechanisms' parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossBlock {
    /// First mechanism variable (rows).
    pub first: String,
    /// Second mechanism variable (columns).
    pub second: String,
    /// Row-major `dim(first) x dim(second)` block.
    pub values: Vec<f64>,
}

/// Everything a typed transported composition is built from.
#[derive(Clone, Debug)]
pub struct TypedTransportRequest {
    /// Population and regime the composition is about.
    pub target: MechanismContext,
    /// The selection diagram from the source population to the target.
    pub diagram: SelectionDiagramDeclaration,
    /// Mechanisms fitted in the source population.
    pub source_mechanisms: Vec<FittedMechanism>,
    /// Mechanisms fitted in the target population: the refits of shifted
    /// mechanisms, and consistency evidence for invariant ones.
    pub target_mechanisms: Vec<FittedMechanism>,
    /// Explicit invariance declarations (needed to carry across regimes).
    pub invariances: Vec<InvarianceDeclaration>,
    /// Declared dependence between the fits' parameter estimates.
    pub dependence: ParameterDependence,
    /// Level of the consistency test of a declared-invariant mechanism.
    pub invariance_level: f64,
}

/// Outcome of the Wald comparison of a source and a target fit of one mechanism.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvarianceCheck {
    /// Wald statistic `d' (S_source + S_target)^-1 d`.
    pub statistic: f64,
    /// Degrees of freedom (number of parameters compared).
    pub degrees_of_freedom: usize,
    /// Upper-tail chi-square probability of the statistic.
    pub p_value: f64,
    /// Level the request tested at.
    pub level: f64,
    /// Whether the declaration was rejected at `level`.
    pub rejected: bool,
}

/// Wald comparison of two fits of the same specification on the same design.
///
/// # Errors
/// [`CompositionError::Incoherent`] when the fits do not share a specification
/// or design (the parameters are then not comparable), or
/// [`CompositionError::Numerical`] when the summed covariance is singular.
pub fn invariance_check(
    source: &FittedMechanism,
    target: &FittedMechanism,
    level: f64,
) -> Result<InvarianceCheck, CompositionError> {
    if source.spec != target.spec {
        return Err(single(
            CoherenceKind::Family,
            DETAIL_TARGET_SPEC_DIFFERS,
            source.variable(),
            "refit the target mechanism with the source mechanism's family, mean and parents",
        ));
    }
    if source.scaling != target.scaling {
        return Err(single(
            CoherenceKind::Family,
            DETAIL_DESIGN_NOT_SHARED,
            source.variable(),
            "refit the target mechanism on the source mechanism's design with refit_in_target",
        ));
    }
    let d = source.theta.len();
    let diff: Vec<f64> = source.theta.iter().zip(&target.theta).map(|(a, b)| a - b).collect();
    let sum: Vec<f64> =
        source.covariance.iter().zip(&target.covariance).map(|(a, b)| a + b).collect();
    let inverse = invert_square(&sum, d).ok_or_else(|| CompositionError::Numerical {
        message: format!("summed fit covariance of `{}` is singular", source.variable()),
    })?;
    let solved = matvec(&inverse, d, &diff);
    let statistic: f64 = diff.iter().zip(&solved).map(|(a, b)| a * b).sum::<f64>().max(0.0);
    let p_value = chi_square_sf(statistic, d);
    Ok(InvarianceCheck {
        statistic,
        degrees_of_freedom: d,
        p_value,
        level,
        rejected: p_value < level,
    })
}

fn single(
    kind: CoherenceKind,
    detail: &'static str,
    subject: &str,
    remedy: &'static str,
) -> CompositionError {
    CompositionError::Incoherent {
        findings: vec![CoherenceFinding::new(kind, detail, subject.to_owned(), remedy)],
    }
}

/// One component of the transported composition and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedDisposition {
    /// The variable the mechanism produces.
    pub variable: String,
    /// Carried or refitted.
    pub role: MechanismRole,
    /// Response family of the component.
    pub family: ResponseFamily,
    /// Population the mechanism was fitted in.
    pub fitted_population: String,
    /// Owning fit.
    pub artifact_id: String,
    /// Evidence the fit used.
    pub evidence_id: String,
    /// Whose fit covariance the component's parameter uncertainty is:
    /// `source_fit` or `target_fit`.
    pub covariance_source: &'static str,
    /// Consistency test of a carried mechanism against supplied target data; `None`
    /// when no target data were supplied (the invariance is then an untested premise).
    pub consistency: Option<InvarianceCheck>,
}

/// Monte Carlo estimate of a contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastEstimate {
    /// The contrast `sum_k w_k E[outcome | do(arm_k)]`.
    pub value: f64,
    /// Monte Carlo standard error of `value` (numerical approximation only).
    pub monte_carlo_se: f64,
    /// Units per arm.
    pub draws: usize,
}

/// First-order parameter uncertainty of a contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedFunctionalUncertainty {
    /// The contrast estimate (with its Monte Carlo error).
    pub estimate: ContrastEstimate,
    /// Gradient with respect to each parameter of each mechanism involved:
    /// `(variable, parameter label, derivative)`.
    pub gradient: Vec<(String, String, f64)>,
    /// First-order parameter variance `g' Sigma g`.
    pub variance: f64,
    /// Square root of `variance`.
    pub standard_error: f64,
    /// How the dependence between mechanisms was licensed.
    pub dependence: &'static str,
    /// What the variance does and does not contain ([`UNCERTAINTY_SCOPE`]).
    pub scope: &'static str,
}

/// An executable target-population composition of typed mechanisms.
#[derive(Clone, Debug)]
pub struct TransportedTypedModel {
    mechanisms: Vec<FittedMechanism>,
    dispositions: Vec<TypedDisposition>,
    target: MechanismContext,
    diagram: SelectionDiagramDeclaration,
    dependence: ParameterDependence,
    identity: [u8; 32],
}

/// The registered reason code a typed-transport finding detail is refused under.
#[must_use]
pub fn typed_refusal_code(detail: &str) -> Option<&'static str> {
    match detail {
        DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE => {
            Some(antecedent_core::reason_code!("transport_missing_evidence"))
        }
        DETAIL_DEPENDENCE_UNKNOWN | DETAIL_DEPENDENCE_INCOMPLETE | DETAIL_DEPENDENCE_INVALID => {
            Some(antecedent_core::reason_code!("uncertainty_representation_insufficient"))
        }
        DETAIL_INVARIANCE_CONTRADICTED
        | DETAIL_TARGET_SPEC_DIFFERS
        | DETAIL_DESIGN_NOT_SHARED
        | DETAIL_PARENTS_CHANGED
        | DETAIL_REGIME_NOT_DECLARED
        | DETAIL_SELECTION_TARGET_UNKNOWN => {
            Some(antecedent_core::reason_code!("quantity_semantics_mismatch"))
        }
        DETAIL_PARENT_WITHOUT_MECHANISM
        | DETAIL_CYCLE
        | DETAIL_INVARIANCE_LEVEL
        | DETAIL_UNKNOWN_VARIABLE => {
            Some(antecedent_core::reason_code!("mechanism_composition_incoherent"))
        }
        other => crate::mechanism_transport::refusal_code(other),
    }
}

fn finding(
    kind: CoherenceKind,
    detail: &'static str,
    subject: &str,
    remedy: &'static str,
) -> CoherenceFinding {
    CoherenceFinding::new(kind, detail, subject.to_owned(), remedy)
}

/// Carry invariant typed mechanisms from the source and use target refits for
/// shifted ones, or refuse with every finding.
///
/// # Errors
/// [`CompositionError::Incoherent`] with every defect: a diagram for other
/// populations, a shifted mechanism with no target fit, a declared invariance
/// the target fit contradicts, fits that do not share a specification or design,
/// a carried mechanism that is not fitted or not from the diagram's source
/// population, a regime change without a declaration, a parent without a
/// mechanism, a cycle, and a malformed dependence declaration.
pub fn transport_typed(
    request: &TypedTransportRequest,
) -> Result<TransportedTypedModel, CompositionError> {
    let shifted: BTreeSet<String> = selection_targets(&request.diagram).into_iter().collect();
    let mut findings: Vec<CoherenceFinding> = Vec::new();

    if !(request.invariance_level > 0.0 && request.invariance_level < 0.5) {
        findings.push(finding(
            CoherenceKind::Shape,
            DETAIL_INVARIANCE_LEVEL,
            "invariance_level",
            "test declared invariances at a level strictly between 0 and 0.5",
        ));
    }
    if request.diagram.target_population != request.target.population {
        findings.push(
            finding(
                CoherenceKind::Population,
                DETAIL_DIAGRAM_TARGET_MISMATCH_LOCAL,
                "selection_diagram",
                "declare the selection diagram between the source population and this target population",
            )
            .comparing(
                request.target.population.clone(),
                request.diagram.target_population.clone(),
            ),
        );
    }

    let mut native: BTreeMap<&str, &FittedMechanism> = BTreeMap::new();
    for m in &request.target_mechanisms {
        if m.context.population != request.target.population
            || m.context.regime != request.target.regime
        {
            findings.push(
                finding(
                    CoherenceKind::Population,
                    DETAIL_TARGET_NOT_NATIVE_LOCAL,
                    m.variable(),
                    "supply as a target mechanism only a mechanism fitted in the target population and regime",
                )
                .comparing(
                    format!("{}/{}", request.target.population, request.target.regime),
                    format!("{}/{}", m.context.population, m.context.regime),
                ),
            );
        }
        if m.provenance.origin != MechanismOrigin::Fitted {
            findings.push(finding(
                CoherenceKind::Provenance,
                DETAIL_ORIGIN_NOT_FITTED_LOCAL,
                m.variable(),
                "only a mechanism estimated from retained evidence is used",
            ));
        }
        if native.insert(m.variable(), m).is_some() {
            findings.push(finding(
                CoherenceKind::VariableIdentity,
                DETAIL_DUPLICATE_LOCAL,
                m.variable(),
                "give each variable exactly one target mechanism",
            ));
        }
    }
    let mut carried: BTreeMap<&str, &FittedMechanism> = BTreeMap::new();
    for m in &request.source_mechanisms {
        if carried.insert(m.variable(), m).is_some() {
            findings.push(finding(
                CoherenceKind::VariableIdentity,
                DETAIL_DUPLICATE_LOCAL,
                m.variable(),
                "give each variable exactly one source mechanism",
            ));
        }
    }

    let mut chosen: BTreeMap<String, FittedMechanism> = BTreeMap::new();
    let mut dispositions: BTreeMap<String, TypedDisposition> = BTreeMap::new();
    let variables: BTreeSet<&str> = native
        .keys()
        .chain(carried.keys())
        .copied()
        .chain(shifted.iter().map(String::as_str))
        .collect();
    for variable in variables {
        let is_shifted = shifted.contains(variable);
        let source = carried.get(variable).copied();
        let target = native.get(variable).copied();
        let (mechanism, role, consistency) = match (is_shifted, source, target) {
            (_, None, None) => {
                findings.push(finding(
                    CoherenceKind::VariableIdentity,
                    DETAIL_SELECTION_TARGET_UNKNOWN,
                    variable,
                    "a selection node points into a variable no supplied mechanism produces",
                ));
                continue;
            }
            (true, _, None) => {
                findings.push(finding(
                    CoherenceKind::Provenance,
                    DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE,
                    variable,
                    "a selection node points into this variable, so its mechanism differs in the target; fit it on target data, it is never carried",
                ));
                continue;
            }
            (true, source, Some(target)) => {
                if let Some(source) = source {
                    if source.spec.parents != target.spec.parents {
                        findings.push(
                            finding(
                                CoherenceKind::CausalParents,
                                DETAIL_PARENTS_CHANGED,
                                variable,
                                "a shifted mechanism's target fit keeps the declared parent set",
                            )
                            .comparing(
                                source.spec.parents.join(","),
                                target.spec.parents.join(","),
                            ),
                        );
                    }
                }
                (target, MechanismRole::NativeInTarget, None)
            }
            (false, None, Some(target)) => (target, MechanismRole::NativeInTarget, None),
            (false, Some(source), target) => {
                check_carried_source(request, source, &mut findings);
                let mut check = None;
                if let Some(target) = target {
                    match invariance_check(source, target, request.invariance_level) {
                        Ok(c) => {
                            if c.rejected {
                                findings.push(
                                    finding(
                                        CoherenceKind::Population,
                                        DETAIL_INVARIANCE_CONTRADICTED,
                                        variable,
                                        "target data reject the declared invariance of this mechanism; declare a selection node into it and refit it in the target",
                                    )
                                    .comparing(
                                        format!("p >= {}", request.invariance_level),
                                        format!("p = {:.3e}", c.p_value),
                                    ),
                                );
                            }
                            check = Some(c);
                        }
                        Err(CompositionError::Incoherent { findings: f }) => findings.extend(f),
                        Err(_) => findings.push(finding(
                            CoherenceKind::UncertaintyAlignment,
                            DETAIL_DEPENDENCE_INVALID,
                            variable,
                            "the summed fit covariance of the consistency test is singular",
                        )),
                    }
                }
                (source, MechanismRole::Transported, check)
            }
        };
        chosen.insert(variable.to_owned(), mechanism.clone());
        dispositions.insert(
            variable.to_owned(),
            TypedDisposition {
                variable: variable.to_owned(),
                role,
                family: mechanism.spec.family,
                fitted_population: mechanism.context.population.clone(),
                artifact_id: mechanism.provenance.artifact_id.clone(),
                evidence_id: mechanism.provenance.evidence_id.clone(),
                covariance_source: match role {
                    MechanismRole::Transported => "source_fit",
                    MechanismRole::NativeInTarget => "target_fit",
                },
                consistency,
            },
        );
    }

    // Structure: every parent has a mechanism; acyclic; topological order.
    let mut order: Vec<String> = Vec::new();
    for (variable, m) in &chosen {
        for parent in &m.spec.parents {
            if !chosen.contains_key(parent) {
                findings.push(finding(
                    CoherenceKind::VariableIdentity,
                    DETAIL_PARENT_WITHOUT_MECHANISM,
                    &format!("{parent}->{variable}"),
                    "supply a source or target mechanism for every parent",
                ));
            }
        }
    }
    if findings.is_empty() {
        let mut placed: BTreeSet<&str> = BTreeSet::new();
        while placed.len() < chosen.len() {
            let ready: Vec<&String> = chosen
                .iter()
                .filter(|(v, m)| {
                    !placed.contains(v.as_str())
                        && m.spec.parents.iter().all(|p| placed.contains(p.as_str()))
                })
                .map(|(v, _)| v)
                .collect();
            if ready.is_empty() {
                findings.push(finding(
                    CoherenceKind::Cycle,
                    DETAIL_CYCLE,
                    "structure",
                    "a typed transported composition is acyclic",
                ));
                break;
            }
            for v in ready {
                placed.insert(v.as_str());
                order.push(v.clone());
            }
        }
    }
    validate_dependence(&request.dependence, &chosen, &mut findings);

    if !findings.is_empty() {
        findings
            .sort_by(|a, b| (a.kind, a.detail, &a.subject).cmp(&(b.kind, b.detail, &b.subject)));
        findings.dedup();
        return Err(CompositionError::Incoherent { findings });
    }

    let mechanisms: Vec<FittedMechanism> = order.iter().map(|v| chosen[v].clone()).collect();
    let dispositions: Vec<TypedDisposition> = dispositions.into_values().collect();
    let identity = identity_of(request, &mechanisms, &dispositions);
    Ok(TransportedTypedModel {
        mechanisms,
        dispositions,
        target: request.target.clone(),
        diagram: request.diagram.clone(),
        dependence: request.dependence.clone(),
        identity,
    })
}

const DETAIL_DIAGRAM_TARGET_MISMATCH_LOCAL: &str =
    crate::mechanism_transport::DETAIL_DIAGRAM_TARGET_MISMATCH;
const DETAIL_TARGET_NOT_NATIVE_LOCAL: &str = crate::mechanism_transport::DETAIL_TARGET_NOT_NATIVE;
const DETAIL_ORIGIN_NOT_FITTED_LOCAL: &str = crate::mechanism_transport::DETAIL_ORIGIN_NOT_FITTED;
const DETAIL_DUPLICATE_LOCAL: &str = crate::mechanism_transport::DETAIL_DUPLICATE_MECHANISM;
const DETAIL_SOURCE_MISMATCH_LOCAL: &str =
    crate::mechanism_transport::DETAIL_DIAGRAM_SOURCE_MISMATCH;

fn check_carried_source(
    request: &TypedTransportRequest,
    source: &FittedMechanism,
    out: &mut Vec<CoherenceFinding>,
) {
    if source.provenance.origin != MechanismOrigin::Fitted {
        out.push(finding(
            CoherenceKind::Provenance,
            DETAIL_ORIGIN_NOT_FITTED_LOCAL,
            source.variable(),
            "only a mechanism estimated from retained evidence is carried",
        ));
    }
    if source.context.population != request.diagram.source_population {
        out.push(
            finding(
                CoherenceKind::Population,
                DETAIL_SOURCE_MISMATCH_LOCAL,
                source.variable(),
                "declare the selection diagram between the population this mechanism was fitted in and the target",
            )
            .comparing(
                request.diagram.source_population.clone(),
                source.context.population.clone(),
            ),
        );
    }
    if source.context.regime != request.target.regime {
        let declared = request.invariances.iter().any(|d| {
            d.variable == source.variable()
                && d.coordinate == ContextCoordinate::Regime
                && d.source == source.context.regime
                && d.target == request.target.regime
        });
        if !declared {
            out.push(
                finding(
                    CoherenceKind::Regime,
                    DETAIL_REGIME_NOT_DECLARED,
                    source.variable(),
                    "declare the invariance of this mechanism across the two regimes explicitly",
                )
                .comparing(request.target.regime.clone(), source.context.regime.clone()),
            );
        }
    }
}

fn validate_dependence(
    dependence: &ParameterDependence,
    chosen: &BTreeMap<String, FittedMechanism>,
    out: &mut Vec<CoherenceFinding>,
) {
    let bad = |subject: &str, remedy: &'static str| {
        finding(CoherenceKind::PosteriorDependence, DETAIL_DEPENDENCE_INVALID, subject, remedy)
    };
    match dependence {
        ParameterDependence::Unknown => {}
        ParameterDependence::IndependentByDesign { justification } => {
            if justification.trim().is_empty() {
                out.push(bad("independence", "state why the fits are independent"));
            }
        }
        ParameterDependence::Joint { variables, covariance } => {
            let mut dim = 0;
            let mut seen = BTreeSet::new();
            for v in variables {
                match chosen.get(v) {
                    Some(m) if seen.insert(v.as_str()) => dim += m.theta.len(),
                    _ => out
                        .push(bad(v, "a joint covariance lists each model mechanism at most once")),
                }
            }
            if covariance.len() != dim * dim || !valid_covariance(covariance, dim) {
                out.push(bad("joint", "a joint covariance is symmetric positive semidefinite and aligned to the listed parameters"));
            }
        }
        ParameterDependence::CrossCovariance { blocks } => {
            let mut seen = BTreeSet::new();
            for b in blocks {
                let pair =
                    if b.first <= b.second { (&b.first, &b.second) } else { (&b.second, &b.first) };
                if !seen.insert(pair) {
                    out.push(bad(&b.first, "declare each unordered cross-covariance pair once"));
                }
                match (chosen.get(&b.first), chosen.get(&b.second)) {
                    (Some(a), Some(c)) if b.first != b.second => {
                        if b.values.len() != a.theta.len() * c.theta.len()
                            || b.values.iter().any(|v| !v.is_finite())
                        {
                            out.push(bad(
                                &b.first,
                                "a cross-covariance block is dim(first) x dim(second) and finite",
                            ));
                        }
                    }
                    _ => out.push(bad(
                        &b.first,
                        "a cross-covariance block names two distinct model mechanisms",
                    )),
                }
            }
        }
    }
}

fn write_mechanism(w: &mut CanonicalWriter, m: &FittedMechanism) {
    w.text(m.variable()).word(m.spec.family.code());
    w.texts(&m.spec.parents);
    match &m.spec.mean {
        MeanSpec::Polynomial { degree, interactions } => {
            w.word(1).word(u64::from(*degree)).word(u64::from(*interactions));
        }
        MeanSpec::Expression { expression, start } => {
            w.word(2).text(&format!("{expression:?}")).reals(start);
        }
    }
    w.word(m.scaling.len() as u64);
    for (c, s) in &m.scaling {
        w.real(*c).real(*s);
    }
    w.word(m.n_obs as u64).reals(&m.theta).reals(&m.covariance);
    w.text(&m.context.population).text(&m.context.regime);
    w.text(&m.provenance.artifact_id).text(&m.provenance.evidence_id);
}

fn identity_of(
    request: &TypedTransportRequest,
    mechanisms: &[FittedMechanism],
    dispositions: &[TypedDisposition],
) -> [u8; 32] {
    let mut w = CanonicalWriter::new(TAG_IDENTITY);
    w.word(u64::from(TYPED_TRANSPORT_VERSION));
    w.text(&request.target.population).text(&request.target.regime);
    w.real(request.invariance_level);
    let mut sorted: Vec<&FittedMechanism> = mechanisms.iter().collect();
    sorted.sort_by(|a, b| a.variable().cmp(b.variable()));
    w.word(sorted.len() as u64);
    for m in sorted {
        let role =
            dispositions.iter().find(|d| d.variable == m.variable()).map_or(0, |d| match d.role {
                MechanismRole::Transported => 1,
                MechanismRole::NativeInTarget => 2,
            });
        w.word(role);
        write_mechanism(&mut w, m);
    }
    w.word(request.invariances.len() as u64);
    let mut invariances: Vec<&InvarianceDeclaration> = request.invariances.iter().collect();
    invariances.sort_by(|a, b| {
        (&a.variable, a.coordinate, &a.source, &a.target).cmp(&(
            &b.variable,
            b.coordinate,
            &b.source,
            &b.target,
        ))
    });
    for i in invariances {
        w.text(&i.variable).word(i.coordinate as u64).text(&i.source).text(&i.target);
        w.text(&i.assumption_id);
    }
    w.text(&request.diagram.source_population).text(&request.diagram.target_population);
    let mut nodes: Vec<(&String, Vec<&String>)> = request
        .diagram
        .selection_nodes
        .iter()
        .map(|n| {
            let mut into: Vec<&String> = n.points_into.iter().collect();
            into.sort();
            (&n.name, into)
        })
        .collect();
    nodes.sort();
    w.word(nodes.len() as u64);
    for (name, into) in nodes {
        w.text(name).word(into.len() as u64);
        for v in into {
            w.text(v);
        }
    }
    match &request.dependence {
        ParameterDependence::Unknown => {
            w.word(0);
        }
        ParameterDependence::IndependentByDesign { justification } => {
            w.word(1).text(justification);
        }
        ParameterDependence::Joint { variables, covariance } => {
            w.word(2).texts(variables).reals(covariance);
        }
        ParameterDependence::CrossCovariance { blocks } => {
            w.word(3).word(blocks.len() as u64);
            for b in blocks {
                w.text(&b.first).text(&b.second).reals(&b.values);
            }
        }
    }
    w.finish()
}

// --------------------------------------------------------------- execution

struct ContrastRun {
    value: f64,
    monte_carlo_se: f64,
    gradient: Vec<f64>,
    involved: BTreeSet<usize>,
}

impl TransportedTypedModel {
    /// Every component, sorted by variable.
    #[must_use]
    pub fn dispositions(&self) -> &[TypedDisposition] {
        &self.dispositions
    }

    /// The mechanisms in topological order.
    #[must_use]
    pub fn mechanisms(&self) -> &[FittedMechanism] {
        &self.mechanisms
    }

    /// The target population and regime.
    #[must_use]
    pub fn target(&self) -> &MechanismContext {
        &self.target
    }

    /// The selection diagram the model was admitted under.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagramDeclaration {
        &self.diagram
    }

    /// Variables whose mechanism was carried from the source, sorted.
    #[must_use]
    pub fn transported_variables(&self) -> Vec<&str> {
        self.dispositions
            .iter()
            .filter(|d| d.role == MechanismRole::Transported)
            .map(|d| d.variable.as_str())
            .collect()
    }

    /// Identity of the transported composition. Equal requests give equal
    /// identities; a changed number at an unchanged shape changes it.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Lowercase hexadecimal [`Self::identity`].
    #[must_use]
    pub fn identity_hex(&self) -> String {
        antecedent_core::identity::to_hex(&self.identity)
    }

    fn offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.mechanisms.len());
        let mut acc = 0;
        for m in &self.mechanisms {
            out.push(acc);
            acc += m.theta.len();
        }
        out
    }

    fn total_dimension(&self) -> usize {
        self.mechanisms.iter().map(|m| m.theta.len()).sum()
    }

    /// `sum_k w_k E[outcome | do(arm_k)]` by Monte Carlo over `draws` paired units per arm.
    ///
    /// # Errors
    /// [`CompositionError::Incoherent`] for an unknown variable or a malformed
    /// contrast; [`CompositionError::Unsupported`] for a draw count outside
    /// `1..=`[`MAX_TYPED_DRAWS`]; [`CompositionError::Numerical`] for a
    /// non-finite draw; [`CompositionError::Cancelled`].
    pub fn mean_contrast(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<ContrastEstimate, CompositionError> {
        let run = self.run_contrast(contrast, draws, ctx, false)?;
        Ok(ContrastEstimate { value: run.value, monte_carlo_se: run.monte_carlo_se, draws })
    }

    /// First-order parameter uncertainty of a contrast, scoped honestly.
    ///
    /// # Errors
    /// As [`Self::mean_contrast`]; additionally a refusal when more than one
    /// mechanism's parameters are involved and the declared dependence is
    /// unknown or omits a pair of them.
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<TypedFunctionalUncertainty, CompositionError> {
        let run = self.run_contrast(contrast, draws, ctx, true)?;
        let (sigma, dependence) = self.parameter_covariance(&run.involved)?;
        let d = run.gradient.len();
        let solved = matvec(&sigma, d, &run.gradient);
        let terms: Vec<f64> = run.gradient.iter().zip(&solved).map(|(a, b)| a * b).collect();
        let variance: f64 = terms.iter().sum();
        let magnitude: f64 = terms.iter().map(|x| x.abs()).sum();
        let tolerance = 64.0 * f64::EPSILON * d as f64 * magnitude;
        if !variance.is_finite() || !magnitude.is_finite() || variance < -tolerance {
            return Err(single(
                CoherenceKind::PosteriorDependence,
                DETAIL_DEPENDENCE_INVALID,
                "parameter_dependence",
                "use a finite positive semidefinite parameter covariance",
            ));
        }
        let variance = variance.max(0.0);
        let offsets = self.offsets();
        let mut gradient = Vec::new();
        for &idx in &run.involved {
            let m = &self.mechanisms[idx];
            for (j, label) in m.parameter_labels().into_iter().enumerate() {
                gradient.push((m.variable().to_owned(), label, run.gradient[offsets[idx] + j]));
            }
        }
        Ok(TypedFunctionalUncertainty {
            estimate: ContrastEstimate {
                value: run.value,
                monte_carlo_se: run.monte_carlo_se,
                draws,
            },
            gradient,
            variance,
            standard_error: variance.sqrt(),
            dependence,
            scope: UNCERTAINTY_SCOPE,
        })
    }

    fn parameter_covariance(
        &self,
        involved: &BTreeSet<usize>,
    ) -> Result<(Vec<f64>, &'static str), CompositionError> {
        let d = self.total_dimension();
        let offsets = self.offsets();
        let mut sigma = vec![0.0; d * d];
        for (idx, m) in self.mechanisms.iter().enumerate() {
            let k = m.theta.len();
            for a in 0..k {
                for b in 0..k {
                    sigma[(offsets[idx] + a) * d + offsets[idx] + b] = m.covariance[a * k + b];
                }
            }
        }
        let index_of = |v: &str| self.mechanisms.iter().position(|m| m.variable() == v);
        let label = match &self.dependence {
            ParameterDependence::Unknown => {
                if involved.len() > 1 {
                    return Err(single(
                        CoherenceKind::PosteriorDependence,
                        DETAIL_DEPENDENCE_UNKNOWN,
                        "parameter_dependence",
                        "declare independence by design, a joint covariance, or cross-covariance blocks",
                    ));
                }
                "unknown_single_mechanism"
            }
            ParameterDependence::IndependentByDesign { .. } => "independent_by_design",
            ParameterDependence::Joint { variables, covariance } => {
                let listed: Vec<usize> = variables.iter().filter_map(|v| index_of(v)).collect();
                if involved.iter().any(|i| !listed.contains(i)) {
                    return Err(single(
                        CoherenceKind::PosteriorDependence,
                        DETAIL_DEPENDENCE_INCOMPLETE,
                        "joint",
                        "the joint covariance lists every mechanism whose parameters the contrast involves",
                    ));
                }
                let joint_dim: usize = listed.iter().map(|&i| self.mechanisms[i].theta.len()).sum();
                let mut joint_offsets = Vec::with_capacity(listed.len());
                let mut acc = 0;
                for &i in &listed {
                    joint_offsets.push(acc);
                    acc += self.mechanisms[i].theta.len();
                }
                for (pa, &ia) in listed.iter().enumerate() {
                    for (pb, &ib) in listed.iter().enumerate() {
                        for a in 0..self.mechanisms[ia].theta.len() {
                            for b in 0..self.mechanisms[ib].theta.len() {
                                sigma[(offsets[ia] + a) * d + offsets[ib] + b] = covariance
                                    [(joint_offsets[pa] + a) * joint_dim + joint_offsets[pb] + b];
                            }
                        }
                    }
                }
                "joint"
            }
            ParameterDependence::CrossCovariance { blocks } => {
                let involved_list: Vec<usize> = involved.iter().copied().collect();
                for (pos, &ia) in involved_list.iter().enumerate() {
                    for &ib in &involved_list[pos + 1..] {
                        let (va, vb) =
                            (self.mechanisms[ia].variable(), self.mechanisms[ib].variable());
                        let (ka, kb) =
                            (self.mechanisms[ia].theta.len(), self.mechanisms[ib].theta.len());
                        let direct = blocks.iter().find(|b| b.first == va && b.second == vb);
                        let reverse = blocks.iter().find(|b| b.first == vb && b.second == va);
                        match (direct, reverse) {
                            (Some(b), _) => {
                                for a in 0..ka {
                                    for c in 0..kb {
                                        let v = b.values[a * kb + c];
                                        sigma[(offsets[ia] + a) * d + offsets[ib] + c] = v;
                                        sigma[(offsets[ib] + c) * d + offsets[ia] + a] = v;
                                    }
                                }
                            }
                            (None, Some(b)) => {
                                for a in 0..kb {
                                    for c in 0..ka {
                                        let v = b.values[a * ka + c];
                                        sigma[(offsets[ib] + a) * d + offsets[ia] + c] = v;
                                        sigma[(offsets[ia] + c) * d + offsets[ib] + a] = v;
                                    }
                                }
                            }
                            (None, None) => {
                                return Err(single(
                                    CoherenceKind::PosteriorDependence,
                                    DETAIL_DEPENDENCE_INCOMPLETE,
                                    &format!("{va},{vb}"),
                                    "declare a cross-covariance block for every pair of involved mechanisms",
                                ));
                            }
                        }
                    }
                }
                "cross_covariance"
            }
        };
        let indices: Vec<usize> = involved
            .iter()
            .flat_map(|&i| offsets[i]..offsets[i] + self.mechanisms[i].theta.len())
            .collect();
        if !indices.is_empty() {
            let principal: Vec<f64> = indices
                .iter()
                .flat_map(|&i| indices.iter().map(move |&j| (i, j)))
                .map(|(i, j)| sigma[i * d + j])
                .collect();
            if !valid_covariance(&principal, indices.len()) {
                return Err(single(
                    CoherenceKind::PosteriorDependence,
                    DETAIL_DEPENDENCE_INVALID,
                    "parameter_dependence",
                    "use a symmetric positive semidefinite parameter covariance",
                ));
            }
        }
        Ok((sigma, label))
    }

    fn run_contrast(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
        with_gradient: bool,
    ) -> Result<ContrastRun, CompositionError> {
        if draws == 0 || draws > MAX_TYPED_DRAWS {
            return Err(CompositionError::Unsupported {
                message: format!("draw count {draws} outside 1..={MAX_TYPED_DRAWS}"),
            });
        }
        if contrast.arms.is_empty() || contrast.arms.iter().any(|a| !a.weight.is_finite()) {
            return Err(single(
                CoherenceKind::Shape,
                DETAIL_UNKNOWN_VARIABLE,
                "contrast",
                "a contrast has at least one arm and finite weights",
            ));
        }
        let index_of: BTreeMap<&str, usize> =
            self.mechanisms.iter().enumerate().map(|(i, m)| (m.variable(), i)).collect();
        let outcome = *index_of.get(contrast.outcome.as_str()).ok_or_else(|| {
            single(
                CoherenceKind::VariableIdentity,
                DETAIL_UNKNOWN_VARIABLE,
                &contrast.outcome,
                "the outcome is a variable of the transported model",
            )
        })?;
        let offsets = self.offsets();
        let total = self.total_dimension();
        let parent_indices: Vec<Vec<usize>> = self
            .mechanisms
            .iter()
            .map(|m| {
                m.spec.parents.iter().filter_map(|p| index_of.get(p.as_str()).copied()).collect()
            })
            .collect();

        let mut arm_values: Vec<Vec<f64>> = Vec::with_capacity(contrast.arms.len());
        let mut gradient = vec![0.0; total];
        let mut involved: BTreeSet<usize> = BTreeSet::new();

        for arm in &contrast.arms {
            let mut fixed: BTreeMap<usize, f64> = BTreeMap::new();
            for (name, value) in &arm.interventions {
                let idx = *index_of.get(name.as_str()).ok_or_else(|| {
                    single(
                        CoherenceKind::VariableIdentity,
                        DETAIL_UNKNOWN_VARIABLE,
                        name,
                        "an intervention names a variable of the transported model",
                    )
                })?;
                if !value.is_finite() || fixed.insert(idx, *value).is_some() {
                    return Err(single(
                        CoherenceKind::Shape,
                        DETAIL_UNKNOWN_VARIABLE,
                        name,
                        "an intervention value is finite and each variable is set once per arm",
                    ));
                }
            }
            // Relevant nodes: ancestors of the outcome, not looking past an intervened node.
            let mut relevant: BTreeSet<usize> = BTreeSet::new();
            let mut stack = vec![outcome];
            while let Some(node) = stack.pop() {
                if !relevant.insert(node) {
                    continue;
                }
                if !fixed.contains_key(&node) {
                    stack.extend(parent_indices[node].iter().copied());
                }
            }
            let order: Vec<usize> =
                (0..self.mechanisms.len()).filter(|i| relevant.contains(i)).collect();
            if !fixed.contains_key(&outcome) {
                for &i in &order {
                    if !fixed.contains_key(&i) {
                        involved.insert(i);
                    }
                }
            }
            let seeds: Vec<u64> = (0..self.mechanisms.len())
                .map(|i| {
                    ctx.rng
                        .stream_for(
                            TYPED_TRANSPORT_STREAM_DOMAIN,
                            noise_stream_index(0, self.mechanisms[i].variable()),
                        )
                        .next_u64()
                })
                .collect();

            let mut values = vec![0.0; self.mechanisms.len()];
            let mut f_units = Vec::with_capacity(draws);
            let mut sum_fs = vec![0.0; total];
            let mut sum_s = vec![0.0; total];
            let mut sum_pathwise = vec![0.0; total];
            let mut unit_scores = vec![0.0; total];
            for unit in 0..draws {
                if unit % 4096 == 4095 && ctx.cancellation.is_cancelled() {
                    return Err(CompositionError::Cancelled);
                }
                if with_gradient {
                    unit_scores.iter_mut().for_each(|s| *s = 0.0);
                }
                let mut f = f64::NAN;
                for &node in &order {
                    if let Some(v) = fixed.get(&node) {
                        values[node] = *v;
                        if node == outcome {
                            f = *v;
                        }
                        continue;
                    }
                    let m = &self.mechanisms[node];
                    let raw: Vec<f64> = parent_indices[node].iter().map(|&p| values[p]).collect();
                    if node == outcome {
                        f = m.mean_with(&m.theta, &raw);
                        if with_gradient {
                            let mut shifted = m.theta.clone();
                            for j in 0..m.theta.len() {
                                let h = 1e-5 * (1.0 + m.theta[j].abs());
                                shifted[j] = m.theta[j] + h;
                                let up = m.mean_with(&shifted, &raw);
                                shifted[j] = m.theta[j] - h;
                                let down = m.mean_with(&shifted, &raw);
                                shifted[j] = m.theta[j];
                                sum_pathwise[offsets[node] + j] += (up - down) / (2.0 * h);
                            }
                        }
                        continue;
                    }
                    let mut rng = CausalRng::from_seed(mix64(
                        seeds[node].wrapping_add(mix64((unit as u64).wrapping_add(1))),
                    ));
                    let x = m.sample_with(&m.theta, &raw, &mut rng);
                    if !x.is_finite() {
                        return Err(CompositionError::Numerical {
                            message: format!("non-finite draw of `{}`", m.variable()),
                        });
                    }
                    values[node] = x;
                    if with_gradient {
                        let mut shifted = m.theta.clone();
                        for j in 0..m.theta.len() {
                            let h = 1e-5 * (1.0 + m.theta[j].abs());
                            shifted[j] = m.theta[j] + h;
                            let up = m.log_density_with(&shifted, &raw, x);
                            shifted[j] = m.theta[j] - h;
                            let down = m.log_density_with(&shifted, &raw, x);
                            shifted[j] = m.theta[j];
                            unit_scores[offsets[node] + j] = (up - down) / (2.0 * h);
                        }
                    }
                }
                if !f.is_finite() {
                    return Err(CompositionError::Numerical {
                        message: format!("non-finite outcome for `{}`", contrast.outcome),
                    });
                }
                if with_gradient {
                    for j in 0..total {
                        sum_fs[j] += f * unit_scores[j];
                        sum_s[j] += unit_scores[j];
                    }
                }
                f_units.push(f);
            }
            if with_gradient {
                let n = draws as f64;
                let mean_f = f_units.iter().sum::<f64>() / n;
                for j in 0..total {
                    gradient[j] +=
                        arm.weight * (sum_fs[j] / n - mean_f * sum_s[j] / n + sum_pathwise[j] / n);
                }
            }
            arm_values.push(f_units);
        }

        let n = draws as f64;
        let combos: Vec<f64> = (0..draws)
            .map(|i| contrast.arms.iter().zip(&arm_values).map(|(a, v)| a.weight * v[i]).sum())
            .collect();
        let value = combos.iter().sum::<f64>() / n;
        let monte_carlo_se = if draws > 1 {
            (combos.iter().map(|c| (c - value) * (c - value)).sum::<f64>() / (n - 1.0) / n).sqrt()
        } else {
            f64::NAN
        };
        Ok(ContrastRun { value, monte_carlo_se, gradient, involved })
    }
}

#[cfg(test)]
mod convergence_regressions {
    use super::*;

    #[test]
    fn exhausted_damping_with_nonfinite_objective_is_not_convergence() {
        let spec = MechanismSpec {
            variable: "y".into(),
            parents: vec![],
            mean: MeanSpec::Polynomial { degree: 1, interactions: false },
            family: ResponseFamily::Gaussian,
        };
        let columns = BTreeMap::from([("y".into(), vec![1e200, -1e200, 1e200, -1e200])]);
        let result = fit_mechanism(
            &spec,
            &columns,
            MechanismContext { population: "p".into(), regime: "observational".into() },
            MechanismProvenance {
                origin: MechanismOrigin::Fitted,
                artifact_id: "a".into(),
                evidence_id: "e".into(),
            },
            None,
        );
        assert!(matches!(result, Err(TypedFitError::NotConverged(_))));
    }
}
