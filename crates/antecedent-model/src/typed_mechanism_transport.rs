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

use antecedent_core::{CausalRng, StreamDomain};
use antecedent_kernels::standard_normal;
use antecedent_stats::gram::invert_square;
use antecedent_stats::special::{chi_square_sf, ln_gamma};

use crate::compose::coherence::{CoherenceFinding, valid_covariance};
use crate::compose::{
    CoherenceKind, CompositionError, ContextCoordinate, InvarianceDeclaration, MechanismContext,
    MechanismOrigin, MechanismProvenance, SelectionDiagramDeclaration,
};
use crate::mechanism_expr::{MechanismExpression, MechanismTape};
pub use crate::mechanism_transport::{MechanismRole, UNCERTAINTY_SCOPE, selection_targets};

mod admission;
mod evaluation;
mod fitting;
pub use admission::transport_typed;
use fitting::matvec;
pub use fitting::{fit_mechanism, fit_mechanism_polled, refit_in_target, refit_in_target_polled};

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
    antecedent_core::reason_details::typed_mechanism_transport::SHIFTED_WITHOUT_TARGET_EVIDENCE;
/// The target fit of a mechanism declared invariant rejects the declaration.
pub const DETAIL_INVARIANCE_CONTRADICTED: &str =
    antecedent_core::reason_details::typed_mechanism_transport::INVARIANCE_CONTRADICTED;
/// A target fit has another family, mean or parent set than the source mechanism it checks.
pub const DETAIL_TARGET_SPEC_DIFFERS: &str =
    antecedent_core::reason_details::typed_mechanism_transport::TARGET_SPECIFICATION_DIFFERS;
/// A target fit was not fitted on the source mechanism's design, so the parameters are not comparable.
pub const DETAIL_DESIGN_NOT_SHARED: &str =
    antecedent_core::reason_details::typed_mechanism_transport::DESIGN_NOT_SHARED;
/// A shifted mechanism's target fit changes the parent set of the source mechanism.
pub const DETAIL_PARENTS_CHANGED: &str =
    antecedent_core::reason_details::typed_mechanism_transport::PARENTS_CHANGED;
/// A mechanism names a parent that has no mechanism in the transported model.
pub const DETAIL_PARENT_WITHOUT_MECHANISM: &str =
    antecedent_core::reason_details::typed_mechanism_transport::PARENT_WITHOUT_MECHANISM;
/// A selection node points into a variable no supplied mechanism produces.
pub const DETAIL_SELECTION_TARGET_UNKNOWN: &str =
    antecedent_core::reason_details::typed_mechanism_transport::SELECTION_TARGET_UNKNOWN_VARIABLE;
/// A carried mechanism was fitted in another regime without an explicit invariance declaration.
pub const DETAIL_REGIME_NOT_DECLARED: &str =
    antecedent_core::reason_details::typed_mechanism_transport::REGIME_NOT_DECLARED_INVARIANT;
/// The consistency-test level is outside `(0, 0.5)`.
pub const DETAIL_INVARIANCE_LEVEL: &str =
    antecedent_core::reason_details::typed_mechanism_transport::INVARIANCE_LEVEL_INVALID;
/// The declared parameter dependence is malformed.
pub const DETAIL_DEPENDENCE_INVALID: &str =
    antecedent_core::reason_details::typed_mechanism_transport::DEPENDENCE_INVALID;
/// The structure has a directed cycle.
pub const DETAIL_CYCLE: &str = antecedent_core::reason_details::typed_mechanism_transport::CYCLE;
/// A claim over more than one mechanism's parameters has no declared dependence.
pub const DETAIL_DEPENDENCE_UNKNOWN: &str =
    antecedent_core::reason_details::typed_mechanism_transport::PARAMETER_DEPENDENCE_UNKNOWN;
/// The declared dependence omits a pair of involved mechanisms.
pub const DETAIL_DEPENDENCE_INCOMPLETE: &str =
    antecedent_core::reason_details::typed_mechanism_transport::PARAMETER_DEPENDENCE_INCOMPLETE;
/// A request names a variable the transported model does not have.
pub const DETAIL_UNKNOWN_VARIABLE: &str =
    antecedent_core::reason_details::typed_mechanism_transport::UNKNOWN_VARIABLE;

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
    /// The caller cancelled the fit; nothing is concluded about the data.
    #[error("the fit was cancelled")]
    Cancelled,
}

impl TypedFitError {
    /// The registered reason code this failure is refused under.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::NotConverged(_) | Self::Singular(_) => {
                antecedent_core::reason_code!("mechanism_fit_not_converged")
            }
            Self::Cancelled => antecedent_core::reason_code!("cancelled_no_claim"),
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
    if !lambda.is_finite() || !(0.0..=1.0e5).contains(&lambda) {
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
    /// How the dependence between mechanisms was supported.
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
