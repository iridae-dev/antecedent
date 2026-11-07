//! Joint-law propagation arithmetic: what a quantity computed from several
//! uncertain inputs is, given what is known about their dependence.
//!
//! Three kinds of information support three different computations, and this
//! module never lets one stand in for another.
//!
//! * **A joint moment.** For a linear combination `sum a_i X_i` the mean needs
//!   only the means and the variance needs the covariance matrix: the
//!   delta method is exact for a linear map ([`linear_combination_moments`]).
//!   A mixture's mean and variance need only the known mixture weights and the
//!   component moments, by the law of total variance ([`mixture_moments`]).
//! * **An aligned joint law.** A product, a ratio, a minimum or any other
//!   nonlinear functional of two inputs is an expectation under their *joint*
//!   law. Draws are row-paired only when the rows are realizations of one joint
//!   law ([`aligned_functional_mean`]); pairing the rows of unrelated marginal
//!   draws by index silently assumes independence and is a different claim.
//!   The caller proves alignment (an alignment identity shared by every
//!   input); this module only does the arithmetic once it is proven.
//! * **Nothing but the marginals.** The set of values a functional can take
//!   over every coupling of two known marginals is an exact structural
//!   envelope (Fréchet 1951, Hoeffding 1940): for a supermodular or
//!   submodular value function the extremes of `E[g(X, Y)]` are attained at
//!   the comonotone and countermonotone couplings of the marginals' quantile
//!   functions (Lorentz 1953, Tchen 1980, Rüschendorf 1980), and the extremes
//!   of `P(X + Y <= t)` are the Makarov (1982) / Rüschendorf (1982) bounds
//!   ([`frechet_expectation_envelope`], [`sum_cdf_envelope`]). An envelope is
//!   neither a point, an interval nor a posterior and it is not a substitute
//!   for an aligned joint law: it states what no dependence assumption could
//!   exceed.
//!
//! A declared correlation range gives a second, weaker kind of range
//! ([`correlation_range_envelope`]); it is an assumption range, labelled as
//! such, and says nothing about what the data support.
//!
//! # Composed spread, interval candidates and decisions
//!
//! A composed quantity's spread is read at its *composed coordinate*: the draws
//! or moments of the composition, not of its components. [`aligned_linear_draws`],
//! [`aligned_functional_draws`] and [`aligned_map_draws`] form the composed rows
//! of columns the caller has proven to be one joint law (a linear transform, a
//! contrast or weighted sum, a product, a nonlinear functional, or the last
//! stage of a chain of Monte Carlo transformations); [`draw_spread`] summarises
//! them, [`mixture_marginal`] and [`mixture_exceedance`] do the same for a
//! mixture with known weights, and [`linear_combination_moments`] gives the
//! delta-method variance of a linear map under a covariance.
//!
//! [`normal_interval_candidate`], [`draw_interval_candidate`] and
//! [`marginal_interval_candidate`] are interval *constructions* at the composed
//! coordinate and [`composed_exceedance_decision`] is a decision about a
//! composed exceedance probability with its Monte Carlo band. Each carries the
//! calibration status [`CALIBRATION_STATUS`] (`not_measured`): a construction is
//! not a coverage statement, and the joint-law rule reports the interval
//! reading with that status until a record measured at that exact coordinate and
//! construction exists.
//!
//! No coverage statement or posterior is produced by anything here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CancellationToken;
use thiserror::Error;

use crate::synthesis::{SynthesisError, check_covariance};
use antecedent_core::named_variants;

mod marginal;
mod projection;
pub use marginal::{
    Marginal, covariance_envelope, difference_cdf_envelope, frechet_expectation_envelope,
    functional_envelope, linear_variance_envelope, product_moment_envelope, sum_cdf_envelope,
};

pub use projection::{
    DRAW_EXCHANGEABLE_RANK_CONSTRUCTION, DRAW_INTERPOLATED_CONSTRUCTION, DecisionVerdict,
    ExceedanceDecision, IntervalCandidate, MIXTURE_EQUAL_TAIL_CONSTRUCTION,
    NORMAL_APPROXIMATION_CONSTRUCTION, composed_exceedance_decision, draw_interval_candidate,
    marginal_interval_candidate, mixture_exceedance, mixture_marginal, normal_interval_candidate,
};

/// Label of a spread computed under a supported dependence route.
pub const UNCALIBRATED_SPREAD_LABEL: &str = "uncalibrated_spread";
/// Calibration status of every interval candidate and decision verdict of this
/// module: the construction is wired, and no coverage record of the composed
/// coordinate exists until one is measured and attested.
pub const CALIBRATION_STATUS: &str = "not_measured";
/// Label of a Fréchet-Hoeffding envelope over every coupling of the marginals.
pub use antecedent_core::STRUCTURAL_ENVELOPE_LABEL;
/// Label of the range implied by a declared correlation range.
pub const CORRELATION_RANGE_LABEL: &str = crate::synthesis::DEPENDENCE_ENVELOPE_LABEL;
/// Most atoms of one marginal.
pub const MAX_MARGINAL_ATOMS: usize = 100_000;
/// Most adjacent cells of the atom lattice a value function is verified on.
pub const MAX_LATTICE_CHECKS: usize = 4_000_000;
/// Most terms of a linear combination or components of a mixture.
pub const MAX_COMPONENTS: usize = 256;

const MASS_TOLERANCE: f64 = 1e-9;
const CANCEL_STRIDE: usize = 4096;
const STRUCTURE_RELATIVE_SLACK: f64 = 1e-12;

/// Why a propagation computation was refused.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum JointLawError {
    /// Lengths or counts do not agree.
    #[error("joint-law shape error: {0}")]
    Shape(&'static str),
    /// An input is NaN or infinite.
    #[error("joint-law input is not finite: {0}")]
    NonFinite(&'static str),
    /// Masses or weights are not a probability vector.
    #[error("invalid weights: {0}")]
    InvalidWeights(&'static str),
    /// A variance or standard deviation is negative.
    #[error("invalid variance: {0}")]
    InvalidVariance(&'static str),
    /// A covariance matrix was refused (not symmetric or not positive
    /// semidefinite).
    #[error("covariance refused: {0}")]
    Covariance(SynthesisError),
    /// A declared correlation range is not inside `[-1, 1]` or is degenerate.
    #[error("invalid correlation range: {0}")]
    InvalidRange(&'static str),
    /// A value function was not supermodular (or submodular) on the product of
    /// the marginals' supports, so the coupling extremes are not its bounds.
    #[error("the value function is not {structure} on the marginals' support")]
    StructureNotVerified {
        /// `supermodular` or `submodular`.
        structure: &'static str,
    },
    /// The value function returned a non-finite value on the support.
    #[error("the value function is not finite on the marginals' support")]
    NonFiniteValue,
    /// A level or probability is not strictly inside `(0, 1)`, or an effective
    /// size is not in `(0, n]`.
    #[error("invalid level: {0}")]
    InvalidLevel(&'static str),
    /// A declared bound was exceeded.
    #[error("bound exceeded: {0}")]
    BudgetExceeded(&'static str),
    /// Cancellation was observed at a work boundary.
    #[error("joint-law computation cancelled")]
    Cancelled,
}

impl From<SynthesisError> for JointLawError {
    fn from(error: SynthesisError) -> Self {
        Self::Covariance(error)
    }
}

fn require_finite(values: &[f64], what: &'static str) -> Result<(), JointLawError> {
    if values.iter().any(|v| !v.is_finite()) { Err(JointLawError::NonFinite(what)) } else { Ok(()) }
}

fn check_cancel(cancel: &CancellationToken) -> Result<(), JointLawError> {
    if cancel.is_cancelled() { Err(JointLawError::Cancelled) } else { Ok(()) }
}

// ------------------------------------------------------------- bounds value

/// Lower and upper extreme of a quantity over a declared set of dependence
/// structures. Not a point, an interval or a posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueBounds {
    /// Smallest value over the set.
    pub lower: f64,
    /// Largest value over the set.
    pub upper: f64,
}

impl ValueBounds {
    fn ordered(a: f64, b: f64) -> Self {
        Self { lower: a.min(b), upper: a.max(b) }
    }

    /// Whether the envelope is a single value (the quantity does not depend on
    /// the coupling at all).
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.lower.total_cmp(&self.upper).is_eq()
    }
}

// ----------------------------------------------------------- linear moments

/// Mean and variance of a linear combination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearMoments {
    /// `sum a_i mu_i`.
    pub mean: f64,
    /// `a' Sigma a` under the supplied covariance.
    pub variance: f64,
}

/// Mean and variance of `sum a_i X_i` under a supplied covariance matrix.
///
/// The delta method is exact for a linear map: the mean is `a' mu` and the
/// variance is `a' Sigma a`. `covariance` is the row-major `k x k` covariance
/// of the inputs, including its cross terms; a diagonal matrix is the
/// statement that the inputs are uncorrelated and is the caller's declaration.
///
/// # Errors
/// Shape mismatch, non-finite inputs, or a covariance that is not symmetric
/// positive semidefinite.
pub fn linear_combination_moments(
    coefficients: &[f64],
    means: &[f64],
    covariance: &[f64],
) -> Result<LinearMoments, JointLawError> {
    let k = coefficients.len();
    if k == 0 || k > MAX_COMPONENTS {
        return Err(JointLawError::Shape("terms outside 1..=MAX_COMPONENTS"));
    }
    if means.len() != k {
        return Err(JointLawError::Shape("coefficients and means differ in length"));
    }
    require_finite(coefficients, "coefficient")?;
    require_finite(means, "mean")?;
    check_covariance(covariance, k)?;
    let mean: f64 = coefficients.iter().zip(means).map(|(a, m)| a * m).sum();
    let mut variance = 0.0;
    for i in 0..k {
        for j in 0..k {
            variance += coefficients[i] * coefficients[j] * covariance[i * k + j];
        }
    }
    if !mean.is_finite() || !variance.is_finite() {
        return Err(JointLawError::NonFinite("combination moment"));
    }
    // A positive semidefinite covariance gives a non-negative quadratic form;
    // negative values here are rounding of that form and are not repaired
    // beyond zero.
    Ok(LinearMoments { mean, variance: variance.max(0.0) })
}

// ---------------------------------------------------------------- mixtures

/// Mean and variance of a finite mixture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixtureMoments {
    /// `sum w_i mu_i`.
    pub mean: f64,
    /// Total variance, `within + between`.
    pub variance: f64,
    /// `sum w_i v_i`: the weighted component variances.
    pub within: f64,
    /// `sum w_i (mu_i - mean)^2`: the spread of the component means.
    pub between: f64,
}

/// Mean and variance of a mixture with known weights, by the law of total
/// variance. Components need no joint law: a mixture selects one component.
///
/// # Errors
/// Shape mismatch, non-finite inputs, weights that are negative or do not sum
/// to one, or a negative component variance.
pub fn mixture_moments(
    weights: &[f64],
    means: &[f64],
    variances: &[f64],
) -> Result<MixtureMoments, JointLawError> {
    let k = weights.len();
    if k == 0 || k > MAX_COMPONENTS {
        return Err(JointLawError::Shape("components outside 1..=MAX_COMPONENTS"));
    }
    if means.len() != k || variances.len() != k {
        return Err(JointLawError::Shape("weights, means and variances differ in length"));
    }
    require_finite(weights, "weight")?;
    require_finite(means, "component mean")?;
    require_finite(variances, "component variance")?;
    if weights.iter().any(|&w| w < 0.0) {
        return Err(JointLawError::InvalidWeights("weights must be non-negative"));
    }
    let total: f64 = weights.iter().sum();
    if (total - 1.0).abs() > MASS_TOLERANCE {
        return Err(JointLawError::InvalidWeights("weights must sum to one"));
    }
    if variances.iter().any(|&v| v < 0.0) {
        return Err(JointLawError::InvalidVariance("component variance is negative"));
    }
    let mean: f64 = weights.iter().zip(means).map(|(w, m)| w * m).sum();
    let within: f64 = weights.iter().zip(variances).map(|(w, v)| w * v).sum();
    let between: f64 = weights.iter().zip(means).map(|(w, m)| w * (m - mean) * (m - mean)).sum();
    let variance = within + between;
    if !variance.is_finite() {
        return Err(JointLawError::NonFinite("mixture moment"));
    }
    Ok(MixtureMoments { mean, variance, within, between })
}

// ------------------------------------------------------- closed functionals

/// Whether the mixed second difference of a value function has one sign.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairStructure {
    /// `g(x2, y2) + g(x1, y1) >= g(x2, y1) + g(x1, y2)` for `x1 < x2`, `y1 < y2`.
    Supermodular,
    /// The reverse inequality.
    Submodular,
}

impl PairStructure {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supermodular => "supermodular",
            Self::Submodular => "submodular",
        }
    }
}

/// A closed set of two-argument value functions. A functional named here is
/// replayable from its name alone; arbitrary callbacks are not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueFunctional {
    /// `x * y`.
    Product,
    /// `x / y`, refused where `y` is zero.
    Ratio,
    /// `min(x, y)`.
    Minimum,
    /// `max(x, y)`.
    Maximum,
    /// `(x - y)^2`.
    SquaredDifference,
}

named_variants! {
    ValueFunctional {
        Product => "product",
        Ratio => "ratio",
        Minimum => "minimum",
        Maximum => "maximum",
        SquaredDifference => "squared_difference",
    }
}

impl ValueFunctional {
    /// The function's value.
    #[must_use]
    pub fn value(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Product => x * y,
            Self::Ratio => x / y,
            Self::Minimum => x.min(y),
            Self::Maximum => x.max(y),
            Self::SquaredDifference => (x - y) * (x - y),
        }
    }

    /// The mixed-difference sign the function is expected to have. It is
    /// verified on the lattice of the marginals' atoms before any coupling
    /// extreme is reported as a bound.
    #[must_use]
    pub const fn structure(self) -> PairStructure {
        match self {
            Self::Product | Self::Minimum => PairStructure::Supermodular,
            Self::Ratio | Self::Maximum | Self::SquaredDifference => PairStructure::Submodular,
        }
    }
}

// ---------------------------------------------------------- aligned draws

/// Population moments of two aligned draw columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedMoments {
    /// Number of rows.
    pub n: usize,
    /// Mean of the first column.
    pub mean_x: f64,
    /// Mean of the second column.
    pub mean_y: f64,
    /// Population variance of the first column.
    pub var_x: f64,
    /// Population variance of the second column.
    pub var_y: f64,
    /// Population covariance of the two columns.
    pub covariance: f64,
}

fn check_columns(x: &[f64], y: &[f64]) -> Result<(), JointLawError> {
    if x.is_empty() {
        return Err(JointLawError::Shape("a draw column needs at least one row"));
    }
    if x.len() != y.len() {
        return Err(JointLawError::Shape("aligned columns differ in length"));
    }
    require_finite(x, "draw")?;
    require_finite(y, "draw")
}

/// Population moments of two columns of rows that the caller has established
/// are realizations of one joint law.
///
/// # Errors
/// Empty or unequal columns, or non-finite draws.
pub fn paired_moments(x: &[f64], y: &[f64]) -> Result<PairedMoments, JointLawError> {
    check_columns(x, y)?;
    let n = x.len();
    let nf = n as f64;
    let mean_x = x.iter().sum::<f64>() / nf;
    let mean_y = y.iter().sum::<f64>() / nf;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (da, db) = (a - mean_x, b - mean_y);
        sxx += da * da;
        syy += db * db;
        sxy += da * db;
    }
    Ok(PairedMoments { n, mean_x, mean_y, var_x: sxx / nf, var_y: syy / nf, covariance: sxy / nf })
}

/// Mean of `functional(x_i, y_i)` over aligned rows.
///
/// # Errors
/// Empty or unequal columns, non-finite draws or function values (a ratio with
/// a zero denominator), or cancellation.
pub fn aligned_functional_mean(
    x: &[f64],
    y: &[f64],
    functional: ValueFunctional,
    cancel: &CancellationToken,
) -> Result<f64, JointLawError> {
    check_columns(x, y)?;
    let mut total = 0.0;
    for (row, (a, b)) in x.iter().zip(y).enumerate() {
        if row % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        let value = functional.value(*a, *b);
        if !value.is_finite() {
            return Err(JointLawError::NonFiniteValue);
        }
        total += value;
    }
    Ok(total / x.len() as f64)
}

// --------------------------------------------------------------- marginals

// ------------------------------------------------- composed spread and draws

/// Rows of `sum a_i X_i` for columns that are realizations of one joint law.
///
/// The caller proves alignment (one alignment identity for every column); this
/// function only forms the composed draws.
///
/// # Errors
/// No columns, more than [`MAX_COMPONENTS`], unequal or empty columns,
/// non-finite coefficients or draws, or a non-finite composed row.
pub fn aligned_linear_draws(
    coefficients: &[f64],
    columns: &[&[f64]],
) -> Result<Vec<f64>, JointLawError> {
    let k = coefficients.len();
    if k == 0 || k > MAX_COMPONENTS {
        return Err(JointLawError::Shape("terms outside 1..=MAX_COMPONENTS"));
    }
    if columns.len() != k {
        return Err(JointLawError::Shape("coefficients and columns differ in count"));
    }
    require_finite(coefficients, "coefficient")?;
    let n = columns[0].len();
    if n == 0 {
        return Err(JointLawError::Shape("a draw column needs at least one row"));
    }
    for column in columns {
        if column.len() != n {
            return Err(JointLawError::Shape("aligned columns differ in length"));
        }
        require_finite(column, "draw")?;
    }
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        let value: f64 = coefficients.iter().zip(columns).map(|(a, c)| a * c[row]).sum();
        if !value.is_finite() {
            return Err(JointLawError::NonFiniteValue);
        }
        rows.push(value);
    }
    Ok(rows)
}

/// Rows of `functional(x_i, y_i)` for aligned columns.
///
/// # Errors
/// Empty or unequal columns, non-finite draws or function values, or
/// cancellation.
pub fn aligned_functional_draws(
    x: &[f64],
    y: &[f64],
    functional: ValueFunctional,
    cancel: &CancellationToken,
) -> Result<Vec<f64>, JointLawError> {
    check_columns(x, y)?;
    let mut rows = Vec::with_capacity(x.len());
    for (row, (a, b)) in x.iter().zip(y).enumerate() {
        if row % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        let value = functional.value(*a, *b);
        if !value.is_finite() {
            return Err(JointLawError::NonFiniteValue);
        }
        rows.push(value);
    }
    Ok(rows)
}

/// Rows of an arbitrary map of the aligned columns: a nonlinear functional of
/// several inputs, or the last stage of a chain of Monte Carlo
/// transformations whose earlier stages the caller already applied row by row.
///
/// The map receives the row's values in column order. A chain is aligned only
/// when each stage consumed the rows of the stage before it; a stage that
/// re-drew its inputs breaks the alignment and its rows must not be passed
/// here.
///
/// # Errors
/// No columns, more than [`MAX_COMPONENTS`], unequal or empty columns,
/// non-finite draws or composed values, or cancellation.
pub fn aligned_map_draws(
    columns: &[&[f64]],
    map: &dyn Fn(&[f64]) -> f64,
    cancel: &CancellationToken,
) -> Result<Vec<f64>, JointLawError> {
    let k = columns.len();
    if k == 0 || k > MAX_COMPONENTS {
        return Err(JointLawError::Shape("columns outside 1..=MAX_COMPONENTS"));
    }
    let n = columns[0].len();
    if n == 0 {
        return Err(JointLawError::Shape("a draw column needs at least one row"));
    }
    for column in columns {
        if column.len() != n {
            return Err(JointLawError::Shape("aligned columns differ in length"));
        }
        require_finite(column, "draw")?;
    }
    let mut buffer = vec![0.0; k];
    let mut rows = Vec::with_capacity(n);
    for row in 0..n {
        if row % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        for (slot, column) in buffer.iter_mut().zip(columns) {
            *slot = column[row];
        }
        let value = map(&buffer);
        if !value.is_finite() {
            return Err(JointLawError::NonFiniteValue);
        }
        rows.push(value);
    }
    Ok(rows)
}

/// The spread of a column of composed draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawSpread {
    /// Number of rows.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Sample variance (divisor `n - 1`).
    pub variance: f64,
    /// Square root of the variance.
    pub standard_deviation: f64,
    /// Monte Carlo standard error of the mean, `sd / sqrt(size)` with the
    /// effective size given (the row count when none was).
    pub mean_standard_error: f64,
}

/// Spread of composed draws.
///
/// `effective_size`, when given, is the effective number of independent rows
/// (autocorrelated or weighted draws); the rows are treated as independent when
/// it is `None`. The spread is the spread of the composed quantity's law
/// represented by the draws; whether that law is a sampling distribution, a
/// posterior or a bootstrap distribution is the producer's declaration and is
/// not decided here.
///
/// # Errors
/// Fewer than two rows, non-finite draws, or an effective size outside `(0, n]`.
pub fn draw_spread(
    draws: &[f64],
    effective_size: Option<f64>,
) -> Result<DrawSpread, JointLawError> {
    let n = draws.len();
    if n < 2 {
        return Err(JointLawError::Shape("a spread needs at least two rows"));
    }
    require_finite(draws, "draw")?;
    let nf = n as f64;
    let size = match effective_size {
        None => nf,
        Some(s) if s.is_finite() && s > 0.0 && s <= nf => s,
        Some(_) => return Err(JointLawError::InvalidLevel("effective size is in (0, n]")),
    };
    let mean = draws.iter().sum::<f64>() / nf;
    let ss: f64 = draws.iter().map(|d| (d - mean) * (d - mean)).sum();
    let variance = ss / (nf - 1.0);
    let standard_deviation = variance.sqrt();
    let mean_standard_error = (variance / size).sqrt();
    if ![mean, variance, standard_deviation, mean_standard_error].into_iter().all(f64::is_finite) {
        return Err(JointLawError::NonFiniteValue);
    }
    Ok(DrawSpread { n, mean, variance, standard_deviation, mean_standard_error })
}

// --------------------------------------------------- declared correlation

/// Moment summaries a declared correlation range is applied to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentSummary {
    /// Mean.
    pub mean: f64,
    /// Standard deviation (non-negative).
    pub standard_deviation: f64,
}

/// Ranges of `E[XY]`, `Cov(X, Y)` and `Var(a X + b Y)` as the correlation
/// ranges over a declared interval. An assumption range, never an interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationRangeEnvelope {
    /// Declared lower correlation.
    pub rho_lo: f64,
    /// Declared upper correlation.
    pub rho_hi: f64,
    /// `E[XY] = mu_x mu_y + rho s_x s_y`.
    pub product_moment: ValueBounds,
    /// `Cov = rho s_x s_y`.
    pub covariance: ValueBounds,
    /// `Var(a X + b Y)`.
    pub linear_variance: ValueBounds,
}

/// The ranges implied by a declared correlation range `[rho_lo, rho_hi]`.
///
/// Each quantity is linear in the correlation, so the extremes are at the
/// endpoints. The range is a declaration about the dependence, not something
/// the data establish; it is not intersected with the correlations the
/// marginals can attain.
///
/// # Errors
/// A negative or non-finite standard deviation, or a range that is not
/// `-1 <= lo < hi <= 1`.
pub fn correlation_range_envelope(
    x: MomentSummary,
    y: MomentSummary,
    a: f64,
    b: f64,
    rho_lo: f64,
    rho_hi: f64,
) -> Result<CorrelationRangeEnvelope, JointLawError> {
    require_finite(
        &[x.mean, x.standard_deviation, y.mean, y.standard_deviation, a, b, rho_lo, rho_hi],
        "correlation-range input",
    )?;
    if x.standard_deviation < 0.0 || y.standard_deviation < 0.0 {
        return Err(JointLawError::InvalidVariance("standard deviation is negative"));
    }
    if rho_lo < -1.0 || rho_hi > 1.0 || rho_lo >= rho_hi {
        return Err(JointLawError::InvalidRange(
            "the range needs -1 <= lo < hi <= 1; a single correlation is a covariance declaration",
        ));
    }
    let scale = x.standard_deviation * y.standard_deviation;
    let centre = x.mean * y.mean;
    let base = a * a * x.standard_deviation * x.standard_deviation
        + b * b * y.standard_deviation * y.standard_deviation;
    let at = |rho: f64| (centre + rho * scale, rho * scale, base + 2.0 * a * b * rho * scale);
    let (p_lo, c_lo, v_lo) = at(rho_lo);
    let (p_hi, c_hi, v_hi) = at(rho_hi);
    if ![scale, centre, base, p_lo, c_lo, v_lo, p_hi, c_hi, v_hi].into_iter().all(f64::is_finite) {
        return Err(JointLawError::NonFiniteValue);
    }
    Ok(CorrelationRangeEnvelope {
        rho_lo,
        rho_hi,
        product_moment: ValueBounds::ordered(p_lo, p_hi),
        covariance: ValueBounds::ordered(c_lo, c_hi),
        linear_variance: ValueBounds::ordered(v_lo.max(0.0), v_hi.max(0.0)),
    })
}

#[cfg(test)]
mod tests;
