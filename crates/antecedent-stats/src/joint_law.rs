//! Joint-law propagation arithmetic: what a quantity computed from several
//! uncertain inputs is, given what is known about their dependence.
//!
//! Three kinds of information license three different computations, and this
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
//! No interval, coverage statement or posterior is produced by anything here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CancellationToken;
use thiserror::Error;

use crate::synthesis::{SynthesisError, check_covariance};

/// Label of a Fréchet-Hoeffding envelope over every coupling of the marginals.
pub const STRUCTURAL_ENVELOPE_LABEL: &str = "structural_envelope";
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

impl ValueFunctional {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Product => "product",
            Self::Ratio => "ratio",
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
            Self::SquaredDifference => "squared_difference",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Product, Self::Ratio, Self::Minimum, Self::Maximum, Self::SquaredDifference]
            .into_iter()
            .find(|f| f.as_str() == name)
    }

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

/// A finite marginal law: distinct ascending atoms with positive masses that
/// sum to one. Draws are equal-mass atoms.
#[derive(Clone, Debug, PartialEq)]
pub struct Marginal {
    atoms: Vec<(f64, f64)>,
    cumulative: Vec<f64>,
}

impl Marginal {
    /// A marginal from draws, each with mass `1 / n`.
    ///
    /// # Errors
    /// Empty input, more than [`MAX_MARGINAL_ATOMS`] draws, or non-finite values.
    pub fn from_draws(draws: &[f64]) -> Result<Self, JointLawError> {
        let masses = vec![1.0 / draws.len().max(1) as f64; draws.len()];
        Self::from_weighted(draws, &masses)
    }

    /// A marginal from values and masses; repeated values are merged.
    ///
    /// # Errors
    /// Empty or unequal inputs, more than [`MAX_MARGINAL_ATOMS`] atoms,
    /// non-finite values, negative masses or masses not summing to one.
    pub fn from_weighted(values: &[f64], masses: &[f64]) -> Result<Self, JointLawError> {
        if values.is_empty() {
            return Err(JointLawError::Shape("a marginal needs at least one atom"));
        }
        if values.len() != masses.len() {
            return Err(JointLawError::Shape("values and masses differ in length"));
        }
        if values.len() > MAX_MARGINAL_ATOMS {
            return Err(JointLawError::BudgetExceeded("more atoms than the marginal bound"));
        }
        require_finite(values, "marginal value")?;
        require_finite(masses, "marginal mass")?;
        if masses.iter().any(|&m| m < 0.0) {
            return Err(JointLawError::InvalidWeights("masses must be non-negative"));
        }
        // Canonical order (value, then mass) before any summation, so the
        // atoms, their masses and every bound derived from them are the same
        // bits for any listing order of the input.
        let mut pairs: Vec<(f64, f64)> = values
            .iter()
            .zip(masses)
            .filter(|(_, m)| **m > 0.0)
            // `+ 0.0` turns negative zero into positive zero.
            .map(|(v, m)| (v + 0.0, *m))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let total: f64 = pairs.iter().map(|(_, m)| m).sum();
        if (total - 1.0).abs() > MASS_TOLERANCE {
            return Err(JointLawError::InvalidWeights("masses must sum to one"));
        }
        for pair in &mut pairs {
            pair.1 /= total;
        }
        let mut atoms: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
        for (value, mass) in pairs {
            match atoms.last_mut() {
                Some(last) if last.0.total_cmp(&value).is_eq() => last.1 += mass,
                _ => atoms.push((value, mass)),
            }
        }
        let mut cumulative = Vec::with_capacity(atoms.len());
        let mut running = 0.0;
        for (_, mass) in &atoms {
            running += mass;
            cumulative.push(running);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }
        Ok(Self { atoms, cumulative })
    }

    /// Ascending `(value, mass)` atoms.
    #[must_use]
    pub fn atoms(&self) -> &[(f64, f64)] {
        &self.atoms
    }

    /// Mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.atoms.iter().map(|(v, m)| v * m).sum()
    }

    /// Population variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        self.atoms.iter().map(|(v, m)| m * (v - mean) * (v - mean)).sum()
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn mass_at_most(&self, x: f64) -> f64 {
        match self.atoms.partition_point(|a| a.0 <= x) {
            0 => 0.0,
            idx => self.cumulative[idx - 1],
        }
    }

    /// `P(X < x)`.
    #[must_use]
    pub fn mass_below(&self, x: f64) -> f64 {
        match self.atoms.partition_point(|a| a.0 < x) {
            0 => 0.0,
            idx => self.cumulative[idx - 1],
        }
    }

    /// The marginal of `-X`.
    #[must_use]
    pub fn negated(&self) -> Self {
        let values: Vec<f64> = self.atoms.iter().rev().map(|(v, _)| -v).collect();
        let masses: Vec<f64> = self.atoms.iter().rev().map(|(_, m)| *m).collect();
        // The mass vector sums to one by construction, so this cannot refuse.
        Self::from_weighted(&values, &masses).unwrap_or_else(|_| self.clone())
    }
}

// ------------------------------------------------------- coupling extremes

fn coupled_expectation(
    x: &[(f64, f64)],
    y: &[(f64, f64)],
    value: &dyn Fn(f64, f64) -> f64,
    cancel: &CancellationToken,
) -> Result<f64, JointLawError> {
    let (mut i, mut j) = (0_usize, 0_usize);
    let (mut rx, mut ry) = (x[0].1, y[0].1);
    let mut terms: Vec<f64> = Vec::new();
    let mut steps = 0_usize;
    loop {
        steps += 1;
        if steps % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        let m = rx.min(ry);
        let v = value(x[i].0, y[j].0);
        if !v.is_finite() {
            return Err(JointLawError::NonFiniteValue);
        }
        terms.push(m * v);
        rx -= m;
        ry -= m;
        // One of `rx` and `ry` is exactly the selected minimum, so its
        // subtraction reaches zero. Do not use a probability tolerance here:
        // even a tiny positive atom is part of the declared marginal and can
        // change an allegedly exact coupling envelope.
        if rx <= 0.0 {
            i += 1;
            if i == x.len() {
                break;
            }
            rx = x[i].1;
        }
        if ry <= 0.0 {
            j += 1;
            if j == y.len() {
                break;
            }
            ry = y[j].1;
        }
    }
    // Sum the cell contributions in a total order so the value does not depend
    // on the direction the coupling was walked.
    terms.sort_by(f64::total_cmp);
    Ok(terms.iter().sum())
}

/// `E[g(X, Y)]` under the comonotone and countermonotone couplings.
fn monotone_couplings(
    x: &Marginal,
    y: &Marginal,
    value: &dyn Fn(f64, f64) -> f64,
    cancel: &CancellationToken,
) -> Result<(f64, f64), JointLawError> {
    let comonotone = coupled_expectation(&x.atoms, &y.atoms, value, cancel)?;
    // The countermonotone coupling is one coupling however it is walked; walk it
    // from the canonically smaller marginal so that swapping the two arguments
    // visits the same cells in the same order with the same masses.
    let swap = x
        .atoms
        .iter()
        .zip(&y.atoms)
        .map(|(a, b)| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))
        .find(|o| o.is_ne())
        .unwrap_or_else(|| x.atoms.len().cmp(&y.atoms.len()))
        == std::cmp::Ordering::Greater;
    let countermonotone = if swap {
        let reversed: Vec<(f64, f64)> = x.atoms.iter().rev().copied().collect();
        coupled_expectation(&y.atoms, &reversed, &|a, b| value(b, a), cancel)?
    } else {
        let reversed: Vec<(f64, f64)> = y.atoms.iter().rev().copied().collect();
        coupled_expectation(&x.atoms, &reversed, value, cancel)?
    };
    Ok((comonotone, countermonotone))
}

/// Verify the mixed second difference of `value` on every adjacent cell of the
/// product of the atom lattices.
///
/// Adjacent-cell sign is equivalent to the global lattice property on a product
/// of chains, so this is a complete check, not a sample.
fn verify_structure(
    x: &Marginal,
    y: &Marginal,
    value: &dyn Fn(f64, f64) -> f64,
    structure: PairStructure,
    cancel: &CancellationToken,
) -> Result<(), JointLawError> {
    let (nx, ny) = (x.atoms.len(), y.atoms.len());
    if (nx.saturating_sub(1)).saturating_mul(ny.saturating_sub(1)) > MAX_LATTICE_CHECKS {
        return Err(JointLawError::BudgetExceeded(
            "the atom lattice is larger than the structure-verification bound; coarsen the marginals",
        ));
    }
    let row = |xv: f64| -> Result<Vec<f64>, JointLawError> {
        y.atoms
            .iter()
            .map(|(yv, _)| {
                let v = value(xv, *yv);
                if v.is_finite() { Ok(v) } else { Err(JointLawError::NonFiniteValue) }
            })
            .collect()
    };
    let mut lower_row = row(x.atoms[0].0)?;
    for &(x_value, _) in x.atoms.iter().skip(1) {
        check_cancel(cancel)?;
        let upper_row = row(x_value)?;
        for j in 1..ny {
            let (a, b, c, d) = (upper_row[j], lower_row[j - 1], lower_row[j], upper_row[j - 1]);
            let mixed = (a + b) - (c + d);
            let slack = STRUCTURE_RELATIVE_SLACK * (a.abs() + b.abs() + c.abs() + d.abs());
            let violated = match structure {
                PairStructure::Supermodular => mixed < -slack,
                PairStructure::Submodular => mixed > slack,
            };
            if violated {
                return Err(JointLawError::StructureNotVerified { structure: structure.as_str() });
            }
        }
        lower_row = upper_row;
    }
    Ok(())
}

/// Exact extremes of `E[g(X, Y)]` over every coupling of the two marginals,
/// for a value function with a verified mixed-difference sign.
///
/// For a supermodular `g` the largest value is attained at the comonotone
/// coupling and the smallest at the countermonotone coupling; for a
/// submodular `g` the roles swap (Lorentz 1953; Tchen 1980). Both couplings are
/// built from the marginals' quantile functions and evaluated exactly on the
/// finite atoms. The sign of the mixed difference is checked on the whole atom
/// lattice first, and the call refuses when it does not hold.
///
/// # Errors
/// A structure that is not verified, a non-finite value, a lattice above
/// [`MAX_LATTICE_CHECKS`], or cancellation.
pub fn frechet_expectation_envelope(
    x: &Marginal,
    y: &Marginal,
    value: &dyn Fn(f64, f64) -> f64,
    structure: PairStructure,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    verify_structure(x, y, value, structure, cancel)?;
    let (co, counter) = monotone_couplings(x, y, value, cancel)?;
    Ok(ValueBounds::ordered(co, counter))
}

/// [`frechet_expectation_envelope`] for one of the closed value functionals.
///
/// # Errors
/// As [`frechet_expectation_envelope`].
pub fn functional_envelope(
    x: &Marginal,
    y: &Marginal,
    functional: ValueFunctional,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    frechet_expectation_envelope(
        x,
        y,
        &|a, b| functional.value(a, b),
        functional.structure(),
        cancel,
    )
}

/// Extremes of `E[X Y]` over every coupling. `x * y` is supermodular on any
/// lattice, so no lattice verification is needed and the cost is linear in the
/// number of atoms.
///
/// # Errors
/// A non-finite product or cancellation.
pub fn product_moment_envelope(
    x: &Marginal,
    y: &Marginal,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    let (co, counter) = monotone_couplings(x, y, &|a, b| a * b, cancel)?;
    Ok(ValueBounds::ordered(co, counter))
}

/// Extremes of `Cov(X, Y)` over every coupling.
///
/// # Errors
/// As [`product_moment_envelope`].
pub fn covariance_envelope(
    x: &Marginal,
    y: &Marginal,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    let product = product_moment_envelope(x, y, cancel)?;
    let centre = x.mean() * y.mean();
    let lower = product.lower - centre;
    let upper = product.upper - centre;
    if !(centre.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(JointLawError::NonFiniteValue);
    }
    Ok(ValueBounds { lower, upper })
}

/// Extremes of `Var(a X + b Y)` over every coupling.
///
/// `Var = a^2 Var X + b^2 Var Y + 2 a b Cov`, linear in the covariance, whose
/// extremes are attained at the comonotone and countermonotone couplings, so
/// the extremes are exact for every sum, difference and weighted sum of two
/// marginals.
///
/// # Errors
/// Non-finite coefficients, or as [`product_moment_envelope`].
pub fn linear_variance_envelope(
    a: f64,
    b: f64,
    x: &Marginal,
    y: &Marginal,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    require_finite(&[a, b], "coefficient")?;
    let cov = covariance_envelope(x, y, cancel)?;
    let base = a * a * x.variance() + b * b * y.variance();
    let cross = 2.0 * a * b;
    let (lo, hi) = (base + cross * cov.lower, base + cross * cov.upper);
    if !(base.is_finite() && cross.is_finite() && lo.is_finite() && hi.is_finite()) {
        return Err(JointLawError::NonFiniteValue);
    }
    let bounds = ValueBounds::ordered(lo, hi);
    Ok(ValueBounds { lower: bounds.lower.max(0.0), upper: bounds.upper.max(0.0) })
}

/// Exact extremes of `P(X + Y <= t)` over every coupling of the marginals
/// (Makarov 1982; Rüschendorf 1982):
/// `sup_x max(F(x) + G(t - x) - 1, 0)` and `inf_x min(F(x) + G(t - x), 1)` with
/// the right-continuous distribution functions `F` and `G`.
///
/// Both functions of `x` are step functions; the extreme is evaluated exactly
/// at each atom of either marginal (its value, the value just before and the
/// value just after), which covers every constancy interval.
///
/// # Errors
/// A non-finite threshold.
pub fn sum_cdf_envelope(
    x: &Marginal,
    y: &Marginal,
    threshold: f64,
) -> Result<ValueBounds, JointLawError> {
    if !threshold.is_finite() {
        return Err(JointLawError::NonFinite("threshold"));
    }
    let mut lower = 0.0_f64;
    let mut upper = 1.0_f64;
    let mut take = |value: f64| {
        lower = lower.max(value - 1.0);
        upper = upper.min(value);
    };
    for &(a, _) in &x.atoms {
        let (f_at, f_left) = (x.mass_at_most(a), x.mass_below(a));
        let u = threshold - a;
        let (g_at, g_after) = (y.mass_at_most(u), y.mass_below(u));
        take(f_at + g_at);
        take(f_at + g_after);
        take(f_left + g_at);
    }
    for &(b, _) in &y.atoms {
        let p = threshold - b;
        let (f_at, f_left) = (x.mass_at_most(p), x.mass_below(p));
        let (g_at, g_after) = (y.mass_at_most(b), y.mass_below(b));
        take(f_at + g_at);
        take(f_at + g_after);
        take(f_left + g_at);
    }
    Ok(ValueBounds { lower: lower.clamp(0.0, 1.0), upper: upper.clamp(0.0, 1.0) })
}

/// Exact extremes of `P(X - Y <= t)` over every coupling.
///
/// # Errors
/// As [`sum_cdf_envelope`].
pub fn difference_cdf_envelope(
    x: &Marginal,
    y: &Marginal,
    threshold: f64,
) -> Result<ValueBounds, JointLawError> {
    sum_cdf_envelope(x, &y.negated(), threshold)
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
mod tests {
    use antecedent_core::CausalRng;

    use super::*;

    fn token() -> CancellationToken {
        CancellationToken::new()
    }

    fn normal_pair(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = CausalRng::from_seed(seed);
        let mut z = Vec::with_capacity(n);
        while z.len() < 2 * n {
            let u1 = 1.0 - rng.next_f64();
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let t = 2.0 * std::f64::consts::PI * u2;
            z.push(r * t.cos());
            z.push(r * t.sin());
        }
        let (z1, z2): (Vec<f64>, Vec<f64>) = z.chunks(2).map(|c| (c[0], c[1])).unzip();
        (z1, z2[..n].to_vec())
    }

    #[test]
    fn linear_variance_of_a_contrast_is_the_closed_form() {
        // Var(X - Y) = sx^2 + sy^2 - 2 cov, with sx^2 = 4, sy^2 = 9, cov = 3.
        let covariance = [4.0, 3.0, 3.0, 9.0];
        let moments =
            linear_combination_moments(&[1.0, -1.0], &[5.0, 2.0], &covariance).expect("moments");
        assert!((moments.mean - 3.0).abs() < 1e-15);
        assert!((moments.variance - (4.0 + 9.0 - 6.0)).abs() < 1e-12);
        // Weighted sum 0.25 X + 0.75 Y.
        let w = linear_combination_moments(&[0.25, 0.75], &[5.0, 2.0], &covariance).expect("w");
        let expected = 0.0625 * 4.0 + 0.5625 * 9.0 + 2.0 * 0.25 * 0.75 * 3.0;
        assert!((w.variance - expected).abs() < 1e-12);
        assert!((w.mean - (1.25 + 1.5)).abs() < 1e-15);
    }

    #[test]
    fn a_covariance_that_is_not_positive_semidefinite_is_refused() {
        let covariance = [1.0, 2.0, 2.0, 1.0];
        let error =
            linear_combination_moments(&[1.0, 1.0], &[0.0, 0.0], &covariance).expect_err("refuse");
        assert!(matches!(
            error,
            JointLawError::Covariance(SynthesisError::NotPositiveSemidefinite { .. })
        ));
    }

    #[test]
    fn known_weight_mixture_matches_the_closed_form() {
        // Components N(0, 1) and N(4, 4) with weights 0.25 and 0.75.
        let m = mixture_moments(&[0.25, 0.75], &[0.0, 4.0], &[1.0, 4.0]).expect("mixture");
        assert!((m.mean - 3.0).abs() < 1e-15);
        assert!((m.within - (0.25 + 3.0)).abs() < 1e-15);
        assert!((m.between - (0.25 * 9.0 + 0.75 * 1.0)).abs() < 1e-14);
        // Second moment route: E[X^2] = 0.25 (1) + 0.75 (4 + 16) = 15.25.
        assert!((m.variance - (15.25 - 9.0)).abs() < 1e-13);
        assert!(matches!(
            mixture_moments(&[0.5, 0.4], &[0.0, 1.0], &[1.0, 1.0]),
            Err(JointLawError::InvalidWeights(_))
        ));
    }

    #[test]
    fn aligned_draws_recover_the_product_moment_and_unpaired_draws_do_not() {
        let (mu_x, mu_y, s_x, s_y, rho) = (1.0, 2.0, 1.5, 0.5, 0.6);
        let n = 200_000;
        let (z1, z2) = normal_pair(20_240_917, n);
        let x: Vec<f64> = z1.iter().map(|z| mu_x + s_x * z).collect();
        let y: Vec<f64> = z1
            .iter()
            .zip(&z2)
            .map(|(a, b)| mu_y + s_y * (rho * a + (1.0 - rho * rho).sqrt() * b))
            .collect();
        let truth = rho * s_x * s_y + mu_x * mu_y;
        let var = mu_x * mu_x * s_y * s_y
            + mu_y * mu_y * s_x * s_x
            + 2.0 * mu_x * mu_y * rho * s_x * s_y
            + s_x * s_x * s_y * s_y * (1.0 + rho * rho);
        let band = 5.0 * (var / n as f64).sqrt();
        let aligned =
            aligned_functional_mean(&x, &y, ValueFunctional::Product, &token()).expect("mean");
        assert!((aligned - truth).abs() < band, "aligned {aligned} truth {truth} band {band}");
        // The same columns shuffled apart by an independent permutation behave
        // as unrelated marginals: the product moment is mu_x mu_y, outside the
        // band of the joint truth.
        let shifted: Vec<f64> = y.iter().cycle().skip(7_919).take(n).copied().collect();
        let paired_by_index =
            aligned_functional_mean(&x, &shifted, ValueFunctional::Product, &token())
                .expect("mean");
        assert!((paired_by_index - truth).abs() > 5.0 * band);
        let moments = paired_moments(&x, &y).expect("moments");
        assert!((moments.covariance - rho * s_x * s_y).abs() < 5.0 * s_x * s_y / (n as f64).sqrt());
    }

    #[test]
    fn exact_finite_joint_law_enumeration_of_a_nonlinear_functional() {
        // Joint law over (X, Y): (0,0) 0.1, (0,2) 0.4, (1,0) 0.3, (1,2) 0.2.
        let x = [0.0, 0.0, 1.0, 1.0];
        let y = [0.0, 2.0, 0.0, 2.0];
        // Replicate rows by mass: ten rows with the stated frequencies.
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for (i, count) in [1, 4, 3, 2].into_iter().enumerate() {
            for _ in 0..count {
                xs.push(x[i]);
                ys.push(y[i]);
            }
        }
        let product =
            aligned_functional_mean(&xs, &ys, ValueFunctional::Product, &token()).expect("product");
        assert!((product - 0.4).abs() < 1e-15, "E[XY] = 1 * 2 * 0.2");
        let minimum =
            aligned_functional_mean(&xs, &ys, ValueFunctional::Minimum, &token()).expect("minimum");
        assert!((minimum - 0.2).abs() < 1e-15);
        let squared =
            aligned_functional_mean(&xs, &ys, ValueFunctional::SquaredDifference, &token())
                .expect("squared");
        // 0.1*0 + 0.4*4 + 0.3*1 + 0.2*1 = 2.1
        assert!((squared - 2.1).abs() < 1e-14);
    }

    #[test]
    fn bernoulli_envelopes_equal_the_closed_form_endpoints() {
        let (p, q) = (0.3, 0.6);
        let x = Marginal::from_weighted(&[0.0, 1.0], &[1.0 - p, p]).expect("x");
        let y = Marginal::from_weighted(&[0.0, 1.0], &[1.0 - q, q]).expect("y");
        let c = token();
        let product = product_moment_envelope(&x, &y, &c).expect("product");
        // P(1,1) in [max(0, p+q-1), min(p, q)] = [0, 0.3].
        assert!((product.lower - 0.0).abs() < 1e-12);
        assert!((product.upper - 0.3).abs() < 1e-12);
        let cov = covariance_envelope(&x, &y, &c).expect("cov");
        assert!((cov.lower + 0.18).abs() < 1e-12 && (cov.upper - 0.12).abs() < 1e-12);
        let var_sum = linear_variance_envelope(1.0, 1.0, &x, &y, &c).expect("var");
        assert!((var_sum.lower - 0.09).abs() < 1e-12 && (var_sum.upper - 0.69).abs() < 1e-12);
        let var_diff = linear_variance_envelope(1.0, -1.0, &x, &y, &c).expect("var");
        assert!((var_diff.lower - 0.21).abs() < 1e-12 && (var_diff.upper - 0.81).abs() < 1e-12);
        // P(X+Y <= 0) = 1 - p - q + c in [0.1, 0.4]; P(X+Y <= 1) = 1 - c in [0.7, 1].
        let at0 = sum_cdf_envelope(&x, &y, 0.0).expect("cdf");
        assert!((at0.lower - 0.1).abs() < 1e-12 && (at0.upper - 0.4).abs() < 1e-12);
        let at1 = sum_cdf_envelope(&x, &y, 1.0).expect("cdf");
        assert!((at1.lower - 0.7).abs() < 1e-12 && (at1.upper - 1.0).abs() < 1e-12);
        let at2 = sum_cdf_envelope(&x, &y, 2.0).expect("cdf");
        assert!(at2.is_degenerate() && (at2.lower - 1.0).abs() < 1e-15);
    }

    #[test]
    fn coupling_envelope_keeps_atoms_below_the_mass_tolerance() {
        let tiny = 1e-13;
        let x = Marginal::from_weighted(&[0.0, 10.0], &[1.0 - tiny, tiny]).expect("x");
        let y = Marginal::from_weighted(&[0.0, 1.0], &[1.0 - tiny, tiny]).expect("y");
        let product = product_moment_envelope(&x, &y, &token()).expect("product envelope");
        // The comonotone coupling pairs the two rare upper atoms, contributing
        // 10 * tiny. The old residual-mass cutoff silently discarded this cell.
        assert!((product.upper - 10.0 * tiny).abs() < 1e-25, "{product:?}");
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        fn go(current: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>) {
            if current.len() == used.len() {
                out.push(current.clone());
                return;
            }
            for i in 0..used.len() {
                if !used[i] {
                    used[i] = true;
                    current.push(i);
                    go(current, used, out);
                    current.pop();
                    used[i] = false;
                }
            }
        }
        let mut out = Vec::new();
        go(&mut Vec::new(), &mut vec![false; n], &mut out);
        out
    }

    #[test]
    fn equal_mass_marginals_envelopes_equal_permutation_enumeration() {
        // Uniform marginals on m atoms: the extreme couplings are permutations.
        let xs = [0.0, 0.25, 0.5, 0.75, 1.0];
        let ys = [0.0, 0.125, 0.5, 0.625, 1.5];
        let m = xs.len();
        let x = Marginal::from_draws(&xs).expect("x");
        let y = Marginal::from_draws(&ys).expect("y");
        let perms = permutations(m);
        let mut prod = (f64::INFINITY, f64::NEG_INFINITY);
        for sigma in &perms {
            let e: f64 = (0..m).map(|i| xs[i] * ys[sigma[i]]).sum::<f64>() / m as f64;
            prod = (prod.0.min(e), prod.1.max(e));
        }
        let env = product_moment_envelope(&x, &y, &token()).expect("product");
        assert!((env.lower - prod.0).abs() < 1e-14 && (env.upper - prod.1).abs() < 1e-14);
        for t in [0.5, 0.75, 1.0, 1.125, 1.5, 2.0, 2.25] {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for sigma in &perms {
                let hits = (0..m).filter(|&i| xs[i] + ys[sigma[i]] <= t).count();
                let p = hits as f64 / m as f64;
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let bounds = sum_cdf_envelope(&x, &y, t).expect("cdf");
            assert!(
                (bounds.lower - lo).abs() < 1e-12 && (bounds.upper - hi).abs() < 1e-12,
                "t={t}: envelope {bounds:?} vs enumeration ({lo}, {hi})"
            );
        }
        // Difference route against the same enumeration.
        for t in [-0.5, 0.0, 0.25] {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for sigma in &perms {
                let hits = (0..m).filter(|&i| xs[i] - ys[sigma[i]] <= t).count();
                let p = hits as f64 / m as f64;
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let bounds = difference_cdf_envelope(&x, &y, t).expect("cdf");
            assert!((bounds.lower - lo).abs() < 1e-12 && (bounds.upper - hi).abs() < 1e-12);
        }
    }

    #[test]
    fn uniform_marginals_makarov_bounds_match_the_continuous_closed_form() {
        // X, Y ~ U(0, 1) as m equal atoms (j + 0.5)/m: P(X+Y <= t) lies in
        // [max(0, t - 1), min(t, 1)] for the continuous law; the atom bounds
        // converge to it at rate 1/m.
        let m = 2000;
        let atoms: Vec<f64> = (0..m).map(|j| (f64::from(j) + 0.5) / f64::from(m)).collect();
        let u = Marginal::from_draws(&atoms).expect("u");
        for t in [0.25, 0.5, 0.9, 1.0, 1.4, 1.75] {
            let b = sum_cdf_envelope(&u, &u, t).expect("cdf");
            let (lo, hi) = ((t - 1.0_f64).max(0.0), t.min(1.0));
            assert!((b.lower - lo).abs() < 2.0 / f64::from(m), "t={t} lower {b:?}");
            assert!((b.upper - hi).abs() < 2.0 / f64::from(m), "t={t} upper {b:?}");
        }
        // E[XY] for uniform marginals: comonotone 1/3, countermonotone 1/6.
        let p = product_moment_envelope(&u, &u, &token()).expect("product");
        assert!((p.upper - 1.0 / 3.0).abs() < 1e-3 && (p.lower - 1.0 / 6.0).abs() < 1e-3);
    }

    #[test]
    fn envelopes_are_invariant_to_listing_order_and_relabelling() {
        let a = [0.3, 1.2, -0.4, 2.0, 0.9, 0.9];
        let mut a_rev = a;
        a_rev.reverse();
        let b = [1.0, 0.1, 0.1, 4.0, -2.0, 0.5];
        let mut b_perm = b;
        b_perm.rotate_left(2);
        let (x1, x2) =
            (Marginal::from_draws(&a).expect("x"), Marginal::from_draws(&a_rev).expect("x"));
        let (y1, y2) =
            (Marginal::from_draws(&b).expect("y"), Marginal::from_draws(&b_perm).expect("y"));
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
        assert_eq!(
            sum_cdf_envelope(&x1, &y1, 1.0).expect("a"),
            sum_cdf_envelope(&x2, &y2, 1.0).expect("b")
        );
        // Swapping the two variables leaves the sum and product envelopes.
        assert_eq!(
            sum_cdf_envelope(&x1, &y1, 1.0).expect("a"),
            sum_cdf_envelope(&y1, &x1, 1.0).expect("b")
        );
        assert_eq!(
            product_moment_envelope(&x1, &y1, &token()).expect("a"),
            product_moment_envelope(&y1, &x1, &token()).expect("b")
        );
    }

    #[test]
    fn a_value_function_without_the_declared_structure_is_refused() {
        let x = Marginal::from_draws(&[0.0, 1.0, 2.0]).expect("x");
        let y = Marginal::from_draws(&[0.0, 1.0, 2.0]).expect("y");
        // sin(x y) is neither super- nor submodular on this lattice.
        let error = frechet_expectation_envelope(
            &x,
            &y,
            &|a, b| (3.0 * a * b).sin(),
            PairStructure::Supermodular,
            &token(),
        )
        .expect_err("refuse");
        assert!(matches!(error, JointLawError::StructureNotVerified { .. }));
        // A submodular functional is bounded with the roles swapped.
        let env = functional_envelope(&x, &y, ValueFunctional::SquaredDifference, &token())
            .expect("envelope");
        // Comonotone coupling gives E[(X-Y)^2] = 0; countermonotone
        // pairs (0,2), (1,1), (2,0): (4 + 0 + 4)/3.
        assert!((env.lower - 0.0).abs() < 1e-15 && (env.upper - 8.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn correlation_range_endpoints_are_the_closed_form() {
        let x = MomentSummary { mean: 1.0, standard_deviation: 2.0 };
        let y = MomentSummary { mean: 3.0, standard_deviation: 0.5 };
        let r = correlation_range_envelope(x, y, 1.0, -1.0, -0.5, 0.5).expect("range");
        assert!((r.product_moment.lower - (3.0 - 0.5)).abs() < 1e-15);
        assert!((r.product_moment.upper - (3.0 + 0.5)).abs() < 1e-15);
        // Var(X - Y) = 4 + 0.25 - 2 rho 1.
        assert!((r.linear_variance.lower - (4.25 - 1.0)).abs() < 1e-15);
        assert!((r.linear_variance.upper - (4.25 + 1.0)).abs() < 1e-15);
        assert!(matches!(
            correlation_range_envelope(x, y, 1.0, 1.0, 0.2, 0.2),
            Err(JointLawError::InvalidRange(_))
        ));
    }

    #[test]
    fn finite_inputs_that_overflow_declared_moment_bounds_are_refused() {
        let x = MomentSummary { mean: 1e308, standard_deviation: 0.0 };
        let y = MomentSummary { mean: 1e308, standard_deviation: 0.0 };
        assert_eq!(
            correlation_range_envelope(x, y, 1.0, 0.0, -0.5, 0.5),
            Err(JointLawError::NonFiniteValue)
        );

        let x = Marginal::from_draws(&[0.0, 1.0]).expect("x");
        let y = Marginal::from_draws(&[0.0, 1.0]).expect("y");
        assert_eq!(
            linear_variance_envelope(1e308, 1.0, &x, &y, &token()),
            Err(JointLawError::NonFiniteValue)
        );
    }

    #[test]
    fn cancellation_is_observed_in_the_coupling_walk() {
        let draws: Vec<f64> = (0..20_000).map(f64::from).collect();
        let m = Marginal::from_draws(&draws).expect("m");
        let c = CancellationToken::new();
        c.cancel();
        assert_eq!(product_moment_envelope(&m, &m, &c), Err(JointLawError::Cancelled));
    }
}
