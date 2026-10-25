//! Arithmetic of a propagation chain: uncertain quantities carried through
//! several transformations, each stage fed by the stages before it.
//!
//! A chain is a list of source columns and a list of stages. A stage reads
//! earlier columns (sources or stages) and applies one operator: an affine map
//! (a measurement mapping or a transport shift, whose slope and intercept may
//! themselves be uncertain columns), a fixed-weight combination (a transport
//! reweighting over strata), a pool, a binary or unary transformation (a
//! derived quantity, a nonlinear decision utility), an expression over several
//! aligned inputs (a measurement correction, a predictive line), or a
//! preposterior update.
//!
//! Nothing here decides whether a combination is licensed. That is the
//! contract's rule (`antecedent_contract::check_propagation_chain`). This module
//! computes what a licensed chain means:
//!
//! * Every stage is evaluated row by row on one aligned draw table, so a
//!   nonlinear stage sees the joint law of its inputs and never a marginal
//!   approximation of it: `g(E[X]) != E[g(X)]` holds here by construction.
//! * Intervals and points can be carried by an enclosure instead of draws
//!   ([`evaluate_enclosures`]). The enclosure is the image of the box of the
//!   inputs, valid under every coupling of them, and it is not an interval, a
//!   spread or a posterior.
//! * A pool takes its weights from the inputs. An independent pool uses inverse
//!   variance weights. A dependent pool uses generalized least squares weights
//!   from the joint covariance of the aligned columns, so evidence that two
//!   inputs share is counted once.
//! * A summary reads the column under row weights. Row weights are how a changed
//!   assumption about a source re-weights exactly the stages that depend on it.
//! * A preposterior update simulates one future study per row from that row's
//!   parameter value, updates the posterior draws by the study's likelihood,
//!   and re-evaluates the decision under the updated draws
//!   ([`preposterior`]).
//!
//! # Numerics
//!
//! A summary of weighted rows has effective size `1 / sum p_i^2`, a variance
//! with the reliability-weights divisor `1 - sum p_i^2` (the usual `n - 1`
//! for equal weights) and a Monte Carlo standard error `sqrt(variance / size)`.
//! A decision verdict compares the expected utility with a reference utility
//! beyond the normal band of that standard error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CancellationToken, CanonicalEncode, CanonicalWriter};
use thiserror::Error;

use crate::joint_law::DecisionVerdict;
use crate::special::normal_ppf;
use crate::synthesis::{SynthesisError, fixed_effect_gls, fixed_effect_inverse_variance};

/// Most rows one chain table may carry.
pub const MAX_CHAIN_ROWS: usize = 200_000;
/// Most sources plus stages one chain may name.
pub const MAX_CHAIN_NODES: usize = 128;
/// Most rows a preposterior update may use: it compares every row with every
/// other row.
pub const MAX_PREPOSTERIOR_ROWS: usize = 2_048;
/// Smallest mean effective size of the updated draws a preposterior update
/// may report. Below it the likelihood of the simulated study leaves too few
/// draws to represent the updated posterior.
pub const MIN_UPDATE_EFFECTIVE_SIZE: f64 = 10.0;
/// Level of the normal band a decision verdict is read against.
pub const DECISION_BAND_LEVEL: f64 = 0.95;

/// Most terms one expression stage may hold.
pub const MAX_EXPRESSION_TERMS: usize = 64;

const CANCEL_STRIDE: usize = 64;
const VARIANCE_FLOOR: f64 = 1e-300;

/// Why a chain computation was refused.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum ChainError {
    /// Lengths or counts do not agree, or a node reference is invalid.
    #[error("propagation chain shape error: {0}")]
    Shape(&'static str),
    /// An input is NaN or infinite.
    #[error("propagation chain input is not finite: {0}")]
    NonFinite(&'static str),
    /// A stage operator is not well formed or is not defined on the
    /// representation it was given.
    #[error("propagation chain operator refused: {0}")]
    Operator(&'static str),
    /// A transformation left its domain on some row (a logarithm of a
    /// non-positive draw, a reciprocal of zero, an overflow).
    #[error("propagation chain stage left its domain: {0}")]
    Domain(&'static str),
    /// A denominator of an expression stage reaches zero, or comes closer to it
    /// than the stage declares it may. A ratio of such a column is not a
    /// number the draws determine, so it is never evaluated.
    #[error("propagation chain denominator is not bounded away from zero: {0}")]
    Denominator(&'static str),
    /// The pooled inputs are perfectly correlated, so no weights exist.
    #[error("the pooled inputs carry the same evidence: their joint covariance is singular")]
    SingularPool,
    /// The pool's covariance or variances were refused.
    #[error("pooling refused: {0}")]
    PoolCovariance(SynthesisError),
    /// An enclosure cannot be bounded (a divisor interval contains zero, a
    /// logarithm of an interval that reaches zero).
    #[error("the enclosure is unbounded: {0}")]
    EnclosureUnbounded(&'static str),
    /// Row weights are not a usable weight vector.
    #[error("invalid row weights: {0}")]
    InvalidWeights(&'static str),
    /// The updated draws of a preposterior update are too few to represent
    /// the updated posterior.
    #[error("the updated draws have effective size {effective:.3}, below the minimum {minimum}")]
    LowEffectiveSize {
        /// Mean effective size over the simulated studies.
        effective: f64,
        /// The minimum the update requires.
        minimum: f64,
    },
    /// A declared budget was exceeded.
    #[error("propagation chain budget exceeded: {0}")]
    Budget(&'static str),
    /// Cancellation was observed at a work boundary.
    #[error("propagation chain cancelled")]
    Cancelled,
}

fn check_cancel(cancel: &CancellationToken) -> Result<(), ChainError> {
    if cancel.is_cancelled() { Err(ChainError::Cancelled) } else { Ok(()) }
}

fn index_word(index: usize) -> u64 {
    u64::try_from(index).unwrap_or(u64::MAX)
}

// ------------------------------------------------------------------ operators

/// A coefficient of an affine stage: a constant, or another input of the
/// stage (an uncertain slope or intercept).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coefficient {
    /// A known constant.
    Fixed(f64),
    /// The stage's input at this position. Position zero is the mapped
    /// quantity and cannot be a coefficient.
    Input(usize),
}

impl CanonicalEncode for Coefficient {
    /// Tags are append-only: `1` fixed, `2` input.
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Fixed(value) => {
                writer.word(1).real(*value);
            }
            Self::Input(position) => {
                writer.word(2).word(index_word(*position));
            }
        }
    }
}

/// A transformation of one column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryFn {
    /// `exp(x)`.
    Exp,
    /// `ln(x)`; refused on a non-positive draw.
    Log,
    /// `x^2`.
    Square,
    /// `1 / x`; refused when a draw is zero.
    Reciprocal,
    /// `1 / (1 + exp(-x))`.
    Logistic,
    /// The exponential (constant absolute risk aversion) utility
    /// `1 - exp(-r x)` with a non-zero risk aversion `r`.
    ExponentialUtility {
        /// Risk aversion `r`.
        risk_aversion: f64,
    },
    /// The kinked linear utility `slope * (x - kink)` with the slope on the
    /// side of the kink `x` falls on.
    PiecewiseLinear {
        /// Where the slope changes.
        kink: f64,
        /// Slope below the kink.
        below_slope: f64,
        /// Slope at and above the kink.
        above_slope: f64,
    },
    /// The step utility: `above` when `x > cutoff`, otherwise `below`.
    Threshold {
        /// The cutoff.
        cutoff: f64,
        /// Value at and below the cutoff.
        below: f64,
        /// Value above the cutoff.
        above: f64,
    },
}

impl UnaryFn {
    /// Value at `x`.
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Exp => x.exp(),
            Self::Log => x.ln(),
            Self::Square => x * x,
            Self::Reciprocal => 1.0 / x,
            Self::Logistic => 1.0 / (1.0 + (-x).exp()),
            Self::ExponentialUtility { risk_aversion } => 1.0 - (-risk_aversion * x).exp(),
            Self::PiecewiseLinear { kink, below_slope, above_slope } => {
                if x >= kink {
                    above_slope * (x - kink)
                } else {
                    below_slope * (x - kink)
                }
            }
            Self::Threshold { cutoff, below, above } => {
                if x > cutoff {
                    above
                } else {
                    below
                }
            }
        }
    }

    /// The point inside the domain where the function stops being monotone,
    /// if it has one.
    fn turning_point(self) -> Option<f64> {
        match self {
            Self::Square => Some(0.0),
            Self::PiecewiseLinear { kink, .. } => Some(kink),
            _ => None,
        }
    }

    fn check(self) -> Result<(), ChainError> {
        let parameters = match self {
            Self::ExponentialUtility { risk_aversion } => {
                if risk_aversion == 0.0 {
                    return Err(ChainError::Operator(
                        "the exponential utility needs a non-zero risk aversion",
                    ));
                }
                vec![risk_aversion]
            }
            Self::PiecewiseLinear { kink, below_slope, above_slope } => {
                vec![kink, below_slope, above_slope]
            }
            Self::Threshold { cutoff, below, above } => vec![cutoff, below, above],
            _ => Vec::new(),
        };
        if parameters.iter().all(|p| p.is_finite()) {
            Ok(())
        } else {
            Err(ChainError::NonFinite("transformation parameter"))
        }
    }
}

impl CanonicalEncode for UnaryFn {
    /// Tags are append-only: `1` exp, `2` log, `3` square, `4` reciprocal,
    /// `5` logistic, `6` exponential utility, `7` piecewise linear, `8`
    /// threshold.
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Exp => {
                writer.word(1);
            }
            Self::Log => {
                writer.word(2);
            }
            Self::Square => {
                writer.word(3);
            }
            Self::Reciprocal => {
                writer.word(4);
            }
            Self::Logistic => {
                writer.word(5);
            }
            Self::ExponentialUtility { risk_aversion } => {
                writer.word(6).real(*risk_aversion);
            }
            Self::PiecewiseLinear { kink, below_slope, above_slope } => {
                writer.word(7).real(*kink).real(*below_slope).real(*above_slope);
            }
            Self::Threshold { cutoff, below, above } => {
                writer.word(8).real(*cutoff).real(*below).real(*above);
            }
        }
    }
}

/// A combination of two columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Combine {
    /// `a + b`.
    Sum,
    /// `a - b`.
    Difference,
    /// `a * b`.
    Product,
    /// `a / b`; refused when a draw of `b` is zero.
    Ratio,
}

impl Combine {
    /// Every combination.
    pub const ALL: [Self; 4] = [Self::Sum, Self::Difference, Self::Product, Self::Ratio];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Difference => "difference",
            Self::Product => "product",
            Self::Ratio => "ratio",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a combination"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|value| value.as_str() == name)
    }

    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Sum => a + b,
            Self::Difference => a - b,
            Self::Product => a * b,
            Self::Ratio => a / b,
        }
    }
}

impl CanonicalEncode for Combine {
    /// Tags are append-only: `1` sum, `2` difference, `3` product, `4` ratio.
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(match self {
            Self::Sum => 1,
            Self::Difference => 2,
            Self::Product => 3,
            Self::Ratio => 4,
        });
    }
}

/// One term of an expression stage, in postfix order.
///
/// An expression stage evaluates a formula over the stage's inputs row by row on
/// the aligned draw table: a measurement correction `(q + sp - 1) / (se + sp - 1)`
/// of the observed rate and the draws of sensitivity and specificity, or a
/// predictive line `x * slope + intercept + residual`. Each term pops its
/// operands from a stack and pushes one column, and the formula leaves exactly
/// one column. A ratio is never silent: [`Self::Divide`] refuses a denominator
/// that reaches zero on any row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExprTerm {
    /// Push the stage's input at this position.
    Input(usize),
    /// Push a constant.
    Constant(f64),
    /// Pop `b`, `a`; push `a + b`.
    Add,
    /// Pop `b`, `a`; push `a - b`.
    Subtract,
    /// Pop `b`, `a`; push `a * b`.
    Multiply,
    /// Pop `b`, `a`; push `a / b`. Refused when the draws of `b` reach zero (a
    /// denominator that crosses or touches zero) or when some draw of `b` has
    /// magnitude below `min_magnitude`.
    Divide {
        /// Smallest magnitude a denominator draw may have; zero asks only that
        /// the denominator never reach zero.
        min_magnitude: f64,
    },
    /// Check the top column lies in `[low, high]` on every row and leave it. A
    /// value outside the range is refused, never clipped.
    RequireRange {
        /// Smallest admissible value.
        low: f64,
        /// Largest admissible value.
        high: f64,
    },
}

impl CanonicalEncode for ExprTerm {
    /// Tags are append-only: `1` input, `2` constant, `3` add, `4` subtract, `5`
    /// multiply, `6` divide, `7` required range.
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Input(position) => {
                writer.word(1).word(index_word(*position));
            }
            Self::Constant(value) => {
                writer.word(2).real(*value);
            }
            Self::Add => {
                writer.word(3);
            }
            Self::Subtract => {
                writer.word(4);
            }
            Self::Multiply => {
                writer.word(5);
            }
            Self::Divide { min_magnitude } => {
                writer.word(6).real(*min_magnitude);
            }
            Self::RequireRange { low, high } => {
                writer.word(7).real(*low).real(*high);
            }
        }
    }
}

/// A future study of the parameter column, updating it before a decision is
/// re-evaluated.
///
/// The study measures the parameter with independent normal noise of standard
/// deviation `noise_sd` per observation and `sample_size` observations, so its
/// sufficient statistic is `N(theta, noise_sd^2 / sample_size)`. `noise` holds
/// one standard-normal draw per row; the simulated study of row `r` observes
/// `theta_r + noise_sd / sqrt(sample_size) * noise_r`.
#[derive(Clone, Debug, PartialEq)]
pub struct Preposterior {
    /// Per-observation standard deviation of the future study.
    pub noise_sd: f64,
    /// Number of observations of the future study.
    pub sample_size: f64,
    /// Utility of the reference action the decision is compared with.
    pub reference_utility: f64,
    /// Standard-normal draws of the simulated studies, one per row.
    pub noise: Vec<f64>,
}

impl CanonicalEncode for Preposterior {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer
            .real(self.noise_sd)
            .real(self.sample_size)
            .real(self.reference_utility)
            .reals(&self.noise);
    }
}

/// The operator of one stage.
#[derive(Clone, Debug, PartialEq)]
pub enum StageOp {
    /// `input_0 * scale + shift`. Inputs: the mapped quantity first, then any
    /// uncertain coefficient columns.
    Affine {
        /// Slope.
        scale: Coefficient,
        /// Intercept.
        shift: Coefficient,
    },
    /// `sum w_i input_i` with fixed weights (a transport reweighting over
    /// strata, a weighted sum).
    Reweight {
        /// One weight per input.
        weights: Vec<f64>,
    },
    /// Pool at least two estimates of one quantity.
    Pool {
        /// Whether the weights come from the joint covariance of the aligned
        /// columns (generalized least squares) instead of the variances alone.
        dependent: bool,
    },
    /// A binary transformation of two inputs.
    Binary(Combine),
    /// A transformation of one input.
    Unary(UnaryFn),
    /// A preposterior update. Inputs: the parameter column, then the utility
    /// column the decision reads. Terminal: no stage may read it.
    Preposterior(Preposterior),
    /// A formula over the inputs in postfix order: the multi-stage ratio of a
    /// measurement correction, a predictive line with an independent residual.
    Expression(Vec<ExprTerm>),
}

impl CanonicalEncode for StageOp {
    /// Tags are append-only: `1` affine, `2` reweight, `3` pool, `4` binary,
    /// `5` unary, `6` preposterior, `7` expression.
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Affine { scale, shift } => {
                writer.word(1);
                scale.encode(writer);
                shift.encode(writer);
            }
            Self::Reweight { weights } => {
                writer.word(2).reals(weights);
            }
            Self::Pool { dependent } => {
                writer.word(3).word(u64::from(*dependent));
            }
            Self::Binary(combine) => {
                writer.word(4);
                combine.encode(writer);
            }
            Self::Unary(function) => {
                writer.word(5);
                function.encode(writer);
            }
            Self::Preposterior(study) => {
                writer.word(6);
                study.encode(writer);
            }
            Self::Expression(terms) => {
                writer.word(7).word(index_word(terms.len()));
                for term in terms {
                    term.encode(writer);
                }
            }
        }
    }
}

/// One stage: the nodes it reads (sources first, then earlier stages, by
/// position) and its operator.
#[derive(Clone, Debug, PartialEq)]
pub struct StageSpec {
    /// Node positions this stage reads. A node below `n_sources` is a source;
    /// otherwise it is stage `node - n_sources`, which must come earlier.
    pub inputs: Vec<usize>,
    /// The operator.
    pub op: StageOp,
}

impl CanonicalEncode for StageSpec {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(index_word(self.inputs.len()));
        for input in &self.inputs {
            writer.word(index_word(*input));
        }
        self.op.encode(writer);
    }
}

/// Check the shape of a chain before any number is formed.
///
/// # Errors
/// A node reference that is not earlier, a wrong input count for an operator,
/// a coefficient that is not an input, a non-finite parameter, a stage that
/// reads a terminal update stage, or too many nodes.
pub fn validate_spec(n_sources: usize, stages: &[StageSpec]) -> Result<(), ChainError> {
    if n_sources == 0 {
        return Err(ChainError::Shape("a chain needs at least one source"));
    }
    if stages.is_empty() {
        return Err(ChainError::Shape("a chain needs at least one stage"));
    }
    if n_sources + stages.len() > MAX_CHAIN_NODES {
        return Err(ChainError::Budget("nodes"));
    }
    let mut terminal = vec![false; stages.len()];
    for (position, stage) in stages.iter().enumerate() {
        let node = n_sources + position;
        if stage.inputs.is_empty() {
            return Err(ChainError::Shape("a stage reads at least one node"));
        }
        for input in &stage.inputs {
            if *input >= node {
                return Err(ChainError::Shape("a stage reads only earlier nodes"));
            }
            if *input >= n_sources && terminal[*input - n_sources] {
                return Err(ChainError::Shape("no stage may read a preposterior update"));
            }
        }
        let arity = stage.inputs.len();
        match &stage.op {
            StageOp::Affine { scale, shift } => {
                for coefficient in [scale, shift] {
                    match coefficient {
                        Coefficient::Fixed(value) if !value.is_finite() => {
                            return Err(ChainError::NonFinite("affine coefficient"));
                        }
                        Coefficient::Input(slot) if *slot == 0 || *slot >= arity => {
                            return Err(ChainError::Shape(
                                "a coefficient is an input after the first",
                            ));
                        }
                        _ => {}
                    }
                }
            }
            StageOp::Reweight { weights } => {
                if weights.len() != arity {
                    return Err(ChainError::Shape("one weight per input"));
                }
                if weights.iter().any(|w| !w.is_finite()) {
                    return Err(ChainError::NonFinite("reweighting weight"));
                }
            }
            StageOp::Pool { .. } => {
                if arity < 2 {
                    return Err(ChainError::Shape("a pool needs at least two inputs"));
                }
            }
            StageOp::Binary(_) => {
                if arity != 2 {
                    return Err(ChainError::Shape("a binary transformation reads two inputs"));
                }
            }
            StageOp::Unary(function) => {
                if arity != 1 {
                    return Err(ChainError::Shape("a unary transformation reads one input"));
                }
                function.check()?;
            }
            StageOp::Preposterior(study) => {
                if arity != 2 {
                    return Err(ChainError::Shape(
                        "an update reads the parameter and the utility column",
                    ));
                }
                let finite = study.noise_sd.is_finite()
                    && study.sample_size.is_finite()
                    && study.reference_utility.is_finite();
                if !finite || study.noise_sd <= 0.0 || study.sample_size <= 0.0 {
                    return Err(ChainError::Operator(
                        "a future study needs a positive noise scale and sample size",
                    ));
                }
                if study.noise.iter().any(|z| !z.is_finite()) {
                    return Err(ChainError::NonFinite("future-study noise"));
                }
                terminal[position] = true;
            }
            StageOp::Expression(terms) => check_expression(terms, arity)?,
        }
    }
    Ok(())
}

/// Check an expression's stack discipline before any number is formed.
fn check_expression(terms: &[ExprTerm], arity: usize) -> Result<(), ChainError> {
    if terms.is_empty() || terms.len() > MAX_EXPRESSION_TERMS {
        return Err(ChainError::Shape("an expression holds between one and 64 terms"));
    }
    let mut depth = 0_usize;
    for term in terms {
        match term {
            ExprTerm::Input(slot) => {
                if *slot >= arity {
                    return Err(ChainError::Shape("an expression reads an input the stage has"));
                }
                depth += 1;
            }
            ExprTerm::Constant(value) => {
                if !value.is_finite() {
                    return Err(ChainError::NonFinite("expression constant"));
                }
                depth += 1;
            }
            ExprTerm::Add | ExprTerm::Subtract | ExprTerm::Multiply => {
                if depth < 2 {
                    return Err(ChainError::Shape("an expression operation reads two columns"));
                }
                depth -= 1;
            }
            ExprTerm::Divide { min_magnitude } => {
                if !(min_magnitude.is_finite() && *min_magnitude >= 0.0) {
                    return Err(ChainError::Operator(
                        "a denominator bound is a finite non-negative magnitude",
                    ));
                }
                if depth < 2 {
                    return Err(ChainError::Shape("an expression operation reads two columns"));
                }
                depth -= 1;
            }
            ExprTerm::RequireRange { low, high } => {
                if !(low.is_finite() && high.is_finite()) {
                    return Err(ChainError::NonFinite("expression range"));
                }
                if low > high {
                    return Err(ChainError::Operator("a range's low end exceeds its high end"));
                }
                if depth < 1 {
                    return Err(ChainError::Shape("a range check reads a column"));
                }
            }
        }
    }
    if depth == 1 {
        Ok(())
    } else {
        Err(ChainError::Shape("an expression leaves exactly one column"))
    }
}

// ----------------------------------------------------------------- evaluation

/// What a pool took its weights from.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolOutcome {
    /// Weights of the inputs, summing to one.
    pub weights: Vec<f64>,
    /// Variance of the pooled estimate if the inputs were independent:
    /// `1 / sum(1 / v_i)`. It is what counting shared evidence twice would
    /// claim.
    pub independent_variance: f64,
    /// Sample variance of the pooled column under the aligned joint law.
    pub joint_variance: f64,
}

/// The columns of an evaluated chain.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluated {
    /// One column per stage, `None` for a terminal preposterior stage.
    pub columns: Vec<Option<Vec<f64>>>,
    /// One pool outcome per stage, `Some` for a pool.
    pub pools: Vec<Option<PoolOutcome>>,
}

fn node_column<'a>(
    sources: &'a [Vec<f64>],
    stages: &'a [Option<Vec<f64>>],
    node: usize,
) -> Result<&'a [f64], ChainError> {
    if node < sources.len() {
        return Ok(&sources[node]);
    }
    stages
        .get(node - sources.len())
        .and_then(|column| column.as_deref())
        .ok_or(ChainError::Shape("a stage reads a column that does not exist"))
}

fn finite_column(column: &[f64]) -> Result<(), ChainError> {
    if column.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(ChainError::Domain(
            "a transformation produced a non-finite value; check its domain on the draws",
        ))
    }
}

fn mean_of(column: &[f64]) -> f64 {
    column.iter().sum::<f64>() / column.len() as f64
}

fn covariance_of(a: &[f64], b: &[f64], mean_a: f64, mean_b: f64) -> f64 {
    let n = a.len() as f64;
    let sum: f64 = a.iter().zip(b).map(|(x, y)| (x - mean_a) * (y - mean_b)).sum();
    sum / (n - 1.0)
}

fn pool_columns(
    columns: &[&[f64]],
    dependent: bool,
) -> Result<(Vec<f64>, PoolOutcome), ChainError> {
    let k = columns.len();
    let n = columns[0].len();
    if n < 2 {
        return Err(ChainError::Shape("a pool needs at least two rows"));
    }
    let means: Vec<f64> = columns.iter().map(|c| mean_of(c)).collect();
    let mut covariance = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let value = covariance_of(columns[i], columns[j], means[i], means[j]);
            covariance[i * k + j] = value;
            covariance[j * k + i] = value;
        }
    }
    let variances: Vec<f64> = (0..k).map(|i| covariance[i * k + i]).collect();
    if variances.iter().any(|v| *v <= VARIANCE_FLOOR) {
        return Err(ChainError::PoolCovariance(SynthesisError::NonFinite(
            "a pooled input has no spread",
        )));
    }
    let weights = if dependent {
        match fixed_effect_gls(&means, &covariance) {
            Ok(pooled) => pooled.weights,
            Err(
                SynthesisError::Singular { .. } | SynthesisError::NotPositiveSemidefinite { .. },
            ) => {
                return Err(ChainError::SingularPool);
            }
            Err(other) => return Err(ChainError::PoolCovariance(other)),
        }
    } else {
        fixed_effect_inverse_variance(&means, &variances)
            .map_err(ChainError::PoolCovariance)?
            .weights
    };
    let mut pooled = vec![0.0; n];
    for (weight, column) in weights.iter().zip(columns) {
        for (out, value) in pooled.iter_mut().zip(*column) {
            *out += weight * value;
        }
    }
    let pooled_mean = mean_of(&pooled);
    let joint_variance = pooled.iter().map(|v| (v - pooled_mean) * (v - pooled_mean)).sum::<f64>()
        / (n as f64 - 1.0);
    let independent_variance = 1.0 / variances.iter().map(|v| 1.0 / v).sum::<f64>();
    Ok((pooled, PoolOutcome { weights, independent_variance, joint_variance }))
}

/// Evaluate every stage of a chain on one aligned draw table.
///
/// `sources` holds one column per source, all of the same length. A terminal
/// preposterior stage has no column: read it with [`preposterior`].
///
/// # Errors
/// A malformed chain, columns of different lengths or more rows than
/// [`MAX_CHAIN_ROWS`], a transformation leaving its domain, a singular pool, or
/// cancellation.
pub fn evaluate_columns(
    sources: &[Vec<f64>],
    stages: &[StageSpec],
    cancel: &CancellationToken,
) -> Result<Evaluated, ChainError> {
    validate_spec(sources.len(), stages)?;
    let rows = sources[0].len();
    if rows == 0 {
        return Err(ChainError::Shape("a chain table needs at least one row"));
    }
    if rows > MAX_CHAIN_ROWS {
        return Err(ChainError::Budget("rows"));
    }
    for column in sources {
        if column.len() != rows {
            return Err(ChainError::Shape("source columns have different lengths"));
        }
        if column.iter().any(|v| !v.is_finite()) {
            return Err(ChainError::NonFinite("source draw"));
        }
    }
    let mut columns: Vec<Option<Vec<f64>>> = Vec::with_capacity(stages.len());
    let mut pools: Vec<Option<PoolOutcome>> = Vec::with_capacity(stages.len());
    for stage in stages {
        check_cancel(cancel)?;
        let inputs: Vec<&[f64]> = stage
            .inputs
            .iter()
            .map(|node| node_column(sources, &columns, *node))
            .collect::<Result<_, _>>()?;
        let mut pool = None;
        let column: Option<Vec<f64>> = match &stage.op {
            StageOp::Affine { scale, shift } => {
                let pick = |coefficient: &Coefficient, row: usize| match coefficient {
                    Coefficient::Fixed(value) => *value,
                    Coefficient::Input(slot) => inputs[*slot][row],
                };
                Some(
                    (0..rows)
                        .map(|row| inputs[0][row] * pick(scale, row) + pick(shift, row))
                        .collect(),
                )
            }
            StageOp::Reweight { weights } => Some(
                (0..rows)
                    .map(|row| weights.iter().zip(&inputs).map(|(w, c)| w * c[row]).sum())
                    .collect(),
            ),
            StageOp::Pool { dependent } => {
                let (column, outcome) = pool_columns(&inputs, *dependent)?;
                pool = Some(outcome);
                Some(column)
            }
            StageOp::Binary(combine) => {
                Some(inputs[0].iter().zip(inputs[1]).map(|(a, b)| combine.apply(*a, *b)).collect())
            }
            StageOp::Unary(function) => {
                Some(inputs[0].iter().map(|x| function.apply(*x)).collect())
            }
            StageOp::Expression(terms) => Some(evaluate_expression(terms, &inputs, rows)?),
            StageOp::Preposterior(_) => None,
        };
        if let Some(column) = &column {
            finite_column(column)?;
        }
        columns.push(column);
        pools.push(pool);
    }
    Ok(Evaluated { columns, pools })
}

/// The guard of a ratio: the denominator column must stay away from zero.
fn check_denominator(denominator: &[f64], min_magnitude: f64) -> Result<(), ChainError> {
    let low = denominator.iter().copied().fold(f64::INFINITY, f64::min);
    let high = denominator.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if low <= 0.0 && 0.0 <= high {
        return Err(ChainError::Denominator(
            "a denominator draw reaches zero or the draws cross it, so the ratio is not determined",
        ));
    }
    let nearest = if low > 0.0 { low } else { -high };
    if nearest < min_magnitude {
        return Err(ChainError::Denominator(
            "a denominator draw is closer to zero than the stage declares it may be",
        ));
    }
    Ok(())
}

/// Evaluate an expression row by row on the aligned columns.
fn evaluate_expression(
    terms: &[ExprTerm],
    inputs: &[&[f64]],
    rows: usize,
) -> Result<Vec<f64>, ChainError> {
    let mut stack: Vec<Vec<f64>> = Vec::with_capacity(terms.len());
    let underflow = ChainError::Shape("an expression operation reads two columns");
    for term in terms {
        match term {
            ExprTerm::Input(slot) => stack.push(inputs[*slot].to_vec()),
            ExprTerm::Constant(value) => stack.push(vec![*value; rows]),
            ExprTerm::Add | ExprTerm::Subtract | ExprTerm::Multiply => {
                let (Some(right), Some(mut left)) = (stack.pop(), stack.pop()) else {
                    return Err(underflow);
                };
                for (a, b) in left.iter_mut().zip(&right) {
                    *a = match term {
                        ExprTerm::Add => *a + b,
                        ExprTerm::Subtract => *a - b,
                        _ => *a * b,
                    };
                }
                stack.push(left);
            }
            ExprTerm::Divide { min_magnitude } => {
                let (Some(right), Some(mut left)) = (stack.pop(), stack.pop()) else {
                    return Err(underflow);
                };
                check_denominator(&right, *min_magnitude)?;
                for (a, b) in left.iter_mut().zip(&right) {
                    *a /= b;
                }
                stack.push(left);
            }
            ExprTerm::RequireRange { low, high } => {
                let Some(top) = stack.last() else {
                    return Err(underflow);
                };
                if top.iter().any(|v| !(*low..=*high).contains(v)) {
                    return Err(ChainError::Domain(
                        "an expression value left its required range; it is refused, not clipped",
                    ));
                }
            }
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(column), true) => Ok(column),
        _ => Err(ChainError::Shape("an expression leaves exactly one column")),
    }
}

/// The independent product of separately drawn columns, as one aligned table.
///
/// Row `r` pairs the value of every column at the digit of `r` in the mixed
/// radix of the column lengths, so the table holds every combination of the
/// columns' values exactly once: the product of their empirical laws, which
/// is the joint law of independent variables. Nothing is paired by index.
///
/// # Errors
/// No columns, an empty or non-finite column, or a product larger than
/// `max_rows` (the next attempt is to thin the columns).
pub fn independent_product_table(
    columns: &[Vec<f64>],
    max_rows: usize,
) -> Result<Vec<Vec<f64>>, ChainError> {
    if columns.is_empty() {
        return Err(ChainError::Shape("a product table needs at least one column"));
    }
    let mut rows: usize = 1;
    for column in columns {
        if column.is_empty() {
            return Err(ChainError::Shape("a product table column has no draws"));
        }
        if column.iter().any(|v| !v.is_finite()) {
            return Err(ChainError::NonFinite("product table draw"));
        }
        rows = rows
            .checked_mul(column.len())
            .filter(|rows| *rows <= max_rows.min(MAX_CHAIN_ROWS))
            .ok_or(ChainError::Budget("product table rows"))?;
    }
    let mut strides = vec![1_usize; columns.len()];
    for index in (0..columns.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * columns[index + 1].len();
    }
    Ok(columns
        .iter()
        .zip(&strides)
        .map(|(column, stride)| {
            (0..rows).map(|row| column[(row / stride) % column.len()]).collect()
        })
        .collect())
}

// ------------------------------------------------------------------ summaries

/// Normalized row weights from log weights (a softmax), stable against large
/// magnitudes.
///
/// # Errors
/// An empty vector, a non-finite log weight, or weights that underflow to zero.
pub fn weights_from_log(log_weights: &[f64]) -> Result<Vec<f64>, ChainError> {
    if log_weights.is_empty() {
        return Err(ChainError::InvalidWeights("no rows"));
    }
    if log_weights.iter().any(|v| !v.is_finite()) {
        return Err(ChainError::InvalidWeights("a log weight is not finite"));
    }
    let top = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let raw: Vec<f64> = log_weights.iter().map(|v| (v - top).exp()).collect();
    let total: f64 = raw.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(ChainError::InvalidWeights("the weights do not normalize"));
    }
    Ok(raw.into_iter().map(|v| v / total).collect())
}

/// The summary of one column under row weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnSummary {
    /// Number of rows.
    pub n: usize,
    /// Weighted mean.
    pub mean: f64,
    /// Weighted variance with the reliability-weights divisor.
    pub variance: f64,
    /// Effective size `1 / sum p_i^2`.
    pub effective_size: f64,
    /// Monte Carlo standard error of the mean, `sqrt(variance / effective_size)`.
    pub mean_standard_error: f64,
}

/// Summarize `column` under normalized `weights` (equal weights when `None`).
///
/// # Errors
/// An empty column, a weight vector of another length, or a non-finite value.
pub fn summarize(column: &[f64], weights: Option<&[f64]>) -> Result<ColumnSummary, ChainError> {
    let n = column.len();
    if n == 0 {
        return Err(ChainError::Shape("a summary needs at least one row"));
    }
    if let Some(weights) = weights {
        if weights.len() != n {
            return Err(ChainError::InvalidWeights("one weight per row"));
        }
    }
    if column.iter().any(|v| !v.is_finite()) {
        return Err(ChainError::NonFinite("summarized value"));
    }
    let uniform = 1.0 / n as f64;
    let weight = |row: usize| weights.map_or(uniform, |w| w[row]);
    let mean: f64 = (0..n).map(|row| weight(row) * column[row]).sum();
    let squares: f64 = (0..n).map(|row| weight(row) * weight(row)).sum();
    let effective_size = 1.0 / squares;
    let denominator = 1.0 - squares;
    let variance = if denominator <= 1e-12 {
        0.0
    } else {
        (0..n).map(|row| weight(row) * (column[row] - mean) * (column[row] - mean)).sum::<f64>()
            / denominator
    };
    let mean_standard_error = (variance / effective_size).sqrt();
    Ok(ColumnSummary { n, mean, variance, effective_size, mean_standard_error })
}

/// A decision read from an expected utility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    /// Expected utility of acting.
    pub expected_utility: f64,
    /// Utility of the reference action.
    pub reference_utility: f64,
    /// Expected utility minus the reference.
    pub gap: f64,
    /// Half-width of the normal band of the Monte Carlo standard error.
    pub band: f64,
    /// Act, withhold, or unresolved when the gap is inside the band.
    pub verdict: DecisionVerdict,
}

/// Read the decision of a utility column summary against `reference_utility`.
#[must_use]
pub fn decide(summary: &ColumnSummary, reference_utility: f64) -> Decision {
    let z = normal_ppf(0.5 + 0.5 * DECISION_BAND_LEVEL);
    let band = z * summary.mean_standard_error;
    let gap = summary.mean - reference_utility;
    let verdict = if gap > band {
        DecisionVerdict::Act
    } else if gap < -band {
        DecisionVerdict::Withhold
    } else {
        DecisionVerdict::Unresolved
    };
    Decision { expected_utility: summary.mean, reference_utility, gap, band, verdict }
}

// ---------------------------------------------------------------- preposterior

/// The decision re-evaluated under the updated posterior of a future study.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreposteriorOutcome {
    /// Expected utility of acting under the current draws.
    pub current_expected_utility: f64,
    /// Expected value of deciding after the study: the mean over simulated
    /// studies of the better of the updated expected utility and the
    /// reference.
    pub expected_value_with_study: f64,
    /// Expected value with the study minus the value of deciding now.
    pub value_of_information: f64,
    /// Mean over simulated studies of the updated expected utility. By the
    /// tower property it equals the current expected utility up to Monte Carlo
    /// error.
    pub mean_updated_expected_utility: f64,
    /// Probability that the study changes which action is better.
    pub probability_decision_changes: f64,
    /// Mean effective size of the updated draws.
    pub mean_effective_size: f64,
    /// Smallest effective size of the updated draws.
    pub min_effective_size: f64,
}

/// Update the parameter draws by each simulated future study and re-evaluate
/// the decision.
///
/// For every row `r`, the study observes `y_r = theta_r + s * noise_r` with
/// `s = noise_sd / sqrt(sample_size)`. The draws are reweighted by the
/// likelihood `N(y_r; theta_j, s^2)` times the current row weights, which is
/// the updated posterior of the parameter, and the expected utility is
/// recomputed under it. The utility column is a function of the same rows, so
/// every stage the parameter feeds is updated with it.
///
/// # Errors
/// Columns of different lengths, more than [`MAX_PREPOSTERIOR_ROWS`] rows, noise
/// of another length, updated draws with mean effective size below
/// [`MIN_UPDATE_EFFECTIVE_SIZE`], or cancellation.
pub fn preposterior(
    parameter: &[f64],
    utility: &[f64],
    weights: Option<&[f64]>,
    study: &Preposterior,
    cancel: &CancellationToken,
) -> Result<PreposteriorOutcome, ChainError> {
    let n = parameter.len();
    if n == 0 || utility.len() != n || study.noise.len() != n {
        return Err(ChainError::Shape(
            "the parameter, utility and noise columns have one entry per row",
        ));
    }
    if n > MAX_PREPOSTERIOR_ROWS {
        return Err(ChainError::Budget("preposterior rows"));
    }
    if let Some(weights) = weights {
        if weights.len() != n {
            return Err(ChainError::InvalidWeights("one weight per row"));
        }
    }
    let uniform = 1.0 / n as f64;
    let prior = |row: usize| weights.map_or(uniform, |w| w[row]);
    let scale = study.noise_sd / study.sample_size.sqrt();
    let current: f64 = (0..n).map(|row| prior(row) * utility[row]).sum();
    let current_acts = current > study.reference_utility;
    let mut value_with = 0.0;
    let mut mean_updated = 0.0;
    let mut changes = 0.0;
    let mut sum_ess = 0.0;
    let mut min_ess = f64::INFINITY;
    let mut log_likelihood = vec![0.0; n];
    for row in 0..n {
        if row % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        let observed = parameter[row] + scale * study.noise[row];
        let mut top = f64::NEG_INFINITY;
        for (slot, theta) in log_likelihood.iter_mut().zip(parameter) {
            let z = (observed - theta) / scale;
            *slot = -0.5 * z * z;
            top = top.max(*slot);
        }
        let mut total = 0.0;
        let mut squares = 0.0;
        let mut weighted_utility = 0.0;
        for (column, slot) in log_likelihood.iter().enumerate() {
            let q = prior(column) * (slot - top).exp();
            total += q;
            squares += q * q;
            weighted_utility += q * utility[column];
        }
        if !(total.is_finite() && total > 0.0) {
            return Err(ChainError::InvalidWeights("a simulated study has no posterior mass"));
        }
        let updated = weighted_utility / total;
        let ess = total * total / squares;
        let outer = prior(row);
        value_with += outer * updated.max(study.reference_utility);
        mean_updated += outer * updated;
        if (updated > study.reference_utility) != current_acts {
            changes += outer;
        }
        sum_ess += outer * ess;
        min_ess = min_ess.min(ess);
    }
    if sum_ess < MIN_UPDATE_EFFECTIVE_SIZE {
        return Err(ChainError::LowEffectiveSize {
            effective: sum_ess,
            minimum: MIN_UPDATE_EFFECTIVE_SIZE,
        });
    }
    let baseline = current.max(study.reference_utility);
    Ok(PreposteriorOutcome {
        current_expected_utility: current,
        expected_value_with_study: value_with,
        value_of_information: value_with - baseline,
        mean_updated_expected_utility: mean_updated,
        probability_decision_changes: changes,
        mean_effective_size: sum_ess,
        min_effective_size: min_ess,
    })
}

// ------------------------------------------------------------------ enclosures

/// A closed interval of values.
///
/// An enclosure is the image of a box of inputs under a chain. It holds under
/// every coupling of the inputs and is not an interval of uncertainty, a
/// spread or a posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enclosure {
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
}

impl Enclosure {
    /// An enclosure from its ends.
    ///
    /// # Errors
    /// An end that is not finite, or ends out of order.
    pub fn new(lower: f64, upper: f64) -> Result<Self, ChainError> {
        if !(lower.is_finite() && upper.is_finite()) {
            return Err(ChainError::NonFinite("enclosure end"));
        }
        if lower > upper {
            return Err(ChainError::Shape("an enclosure's lower end exceeds its upper end"));
        }
        Ok(Self { lower, upper })
    }

    fn point(value: f64) -> Self {
        Self { lower: value, upper: value }
    }

    fn hull(values: &[f64]) -> Self {
        let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self { lower, upper }
    }

    fn contains_zero(self) -> bool {
        self.lower <= 0.0 && 0.0 <= self.upper
    }

    fn add(self, other: Self) -> Self {
        Self { lower: self.lower + other.lower, upper: self.upper + other.upper }
    }

    fn negate(self) -> Self {
        Self { lower: -self.upper, upper: -self.lower }
    }

    fn scale(self, factor: f64) -> Self {
        Self::hull(&[self.lower * factor, self.upper * factor])
    }

    fn mul(self, other: Self) -> Self {
        Self::hull(&[
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ])
    }

    fn reciprocal(self) -> Result<Self, ChainError> {
        if self.contains_zero() {
            return Err(ChainError::EnclosureUnbounded("a divisor interval contains zero"));
        }
        Ok(Self::hull(&[1.0 / self.lower, 1.0 / self.upper]))
    }

    fn map(self, function: UnaryFn) -> Result<Self, ChainError> {
        match function {
            UnaryFn::Log if self.lower <= 0.0 => {
                return Err(ChainError::EnclosureUnbounded(
                    "a logarithm of an interval reaching zero",
                ));
            }
            UnaryFn::Reciprocal => return self.reciprocal(),
            _ => {}
        }
        let mut values = vec![function.apply(self.lower), function.apply(self.upper)];
        if let Some(turn) = function.turning_point() {
            if self.lower < turn && turn < self.upper {
                values.push(function.apply(turn));
            }
        }
        Ok(Self::hull(&values))
    }
}

fn check_enclosure(enclosure: Enclosure) -> Result<Enclosure, ChainError> {
    if enclosure.lower.is_finite() && enclosure.upper.is_finite() {
        Ok(enclosure)
    } else {
        Err(ChainError::Domain("an enclosure left the finite numbers"))
    }
}

/// Interval arithmetic of an expression. An input that appears twice is two
/// independent boxes, so the result holds under every coupling and may be wider
/// than the image of the box.
fn enclose_expression(terms: &[ExprTerm], inputs: &[Enclosure]) -> Result<Enclosure, ChainError> {
    let mut stack: Vec<Enclosure> = Vec::with_capacity(terms.len());
    let underflow = ChainError::Shape("an expression operation reads two columns");
    for term in terms {
        match term {
            ExprTerm::Input(slot) => stack.push(inputs[*slot]),
            ExprTerm::Constant(value) => stack.push(Enclosure::point(*value)),
            ExprTerm::Add | ExprTerm::Subtract | ExprTerm::Multiply => {
                let (Some(right), Some(left)) = (stack.pop(), stack.pop()) else {
                    return Err(underflow);
                };
                stack.push(match term {
                    ExprTerm::Add => left.add(right),
                    ExprTerm::Subtract => left.add(right.negate()),
                    _ => left.mul(right),
                });
            }
            ExprTerm::Divide { min_magnitude } => {
                let (Some(right), Some(left)) = (stack.pop(), stack.pop()) else {
                    return Err(underflow);
                };
                let nearest = if right.lower > 0.0 { right.lower } else { -right.upper };
                if right.contains_zero() || nearest < *min_magnitude {
                    return Err(ChainError::EnclosureUnbounded(
                        "a divisor interval reaches zero or the declared bound",
                    ));
                }
                stack.push(left.mul(right.reciprocal()?));
            }
            ExprTerm::RequireRange { low, high } => {
                let Some(top) = stack.last() else {
                    return Err(underflow);
                };
                if top.lower < *low || top.upper > *high {
                    return Err(ChainError::Domain(
                        "an expression enclosure leaves its required range",
                    ));
                }
            }
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(result), true) => Ok(result),
        _ => Err(ChainError::Shape("an expression leaves exactly one column")),
    }
}

/// Carry source intervals through a chain by interval arithmetic.
///
/// A point is an interval with equal ends. The result holds under every
/// coupling of the sources. A pool and a preposterior update are defined on
/// draws, not on intervals, and are refused: an interval is not a law.
///
/// # Errors
/// A malformed chain, a pool or update stage, or a divisor or logarithm
/// interval that cannot be bounded.
pub fn evaluate_enclosures(
    sources: &[Enclosure],
    stages: &[StageSpec],
) -> Result<Vec<Enclosure>, ChainError> {
    validate_spec(sources.len(), stages)?;
    let mut out: Vec<Enclosure> = Vec::with_capacity(stages.len());
    let node = |nodes: &[Enclosure], index: usize| -> Enclosure {
        if index < sources.len() { sources[index] } else { nodes[index - sources.len()] }
    };
    for stage in stages {
        let inputs: Vec<Enclosure> = stage.inputs.iter().map(|i| node(&out, *i)).collect();
        let result = match &stage.op {
            StageOp::Affine { scale, shift } => {
                let pick = |coefficient: &Coefficient| match coefficient {
                    Coefficient::Fixed(value) => Enclosure::point(*value),
                    Coefficient::Input(slot) => inputs[*slot],
                };
                inputs[0].mul(pick(scale)).add(pick(shift))
            }
            StageOp::Reweight { weights } => {
                let mut total = Enclosure::point(0.0);
                for (weight, input) in weights.iter().zip(&inputs) {
                    total = total.add(input.scale(*weight));
                }
                total
            }
            StageOp::Binary(combine) => match combine {
                Combine::Sum => inputs[0].add(inputs[1]),
                Combine::Difference => inputs[0].add(inputs[1].negate()),
                Combine::Product => inputs[0].mul(inputs[1]),
                Combine::Ratio => inputs[0].mul(inputs[1].reciprocal()?),
            },
            StageOp::Unary(function) => inputs[0].map(*function)?,
            StageOp::Expression(terms) => enclose_expression(terms, &inputs)?,
            StageOp::Pool { .. } => {
                return Err(ChainError::Operator(
                    "an interval is not a law: pooling needs aligned draws",
                ));
            }
            StageOp::Preposterior(_) => {
                return Err(ChainError::Operator(
                    "an interval is not a posterior: a future-study update needs posterior draws",
                ));
            }
        };
        out.push(check_enclosure(result)?);
    }
    Ok(out)
}

// ------------------------------------------------------------- family tables

/// Most variables one joint normal table may carry.
pub const MAX_TABLE_VARIABLES: usize = 64;

/// `SplitMix64` stream, so a seed reproduces the table on every platform.
struct SeedStream(u64);

impl SeedStream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / 9_007_199_254_740_992.0
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        (radius * angle.cos(), radius * angle.sin())
    }
}

/// Lower-triangular `L` with `L Lᵀ = covariance`, allowing zero-variance
/// directions; `None` when the matrix is not positive semi-definite.
fn semidefinite_factor(covariance: &[f64], n: usize) -> Option<Vec<f64>> {
    let scale = (0..n).map(|i| covariance[i * n + i]).fold(0.0_f64, f64::max);
    let degenerate = 1e-13 * scale;
    let slack = 1e-9 * scale;
    let mut factor = vec![0.0; n * n];
    for column in 0..n {
        let mut pivot = covariance[column * n + column];
        for k in 0..column {
            pivot -= factor[column * n + k] * factor[column * n + k];
        }
        if pivot < -slack {
            return None;
        }
        let diagonal = if pivot <= degenerate { 0.0 } else { pivot.sqrt() };
        factor[column * n + column] = diagonal;
        for row in column + 1..n {
            let mut residual = covariance[row * n + column];
            for k in 0..column {
                residual -= factor[row * n + k] * factor[column * n + k];
            }
            if diagonal == 0.0 {
                if residual.abs() > slack {
                    return None;
                }
            } else {
                factor[row * n + column] = residual / diagonal;
            }
        }
    }
    Some(factor)
}

/// Correlation of two equal-weight mean-type estimators computed from
/// `rows_a` and `rows_b` rows of which `shared` are the same rows: the
/// covariance of the two means is `shared / (rows_a rows_b)` times the common
/// per-row variance, so the correlation is `shared / sqrt(rows_a rows_b)`.
///
/// # Errors
/// An empty row set, or more shared rows than either estimator uses.
pub fn shared_row_correlation(
    rows_a: usize,
    rows_b: usize,
    shared: usize,
) -> Result<f64, ChainError> {
    if rows_a == 0 || rows_b == 0 {
        return Err(ChainError::Shape("an estimator uses at least one row"));
    }
    if shared > rows_a.min(rows_b) {
        return Err(ChainError::Shape("shared rows cannot exceed the rows either estimator uses"));
    }
    Ok(shared as f64 / ((rows_a as f64) * (rows_b as f64)).sqrt())
}

/// Joint normal draws whose sample mean and sample covariance (divisor
/// `rows - 1`) equal `means` and `covariance` up to rounding.
///
/// The base draws are independent standard normals, centered and whitened by
/// their own sample covariance before the declared covariance is applied, so a
/// downstream stage sees the declared dependence without Monte Carlo error in
/// the first two moments. Returns one column per variable.
///
/// # Errors
/// No variables or too many; a covariance of the wrong size, not finite, not
/// symmetric or not positive semidefinite; fewer than `variables + 2` rows or
/// more than [`MAX_CHAIN_ROWS`].
pub fn moment_matched_normal_table(
    means: &[f64],
    covariance: &[f64],
    rows: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, ChainError> {
    let k = means.len();
    if k == 0 || k > MAX_TABLE_VARIABLES || covariance.len() != k * k {
        return Err(ChainError::Shape(
            "a joint table needs one mean per variable and a k by k covariance",
        ));
    }
    if rows > MAX_CHAIN_ROWS {
        return Err(ChainError::Budget("table rows"));
    }
    if rows < k + 2 {
        return Err(ChainError::Shape("a moment-matched table needs more rows than variables"));
    }
    if means.iter().chain(covariance).any(|value| !value.is_finite()) {
        return Err(ChainError::NonFinite("joint table moment"));
    }
    let scale = (0..k).map(|i| covariance[i * k + i]).fold(0.0_f64, f64::max);
    for row in 0..k {
        if covariance[row * k + row] < 0.0 {
            return Err(ChainError::Operator("a variance is negative"));
        }
        for column in 0..row {
            if (covariance[row * k + column] - covariance[column * k + row]).abs() > 1e-9 * scale {
                return Err(ChainError::Operator("the covariance is not symmetric"));
            }
        }
    }
    let target = semidefinite_factor(covariance, k)
        .ok_or(ChainError::Operator("the covariance is not positive semidefinite"))?;
    let mut stream = SeedStream(seed);
    let mut base: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut column = Vec::with_capacity(rows + 1);
        while column.len() < rows {
            let (first, second) = stream.normal_pair();
            column.push(first);
            column.push(second);
        }
        column.truncate(rows);
        let mean = column.iter().sum::<f64>() / rows as f64;
        for value in &mut column {
            *value -= mean;
        }
        base.push(column);
    }
    let divisor = (rows - 1) as f64;
    let mut sample = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let cross = base[i].iter().zip(&base[j]).map(|(a, b)| a * b).sum::<f64>() / divisor;
            sample[i * k + j] = cross;
            sample[j * k + i] = cross;
        }
    }
    // The base draws are PD with probability one; a zero pivot means the seed
    // produced collinear columns and the table cannot be whitened.
    let whitener = semidefinite_factor(&sample, k)
        .filter(|factor| (0..k).all(|i| factor[i * k + i] > 0.0))
        .ok_or(ChainError::Operator("the base draws are degenerate; change the seed"))?;
    let mut table = vec![Vec::with_capacity(rows); k];
    let mut whitened = vec![0.0; k];
    for row in 0..rows {
        for i in 0..k {
            let mut value = base[i][row];
            for j in 0..i {
                value -= whitener[i * k + j] * whitened[j];
            }
            whitened[i] = value / whitener[i * k + i];
        }
        for i in 0..k {
            let mut value = means[i];
            for j in 0..=i {
                value += target[i * k + j] * whitened[j];
            }
            table[i].push(value);
        }
    }
    Ok(table)
}

#[cfg(test)]
#[path = "propagation_chain_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "propagation_chain_table_tests.rs"]
mod table_tests;
