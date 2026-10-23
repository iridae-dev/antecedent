//! Classical additive error in a continuous regressor: regression calibration
//! for linear and logistic outcome models, SIMEX, and an error variance
//! derived from a declared reliability ratio.
//!
//! The error-prone measurement is `W = X + U`, `U` independent of `(X, Z, Y)`
//! with mean zero and variance `sigma_u^2` per measurement; a row averaging
//! `k_i` replicates carries error variance `sigma_u^2 / k_i`. The design is
//! `(1, W, Z_1, ..., Z_p)` throughout, so coefficient index 1 is the
//! error-prone regressor and `2 + j` is covariate `j`, exactly as in
//! [`crate::measurement_error::LinearErrorData`].
//!
//! # Regression calibration
//!
//! [`regression_calibration`] replaces `W` by the best linear predictor
//! `E[X | W, Z]` computed from the sample moments of `(W, Z)` with `Var(W)`
//! replaced by `Var(W) - sigma_u^2 mean(1/k)` (and, row by row, the error
//! variance of that row's mean added back), then fits the outcome model on
//! `(1, X_hat, Z)`. For a linear outcome model this is the corrected
//! normal-equation estimate of
//! [`crate::measurement_error::LinearMoments`]; for a logistic model it is
//! the usual first-order approximation (Carroll, Ruppert, Stefanski and
//! Crainiceanu 2006, ch. 4), accurate when the measurement error or the
//! coefficient is moderate.
//!
//! # SIMEX
//!
//! [`simex`] (Cook and Stefanski 1994, doi:10.1080/01621459.1994.10476871)
//! adds further error of variance `lambda sigma_u^2 / k_i` to each row, refits
//! the outcome model `B` times per `lambda`, averages the coefficients, fits
//! the averages as a function of `lambda` (quadratic by default, or linear) by
//! least squares over the grid including `lambda = 0`, and evaluates the fit
//! at `lambda = -1`, the error-free point. The extrapolant is an
//! approximation: for a linear model the exact curve is rational in `lambda`,
//! so a quadratic extrapolant under-corrects slightly. The per-`lambda` means
//! are returned so the path can be inspected.
//!
//! # Uncertainty
//!
//! Nothing here reports a standard error for logistic calibration or SIMEX:
//! the estimator layer bootstraps the whole procedure. Every result depends
//! only on its inputs and, for SIMEX, the supplied random stream.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;

use crate::gram::{chol_solve, cholesky_spd};
use crate::measurement_error::{
    ErrorVariance, LinearErrorData, MAX_MEASUREMENT_COVARIATES, MAX_MEASUREMENT_ROWS,
    MeasurementErrorError,
};
use crate::{FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm};

/// Default SIMEX grid of added-error multiples (the naive fit at `lambda = 0`
/// is always included as well).
pub const DEFAULT_SIMEX_LAMBDAS: [f64; 4] = [0.5, 1.0, 1.5, 2.0];
/// Default remeasurement replicates per `lambda`.
pub const DEFAULT_SIMEX_REPLICATES: u32 = 50;
/// Most remeasurement replicates per `lambda`.
pub const MAX_SIMEX_REPLICATES: u32 = 10_000;
/// Most `lambda` values in one SIMEX grid.
pub const MAX_SIMEX_LAMBDAS: usize = 32;

/// Outcome model fitted on the (calibrated or remeasured) design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutcomeFamily {
    /// Least squares (identity link).
    Linear,
    /// Logistic regression of a 0/1 outcome (logit link).
    Logistic,
}

impl OutcomeFamily {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Logistic => "logistic",
        }
    }
}

/// Extrapolant of the SIMEX coefficient path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SimexExtrapolant {
    /// `a + b lambda`.
    Linear,
    /// `a + b lambda + c lambda^2` (the default).
    Quadratic,
}

impl SimexExtrapolant {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
        }
    }

    const fn width(self) -> usize {
        match self {
            Self::Linear => 2,
            Self::Quadratic => 3,
        }
    }
}

/// SIMEX grid, replicates and extrapolant.
#[derive(Clone, Debug, PartialEq)]
pub struct SimexConfig {
    /// Positive, distinct added-error multiples; `lambda = 0` is added.
    pub lambdas: Vec<f64>,
    /// Remeasurement replicates per `lambda`.
    pub replicates: u32,
    /// Extrapolant fitted by least squares and evaluated at `lambda = -1`.
    pub extrapolant: SimexExtrapolant,
}

impl Default for SimexConfig {
    fn default() -> Self {
        Self {
            lambdas: DEFAULT_SIMEX_LAMBDAS.to_vec(),
            replicates: DEFAULT_SIMEX_REPLICATES,
            extrapolant: SimexExtrapolant::Quadratic,
        }
    }
}

impl SimexConfig {
    /// Check the grid, the replicate count and that the extrapolant has
    /// enough grid points.
    ///
    /// # Errors
    /// A non-positive, non-finite or repeated `lambda`, too many or too few
    /// grid points for the extrapolant, or a replicate count outside
    /// `1..=MAX_SIMEX_REPLICATES`.
    pub fn validate(&self) -> Result<(), MeasurementErrorError> {
        if self.lambdas.is_empty() || self.lambdas.len() > MAX_SIMEX_LAMBDAS {
            return Err(MeasurementErrorError::InvalidParameter(
                "the SIMEX grid needs between one and 32 lambda values",
            ));
        }
        if self.lambdas.iter().any(|l| !(l.is_finite() && *l > 0.0)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "every SIMEX lambda must be finite and positive",
            ));
        }
        let mut sorted = self.lambdas.clone();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|pair| pair[1] - pair[0] <= 0.0) {
            return Err(MeasurementErrorError::InvalidParameter("SIMEX lambdas must be distinct"));
        }
        if self.lambdas.len() + 1 < self.extrapolant.width() {
            return Err(MeasurementErrorError::InvalidParameter(
                "the extrapolant needs at least as many grid points (with lambda = 0) as it has coefficients",
            ));
        }
        if self.replicates == 0 || self.replicates > MAX_SIMEX_REPLICATES {
            return Err(MeasurementErrorError::InvalidParameter(
                "SIMEX replicates must lie in 1..=10000",
            ));
        }
        Ok(())
    }
}

/// A regression-calibration fit.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedFit {
    /// Outcome model.
    pub family: OutcomeFamily,
    /// Coefficients on `(1, E[X | W, Z], Z)`.
    pub coefficients: Vec<f64>,
    /// Coefficients on `(1, W, Z)`, uncorrected.
    pub naive_coefficients: Vec<f64>,
    /// Implied reliability of `W` given the covariates,
    /// `1 - sigma_u^2 mean(1/k) / Var(W | Z)`.
    pub reliability: f64,
    /// Error variance of one measurement used.
    pub error_variance: f64,
}

/// A SIMEX fit and its coefficient path.
#[derive(Clone, Debug, PartialEq)]
pub struct SimexFit {
    /// Outcome model.
    pub family: OutcomeFamily,
    /// Extrapolated coefficients at `lambda = -1`.
    pub coefficients: Vec<f64>,
    /// Coefficients of the naive fit (`lambda = 0`).
    pub naive_coefficients: Vec<f64>,
    /// The grid, `0` first and then the configured values in ascending order.
    pub lambdas: Vec<f64>,
    /// Mean coefficient vector at each grid value, aligned with `lambdas`.
    pub lambda_means: Vec<Vec<f64>>,
    /// Extrapolant used.
    pub extrapolant: SimexExtrapolant,
    /// Remeasurement replicates per `lambda`.
    pub replicates: u32,
    /// Implied reliability of `W` given the covariates.
    pub reliability: f64,
    /// Error variance of one measurement used.
    pub error_variance: f64,
}

/// An error variance derived from a declared reliability ratio.
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityErrorVariance {
    /// `sigma_u^2 = (1 - reliability) Var(W)` with its delta-method variance.
    pub error_variance: ErrorVariance,
    /// Sample variance of the observed measurement used.
    pub observed_variance: f64,
    /// Declared reliability ratio.
    pub reliability: f64,
    /// Declared standard error of the reliability ratio.
    pub reliability_standard_error: f64,
}

/// Validated shape of one error-prone design.
struct Shape {
    n: usize,
    p: usize,
}

fn check_data(data: &LinearErrorData<'_>) -> Result<Shape, MeasurementErrorError> {
    let n = data.y.len();
    let p = data.covariate_count;
    if p > MAX_MEASUREMENT_COVARIATES || n > MAX_MEASUREMENT_ROWS {
        return Err(MeasurementErrorError::Shape("exceeds the declared row or covariate bound"));
    }
    if data.w.len() != n || p.checked_mul(n) != Some(data.covariates.len()) {
        return Err(MeasurementErrorError::Shape("columns differ in length"));
    }
    if let Some(counts) = data.replicate_counts {
        if counts.len() != n {
            return Err(MeasurementErrorError::Shape("replicate counts differ in length"));
        }
        if counts.contains(&0) {
            return Err(MeasurementErrorError::Shape("a row averages zero replicates"));
        }
    }
    let d = p + 2;
    if n < d + 1 {
        return Err(MeasurementErrorError::TooFewRows { rows: n, minimum: d + 1 });
    }
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    if !(finite(data.y) && finite(data.w) && finite(data.covariates)) {
        return Err(MeasurementErrorError::NonFinite("outcome, regressor or covariates"));
    }
    Ok(Shape { n, p })
}

fn check_error_inputs(error_variance: f64, floor: f64) -> Result<(), MeasurementErrorError> {
    if !(error_variance.is_finite() && error_variance >= 0.0) {
        return Err(MeasurementErrorError::InvalidParameter("error variance"));
    }
    if !(floor.is_finite() && (0.0..1.0).contains(&floor)) {
        return Err(MeasurementErrorError::InvalidParameter(
            "reliability floor must lie in [0, 1)",
        ));
    }
    Ok(())
}

fn inverse_k(data: &LinearErrorData<'_>, i: usize) -> f64 {
    data.replicate_counts.map_or(1.0, |counts| 1.0 / f64::from(counts[i]))
}

/// Column `c` of `V = (W, Z)`.
fn v_column<'a>(data: &LinearErrorData<'a>, n: usize, c: usize) -> &'a [f64] {
    if c == 0 { data.w } else { &data.covariates[(c - 1) * n..c * n] }
}

/// Means and covariance (divisor `n`) of `V = (W, Z)`, row-major `q x q`.
fn moments_of_v(data: &LinearErrorData<'_>, shape: &Shape) -> (Vec<f64>, Vec<f64>) {
    let Shape { n, p } = *shape;
    let q = p + 1;
    let n_f = n as f64;
    let means: Vec<f64> = (0..q).map(|c| v_column(data, n, c).iter().sum::<f64>() / n_f).collect();
    let mut cov = vec![0.0; q * q];
    for a in 0..q {
        for b in a..q {
            let (ca, cb) = (v_column(data, n, a), v_column(data, n, b));
            let s: f64 =
                ca.iter().zip(cb).map(|(x, y)| (x - means[a]) * (y - means[b])).sum::<f64>() / n_f;
            cov[a * q + b] = s;
            cov[b * q + a] = s;
        }
    }
    (means, cov)
}

fn solve_spd(a: &[f64], d: usize, b: &[f64]) -> Option<Vec<f64>> {
    let chol = cholesky_spd(a, d)?;
    chol_solve(&chol, d, b)
}

/// `1 - shift (S^{-1})_{ww}` from the covariance of `(W, Z)`.
fn reliability_from(cov: &[f64], q: usize, shift: f64) -> Result<f64, MeasurementErrorError> {
    let mut e = vec![0.0; q];
    e[0] = 1.0;
    let column = solve_spd(cov, q, &e).ok_or(MeasurementErrorError::SingularDesign)?;
    let inv_ww = column[0];
    if !(inv_ww.is_finite() && inv_ww > 0.0) {
        return Err(MeasurementErrorError::SingularDesign);
    }
    Ok(1.0 - shift * inv_ww)
}

/// Implied reliability of the error-prone regressor given the covariates,
/// refused at or below `floor`.
fn checked_reliability(
    data: &LinearErrorData<'_>,
    shape: &Shape,
    error_variance: f64,
    floor: f64,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), MeasurementErrorError> {
    let (means, cov) = moments_of_v(data, shape);
    let mean_inverse = (0..shape.n).map(|i| inverse_k(data, i)).sum::<f64>() / shape.n as f64;
    let reliability = reliability_from(&cov, shape.p + 1, error_variance * mean_inverse)?;
    if !(reliability > 0.0 && reliability >= floor) {
        return Err(MeasurementErrorError::ReliabilityNotPositive { reliability, floor });
    }
    Ok((means, cov, mean_inverse, reliability))
}

/// Column-major design `(1, column, Z)` with `column` at index 1.
fn design_with(column: &[f64], data: &LinearErrorData<'_>, n: usize, d: usize) -> Vec<f64> {
    let mut design = vec![1.0; n * d];
    design[n..2 * n].copy_from_slice(column);
    design[2 * n..].copy_from_slice(data.covariates);
    design
}

/// Fit the outcome model on a column-major design.
fn fit_family(
    family: OutcomeFamily,
    design: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
) -> Result<Vec<f64>, MeasurementErrorError> {
    match family {
        OutcomeFamily::Linear => {
            // Normal equations on a design of at most 66 columns; a
            // Cholesky failure is a rank-deficient design.
            let mut gram = vec![0.0; d * d];
            let mut cross = vec![0.0; d];
            for a in 0..d {
                let ca = &design[a * n..(a + 1) * n];
                cross[a] = ca.iter().zip(y).map(|(x, v)| x * v).sum();
                for b in a..d {
                    let cb = &design[b * n..(b + 1) * n];
                    let s: f64 = ca.iter().zip(cb).map(|(x, z)| x * z).sum();
                    gram[a * d + b] = s;
                    gram[b * d + a] = s;
                }
            }
            solve_spd(&gram, d, &cross).ok_or(MeasurementErrorError::SingularDesign)
        }
        OutcomeFamily::Logistic => {
            let mut workspace = LeastSquaresWorkspace::default();
            let fit = fit_glm(
                GlmFamily::BinomialLogit,
                GlmDesignRef { x_colmajor: design, nrows: n, ncols: d, y },
                &FaerBackend,
                &mut workspace,
                &GlmOptions::default().without_separation_ridge(),
            )
            .map_err(|e| MeasurementErrorError::OutcomeModelNotFitted(e.to_string()))?;
            if fit.separated {
                return Err(MeasurementErrorError::OutcomeModelNotFitted(
                    "the logistic outcome model is (quasi-)completely separated; no finite maximum-likelihood estimate exists".to_owned(),
                ));
            }
            if !fit.converged {
                return Err(MeasurementErrorError::OutcomeModelNotFitted(
                    "the logistic outcome model did not converge".to_owned(),
                ));
            }
            if fit.coefficients.len() != d || fit.coefficients.iter().any(|c| !c.is_finite()) {
                return Err(MeasurementErrorError::OutcomeModelNotFitted(
                    "the outcome model returned non-finite coefficients".to_owned(),
                ));
            }
            Ok(fit.coefficients)
        }
    }
}

fn check_family_outcome(family: OutcomeFamily, y: &[f64]) -> Result<(), MeasurementErrorError> {
    if family == OutcomeFamily::Logistic {
        #[allow(
            clippy::float_cmp,
            reason = "binary outcomes must be exactly the coded values 0 or 1"
        )]
        let binary = y.iter().all(|&v| v == 0.0 || v == 1.0);
        if !binary {
            return Err(MeasurementErrorError::InvalidParameter(
                "a logistic outcome model needs outcomes coded exactly 0 or 1",
            ));
        }
    }
    Ok(())
}

/// Regression calibration: fit the outcome model on `(1, E[X | W, Z], Z)`.
///
/// `E[X | W, Z]` is the best linear predictor from the sample moments of
/// `(W, Z)` with `Var(X) = Var(W) - sigma_u^2 mean(1/k)`; a row averaging
/// `k_i` replicates uses the covariance of its own mean, so rows with more
/// replicates are shrunk less.
///
/// # Errors
/// Shape, finiteness or rank refusals; an implied reliability at or below
/// `reliability_floor` ([`MeasurementErrorError::ReliabilityNotPositive`]);
/// an outcome that is not 0/1 for the logistic family; or an outcome model
/// that does not fit ([`MeasurementErrorError::OutcomeModelNotFitted`]).
pub fn regression_calibration(
    data: &LinearErrorData<'_>,
    family: OutcomeFamily,
    error_variance: f64,
    reliability_floor: f64,
) -> Result<CalibratedFit, MeasurementErrorError> {
    let shape = check_data(data)?;
    check_error_inputs(error_variance, reliability_floor)?;
    check_family_outcome(family, data.y)?;
    let (means, cov, mean_inverse, reliability) =
        checked_reliability(data, &shape, error_variance, reliability_floor)?;
    let Shape { n, p } = shape;
    let q = p + 1;
    let d = p + 2;
    // Cov(X, V) = (Var(X), Cov(W, Z)).
    let var_x = cov[0] - error_variance * mean_inverse;
    let mut target = cov[..q].to_vec();
    target[0] = var_x;
    // One predictor per distinct replicate count.
    let mut predictors: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut x_hat = Vec::with_capacity(n);
    let mut deviation = vec![0.0; q];
    for i in 0..n {
        let k = data.replicate_counts.map_or(1, |counts| counts[i]);
        let c = match predictors.entry(k) {
            Entry::Occupied(slot) => slot.into_mut(),
            Entry::Vacant(slot) => {
                let mut row_cov = cov.clone();
                row_cov[0] = var_x + error_variance / f64::from(k);
                let c =
                    solve_spd(&row_cov, q, &target).ok_or(MeasurementErrorError::SingularDesign)?;
                slot.insert(c)
            }
        };
        deviation[0] = data.w[i] - means[0];
        for j in 0..p {
            deviation[1 + j] = data.covariates[j * n + i] - means[1 + j];
        }
        let shift: f64 = c.iter().zip(&deviation).map(|(a, b)| a * b).sum();
        x_hat.push(means[0] + shift);
    }
    let calibrated = design_with(&x_hat, data, n, d);
    let coefficients = fit_family(family, &calibrated, n, d, data.y)?;
    let observed = design_with(data.w, data, n, d);
    let naive_coefficients = fit_family(family, &observed, n, d, data.y)?;
    Ok(CalibratedFit { family, coefficients, naive_coefficients, reliability, error_variance })
}

/// Least-squares fit of `means[g][c]` on the extrapolant basis in `lambda`,
/// evaluated at `lambda = -1`, for every coefficient `c`.
#[allow(
    clippy::needless_range_loop,
    reason = "the coefficient index selects one entry of every grid point's mean vector"
)]
fn extrapolate(
    lambdas: &[f64],
    means: &[Vec<f64>],
    extrapolant: SimexExtrapolant,
    d: usize,
) -> Result<Vec<f64>, MeasurementErrorError> {
    let m = extrapolant.width();
    let basis = |l: f64| -> Vec<f64> {
        let mut row = vec![1.0, l];
        if m == 3 {
            row.push(l * l);
        }
        row
    };
    let mut gram = vec![0.0; m * m];
    for &l in lambdas {
        let row = basis(l);
        for a in 0..m {
            for b in 0..m {
                gram[a * m + b] += row[a] * row[b];
            }
        }
    }
    let chol = cholesky_spd(&gram, m).ok_or(MeasurementErrorError::SingularDesign)?;
    let at = basis(-1.0);
    let mut out = Vec::with_capacity(d);
    for c in 0..d {
        let mut cross = vec![0.0; m];
        for (l, mean) in lambdas.iter().zip(means) {
            for (slot, r) in cross.iter_mut().zip(basis(*l)) {
                *slot += r * mean[c];
            }
        }
        let coef = chol_solve(&chol, m, &cross).ok_or(MeasurementErrorError::SingularDesign)?;
        out.push(coef.iter().zip(&at).map(|(a, b)| a * b).sum());
    }
    Ok(out)
}

/// SIMEX for a linear or logistic outcome model.
///
/// For each `lambda` in the grid, `config.replicates` remeasured columns
/// `W + sqrt(lambda sigma_u^2 / k_i) e_i`, `e_i ~ N(0, 1)` from `rng`, are
/// fitted and their coefficients averaged; the averages (with the naive fit
/// at `lambda = 0`) are extrapolated to `lambda = -1`. Draws are taken from
/// `rng` in grid order, then replicate order, then row order, so the result
/// is a function of the inputs and the stream.
///
/// # Errors
/// As [`regression_calibration`], and an invalid [`SimexConfig`].
pub fn simex(
    data: &LinearErrorData<'_>,
    family: OutcomeFamily,
    error_variance: f64,
    reliability_floor: f64,
    config: &SimexConfig,
    rng: &mut CausalRng,
) -> Result<SimexFit, MeasurementErrorError> {
    let shape = check_data(data)?;
    check_error_inputs(error_variance, reliability_floor)?;
    check_family_outcome(family, data.y)?;
    config.validate()?;
    let (_, _, _, reliability) =
        checked_reliability(data, &shape, error_variance, reliability_floor)?;
    let Shape { n, p } = shape;
    let d = p + 2;
    let mut design = design_with(data.w, data, n, d);
    let naive_coefficients = fit_family(family, &design, n, d, data.y)?;
    let mut grid = config.lambdas.clone();
    grid.sort_by(f64::total_cmp);
    let mut lambdas = Vec::with_capacity(grid.len() + 1);
    lambdas.push(0.0);
    let mut lambda_means = Vec::with_capacity(grid.len() + 1);
    lambda_means.push(naive_coefficients.clone());
    let scale: Vec<f64> = (0..n).map(|i| (error_variance * inverse_k(data, i)).sqrt()).collect();
    let replicates = f64::from(config.replicates);
    for &lambda in &grid {
        let root = lambda.sqrt();
        let mut sum = vec![0.0; d];
        for _ in 0..config.replicates {
            for (i, slot) in design[n..2 * n].iter_mut().enumerate() {
                *slot = data.w[i] + root * scale[i] * standard_normal(rng);
            }
            let coefficients = fit_family(family, &design, n, d, data.y)?;
            for (s, c) in sum.iter_mut().zip(&coefficients) {
                *s += c;
            }
        }
        lambdas.push(lambda);
        lambda_means.push(sum.into_iter().map(|s| s / replicates).collect());
    }
    let coefficients = extrapolate(&lambdas, &lambda_means, config.extrapolant, d)?;
    Ok(SimexFit {
        family,
        coefficients,
        naive_coefficients,
        lambdas,
        lambda_means,
        extrapolant: config.extrapolant,
        replicates: config.replicates,
        reliability,
        error_variance,
    })
}

/// The error variance implied by a declared reliability ratio
/// `lambda_r = Var(X) / Var(W)`: `sigma_u^2 = (1 - lambda_r) Var(W)`.
///
/// The value is **data dependent**: it is computed from the same observed
/// measurement it later corrects. The reported variance of the estimate is
/// the delta-method sum `Var(W)^2 se(lambda_r)^2 + (1 - lambda_r)^2
/// Var(s_W^2)`, with `Var(s_W^2) = (m_4 - s^4) / n` from the sample fourth
/// moment; it treats the declared reliability as independent of the sample
/// and ignores the dependence of `s_W^2` on the outcome score, so it is a
/// first-order, uncalibrated variance.
///
/// # Errors
/// A reliability outside `(0, 1]`, a negative or non-finite standard error,
/// fewer than three observations, non-finite values, or a constant
/// measurement refuse.
pub fn error_variance_from_reliability(
    reliability: f64,
    reliability_standard_error: f64,
    observed: &[f64],
) -> Result<ReliabilityErrorVariance, MeasurementErrorError> {
    if !(reliability.is_finite() && reliability > 0.0 && reliability <= 1.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "a reliability ratio must lie in (0, 1]",
        ));
    }
    if !(reliability_standard_error.is_finite() && reliability_standard_error >= 0.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "the reliability standard error must be finite and non-negative",
        ));
    }
    let n = observed.len();
    if n < 3 {
        return Err(MeasurementErrorError::TooFewRows { rows: n, minimum: 3 });
    }
    if n > MAX_MEASUREMENT_ROWS {
        return Err(MeasurementErrorError::Shape("exceeds the declared row bound"));
    }
    if observed.iter().any(|v| !v.is_finite()) {
        return Err(MeasurementErrorError::NonFinite("observed measurement"));
    }
    let n_f = n as f64;
    let mean = observed.iter().sum::<f64>() / n_f;
    let squares: Vec<f64> = observed.iter().map(|v| (v - mean) * (v - mean)).collect();
    let variance = squares.iter().sum::<f64>() / (n_f - 1.0);
    if variance.is_nan() || variance <= 0.0 {
        return Err(MeasurementErrorError::NonPositiveVariance("observed measurement"));
    }
    let m2 = squares.iter().sum::<f64>() / n_f;
    let m4 = squares.iter().map(|s| s * s).sum::<f64>() / n_f;
    let variance_of_variance = ((m4 - m2 * m2) / n_f).max(0.0);
    let one_minus = 1.0 - reliability;
    let estimate_variance = variance * variance * reliability_standard_error.powi(2)
        + one_minus * one_minus * variance_of_variance;
    Ok(ReliabilityErrorVariance {
        error_variance: ErrorVariance {
            variance: one_minus * variance,
            estimate_variance,
            influence: None,
        },
        observed_variance: variance,
        reliability,
        reliability_standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement_error::LinearMoments;

    fn normal(rng: &mut CausalRng) -> f64 {
        standard_normal(rng)
    }

    fn sigmoid(eta: f64) -> f64 {
        1.0 / (1.0 + (-eta).exp())
    }

    /// `X ~ N(0, 1)`, `W = X + N(0, sigma_u^2)`, `Z = 0.5 X + N(0, 1)`.
    struct Sim {
        y: Vec<f64>,
        w: Vec<f64>,
        z: Vec<f64>,
    }

    fn linear_sim(n: usize, seed: u64, beta: f64, sigma_u2: f64) -> Sim {
        let mut rng = CausalRng::from_seed(seed);
        let mut sim = Sim { y: Vec::new(), w: Vec::new(), z: Vec::new() };
        for _ in 0..n {
            let x = normal(&mut rng);
            let z = normal(&mut rng);
            sim.w.push(x + sigma_u2.sqrt() * normal(&mut rng));
            sim.z.push(z);
            sim.y.push(1.0 + beta * x + 0.5 * z + 0.5 * normal(&mut rng));
        }
        sim
    }

    fn logistic_sim(n: usize, seed: u64, beta: f64, sigma_u2: f64) -> Sim {
        let mut rng = CausalRng::from_seed(seed);
        let mut sim = Sim { y: Vec::new(), w: Vec::new(), z: Vec::new() };
        for _ in 0..n {
            let x = normal(&mut rng);
            let z = 0.5 * x + normal(&mut rng);
            sim.w.push(x + sigma_u2.sqrt() * normal(&mut rng));
            sim.z.push(z);
            let p = sigmoid(-0.5 + beta * x + 0.3 * z);
            sim.y.push(if rng.next_f64() < p { 1.0 } else { 0.0 });
        }
        sim
    }

    fn data(sim: &Sim, with_z: bool) -> LinearErrorData<'_> {
        let none: &[f64] = &[];
        LinearErrorData {
            y: &sim.y,
            w: &sim.w,
            covariates: if with_z { sim.z.as_slice() } else { none },
            covariate_count: usize::from(with_z),
            replicate_counts: None,
        }
    }

    #[test]
    fn linear_calibration_equals_the_corrected_normal_equations() {
        let sim = linear_sim(800, 3, 2.0, 0.25);
        let d = data(&sim, true);
        let rc = regression_calibration(&d, OutcomeFamily::Linear, 0.25, 0.1).unwrap();
        let moments = LinearMoments::accumulate(&d, &ErrorVariance::known(0.25, 0.0).unwrap(), 0.1)
            .unwrap()
            .solve()
            .unwrap();
        for (a, b) in rc.coefficients.iter().zip(&moments.coefficients) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        for (a, b) in rc.naive_coefficients.iter().zip(&moments.naive_coefficients) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        assert!((rc.reliability - moments.reliability).abs() < 1e-12);
    }

    #[test]
    fn logistic_calibration_recovers_the_slope() {
        let sim = logistic_sim(6000, 17, 1.0, 0.25);
        let d = data(&sim, true);
        let rc = regression_calibration(&d, OutcomeFamily::Logistic, 0.25, 0.1).unwrap();
        let naive = rc.naive_coefficients[1];
        let corrected = rc.coefficients[1];
        assert!((corrected - 1.0).abs() < 0.15, "corrected {corrected}");
        assert!((corrected - 1.0).abs() < (naive - 1.0).abs(), "{corrected} vs naive {naive}");
        assert!(naive < 0.9, "naive {naive} should be attenuated");
    }

    #[test]
    fn simex_quadratic_removes_most_linear_attenuation() {
        // Reliability 1 / (1 + 0.25) = 0.8: the naive slope is about 1.6.
        let sim = linear_sim(2000, 5, 2.0, 0.25);
        let d = data(&sim, false);
        let mut rng = CausalRng::from_seed(99);
        let fit =
            simex(&d, OutcomeFamily::Linear, 0.25, 0.1, &SimexConfig::default(), &mut rng).unwrap();
        let naive = fit.naive_coefficients[1];
        let corrected = fit.coefficients[1];
        assert!((naive - 1.6).abs() < 0.1, "naive {naive}");
        assert!((corrected - 2.0).abs() < 0.1, "corrected {corrected}");
        assert!((corrected - 2.0).abs() < (naive - 2.0).abs());
        assert_eq!(fit.lambdas, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(fit.lambda_means.len(), 5);
        // The path decreases as error is added.
        assert!(fit.lambda_means.windows(2).all(|w| w[1][1] < w[0][1]));
    }

    #[test]
    fn simex_logistic_recovers_the_slope_and_is_deterministic() {
        let sim = logistic_sim(4000, 23, 1.0, 0.25);
        let d = data(&sim, true);
        let config = SimexConfig { replicates: 20, ..SimexConfig::default() };
        let mut rng = CausalRng::from_seed(7);
        let fit = simex(&d, OutcomeFamily::Logistic, 0.25, 0.1, &config, &mut rng).unwrap();
        let corrected = fit.coefficients[1];
        assert!((corrected - 1.0).abs() < 0.15, "corrected {corrected}");
        assert!((corrected - 1.0).abs() < (fit.naive_coefficients[1] - 1.0).abs());
        let mut again = CausalRng::from_seed(7);
        let repeat = simex(&d, OutcomeFamily::Logistic, 0.25, 0.1, &config, &mut again).unwrap();
        assert_eq!(fit, repeat);
    }

    #[test]
    fn calibration_refuses_at_the_reliability_floor() {
        let sim = linear_sim(500, 9, 2.0, 0.25);
        let d = data(&sim, false);
        assert!(matches!(
            regression_calibration(&d, OutcomeFamily::Linear, 10.0, 0.1),
            Err(MeasurementErrorError::ReliabilityNotPositive { .. })
        ));
        assert!(matches!(
            regression_calibration(&d, OutcomeFamily::Linear, 0.25, 0.95),
            Err(MeasurementErrorError::ReliabilityNotPositive { .. })
        ));
    }

    #[test]
    fn logistic_family_refuses_a_non_binary_outcome() {
        let sim = linear_sim(200, 2, 1.0, 0.25);
        assert!(matches!(
            regression_calibration(&data(&sim, false), OutcomeFamily::Logistic, 0.25, 0.1),
            Err(MeasurementErrorError::InvalidParameter(_))
        ));
    }

    #[test]
    fn simex_config_refuses_a_bad_grid() {
        let bad = SimexConfig { lambdas: vec![1.0, 1.0], ..SimexConfig::default() };
        assert!(bad.validate().is_err());
        let short = SimexConfig { lambdas: vec![1.0], ..SimexConfig::default() };
        assert!(short.validate().is_err());
        let linear = SimexConfig {
            lambdas: vec![1.0],
            extrapolant: SimexExtrapolant::Linear,
            replicates: 5,
        };
        assert!(linear.validate().is_ok());
        let negative = SimexConfig { lambdas: vec![-0.5, 1.0], ..SimexConfig::default() };
        assert!(negative.validate().is_err());
    }

    #[test]
    fn quadratic_extrapolation_is_exact_for_a_quadratic_path() {
        let lambdas = [0.0, 0.5, 1.0, 1.5, 2.0];
        let means: Vec<Vec<f64>> =
            lambdas.iter().map(|l| vec![3.0 - 0.5 * l + 0.25 * l * l]).collect();
        let at = extrapolate(&lambdas, &means, SimexExtrapolant::Quadratic, 1).unwrap();
        assert!((at[0] - (3.0 + 0.5 + 0.25)).abs() < 1e-12, "{at:?}");
    }

    #[test]
    fn reliability_ratio_gives_the_error_variance_and_its_delta_variance() {
        // Values -2, -1, 0, 1, 2: s^2 = 2.5, m2 = 2, m4 = (16 + 1 + 0 + 1 + 16) / 5 = 6.8.
        let observed = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let derived = error_variance_from_reliability(0.8, 0.05, &observed).unwrap();
        assert!((derived.observed_variance - 2.5).abs() < 1e-12);
        assert!((derived.error_variance.variance - 0.5).abs() < 1e-12);
        let expected = 2.5 * 2.5 * 0.05 * 0.05 + 0.2 * 0.2 * (6.8 - 4.0) / 5.0;
        assert!((derived.error_variance.estimate_variance - expected).abs() < 1e-12);
        assert!(error_variance_from_reliability(0.0, 0.0, &observed).is_err());
        assert!(error_variance_from_reliability(1.2, 0.0, &observed).is_err());
        assert!(error_variance_from_reliability(0.8, 0.0, &[1.0, 1.0, 1.0]).is_err());
    }
}
