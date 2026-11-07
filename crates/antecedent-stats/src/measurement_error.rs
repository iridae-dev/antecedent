//! Pure corrections for uncertain measurements.
//!
//! Every function here is closed-form or a single linear solve over declared
//! measurement parameters. Nothing here decides whether a correction is
//! *supported*: the semantic contract (`antecedent-contract`) and the estimator
//! layer (`antecedent-estimate`) own population, regime and arm binding, the
//! differential versus non-differential declaration, and the refusal codes.
//!
//! # Routes
//!
//! * **Classical continuous error** ([`LinearMoments`]). The error-prone
//!   measurement is `W = X + U` with `U` independent of `(X, Z, Y)`, mean zero
//!   and variance `sigma_u^2` (per replicate). With `k_i` replicates averaged
//!   in row `i`, the error of the row mean has variance `sigma_u^2 / k_i`.
//!   For a linear outcome model `E[Y | X, Z] = V' beta`, `V = (1, X, Z)`, the
//!   corrected score is `V (Y - V' beta) + S beta` with `S = diag(0, sigma_u^2 / k, 0)`,
//!   whose root is `(M - mean(S))^{-1} r` (`M = mean(V V')`, `r = mean(V Y)`).
//!   For a linear outcome model this root is algebraically the regression
//!   calibration estimate: replace `X` by `E[X | W, Z]` computed from the same
//!   sample moments and run least squares (Carroll, Ruppert, Stefanski and
//!   Crainiceanu 2006, ch. 4; the identity is checked in the tests).
//! * **Binary misclassification.** The Rogan-Gladen inversion
//!   `pi = (p* + Sp - 1) / (Se + Sp - 1)` and the risk difference
//!   `(p1* - p0*) / J`, with `J = Se + Sp - 1` (Youden's index).
//! * **Categorical misclassification.** `pi = C^{-1} p*` for a column
//!   stochastic confusion matrix `C[observed, true]`.
//!
//! # Variance
//!
//! Variances are delta-method (first-order) variances that include the
//! sampling variance of the downstream quantity *and* the estimated
//! measurement parameters (error variance, sensitivity/specificity, confusion
//! columns). They carry no coverage claim: a standard error from this module
//! is an uncalibrated estimate of an asymptotic standard deviation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

use crate::gram::{invert_square, solve_spd};

mod classification;
mod imputation;
pub use classification::{
    BinaryErrorRates, ConfusionMatrix, CorrectedCategorical, CorrectedProportion,
    CorrectedRiskDifference, ObservedClassSummary, TwoByTwoCounts, corrected_categories,
    corrected_prevalence, corrected_risk_difference_differential,
    corrected_risk_difference_nondifferential,
};

pub use imputation::{
    CombinedImputations, combine_imputations_linear, posterior_means, rubin_combine,
};

/// Most rows one measurement-error correction accepts.
pub const MAX_MEASUREMENT_ROWS: usize = 1 << 26;
/// Most error-free covariates the linear correction accepts.
pub const MAX_MEASUREMENT_COVARIATES: usize = 64;
/// Most categories a confusion matrix may have.
pub const MAX_MEASUREMENT_CATEGORIES: usize = 16;

/// Why a measurement-error correction was refused.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum MeasurementErrorError {
    /// Inputs disagree in length or exceed a declared bound.
    #[error("shape error: {0}")]
    Shape(&'static str),
    /// A non-finite value where a finite one is required.
    #[error("non-finite input: {0}")]
    NonFinite(&'static str),
    /// A declared parameter is outside its range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(&'static str),
    /// Too few rows for the design.
    #[error("too few rows: {rows} (need {minimum})")]
    TooFewRows {
        /// Rows supplied.
        rows: usize,
        /// Rows needed.
        minimum: usize,
    },
    /// The design (without correction) is rank deficient.
    #[error("the observed design is singular")]
    SingularDesign,
    /// The error variance is too large for the observed variance: the implied
    /// reliability of the measurement is not above the declared floor.
    #[error("implied reliability {reliability} is not above the declared floor {floor}")]
    ReliabilityNotPositive {
        /// Implied reliability of the error-prone measure given the covariates.
        reliability: f64,
        /// Declared floor.
        floor: f64,
    },
    /// Replicate data cannot identify a within-unit variance.
    #[error("replicates cannot identify an error variance: {0}")]
    InsufficientReplicates(&'static str),
    /// Youden's index is not above the declared minimum.
    #[error("Youden index {youden} is not above the declared minimum {minimum}")]
    YoudenTooSmall {
        /// `sensitivity + specificity - 1`.
        youden: f64,
        /// Declared minimum.
        minimum: f64,
    },
    /// The confusion matrix is singular or worse conditioned than declared.
    #[error(
        "confusion matrix is singular or ill-conditioned (condition {condition}, maximum {maximum})"
    )]
    ConfusionIllConditioned {
        /// 1-norm condition number, infinite when singular.
        condition: f64,
        /// Declared maximum.
        maximum: f64,
    },
    /// A latent category has zero corrected mass so its outcome mean is not defined.
    #[error("a latent category has zero corrected mass")]
    EmptyLatentCategory,
    /// A variance came out materially non-positive.
    #[error("non-positive variance: {0}")]
    NonPositiveVariance(&'static str),
    /// The outcome model could not be fitted: a logistic fit that did not
    /// converge or is separated, or an outcome outside the family's support.
    #[error("outcome model not fitted: {0}")]
    OutcomeModelNotFitted(String),
}

fn finite_all(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn matmul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            for j in 0..d {
                out[i * d + j] += aik * b[k * d + j];
            }
        }
    }
    out
}

fn transpose(a: &[f64], d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            out[j * d + i] = a[i * d + j];
        }
    }
    out
}

fn matvec(a: &[f64], v: &[f64], d: usize) -> Vec<f64> {
    (0..d).map(|i| (0..d).map(|j| a[i * d + j] * v[j]).sum()).collect()
}

// ---------------------------------------------------------------------------
// Replicate variance components and error-variance estimates
// ---------------------------------------------------------------------------

/// One-way random-effects variance components from replicated measurements.
#[derive(Clone, Debug, PartialEq)]
pub struct VarianceComponents {
    /// Units (subjects) with at least one measurement.
    pub units: usize,
    /// Total measurements.
    pub observations: usize,
    /// Grand mean of all measurements.
    pub grand_mean: f64,
    /// Within-unit (measurement-error) variance, the pooled mean square within.
    pub within_variance: f64,
    /// Between-unit variance of the latent value `(MSB - MSW) / n0`; may be negative.
    pub between_variance: f64,
    /// Intraclass correlation of a single measurement, `between / (between + within)`.
    pub intraclass_correlation: f64,
    /// Degrees of freedom of the within variance, `N - m`.
    pub within_df: f64,
    /// Effective replicates per unit `n0` for unbalanced designs (Searle).
    pub effective_replicates: f64,
    /// Variance of the estimated within variance (ratio-estimator linearization;
    /// distribution free).
    pub within_variance_estimate_variance: f64,
}

/// Estimated or declared classical error variance of one measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct ErrorVariance {
    /// Error variance of one measurement.
    pub variance: f64,
    /// Variance of the estimate of `variance`; zero when declared exact.
    pub estimate_variance: f64,
    /// Per-row influence of the estimate when it was computed from the main
    /// sample's own replicates (so its dependence with the outcome score is
    /// retained); `None` when it comes from an independent sample or is declared.
    pub influence: Option<Vec<f64>>,
}

impl ErrorVariance {
    /// A declared error variance with a standard error of the declaration
    /// (use `0.0` for a value declared exact).
    ///
    /// # Errors
    /// A negative or non-finite variance or standard error refuses.
    pub fn known(variance: f64, standard_error: f64) -> Result<Self, MeasurementErrorError> {
        if !(variance.is_finite() && variance >= 0.0) {
            return Err(MeasurementErrorError::InvalidParameter(
                "error variance must be finite and non-negative",
            ));
        }
        if !(standard_error.is_finite() && standard_error >= 0.0) {
            return Err(MeasurementErrorError::InvalidParameter(
                "error variance standard error must be finite and non-negative",
            ));
        }
        Ok(Self { variance, estimate_variance: standard_error * standard_error, influence: None })
    }
}

/// Per-unit within sum of squares and replicate counts for flattened groups.
fn group_statistics(
    values: &[f64],
    sizes: &[u32],
) -> Result<(Vec<f64>, Vec<f64>, usize), MeasurementErrorError> {
    let total: usize = sizes.iter().map(|&k| k as usize).sum();
    if sizes.is_empty() || values.len() != total {
        return Err(MeasurementErrorError::Shape("replicate values do not match the group sizes"));
    }
    if sizes.contains(&0) {
        return Err(MeasurementErrorError::Shape("every unit needs at least one measurement"));
    }
    if !finite_all(values) {
        return Err(MeasurementErrorError::NonFinite("replicate values"));
    }
    let mut means = Vec::with_capacity(sizes.len());
    let mut ssw = Vec::with_capacity(sizes.len());
    let mut at = 0_usize;
    for &k in sizes {
        let k = k as usize;
        let group = &values[at..at + k];
        let mean = group.iter().sum::<f64>() / k as f64;
        means.push(mean);
        ssw.push(group.iter().map(|v| (v - mean) * (v - mean)).sum());
        at += k;
    }
    Ok((means, ssw, total))
}

/// One-way random-effects variance components, unbalanced designs allowed.
///
/// Returns the components and the per-unit influence of the within variance
/// estimate (`(SSW_j - (k_j - 1) sigma^2) / mean(k - 1)`).
///
/// # Errors
/// Fewer than two units, no replicated unit, non-finite values or a zero
/// group size refuse.
pub fn replicate_variance_components(
    values: &[f64],
    sizes: &[u32],
) -> Result<(VarianceComponents, Vec<f64>), MeasurementErrorError> {
    let (means, ssw, total) = group_statistics(values, sizes)?;
    let m = sizes.len();
    if m < 2 {
        return Err(MeasurementErrorError::InsufficientReplicates("fewer than two units"));
    }
    if m > MAX_MEASUREMENT_ROWS {
        return Err(MeasurementErrorError::Shape("too many units"));
    }
    let df_within = total - m;
    if df_within == 0 {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "no unit has more than one measurement",
        ));
    }
    let n_obs = total as f64;
    let m_f = m as f64;
    let grand = values.iter().sum::<f64>() / n_obs;
    let ssb: f64 = sizes
        .iter()
        .zip(&means)
        .map(|(&k, &mean)| f64::from(k) * (mean - grand) * (mean - grand))
        .sum();
    let ssw_total: f64 = ssw.iter().sum();
    let msw = ssw_total / df_within as f64;
    let msb = ssb / (m_f - 1.0);
    let sum_k2: f64 = sizes.iter().map(|&k| f64::from(k) * f64::from(k)).sum();
    let n0 = (n_obs - sum_k2 / n_obs) / (m_f - 1.0);
    let between = (msb - msw) / n0;
    let nu_bar = df_within as f64 / m_f;
    let influence: Vec<f64> =
        sizes.iter().zip(&ssw).map(|(&k, &s)| (s - f64::from(k - 1) * msw) / nu_bar).collect();
    let estimate_variance = influence.iter().map(|v| v * v).sum::<f64>() / (m_f * m_f);
    let denom = between + msw;
    let icc = if denom > 0.0 { between / denom } else { f64::NAN };
    Ok((
        VarianceComponents {
            units: m,
            observations: total,
            grand_mean: grand,
            within_variance: msw,
            between_variance: between,
            intraclass_correlation: icc,
            within_df: df_within as f64,
            effective_replicates: n0,
            within_variance_estimate_variance: estimate_variance,
        },
        influence,
    ))
}

impl ErrorVariance {
    /// Estimate the error variance from the main sample's own replicates.
    ///
    /// The returned value keeps the per-unit influence, so a downstream
    /// [`LinearMoments`] retains the covariance between the outcome score and
    /// the variance estimate. `sizes` must be the replicate counts of the main
    /// sample rows, in row order.
    ///
    /// # Errors
    /// See [`replicate_variance_components`].
    pub fn from_main_sample_replicates(
        values: &[f64],
        sizes: &[u32],
    ) -> Result<(Self, VarianceComponents), MeasurementErrorError> {
        let (components, influence) = replicate_variance_components(values, sizes)?;
        Ok((
            Self {
                variance: components.within_variance,
                estimate_variance: components.within_variance_estimate_variance,
                influence: Some(influence),
            },
            components,
        ))
    }

    /// Estimate the error variance from an independent replicate study.
    ///
    /// # Errors
    /// See [`replicate_variance_components`].
    pub fn from_external_replicates(
        values: &[f64],
        sizes: &[u32],
    ) -> Result<(Self, VarianceComponents), MeasurementErrorError> {
        let (components, _) = replicate_variance_components(values, sizes)?;
        Ok((
            Self {
                variance: components.within_variance,
                estimate_variance: components.within_variance_estimate_variance,
                influence: None,
            },
            components,
        ))
    }
}

/// Summary of a validation sample with a reference measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationSummary {
    /// Validation units.
    pub units: usize,
    /// Mean of `W - X`; zero under the classical (unbiased) error model.
    pub error_mean: f64,
    /// Standard error of `error_mean`.
    pub error_mean_standard_error: f64,
    /// Sample variance of `W - X`.
    pub error_variance: f64,
}

impl ErrorVariance {
    /// Estimate the error variance from a validation sample holding both the
    /// error-prone measure `w` and a reference `x`, independent of the main sample.
    ///
    /// # Errors
    /// Fewer than three units, mismatched lengths or non-finite values refuse.
    pub fn from_validation(
        w: &[f64],
        x: &[f64],
    ) -> Result<(Self, ValidationSummary), MeasurementErrorError> {
        if w.len() != x.len() {
            return Err(MeasurementErrorError::Shape("validation vectors differ in length"));
        }
        if w.len() < 3 {
            return Err(MeasurementErrorError::TooFewRows { rows: w.len(), minimum: 3 });
        }
        if !finite_all(w) || !finite_all(x) {
            return Err(MeasurementErrorError::NonFinite("validation values"));
        }
        let n = w.len() as f64;
        let diffs: Vec<f64> = w.iter().zip(x).map(|(a, b)| a - b).collect();
        let mean = diffs.iter().sum::<f64>() / n;
        let sq: Vec<f64> = diffs.iter().map(|d| (d - mean) * (d - mean)).collect();
        let variance = sq.iter().sum::<f64>() / (n - 1.0);
        // Variance of the mean of the squared deviations (nonparametric).
        let sq_mean = sq.iter().sum::<f64>() / n;
        let sq_var = sq.iter().map(|s| (s - sq_mean) * (s - sq_mean)).sum::<f64>() / (n - 1.0);
        Ok((
            Self { variance, estimate_variance: sq_var / n, influence: None },
            ValidationSummary {
                units: w.len(),
                error_mean: mean,
                error_mean_standard_error: (variance / n).sqrt(),
                error_variance: variance,
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// Classical continuous error: corrected linear outcome model
// ---------------------------------------------------------------------------

/// Columns of one linear outcome analysis with an error-prone regressor.
///
/// The design is `V = (1, W, Z_1, ..., Z_p)`; `W` sits at coefficient index 1.
#[derive(Clone, Copy, Debug)]
pub struct LinearErrorData<'a> {
    /// Outcome.
    pub y: &'a [f64],
    /// Error-prone regressor; the row mean of the replicates when replicated.
    pub w: &'a [f64],
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: &'a [f64],
    /// Number of covariate columns.
    pub covariate_count: usize,
    /// Replicates averaged in each row of `w`; `None` means one each.
    pub replicate_counts: Option<&'a [u32]>,
}

/// Sufficient moments of a corrected linear fit.
///
/// Everything needed to recompute the corrected coefficients and their
/// covariance is here; [`LinearMoments::solve`] is a pure function of these
/// numbers, so a consumer holding only the moments reproduces the result.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearMoments {
    /// Rows.
    pub n: usize,
    /// Design width `p + 2`.
    pub dim: usize,
    /// `M = mean(V V')`, row-major `dim x dim`.
    pub gram: Vec<f64>,
    /// `r = mean(V Y)`.
    pub cross: Vec<f64>,
    /// Meat of the corrected score at the corrected coefficients,
    /// `mean(psi psi')`, row-major.
    pub meat: Vec<f64>,
    /// Meat of the uncorrected least-squares score, for the naive comparison.
    pub naive_meat: Vec<f64>,
    /// `mean(psi_i xi_i)`, the covariance of the corrected score with the
    /// influence of the error-variance estimate (zero when independent).
    pub influence_cross: Vec<f64>,
    /// `n * Var(estimated error variance)` for an independent estimate, or
    /// `mean(xi_i^2)` when the influence is retained.
    pub influence_variance: f64,
    /// Error variance of one measurement.
    pub error_variance: f64,
    /// `mean(1 / k_i)`.
    pub mean_inverse_replicates: f64,
    /// Floor on the implied reliability.
    pub reliability_floor: f64,
}

/// A corrected linear fit with its covariance components.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedLinear {
    /// Rows.
    pub n: usize,
    /// Design width; coefficient index 0 is the intercept and 1 the corrected regressor.
    pub dim: usize,
    /// Corrected coefficients.
    pub coefficients: Vec<f64>,
    /// Uncorrected least-squares coefficients.
    pub naive_coefficients: Vec<f64>,
    /// Full covariance, including the error-variance estimate's uncertainty.
    pub covariance: Vec<f64>,
    /// Covariance treating the error variance as known exactly.
    pub covariance_given_error_variance: Vec<f64>,
    /// HC0 covariance of the uncorrected fit.
    pub naive_covariance: Vec<f64>,
    /// Implied reliability of the error-prone regressor given the covariates,
    /// `1 - sigma_u^2 mean(1/k) / Var(W | Z)`.
    pub reliability: f64,
    /// Error variance used.
    pub error_variance: f64,
}

impl CorrectedLinear {
    /// Standard error of coefficient `index` from the full covariance.
    #[must_use]
    pub fn standard_error(&self, index: usize) -> f64 {
        self.covariance[index * self.dim + index].sqrt()
    }

    /// Value and variance of a linear functional `weights' beta`.
    ///
    /// This is how the measurement uncertainty enters any downstream linear
    /// functional of the coefficients: the variance is `w' Cov w` under the
    /// joint covariance, never the downstream sampling variance alone.
    ///
    /// # Errors
    /// A weight vector of the wrong length refuses.
    pub fn linear_functional(&self, weights: &[f64]) -> Result<(f64, f64), MeasurementErrorError> {
        if weights.len() != self.dim || !finite_all(weights) {
            return Err(MeasurementErrorError::Shape("functional weights do not match the design"));
        }
        let value: f64 = weights.iter().zip(&self.coefficients).map(|(w, b)| w * b).sum();
        let cov_w = matvec(&self.covariance, weights, self.dim);
        let variance: f64 = weights.iter().zip(&cov_w).map(|(w, c)| w * c).sum();
        Ok((value, variance))
    }
}

impl LinearMoments {
    /// Accumulate the sufficient moments from row data.
    ///
    /// # Errors
    /// Shape, finiteness, rank, or reliability refusals; see
    /// [`MeasurementErrorError`]. A reliability at or below
    /// `reliability_floor` refuses: the error variance leaves no latent
    /// variance to correct toward.
    #[allow(
        clippy::too_many_lines,
        reason = "one pass accumulating every sufficient moment from the same rows, after a flat run of independent shape and finiteness checks"
    )]
    pub fn accumulate(
        data: &LinearErrorData<'_>,
        error: &ErrorVariance,
        reliability_floor: f64,
    ) -> Result<Self, MeasurementErrorError> {
        let n = data.y.len();
        let p = data.covariate_count;
        if p > MAX_MEASUREMENT_COVARIATES || n > MAX_MEASUREMENT_ROWS {
            return Err(MeasurementErrorError::Shape(
                "exceeds the declared row or covariate bound",
            ));
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
        if !(finite_all(data.y) && finite_all(data.w) && finite_all(data.covariates)) {
            return Err(MeasurementErrorError::NonFinite("outcome, regressor or covariates"));
        }
        let variance_ok = |v: f64| v.is_finite() && v >= 0.0;
        if !variance_ok(error.variance) || !variance_ok(error.estimate_variance) {
            return Err(MeasurementErrorError::InvalidParameter("error variance"));
        }
        if !(reliability_floor.is_finite() && (0.0..1.0).contains(&reliability_floor)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "reliability floor must lie in [0, 1)",
            ));
        }
        if let Some(influence) = &error.influence {
            if influence.len() != n || !finite_all(influence) {
                return Err(MeasurementErrorError::Shape("influence differs in length"));
            }
        }
        let n_f = n as f64;
        let inverse_k = |i: usize| -> f64 {
            data.replicate_counts.map_or(1.0, |counts| 1.0 / f64::from(counts[i]))
        };
        let mean_inverse = (0..n).map(inverse_k).sum::<f64>() / n_f;

        let fill = |i: usize, row: &mut [f64]| {
            row[0] = 1.0;
            row[1] = data.w[i];
            for j in 0..p {
                row[2 + j] = data.covariates[j * n + i];
            }
        };
        let mut row = vec![0.0; d];
        let mut gram = vec![0.0; d * d];
        let mut cross = vec![0.0; d];
        for i in 0..n {
            fill(i, &mut row[..]);
            for a in 0..d {
                cross[a] += row[a] * data.y[i];
                for b in 0..d {
                    gram[a * d + b] += row[a] * row[b];
                }
            }
        }
        for v in &mut gram {
            *v /= n_f;
        }
        for v in &mut cross {
            *v /= n_f;
        }
        let reliability = implied_reliability(&gram, d, error.variance * mean_inverse)?;
        if !(reliability > 0.0 && reliability >= reliability_floor) {
            return Err(MeasurementErrorError::ReliabilityNotPositive {
                reliability,
                floor: reliability_floor,
            });
        }
        let mut a = gram.clone();
        a[d + 1] -= error.variance * mean_inverse;
        let beta =
            solve_spd(&a, d, &cross).ok_or(MeasurementErrorError::ReliabilityNotPositive {
                reliability,
                floor: reliability_floor,
            })?;
        let naive = solve_spd(&gram, d, &cross).ok_or(MeasurementErrorError::SingularDesign)?;

        let mut meat = vec![0.0; d * d];
        let mut naive_meat = vec![0.0; d * d];
        let mut influence_cross = vec![0.0; d];
        let mut influence_sq = 0.0;
        let mut psi = vec![0.0; d];
        for i in 0..n {
            fill(i, &mut row[..]);
            let fitted: f64 = row.iter().zip(&beta).map(|(v, b)| v * b).sum();
            let fitted_naive: f64 = row.iter().zip(&naive).map(|(v, b)| v * b).sum();
            let e = data.y[i] - fitted;
            let e_naive = data.y[i] - fitted_naive;
            for a in 0..d {
                psi[a] = row[a] * e;
            }
            psi[1] += error.variance * inverse_k(i) * beta[1];
            for a in 0..d {
                for b in 0..d {
                    meat[a * d + b] += psi[a] * psi[b];
                    naive_meat[a * d + b] += row[a] * row[b] * e_naive * e_naive;
                }
            }
            if let Some(influence) = &error.influence {
                for a in 0..d {
                    influence_cross[a] += psi[a] * influence[i];
                }
                influence_sq += influence[i] * influence[i];
            }
        }
        for v in &mut meat {
            *v /= n_f;
        }
        for v in &mut naive_meat {
            *v /= n_f;
        }
        let influence_variance = if error.influence.is_some() {
            for v in &mut influence_cross {
                *v /= n_f;
            }
            influence_sq / n_f
        } else {
            n_f * error.estimate_variance
        };
        Ok(Self {
            n,
            dim: d,
            gram,
            cross,
            meat,
            naive_meat,
            influence_cross,
            influence_variance,
            error_variance: error.variance,
            mean_inverse_replicates: mean_inverse,
            reliability_floor,
        })
    }

    /// Solve the corrected normal equations and form the covariance.
    ///
    /// # Errors
    /// Inconsistent moments, a singular design, a reliability at or below the
    /// floor, or a non-positive coefficient variance refuse.
    pub fn solve(&self) -> Result<CorrectedLinear, MeasurementErrorError> {
        let d = self.dim;
        if d < 2
            || self.n == 0
            || self.gram.len() != d * d
            || self.meat.len() != d * d
            || self.naive_meat.len() != d * d
            || self.cross.len() != d
            || self.influence_cross.len() != d
        {
            return Err(MeasurementErrorError::Shape("moments are inconsistent"));
        }
        let all = [&self.gram, &self.cross, &self.meat, &self.naive_meat, &self.influence_cross];
        let moments_valid = all.iter().all(|v| finite_all(v))
            && self.influence_variance.is_finite()
            && self.influence_variance >= 0.0
            && self.error_variance.is_finite()
            && self.error_variance >= 0.0
            && self.mean_inverse_replicates > 0.0
            && self.mean_inverse_replicates <= 1.0;
        if !moments_valid {
            return Err(MeasurementErrorError::NonFinite("moments"));
        }
        let n_f = self.n as f64;
        let shift = self.error_variance * self.mean_inverse_replicates;
        let reliability = implied_reliability(&self.gram, d, shift)?;
        if !(reliability > 0.0 && reliability >= self.reliability_floor) {
            return Err(MeasurementErrorError::ReliabilityNotPositive {
                reliability,
                floor: self.reliability_floor,
            });
        }
        let mut a = self.gram.clone();
        a[d + 1] -= shift;
        let a_inv = invert_square(&a, d).ok_or(MeasurementErrorError::ReliabilityNotPositive {
            reliability,
            floor: self.reliability_floor,
        })?;
        let m_inv = invert_square(&self.gram, d).ok_or(MeasurementErrorError::SingularDesign)?;
        let beta = matvec(&a_inv, &self.cross, d);
        let naive = matvec(&m_inv, &self.cross, d);

        let sandwich = |inv: &[f64], meat: &[f64]| -> Vec<f64> {
            let left = matmul(inv, meat, d);
            let mut out = matmul(&left, &transpose(inv, d), d);
            for v in &mut out {
                *v /= n_f;
            }
            out
        };
        let given = sandwich(&a_inv, &self.meat);
        let naive_cov = sandwich(&m_inv, &self.naive_meat);

        // d beta / d sigma^2 = A^{-1} e_x beta_x mean(1/k).
        let c: Vec<f64> =
            (0..d).map(|i| a_inv[i * d + 1] * beta[1] * self.mean_inverse_replicates).collect();
        let u = matvec(&a_inv, &self.influence_cross, d);
        let mut total = given.clone();
        for i in 0..d {
            for j in 0..d {
                total[i * d + j] +=
                    (c[i] * u[j] + u[i] * c[j] + c[i] * c[j] * self.influence_variance) / n_f;
            }
        }
        if !(total[d + 1] > 0.0 && given[d + 1] > 0.0 && naive_cov[d + 1] > 0.0) {
            return Err(MeasurementErrorError::NonPositiveVariance(
                "corrected regressor coefficient",
            ));
        }
        Ok(CorrectedLinear {
            n: self.n,
            dim: d,
            coefficients: beta,
            naive_coefficients: naive,
            covariance: total,
            covariance_given_error_variance: given,
            naive_covariance: naive_cov,
            reliability,
            error_variance: self.error_variance,
        })
    }
}

/// `1 - shift / s`, with `s = 1 / (M^{-1})_{xx}` the residual variance of the
/// error-prone regressor given the rest of the design.
fn implied_reliability(gram: &[f64], d: usize, shift: f64) -> Result<f64, MeasurementErrorError> {
    let mut e_x = vec![0.0; d];
    e_x[1] = 1.0;
    let column = solve_spd(gram, d, &e_x).ok_or(MeasurementErrorError::SingularDesign)?;
    let inv_xx = column[1];
    if !(inv_xx.is_finite() && inv_xx > 0.0) {
        return Err(MeasurementErrorError::SingularDesign);
    }
    Ok(1.0 - shift * inv_xx)
}

#[cfg(test)]
mod tests;
