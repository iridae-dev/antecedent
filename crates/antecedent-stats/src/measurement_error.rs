//! Pure corrections for uncertain measurements.
//!
//! Every function here is closed-form or a single linear solve over declared
//! measurement parameters. Nothing here decides whether a correction is
//! *licensed*: the semantic contract (`antecedent-contract`) and the estimator
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

use crate::gram::{chol_solve, cholesky_spd, invert_square};

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
    #[error("confusion matrix is singular or ill-conditioned (condition {condition}, maximum {maximum})")]
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
}

fn finite_all(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn solve_spd(a: &[f64], d: usize, b: &[f64]) -> Option<Vec<f64>> {
    let chol = cholesky_spd(a, d)?;
    chol_solve(&chol, d, b)
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
        Ok(Self {
            variance,
            estimate_variance: standard_error * standard_error,
            influence: None,
        })
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
    let influence: Vec<f64> = sizes
        .iter()
        .zip(&ssw)
        .map(|(&k, &s)| (s - f64::from(k - 1) * msw) / nu_bar)
        .collect();
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
    pub fn accumulate(
        data: &LinearErrorData<'_>,
        error: &ErrorVariance,
        reliability_floor: f64,
    ) -> Result<Self, MeasurementErrorError> {
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
        if !(finite_all(data.y) && finite_all(data.w) && finite_all(data.covariates)) {
            return Err(MeasurementErrorError::NonFinite("outcome, regressor or covariates"));
        }
        if !(error.variance.is_finite() && error.variance >= 0.0)
            || !(error.estimate_variance.is_finite() && error.estimate_variance >= 0.0)
        {
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
        let beta = solve_spd(&a, d, &cross).ok_or(MeasurementErrorError::ReliabilityNotPositive {
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
        let all = [
            &self.gram,
            &self.cross,
            &self.meat,
            &self.naive_meat,
            &self.influence_cross,
        ];
        if !all.iter().all(|v| finite_all(v))
            || !self.influence_variance.is_finite()
            || self.influence_variance < 0.0
            || !self.error_variance.is_finite()
            || self.error_variance < 0.0
            || !(self.mean_inverse_replicates > 0.0 && self.mean_inverse_replicates <= 1.0)
        {
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
        let c: Vec<f64> = (0..d)
            .map(|i| a_inv[i * d + 1] * beta[1] * self.mean_inverse_replicates)
            .collect();
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

// ---------------------------------------------------------------------------
// Binary misclassification
// ---------------------------------------------------------------------------

/// Sensitivity and specificity with the sampling variances of their estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryErrorRates {
    /// `P(observed = 1 | true = 1)`.
    pub sensitivity: f64,
    /// `P(observed = 0 | true = 0)`.
    pub specificity: f64,
    /// Variance of the sensitivity estimate; zero when declared exact.
    pub sensitivity_variance: f64,
    /// Variance of the specificity estimate; zero when declared exact.
    pub specificity_variance: f64,
}

impl BinaryErrorRates {
    /// Validated rates.
    ///
    /// # Errors
    /// Rates outside `[0, 1]` or invalid variances refuse.
    pub fn new(
        sensitivity: f64,
        specificity: f64,
        sensitivity_variance: f64,
        specificity_variance: f64,
    ) -> Result<Self, MeasurementErrorError> {
        let unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        let var = |v: f64| v.is_finite() && v >= 0.0;
        if !(unit(sensitivity) && unit(specificity)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "sensitivity and specificity must lie in [0, 1]",
            ));
        }
        if !(var(sensitivity_variance) && var(specificity_variance)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "rate variances must be finite and non-negative",
            ));
        }
        Ok(Self { sensitivity, specificity, sensitivity_variance, specificity_variance })
    }

    /// Rates estimated from a validation sample against a reference standard.
    ///
    /// `true_positive + false_negative` units are truly positive and
    /// `true_negative + false_positive` truly negative. Variances are the
    /// binomial `p (1 - p) / n`.
    ///
    /// # Errors
    /// An empty reference class refuses.
    pub fn from_validation_counts(
        true_positive: u64,
        false_negative: u64,
        true_negative: u64,
        false_positive: u64,
    ) -> Result<Self, MeasurementErrorError> {
        let positives = true_positive + false_negative;
        let negatives = true_negative + false_positive;
        if positives == 0 || negatives == 0 {
            return Err(MeasurementErrorError::InvalidParameter(
                "validation sample needs both truly positive and truly negative units",
            ));
        }
        let se = true_positive as f64 / positives as f64;
        let sp = true_negative as f64 / negatives as f64;
        Self::new(se, sp, se * (1.0 - se) / positives as f64, sp * (1.0 - sp) / negatives as f64)
    }

    /// Youden's index `sensitivity + specificity - 1`.
    #[must_use]
    pub fn youden(&self) -> f64 {
        self.sensitivity + self.specificity - 1.0
    }

    fn require_youden(&self, minimum: f64) -> Result<f64, MeasurementErrorError> {
        if !(minimum.is_finite() && minimum > 0.0 && minimum < 1.0) {
            return Err(MeasurementErrorError::InvalidParameter(
                "minimum Youden index must lie in (0, 1)",
            ));
        }
        let j = self.youden();
        if !(j > minimum) {
            return Err(MeasurementErrorError::YoudenTooSmall { youden: j, minimum });
        }
        Ok(j)
    }
}

/// A corrected proportion with its variance components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedProportion {
    /// Observed (misclassified) proportion.
    pub observed: f64,
    /// Rogan-Gladen corrected proportion, not truncated to `[0, 1]`.
    pub corrected: f64,
    /// Youden's index used.
    pub youden: f64,
    /// Total delta-method variance.
    pub variance: f64,
    /// Part due to sampling of the observed proportion.
    pub variance_sampling: f64,
    /// Part due to the sensitivity and specificity estimates.
    pub variance_rates: f64,
    /// The raw correction left `[0, 1]`: a sign of sampling noise or a wrong
    /// declared rate; the value is reported unclipped.
    pub outside_unit_interval: bool,
}

fn check_count(events: u64, total: u64) -> Result<(), MeasurementErrorError> {
    if total == 0 || events > total {
        return Err(MeasurementErrorError::InvalidParameter(
            "events must not exceed a positive total",
        ));
    }
    Ok(())
}

/// Corrected prevalence (Rogan and Gladen 1978) with delta-method variance
/// including the uncertainty of the sensitivity and specificity.
///
/// # Errors
/// Invalid counts, or Youden's index at or below `min_youden`, refuse.
pub fn corrected_prevalence(
    events: u64,
    total: u64,
    rates: &BinaryErrorRates,
    min_youden: f64,
) -> Result<CorrectedProportion, MeasurementErrorError> {
    check_count(events, total)?;
    let j = rates.require_youden(min_youden)?;
    let n = total as f64;
    let p = events as f64 / n;
    let pi = (p + rates.specificity - 1.0) / j;
    let var_p = p * (1.0 - p) / n;
    let sampling = var_p / (j * j);
    let from_rates = ((1.0 - pi) * (1.0 - pi) * rates.specificity_variance
        + pi * pi * rates.sensitivity_variance)
        / (j * j);
    Ok(CorrectedProportion {
        observed: p,
        corrected: pi,
        youden: j,
        variance: sampling + from_rates,
        variance_sampling: sampling,
        variance_rates: from_rates,
        outside_unit_interval: !(0.0..=1.0).contains(&pi),
    })
}

/// Outcome events by exposure arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TwoByTwoCounts {
    /// Observed events among the exposed.
    pub exposed_events: u64,
    /// Exposed units.
    pub exposed_total: u64,
    /// Observed events among the unexposed.
    pub unexposed_events: u64,
    /// Unexposed units.
    pub unexposed_total: u64,
}

/// A corrected risk difference with its variance components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedRiskDifference {
    /// Observed risk difference.
    pub naive: f64,
    /// Corrected risk difference.
    pub corrected: f64,
    /// Corrected exposed-arm risk.
    pub risk_exposed: f64,
    /// Corrected unexposed-arm risk.
    pub risk_unexposed: f64,
    /// Total delta-method variance.
    pub variance: f64,
    /// Part due to sampling of the observed arm proportions.
    pub variance_sampling: f64,
    /// Part due to the error-rate estimates.
    pub variance_rates: f64,
    /// Youden's index of the exposed arm (the shared one when non-differential).
    pub youden_exposed: f64,
    /// Youden's index of the unexposed arm (equal to the exposed one when non-differential).
    pub youden_unexposed: f64,
    /// Either corrected arm risk left `[0, 1]`.
    pub outside_unit_interval: bool,
}

/// Risk difference under **non-differential** outcome misclassification: the
/// same sensitivity and specificity, from one validation sample, apply to both
/// exposure arms.
///
/// `RD = (p1* - p0*) / J`; the variance is
/// `[Var(p1*) + Var(p0*)] / J^2 + RD^2 (Var(Se) + Var(Sp)) / J^2`, which keeps
/// the covariance that the shared rates induce between the two arms.
///
/// # Errors
/// Invalid counts or Youden's index at or below `min_youden` refuse.
pub fn corrected_risk_difference_nondifferential(
    table: &TwoByTwoCounts,
    rates: &BinaryErrorRates,
    min_youden: f64,
) -> Result<CorrectedRiskDifference, MeasurementErrorError> {
    check_count(table.exposed_events, table.exposed_total)?;
    check_count(table.unexposed_events, table.unexposed_total)?;
    let j = rates.require_youden(min_youden)?;
    let n1 = table.exposed_total as f64;
    let n0 = table.unexposed_total as f64;
    let p1 = table.exposed_events as f64 / n1;
    let p0 = table.unexposed_events as f64 / n0;
    let naive = p1 - p0;
    let corrected = naive / j;
    let risk1 = (p1 + rates.specificity - 1.0) / j;
    let risk0 = (p0 + rates.specificity - 1.0) / j;
    let sampling = (p1 * (1.0 - p1) / n1 + p0 * (1.0 - p0) / n0) / (j * j);
    let from_rates =
        corrected * corrected * (rates.sensitivity_variance + rates.specificity_variance) / (j * j);
    Ok(CorrectedRiskDifference {
        naive,
        corrected,
        risk_exposed: risk1,
        risk_unexposed: risk0,
        variance: sampling + from_rates,
        variance_sampling: sampling,
        variance_rates: from_rates,
        youden_exposed: j,
        youden_unexposed: j,
        outside_unit_interval: !(0.0..=1.0).contains(&risk1) || !(0.0..=1.0).contains(&risk0),
    })
}

/// Risk difference under **differential** outcome misclassification: each
/// exposure arm has its own sensitivity and specificity, estimated from its
/// own independent validation sample.
///
/// # Errors
/// Invalid counts or either arm's Youden index at or below `min_youden` refuse.
pub fn corrected_risk_difference_differential(
    table: &TwoByTwoCounts,
    rates_exposed: &BinaryErrorRates,
    rates_unexposed: &BinaryErrorRates,
    min_youden: f64,
) -> Result<CorrectedRiskDifference, MeasurementErrorError> {
    check_count(table.exposed_events, table.exposed_total)?;
    check_count(table.unexposed_events, table.unexposed_total)?;
    let exposed =
        corrected_prevalence(table.exposed_events, table.exposed_total, rates_exposed, min_youden)?;
    let unexposed = corrected_prevalence(
        table.unexposed_events,
        table.unexposed_total,
        rates_unexposed,
        min_youden,
    )?;
    Ok(CorrectedRiskDifference {
        naive: exposed.observed - unexposed.observed,
        corrected: exposed.corrected - unexposed.corrected,
        risk_exposed: exposed.corrected,
        risk_unexposed: unexposed.corrected,
        variance: exposed.variance + unexposed.variance,
        variance_sampling: exposed.variance_sampling + unexposed.variance_sampling,
        variance_rates: exposed.variance_rates + unexposed.variance_rates,
        youden_exposed: exposed.youden,
        youden_unexposed: unexposed.youden,
        outside_unit_interval: exposed.outside_unit_interval || unexposed.outside_unit_interval,
    })
}

// ---------------------------------------------------------------------------
// Categorical misclassification
// ---------------------------------------------------------------------------

/// Column-stochastic confusion matrix `C[observed, true] = P(observed | true)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfusionMatrix {
    levels: usize,
    matrix: Vec<f64>,
    column_counts: Option<Vec<u64>>,
}

impl ConfusionMatrix {
    /// A matrix from row-major `[observed][true]` probabilities.
    ///
    /// `column_counts[k]` is the number of validation units whose reference
    /// category is `k` (the sample behind column `k`); `None` declares the
    /// matrix exact.
    ///
    /// # Errors
    /// A wrong shape, entries outside `[0, 1]`, columns not summing to one
    /// (tolerance `1e-9`), or zero validation counts refuse.
    pub fn new(
        levels: usize,
        matrix: Vec<f64>,
        column_counts: Option<Vec<u64>>,
    ) -> Result<Self, MeasurementErrorError> {
        if !(2..=MAX_MEASUREMENT_CATEGORIES).contains(&levels) {
            return Err(MeasurementErrorError::Shape("category count outside the declared bound"));
        }
        if matrix.len() != levels * levels {
            return Err(MeasurementErrorError::Shape("confusion matrix is not square"));
        }
        if !matrix.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "confusion entries must be probabilities",
            ));
        }
        for k in 0..levels {
            let sum: f64 = (0..levels).map(|j| matrix[j * levels + k]).sum();
            if (sum - 1.0).abs() > 1e-9 {
                return Err(MeasurementErrorError::InvalidParameter(
                    "each confusion column must sum to one",
                ));
            }
        }
        if let Some(counts) = &column_counts {
            if counts.len() != levels || counts.contains(&0) {
                return Err(MeasurementErrorError::InvalidParameter(
                    "validation counts need one positive count per true category",
                ));
            }
        }
        Ok(Self { levels, matrix, column_counts })
    }

    /// A matrix estimated from validation counts `[observed][true]`, row-major.
    ///
    /// # Errors
    /// An empty reference category refuses.
    pub fn from_validation_counts(
        levels: usize,
        counts: &[u64],
    ) -> Result<Self, MeasurementErrorError> {
        if !(2..=MAX_MEASUREMENT_CATEGORIES).contains(&levels) || counts.len() != levels * levels {
            return Err(MeasurementErrorError::Shape("validation counts are not square"));
        }
        let mut column_totals = vec![0_u64; levels];
        for j in 0..levels {
            for k in 0..levels {
                column_totals[k] += counts[j * levels + k];
            }
        }
        if column_totals.contains(&0) {
            return Err(MeasurementErrorError::InvalidParameter(
                "a true category has no validation units",
            ));
        }
        let matrix = (0..levels * levels)
            .map(|idx| counts[idx] as f64 / column_totals[idx % levels] as f64)
            .collect();
        Self::new(levels, matrix, Some(column_totals))
    }

    /// Number of categories.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Row-major `[observed][true]` probabilities.
    #[must_use]
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Validation units per true category, when the matrix is estimated.
    #[must_use]
    pub fn column_counts(&self) -> Option<&[u64]> {
        self.column_counts.as_deref()
    }
}

/// Observed-category summary of the outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedClassSummary {
    /// Units observed in the category.
    pub count: u64,
    /// Mean outcome among them.
    pub mean: f64,
    /// Sample variance of the outcome among them.
    pub variance: f64,
}

/// Corrected category proportions and per-category outcome means.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedCategorical {
    /// Observed category proportions.
    pub observed_proportions: Vec<f64>,
    /// Corrected true-category proportions (not projected onto the simplex).
    pub proportions: Vec<f64>,
    /// Covariance of the corrected proportions, row-major, including the
    /// uncertainty of the confusion columns.
    pub proportion_covariance: Vec<f64>,
    /// Corrected expected counts `n * proportions`.
    pub expected_counts: Vec<f64>,
    /// Mean outcome among units in each true category.
    pub outcome_means: Vec<f64>,
    /// Covariance of the outcome means, row-major.
    pub outcome_mean_covariance: Vec<f64>,
    /// 1-norm condition number of the confusion matrix.
    pub condition_number: f64,
    /// A corrected proportion left `[0, 1]`.
    pub outside_unit_interval: bool,
}

fn one_norm(a: &[f64], d: usize) -> f64 {
    (0..d).map(|j| (0..d).map(|i| a[i * d + j].abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// Outcome means by true category from the model map. `theta` packs the
/// observed-class means (K), observed proportions (K) and confusion entries
/// (K x K, row-major).
fn category_outcome_means(theta: &[f64], k: usize) -> Option<Vec<f64>> {
    let means = &theta[..k];
    let p_star = &theta[k..2 * k];
    let c = &theta[2 * k..];
    let c_inv = invert_square(c, k)?;
    let pi = matvec(&c_inv, p_star, k);
    // Q[j][t] = C[j][t] pi_t / p*_j
    let mut q = vec![0.0; k * k];
    for j in 0..k {
        if !(p_star[j] > 0.0) {
            return None;
        }
        for t in 0..k {
            q[j * k + t] = c[j * k + t] * pi[t] / p_star[j];
        }
    }
    let q_inv = invert_square(&q, k)?;
    Some(matvec(&q_inv, means, k))
}

/// Correct category proportions and outcome means for declared, known
/// misclassification.
///
/// The observed-class means obey `ybar = Q mu` with
/// `Q[j][k] = P(true = k | observed = j) = C[j][k] pi_k / p*_j`, so
/// `mu = Q^{-1} ybar`. The covariance is delta-method over the observed
/// proportions (multinomial), the observed-class means (independent) and the
/// confusion columns (independent multinomials from the validation counts);
/// the outcome-mean Jacobian is a central finite difference of the exact map.
///
/// # Errors
/// An empty observed class, a singular or ill-conditioned confusion matrix
/// (1-norm condition above `max_condition`) or an empty latent category refuse.
pub fn corrected_categories(
    classes: &[ObservedClassSummary],
    confusion: &ConfusionMatrix,
    max_condition: f64,
) -> Result<CorrectedCategorical, MeasurementErrorError> {
    let k = confusion.levels;
    if classes.len() != k {
        return Err(MeasurementErrorError::Shape("observed classes differ from confusion levels"));
    }
    if !(max_condition.is_finite() && max_condition >= 1.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "maximum condition number must be at least one",
        ));
    }
    if classes.iter().any(|c| c.count < 2) {
        return Err(MeasurementErrorError::TooFewRows {
            rows: classes.iter().map(|c| c.count as usize).min().unwrap_or(0),
            minimum: 2,
        });
    }
    if classes.iter().any(|c| !(c.mean.is_finite() && c.variance.is_finite() && c.variance >= 0.0))
    {
        return Err(MeasurementErrorError::NonFinite("observed class summaries"));
    }
    let total: u64 = classes.iter().map(|c| c.count).sum();
    let n = total as f64;
    let p_star: Vec<f64> = classes.iter().map(|c| c.count as f64 / n).collect();
    let c = &confusion.matrix;
    let c_inv = invert_square(c, k).ok_or(MeasurementErrorError::ConfusionIllConditioned {
        condition: f64::INFINITY,
        maximum: max_condition,
    })?;
    let condition = one_norm(c, k) * one_norm(&c_inv, k);
    if !(condition.is_finite() && condition <= max_condition) {
        return Err(MeasurementErrorError::ConfusionIllConditioned {
            condition,
            maximum: max_condition,
        });
    }
    let pi = matvec(&c_inv, &p_star, k);

    // Covariance of theta = (ybar, p*, C).
    let dim = 2 * k + k * k;
    let mut cov = vec![0.0; dim * dim];
    for j in 0..k {
        cov[j * dim + j] = classes[j].variance / classes[j].count as f64;
    }
    for a in 0..k {
        for b in 0..k {
            let value = if a == b { p_star[a] * (1.0 - p_star[a]) } else { -p_star[a] * p_star[b] };
            cov[(k + a) * dim + (k + b)] = value / n;
        }
    }
    if let Some(counts) = &confusion.column_counts {
        for t in 0..k {
            let nt = counts[t] as f64;
            for a in 0..k {
                for b in 0..k {
                    let ca = c[a * k + t];
                    let cb = c[b * k + t];
                    let value = if a == b { ca * (1.0 - ca) } else { -ca * cb };
                    cov[(2 * k + a * k + t) * dim + (2 * k + b * k + t)] = value / nt;
                }
            }
        }
    }

    // Proportions: pi = C^{-1} p*; d pi = C^{-1} d p* - C^{-1} (d C) pi.
    let mut jac_pi = vec![0.0; k * dim];
    for i in 0..k {
        for j in 0..k {
            jac_pi[i * dim + k + j] = c_inv[i * k + j];
            for t in 0..k {
                // d pi_i / d C[j][t] = - Cinv[i][j] pi_t
                jac_pi[i * dim + 2 * k + j * k + t] = -c_inv[i * k + j] * pi[t];
            }
        }
    }
    let proportion_covariance = congruence(&jac_pi, &cov, k, dim);

    // Outcome means via central differences of the exact map.
    let mut theta = Vec::with_capacity(dim);
    theta.extend(classes.iter().map(|c| c.mean));
    theta.extend_from_slice(&p_star);
    theta.extend_from_slice(c);
    let mu = category_outcome_means(&theta, k).ok_or(MeasurementErrorError::EmptyLatentCategory)?;
    let mut jac_mu = vec![0.0; k * dim];
    for idx in 0..dim {
        if (0..dim).all(|other| cov[idx * dim + other] == 0.0 && cov[other * dim + idx] == 0.0) {
            continue;
        }
        let h = 1e-6 * theta[idx].abs().max(1.0);
        let mut up = theta.clone();
        up[idx] += h;
        let mut down = theta.clone();
        down[idx] -= h;
        let f_up = category_outcome_means(&up, k).ok_or(MeasurementErrorError::EmptyLatentCategory)?;
        let f_down =
            category_outcome_means(&down, k).ok_or(MeasurementErrorError::EmptyLatentCategory)?;
        for i in 0..k {
            jac_mu[i * dim + idx] = (f_up[i] - f_down[i]) / (2.0 * h);
        }
    }
    let outcome_mean_covariance = congruence(&jac_mu, &cov, k, dim);

    if (0..k).any(|i| {
        !(proportion_covariance[i * k + i] >= 0.0 && outcome_mean_covariance[i * k + i] >= 0.0)
    }) {
        return Err(MeasurementErrorError::NonPositiveVariance("corrected category summary"));
    }
    let expected_counts = pi.iter().map(|v| v * n).collect();
    let outside = pi.iter().any(|v| !(0.0..=1.0).contains(v));
    Ok(CorrectedCategorical {
        observed_proportions: p_star,
        proportions: pi,
        proportion_covariance,
        expected_counts,
        outcome_means: mu,
        outcome_mean_covariance,
        condition_number: condition,
        outside_unit_interval: outside,
    })
}

/// `J Sigma J'` for `J` of shape `rows x cols` and `Sigma` `cols x cols`.
fn congruence(jac: &[f64], sigma: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut tmp = vec![0.0; rows * cols];
    for i in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for m in 0..cols {
                acc += jac[i * cols + m] * sigma[m * cols + c];
            }
            tmp[i * cols + c] = acc;
        }
    }
    let mut out = vec![0.0; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            let mut acc = 0.0;
            for c in 0..cols {
                acc += tmp[i * cols + c] * jac[j * cols + c];
            }
            out[i * rows + j] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Externally supplied latent-variable posterior
// ---------------------------------------------------------------------------

/// Posterior means of a latent regressor from joint posterior draws.
///
/// `draws` is draw-major: draw `s` occupies `draws[s * n..(s + 1) * n]`.
///
/// # Errors
/// A shape mismatch, no draws or non-finite values refuse.
pub fn posterior_means(
    draws: &[f64],
    n: usize,
    draw_count: usize,
) -> Result<Vec<f64>, MeasurementErrorError> {
    if n == 0 || draw_count == 0 || draws.len() != n.saturating_mul(draw_count) {
        return Err(MeasurementErrorError::Shape("draws do not form draw_count x n"));
    }
    if !finite_all(draws) {
        return Err(MeasurementErrorError::NonFinite("posterior draws"));
    }
    let mut means = vec![0.0; n];
    for s in 0..draw_count {
        for i in 0..n {
            means[i] += draws[s * n + i];
        }
    }
    let inv = 1.0 / draw_count as f64;
    for v in &mut means {
        *v *= inv;
    }
    Ok(means)
}

/// Result of combining a linear outcome fit over posterior imputations.
#[derive(Clone, Debug, PartialEq)]
pub struct CombinedImputations {
    /// Number of draws combined.
    pub draws: usize,
    /// Design width.
    pub dim: usize,
    /// Mean coefficients.
    pub coefficients: Vec<f64>,
    /// Total covariance `W + (1 + 1/S) B`.
    pub covariance: Vec<f64>,
    /// Mean within-imputation covariance `W`.
    pub within: Vec<f64>,
    /// Between-imputation covariance `B`.
    pub between: Vec<f64>,
    /// Coefficients of each imputation's fit, `draws` vectors of length `dim`.
    pub draw_coefficients: Vec<Vec<f64>>,
    /// HC0 covariance of each imputation's fit, `draws` row-major `dim x dim` blocks.
    pub draw_covariances: Vec<Vec<f64>>,
}

/// Rubin's combination of per-imputation coefficients and covariances.
///
/// # Errors
/// Fewer than two imputations or inconsistent shapes refuse.
pub fn rubin_combine(
    coefficients: &[Vec<f64>],
    covariances: &[Vec<f64>],
) -> Result<CombinedImputations, MeasurementErrorError> {
    let draws = coefficients.len();
    if draws < 2 {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "Rubin's rules need at least two imputations",
        ));
    }
    let d = coefficients[0].len();
    if d == 0
        || covariances.len() != draws
        || coefficients.iter().any(|c| c.len() != d || !finite_all(c))
        || covariances.iter().any(|c| c.len() != d * d || !finite_all(c))
    {
        return Err(MeasurementErrorError::Shape("imputation fits differ in shape"));
    }
    let s_f = draws as f64;
    let mut within = vec![0.0; d * d];
    for cov in covariances {
        for (acc, v) in within.iter_mut().zip(cov) {
            *acc += v / s_f;
        }
    }
    let mut mean = vec![0.0; d];
    for beta in coefficients {
        for (acc, v) in mean.iter_mut().zip(beta) {
            *acc += v / s_f;
        }
    }
    let mut between = vec![0.0; d * d];
    for beta in coefficients {
        for a in 0..d {
            for b in 0..d {
                between[a * d + b] += (beta[a] - mean[a]) * (beta[b] - mean[b]) / (s_f - 1.0);
            }
        }
    }
    let covariance = within
        .iter()
        .zip(&between)
        .map(|(w, b)| w + (1.0 + 1.0 / s_f) * b)
        .collect();
    Ok(CombinedImputations {
        draws,
        dim: d,
        coefficients: mean,
        covariance,
        within,
        between,
        draw_coefficients: coefficients.to_vec(),
        draw_covariances: covariances.to_vec(),
    })
}

/// Combine linear fits over joint posterior draws with Rubin's rules.
///
/// Valid only when the draws are proper imputations conditional on the
/// outcome (a posterior that already conditions on `y`); a posterior of the
/// latent value given the error-prone data alone makes the draws
/// Berkson-type noise and the regression of `y` on them is attenuated, so that
/// case uses the posterior mean instead ([`posterior_means`]).
///
/// # Errors
/// Fewer than two draws, a shape mismatch or a singular fit refuse.
pub fn combine_imputations_linear(
    y: &[f64],
    draws: &[f64],
    draw_count: usize,
    covariates: &[f64],
    covariate_count: usize,
) -> Result<CombinedImputations, MeasurementErrorError> {
    let n = y.len();
    if draw_count < 2 {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "Rubin's rules need at least two imputations",
        ));
    }
    if draws.len() != n.saturating_mul(draw_count) {
        return Err(MeasurementErrorError::Shape("draws do not form draw_count x n"));
    }
    let exact = ErrorVariance::known(0.0, 0.0)?;
    let mut betas = Vec::with_capacity(draw_count);
    let mut covs = Vec::with_capacity(draw_count);
    for s in 0..draw_count {
        let data = LinearErrorData {
            y,
            w: &draws[s * n..(s + 1) * n],
            covariates,
            covariate_count,
            replicate_counts: None,
        };
        let fit = LinearMoments::accumulate(&data, &exact, 0.0)?.solve()?;
        betas.push(fit.naive_coefficients);
        covs.push(fit.naive_covariance);
    }
    rubin_combine(&betas, &covs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::CausalRng;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Standard normal by Box-Muller from the library SplitMix stream.
    fn normal(rng: &mut CausalRng) -> f64 {
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Full factorial over three balanced binary factors X in {-a,a}, U in
    /// {-b,b}, E in {-c,c} (n = 8): all pairwise sample covariances are
    /// exactly zero, so every moment has a closed form.
    fn factorial(a: f64, b: f64, c: f64, beta: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = Vec::new();
        let mut u = Vec::new();
        let mut e = Vec::new();
        for sx in [-1.0, 1.0] {
            for su in [-1.0, 1.0] {
                for se in [-1.0, 1.0] {
                    x.push(sx * a);
                    u.push(su * b);
                    e.push(se * c);
                }
            }
        }
        let w: Vec<f64> = x.iter().zip(&u).map(|(x, u)| x + u).collect();
        let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 2.0 + beta * x + e).collect();
        (y, w, x)
    }

    #[test]
    fn closed_form_attenuation_is_removed_exactly() {
        // lambda = a^2 / (a^2 + b^2) = 1 / (1 + 0.25) = 0.8 with a = 1, b = 0.5.
        let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
        let data = LinearErrorData {
            y: &y,
            w: &w,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        };
        let error = ErrorVariance::known(0.25, 0.0).unwrap();
        let moments = LinearMoments::accumulate(&data, &error, 0.01).unwrap();
        let fit = moments.solve().unwrap();
        assert!(close(fit.naive_coefficients[1], 3.0 * 0.8, 1e-12), "{:?}", fit.naive_coefficients);
        assert!(close(fit.coefficients[1], 3.0, 1e-12), "{:?}", fit.coefficients);
        assert!(close(fit.coefficients[0], 2.0, 1e-12));
        assert!(close(fit.reliability, 0.8, 1e-12));
    }

    #[test]
    fn corrected_normal_equations_equal_regression_calibration() {
        // Independent route: substitute E[X | W] = mean(W) + lambda (W - mean(W)),
        // lambda = (s_w^2 - sigma_u^2) / s_w^2, then run simple least squares.
        let mut rng = CausalRng::from_seed(11);
        let n = 400;
        let sigma_u2: f64 = 0.6;
        let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let w: Vec<f64> = x.iter().map(|v| v + sigma_u2.sqrt() * normal(&mut rng)).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v + normal(&mut rng)).collect();
        let nf = n as f64;
        let w_bar = w.iter().sum::<f64>() / nf;
        let s_w2 = w.iter().map(|v| (v - w_bar) * (v - w_bar)).sum::<f64>() / nf;
        let lambda = (s_w2 - sigma_u2) / s_w2;
        let x_hat: Vec<f64> = w.iter().map(|v| w_bar + lambda * (v - w_bar)).collect();
        let xh_bar = x_hat.iter().sum::<f64>() / nf;
        let y_bar = y.iter().sum::<f64>() / nf;
        let sxy: f64 = x_hat.iter().zip(&y).map(|(a, b)| (a - xh_bar) * (b - y_bar)).sum();
        let sxx: f64 = x_hat.iter().map(|a| (a - xh_bar) * (a - xh_bar)).sum();
        let rc_slope = sxy / sxx;
        let data = LinearErrorData {
            y: &y,
            w: &w,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        };
        let moments =
            LinearMoments::accumulate(&data, &ErrorVariance::known(sigma_u2, 0.0).unwrap(), 0.01)
                .unwrap();
        let fit = moments.solve().unwrap();
        assert!(close(fit.coefficients[1], rc_slope, 1e-10), "{} vs {rc_slope}", fit.coefficients[1]);
    }

    #[test]
    fn variance_includes_error_variance_uncertainty() {
        let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
        let data = LinearErrorData {
            y: &y,
            w: &w,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        };
        let exact = LinearMoments::accumulate(&data, &ErrorVariance::known(0.25, 0.0).unwrap(), 0.01)
            .unwrap()
            .solve()
            .unwrap();
        let uncertain =
            LinearMoments::accumulate(&data, &ErrorVariance::known(0.25, 0.05).unwrap(), 0.01)
                .unwrap()
                .solve()
                .unwrap();
        assert_eq!(exact.coefficients, uncertain.coefficients);
        assert!(close(
            exact.covariance[3],
            exact.covariance_given_error_variance[3],
            1e-15
        ));
        // d beta / d sigma^2 for the scalar slope: beta / (s_w^2 - sigma^2) = 3 / 1.0.
        let added = uncertain.covariance[3] - uncertain.covariance_given_error_variance[3];
        assert!(close(added, (3.0_f64 / 1.0).powi(2) * 0.05 * 0.05, 1e-10), "{added}");
    }

    #[test]
    fn reliability_at_or_below_floor_refuses() {
        let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
        let data = LinearErrorData {
            y: &y,
            w: &w,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        };
        // Var(W) = 1.25: an error variance of 1.25 leaves no latent variance.
        let at_total = ErrorVariance::known(1.25, 0.0).unwrap();
        assert!(matches!(
            LinearMoments::accumulate(&data, &at_total, 0.0),
            Err(MeasurementErrorError::ReliabilityNotPositive { .. })
        ));
        // 0.8 reliability passes a floor of 0.5 and fails a floor of 0.9.
        let ok = ErrorVariance::known(0.25, 0.0).unwrap();
        assert!(LinearMoments::accumulate(&data, &ok, 0.5).is_ok());
        assert!(matches!(
            LinearMoments::accumulate(&data, &ok, 0.9),
            Err(MeasurementErrorError::ReliabilityNotPositive { .. })
        ));
    }

    #[test]
    fn replicate_components_match_the_balanced_anova_closed_form() {
        // Two units, k = 2: values (1, 3) and (5, 9).
        // unit means 2, 7; grand 4.5; SSW = 2 + 8 = 10 on 2 df -> MSW = 5.
        // SSB = 2*(2.5^2) + 2*(2.5^2) = 25 on 1 df -> MSB = 25; n0 = 2.
        // between = (25 - 5) / 2 = 10; ICC = 10 / 15.
        let (components, influence) =
            replicate_variance_components(&[1.0, 3.0, 5.0, 9.0], &[2, 2]).unwrap();
        assert!(close(components.within_variance, 5.0, 1e-12));
        assert!(close(components.between_variance, 10.0, 1e-12));
        assert!(close(components.intraclass_correlation, 10.0 / 15.0, 1e-12));
        assert!(close(components.effective_replicates, 2.0, 1e-12));
        // Influence: (SSW_j - 1 * 5) / mean(k - 1) = (2 - 5, 8 - 5) / 1.
        assert!(close(influence[0], -3.0, 1e-12) && close(influence[1], 3.0, 1e-12));
        assert!(replicate_variance_components(&[1.0, 2.0], &[1, 1]).is_err());
    }

    #[test]
    fn simulated_replicates_recover_truth_and_the_reported_se_matches_the_spread() {
        // DGP: X ~ N(0,1), Z = 0.5 X + N(0,1), Y = 1 + 1.5 X + 0.8 Z + N(0,1),
        // W_ij = X_i + U_ij with var(U) = 0.7, k = 3 replicates.
        // lambda (single) = 1 / 1.7; naive slope is attenuated well below 1.5.
        let reps = 300;
        let n = 1000;
        let k = 3_u32;
        let mut estimates = Vec::new();
        let mut se_sum = 0.0;
        let mut naive_sum = 0.0;
        for rep in 0..reps {
            let mut rng = CausalRng::from_seed(1_000 + rep as u64);
            let mut y = Vec::with_capacity(n);
            let mut z = Vec::with_capacity(n);
            let mut wbar = Vec::with_capacity(n);
            let mut flat = Vec::with_capacity(n * k as usize);
            for _ in 0..n {
                let x = normal(&mut rng);
                let zi = 0.5 * x + normal(&mut rng);
                y.push(1.0 + 1.5 * x + 0.8 * zi + normal(&mut rng));
                z.push(zi);
                let mut sum = 0.0;
                for _ in 0..k {
                    let w = x + 0.7_f64.sqrt() * normal(&mut rng);
                    flat.push(w);
                    sum += w;
                }
                wbar.push(sum / f64::from(k));
            }
            let sizes = vec![k; n];
            let (error, _) = ErrorVariance::from_main_sample_replicates(&flat, &sizes).unwrap();
            let data = LinearErrorData {
                y: &y,
                w: &wbar,
                covariates: &z,
                covariate_count: 1,
                replicate_counts: Some(&sizes),
            };
            let fit = LinearMoments::accumulate(&data, &error, 0.05).unwrap().solve().unwrap();
            estimates.push(fit.coefficients[1]);
            se_sum += fit.standard_error(1);
            naive_sum += fit.naive_coefficients[1];
        }
        let mean = estimates.iter().sum::<f64>() / reps as f64;
        let sd = (estimates.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>()
            / (reps as f64 - 1.0))
            .sqrt();
        // Predeclared: |mean - 1.5| within 4 spreads of the mean of `reps` fits.
        assert!(close(mean, 1.5, 4.0 * sd / (reps as f64).sqrt() + 0.01), "mean {mean} sd {sd}");
        // The naive coefficient is attenuated: strictly below the truth by a wide margin.
        assert!(naive_sum / (reps as f64) < 1.3, "naive {}", naive_sum / reps as f64);
        // Mean reported SE tracks the empirical spread (predeclared band 0.8..1.25).
        let ratio = (se_sum / reps as f64) / sd;
        assert!((0.8..=1.25).contains(&ratio), "se/sd ratio {ratio}");
    }

    #[test]
    fn linear_functional_variance_is_the_quadratic_form_of_the_joint_covariance() {
        let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
        let data = LinearErrorData {
            y: &y,
            w: &w,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        };
        let fit = LinearMoments::accumulate(&data, &ErrorVariance::known(0.25, 0.1).unwrap(), 0.01)
            .unwrap()
            .solve()
            .unwrap();
        let (value, variance) = fit.linear_functional(&[0.0, 2.0]).unwrap();
        assert!(close(value, 6.0, 1e-12));
        assert!(close(variance, 4.0 * fit.covariance[3], 1e-12));
        let (v2, var2) = fit.linear_functional(&[1.0, 1.0]).unwrap();
        let expect = fit.covariance[0] + 2.0 * fit.covariance[1] + fit.covariance[3];
        assert!(close(v2, 5.0, 1e-12) && close(var2, expect, 1e-12));
        assert!(fit.linear_functional(&[1.0]).is_err());
    }

    #[test]
    fn rogan_gladen_inverts_the_misclassification_algebra() {
        // pi = 0.3, Se = 0.9, Sp = 0.8: p* = 0.9*0.3 + 0.2*0.7 = 0.41.
        let rates = BinaryErrorRates::new(0.9, 0.8, 0.0001, 0.0004).unwrap();
        let result = corrected_prevalence(410, 1000, &rates, 0.1).unwrap();
        assert!(close(result.observed, 0.41, 1e-15));
        assert!(close(result.corrected, 0.3, 1e-12));
        assert!(close(result.youden, 0.7, 1e-12));
        // Var = [p*(1-p*)/n + (1-pi)^2 Var(Sp) + pi^2 Var(Se)] / J^2.
        let expect = (0.41 * 0.59 / 1000.0 + 0.49 * 0.0004 + 0.09 * 0.0001) / 0.49;
        assert!(close(result.variance, expect, 1e-12), "{} vs {expect}", result.variance);
        assert!(close(result.variance_sampling + result.variance_rates, result.variance, 1e-15));
        assert!(!result.outside_unit_interval);
    }

    #[test]
    fn youden_at_or_below_the_declared_minimum_refuses() {
        // Se + Sp - 1 = 0 (a coin flip) and a negative index both refuse.
        for (se, sp) in [(0.5, 0.5), (0.3, 0.6)] {
            let rates = BinaryErrorRates::new(se, sp, 0.0, 0.0).unwrap();
            assert!(matches!(
                corrected_prevalence(10, 100, &rates, 0.05),
                Err(MeasurementErrorError::YoudenTooSmall { .. })
            ));
        }
        // 0.12 passes a minimum of 0.1 and fails a minimum of 0.2.
        let weak = BinaryErrorRates::new(0.56, 0.56, 0.0, 0.0).unwrap();
        assert!(corrected_prevalence(10, 100, &weak, 0.1).is_ok());
        assert!(corrected_prevalence(10, 100, &weak, 0.2).is_err());
    }

    #[test]
    fn nondifferential_risk_difference_has_closed_form_and_shared_rate_variance() {
        // pi1 = 0.4, pi0 = 0.2, Se = 0.9, Sp = 0.8:
        // p1* = 0.36 + 0.12 = 0.48, p0* = 0.18 + 0.16 = 0.34.
        let rates = BinaryErrorRates::new(0.9, 0.8, 0.0004, 0.0009).unwrap();
        let table = TwoByTwoCounts {
            exposed_events: 480,
            exposed_total: 1000,
            unexposed_events: 340,
            unexposed_total: 1000,
        };
        let rd = corrected_risk_difference_nondifferential(&table, &rates, 0.1).unwrap();
        assert!(close(rd.naive, 0.14, 1e-12));
        assert!(close(rd.corrected, 0.2, 1e-12));
        assert!(close(rd.risk_exposed, 0.4, 1e-12) && close(rd.risk_unexposed, 0.2, 1e-12));
        let expect_rates = 0.2 * 0.2 * (0.0004 + 0.0009) / 0.49;
        assert!(close(rd.variance_rates, expect_rates, 1e-12));
        // Independent-arm (differential) propagation of the same numbers differs,
        // because the shared rates correlate the arms.
        let split =
            corrected_risk_difference_differential(&table, &rates, &rates, 0.1).unwrap();
        assert!(close(split.corrected, rd.corrected, 1e-12));
        assert!(!close(split.variance, rd.variance, 1e-9));
    }

    #[test]
    fn differential_misclassification_is_not_recovered_by_pooled_rates() {
        // Exposed arm Se 0.9 / Sp 0.8, unexposed Se 0.7 / Sp 0.95; truth 0.4 vs 0.2.
        // p1* = 0.48, p0* = 0.7*0.2 + 0.05*0.8 = 0.18.
        let e1 = BinaryErrorRates::new(0.9, 0.8, 0.0, 0.0).unwrap();
        let e0 = BinaryErrorRates::new(0.7, 0.95, 0.0, 0.0).unwrap();
        let table = TwoByTwoCounts {
            exposed_events: 480,
            exposed_total: 1000,
            unexposed_events: 180,
            unexposed_total: 1000,
        };
        let diff = corrected_risk_difference_differential(&table, &e1, &e0, 0.1).unwrap();
        assert!(close(diff.corrected, 0.2, 1e-12));
        let pooled = corrected_risk_difference_nondifferential(&table, &e1, 0.1).unwrap();
        // (0.48 - 0.18) / 0.7 = 0.4286: the wrong answer is far from 0.2.
        assert!(close(pooled.corrected, 0.3 / 0.7, 1e-12));
        assert!((pooled.corrected - 0.2).abs() > 0.2);
    }

    #[test]
    fn categorical_inversion_recovers_proportions_and_outcome_means() {
        // True categories 0,1,2 with pi = (0.5, 0.3, 0.2) and mu = (1, 2, 4).
        // Columns are P(observed | true).
        let c = vec![
            0.8, 0.1, 0.0, //
            0.1, 0.8, 0.2, //
            0.1, 0.1, 0.8,
        ];
        let pi = [0.5, 0.3, 0.2];
        let mu = [1.0, 2.0, 4.0];
        let confusion = ConfusionMatrix::new(3, c.clone(), Some(vec![500, 500, 500])).unwrap();
        let n = 10_000.0;
        let mut classes = Vec::new();
        for j in 0..3 {
            let p_star: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t]).sum();
            let ybar: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t] * mu[t]).sum::<f64>() / p_star;
            classes.push(ObservedClassSummary {
                count: (p_star * n).round() as u64,
                mean: ybar,
                variance: 1.0,
            });
        }
        let counts: Vec<u64> = classes.iter().map(|c| c.count).collect();
        assert_eq!(counts.iter().sum::<u64>(), 10_000);
        let out = corrected_categories(&classes, &confusion, 50.0).unwrap();
        for t in 0..3 {
            assert!(close(out.proportions[t], pi[t], 1e-9), "{:?}", out.proportions);
            assert!(close(out.outcome_means[t], mu[t], 1e-6), "{:?}", out.outcome_means);
        }
        // The naive observed-class means are biased away from mu.
        assert!(!close(classes[2].mean, 4.0, 0.1));
        // Variance is strictly larger than ignoring confusion-matrix uncertainty.
        let exact = ConfusionMatrix::new(3, c, None).unwrap();
        let out_exact = corrected_categories(&classes, &exact, 50.0).unwrap();
        for t in 0..3 {
            assert!(
                out.proportion_covariance[t * 3 + t] > out_exact.proportion_covariance[t * 3 + t]
            );
        }
    }

    #[test]
    fn binary_case_of_the_categorical_route_matches_rogan_gladen() {
        // K = 2: C = [[Se, 1-Sp], [1-Se, Sp]], observed positives are class 0.
        let (se, sp) = (0.9, 0.8);
        let confusion =
            ConfusionMatrix::new(2, vec![se, 1.0 - sp, 1.0 - se, sp], Some(vec![200, 300])).unwrap();
        let classes = [
            ObservedClassSummary { count: 410, mean: 1.0, variance: 0.5 },
            ObservedClassSummary { count: 590, mean: 0.0, variance: 0.5 },
        ];
        let out = corrected_categories(&classes, &confusion, 20.0).unwrap();
        let rates = BinaryErrorRates::new(
            se,
            sp,
            se * (1.0 - se) / 200.0,
            sp * (1.0 - sp) / 300.0,
        )
        .unwrap();
        let rg = corrected_prevalence(410, 1000, &rates, 0.1).unwrap();
        assert!(close(out.proportions[0], rg.corrected, 1e-12));
        assert!(close(out.proportion_covariance[0], rg.variance, 1e-10));
    }

    #[test]
    fn ill_conditioned_confusion_refuses() {
        let c = vec![0.5, 0.5, 0.5, 0.5];
        let confusion = ConfusionMatrix::new(2, c, None).unwrap();
        let classes = [
            ObservedClassSummary { count: 50, mean: 0.0, variance: 1.0 },
            ObservedClassSummary { count: 50, mean: 0.0, variance: 1.0 },
        ];
        assert!(matches!(
            corrected_categories(&classes, &confusion, 100.0),
            Err(MeasurementErrorError::ConfusionIllConditioned { .. })
        ));
    }

    #[test]
    fn posterior_mean_route_removes_berkson_noise_that_raw_draws_would_add() {
        // X_i ~ N(0,1); the posterior given data is N(m_i, v) with m_i the
        // posterior mean. Regressing y on a *draw* is attenuated by
        // Var(m) / (Var(m) + v); regressing on the posterior mean is not.
        let n = 4000;
        let draws_count = 40;
        let mut rng = CausalRng::from_seed(7);
        let v: f64 = 0.5;
        let mut means_true = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut draws = vec![0.0; n * draws_count];
        for i in 0..n {
            // latent x = m + sqrt(v) * eps with m ~ N(0, 0.5): Var(x) = 1.
            let m = 0.5_f64.sqrt() * normal(&mut rng);
            let x = m + v.sqrt() * normal(&mut rng);
            means_true.push(m);
            y.push(1.0 + 2.0 * x + normal(&mut rng));
            for s in 0..draws_count {
                draws[s * n + i] = m + v.sqrt() * normal(&mut rng);
            }
        }
        let m_hat = posterior_means(&draws, n, draws_count).unwrap();
        let exact = ErrorVariance::known(0.0, 0.0).unwrap();
        let on_mean = LinearMoments::accumulate(
            &LinearErrorData {
                y: &y,
                w: &m_hat,
                covariates: &[],
                covariate_count: 0,
                replicate_counts: None,
            },
            &exact,
            0.0,
        )
        .unwrap()
        .solve()
        .unwrap();
        // Truth 2.0; the posterior-mean MC error adds v/S noise to m (small).
        assert!(close(on_mean.naive_coefficients[1], 2.0, 4.0 * on_mean.naive_covariance[3].sqrt() + 0.05));
        let on_draw = LinearMoments::accumulate(
            &LinearErrorData {
                y: &y,
                w: &draws[..n],
                covariates: &[],
                covariate_count: 0,
                replicate_counts: None,
            },
            &exact,
            0.0,
        )
        .unwrap()
        .solve()
        .unwrap();
        // Attenuation factor 0.5 / (0.5 + 0.5) = 0.5 -> about 1.0.
        assert!(on_draw.naive_coefficients[1] < 1.2, "{}", on_draw.naive_coefficients[1]);
    }

    #[test]
    fn rubin_combination_has_between_imputation_component() {
        // Three exact imputations with slopes differing across draws.
        let n = 60;
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 + ((i * 7) % 5) as f64 * 0.05).collect();
        let base: Vec<f64> = (0..n).map(|i| ((i * 3) % 11) as f64 * 0.2 + i as f64 * 0.1).collect();
        let mut draws = Vec::new();
        for s in 0..3 {
            for i in 0..n {
                draws.push(base[i] + 0.1 * s as f64 * (((i * 5 + s) % 7) as f64 - 3.0));
            }
        }
        let combined = combine_imputations_linear(&y, &draws, 3, &[], 0).unwrap();
        assert_eq!(combined.draws, 3);
        let total = combined.covariance[3];
        let expect = combined.within[3] + (1.0 + 1.0 / 3.0) * combined.between[3];
        assert!(close(total, expect, 1e-15));
        assert!(combined.between[3] > 0.0);
        assert!(combine_imputations_linear(&y, &draws[..n], 1, &[], 0).is_err());
    }

    #[test]
    fn validation_sample_estimates_the_error_variance() {
        // W - X alternates +0.5 / -0.5 around zero: mean 0, variance 0.25 * n/(n-1).
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let w: Vec<f64> = x.iter().enumerate().map(|(i, v)| v + if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let (err, summary) = ErrorVariance::from_validation(&w, &x).unwrap();
        assert!(close(summary.error_mean, 0.0, 1e-15));
        assert!(close(err.variance, 0.25 * 10.0 / 9.0, 1e-12));
        assert!(err.influence.is_none());
    }
}
