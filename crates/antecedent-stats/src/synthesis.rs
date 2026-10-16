//! Summary-data synthesis: covariance-preserving pooling, random effects,
//! hierarchical borrowing and a dependence-range envelope.
//!
//! Every routine takes study summaries (an estimate and its sampling variance
//! or covariance) and returns a point-only number. None of them estimates
//! sampling dependence from the summaries: dependence is either supplied (a
//! full covariance matrix, preserved by generalized least squares) or declared
//! as an assumption range, for which [`pooled_dependence_envelope`] returns the
//! exact extremes of the pooled value over the range and nothing else.
//!
//! * [`fixed_effect_inverse_variance`] and [`fixed_effect_gls`]: the minimum
//!   variance linear unbiased pooled estimate `1' S^-1 y / 1' S^-1 1` with
//!   variance `1 / 1' S^-1 1` (Borenstein, Hedges, Higgins, Rothstein 2009,
//!   ch. 11 for the diagonal case; Aitken 1935 for the general covariance).
//! * [`random_effects_dersimonian_laird`] and [`random_effects_reml`]:
//!   between-study variance by the method of moments (`DerSimonian` and Laird
//!   1986) and by restricted maximum likelihood, with Cochran's Q.
//! * [`hierarchical_normal_borrowing`]: normal-normal borrowing with a
//!   declared (discrete) prior on the between-study standard deviation,
//!   evaluated by an exact finite mixture of closed-form conditional
//!   posteriors.
//! * [`pooled_dependence_envelope`]: the pooled estimate and variance as the
//!   common pairwise correlation ranges over a declared interval.
//!
//! No interval, coverage statement or posterior-calibration claim is made.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Most studies a synthesis routine accepts.
pub const MAX_STUDIES: usize = 256;
/// Most support points of a declared between-study prior.
pub const MAX_TAU_SUPPORT: usize = 4096;
/// Label every [`DependenceEnvelope`] carries: it is an assumption range.
pub const DEPENDENCE_ENVELOPE_LABEL: &str = "assumption_range";

const PSD_RELATIVE_TOLERANCE: f64 = 1e-10;
const SYMMETRY_RELATIVE_TOLERANCE: f64 = 1e-12;
const MAX_JACOBI_SWEEPS: usize = 100;
const REML_BRACKET_DOUBLINGS: usize = 400;
const REML_BISECTIONS: usize = 400;
const PRIOR_SUM_TOLERANCE: f64 = 1e-9;

/// Why a synthesis routine refused its input.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum SynthesisError {
    /// Lengths or counts do not agree, or a bound is exceeded.
    #[error("synthesis shape error: {0}")]
    Shape(&'static str),
    /// An input is NaN or infinite.
    #[error("synthesis input is not finite: {0}")]
    NonFinite(&'static str),
    /// A variance or standard error is not finite and positive.
    #[error("synthesis variance must be finite and positive: {0}")]
    NonPositiveVariance(&'static str),
    /// The covariance matrix is not symmetric.
    #[error("covariance is not symmetric (largest asymmetry {asymmetry})")]
    NotSymmetric {
        /// Largest `|S_ij - S_ji|`.
        asymmetry: f64,
    },
    /// The covariance matrix has a materially negative eigenvalue.
    #[error("covariance is not positive semidefinite (smallest eigenvalue {min_eigenvalue})")]
    NotPositiveSemidefinite {
        /// Smallest eigenvalue found.
        min_eigenvalue: f64,
    },
    /// The covariance matrix is semidefinite but singular.
    #[error(
        "covariance is singular (smallest eigenvalue {min_eigenvalue}, largest {max_eigenvalue}); \
         generalized least squares needs a positive definite covariance"
    )]
    Singular {
        /// Smallest eigenvalue found.
        min_eigenvalue: f64,
        /// Largest eigenvalue found.
        max_eigenvalue: f64,
    },
    /// The declared correlation range is not a valid open-interval sub-range.
    #[error("invalid correlation range: {0}")]
    InvalidCorrelationRange(&'static str),
    /// The declared prior is malformed.
    #[error("invalid prior declaration: {0}")]
    InvalidPrior(&'static str),
    /// Fewer studies than the method needs.
    #[error("too few studies: needed {needed}, got {got}")]
    TooFewStudies {
        /// Minimum count.
        needed: usize,
        /// Count supplied.
        got: usize,
    },
    /// An iterative step did not bracket or converge.
    #[error("synthesis did not converge: {0}")]
    NotConverged(&'static str),
}

/// A pooled point estimate with its variance under the stated covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct PooledEstimate {
    /// Pooled estimate `sum w_i y_i`.
    pub estimate: f64,
    /// Variance of the pooled estimate under the supplied covariance.
    pub variance: f64,
    /// Generalized least squares weights (sum to one; may be negative under
    /// strong positive correlation).
    pub weights: Vec<f64>,
    /// Cochran's Q generalized to the full covariance: `r' S^-1 r`, `r = y - estimate`.
    pub heterogeneity_q: f64,
    /// `k - 1`.
    pub degrees_of_freedom: usize,
}

/// What was learned about a covariance matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CovarianceCheck {
    /// Smallest eigenvalue.
    pub min_eigenvalue: f64,
    /// Largest eigenvalue.
    pub max_eigenvalue: f64,
    /// Whether any off-diagonal entry is nonzero.
    pub has_cross_covariance: bool,
}

fn check_summaries(estimates: &[f64], variances: &[f64]) -> Result<usize, SynthesisError> {
    let k = estimates.len();
    if k == 0 {
        return Err(SynthesisError::TooFewStudies { needed: 1, got: 0 });
    }
    if variances.len() != k {
        return Err(SynthesisError::Shape("estimates and variances differ in length"));
    }
    if k > MAX_STUDIES {
        return Err(SynthesisError::Shape("more studies than the synthesis bound"));
    }
    if estimates.iter().any(|v| !v.is_finite()) {
        return Err(SynthesisError::NonFinite("estimate"));
    }
    if variances.iter().any(|v| !v.is_finite()) {
        return Err(SynthesisError::NonFinite("variance"));
    }
    if variances.iter().any(|&v| v <= 0.0) {
        return Err(SynthesisError::NonPositiveVariance("study variance"));
    }
    Ok(k)
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations.
fn symmetric_eigenvalues(matrix: &[f64], k: usize) -> Result<Vec<f64>, SynthesisError> {
    let mut a = matrix.to_vec();
    let scale: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if scale == 0.0 {
        return Ok(vec![0.0; k]);
    }
    for _ in 0..MAX_JACOBI_SWEEPS {
        let mut off = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off += a[p * k + q] * a[p * k + q];
            }
        }
        if off.sqrt() <= 1e-15 * scale {
            return Ok((0..k).map(|i| a[i * k + i]).collect());
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + theta.mul_add(theta, 1.0).sqrt());
                let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                let s = t * c;
                let app = a[p * k + p];
                let aqq = a[q * k + q];
                a[p * k + p] = t.mul_add(-apq, app);
                a[q * k + q] = t.mul_add(apq, aqq);
                a[p * k + q] = 0.0;
                a[q * k + p] = 0.0;
                for r in 0..k {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * k + p];
                    let arq = a[r * k + q];
                    let new_rp = c * arp - s * arq;
                    let new_rq = c * arq + s * arp;
                    a[r * k + p] = new_rp;
                    a[p * k + r] = new_rp;
                    a[r * k + q] = new_rq;
                    a[q * k + r] = new_rq;
                }
            }
        }
    }
    Err(SynthesisError::NotConverged("symmetric eigenvalue iteration"))
}

/// Validate a row-major `k x k` covariance matrix: finite, symmetric within a
/// relative tolerance, nonnegative diagonal and positive semidefinite.
///
/// # Errors
/// [`SynthesisError`] naming the first defect; a non-positive-semidefinite
/// matrix is refused, never repaired.
pub fn check_covariance(covariance: &[f64], k: usize) -> Result<CovarianceCheck, SynthesisError> {
    if k == 0 || k > MAX_STUDIES {
        return Err(SynthesisError::Shape("study count outside 1..=MAX_STUDIES"));
    }
    if covariance.len() != k * k {
        return Err(SynthesisError::Shape("covariance is not k x k"));
    }
    if covariance.iter().any(|v| !v.is_finite()) {
        return Err(SynthesisError::NonFinite("covariance"));
    }
    let scale = covariance.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut asymmetry = 0.0_f64;
    let mut cross = false;
    for i in 0..k {
        for j in (i + 1)..k {
            asymmetry = asymmetry.max((covariance[i * k + j] - covariance[j * k + i]).abs());
            if covariance[i * k + j] != 0.0 || covariance[j * k + i] != 0.0 {
                cross = true;
            }
        }
    }
    if asymmetry > SYMMETRY_RELATIVE_TOLERANCE * scale.max(f64::MIN_POSITIVE) {
        return Err(SynthesisError::NotSymmetric { asymmetry });
    }
    let mut symmetric = covariance.to_vec();
    for i in 0..k {
        for j in (i + 1)..k {
            let mean = 0.5 * (covariance[i * k + j] + covariance[j * k + i]);
            symmetric[i * k + j] = mean;
            symmetric[j * k + i] = mean;
        }
    }
    let eigenvalues = symmetric_eigenvalues(&symmetric, k)?;
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let max_eigenvalue = eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min_eigenvalue < -PSD_RELATIVE_TOLERANCE * max_eigenvalue.max(f64::MIN_POSITIVE) {
        return Err(SynthesisError::NotPositiveSemidefinite { min_eigenvalue });
    }
    Ok(CovarianceCheck { min_eigenvalue, max_eigenvalue, has_cross_covariance: cross })
}

/// Cholesky factor (lower, row-major) of a positive definite matrix.
fn cholesky(matrix: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = matrix[i * k + j];
            for m in 0..j {
                sum -= l[i * k + m] * l[j * k + m];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    Some(l)
}

fn cholesky_solve(l: &[f64], k: usize, rhs: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; k];
    for i in 0..k {
        let mut sum = rhs[i];
        for m in 0..i {
            sum -= l[i * k + m] * z[m];
        }
        z[i] = sum / l[i * k + i];
    }
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = z[i];
        for m in (i + 1)..k {
            sum -= l[m * k + i] * x[m];
        }
        x[i] = sum / l[i * k + i];
    }
    x
}

/// Pool estimates with a full covariance matrix by generalized least squares.
///
/// The estimate is `1' S^-1 y / 1' S^-1 1` and its variance `1 / 1' S^-1 1`.
/// The covariance is validated as symmetric positive definite and refused
/// otherwise; a singular covariance (perfectly correlated studies) has no
/// generalized least squares weights and is refused with [`SynthesisError::Singular`].
///
/// # Errors
/// Shape, finiteness, symmetry, semidefiniteness or definiteness refusals.
pub fn fixed_effect_gls(
    estimates: &[f64],
    covariance: &[f64],
) -> Result<PooledEstimate, SynthesisError> {
    let k = estimates.len();
    if k == 0 {
        return Err(SynthesisError::TooFewStudies { needed: 1, got: 0 });
    }
    if estimates.iter().any(|v| !v.is_finite()) {
        return Err(SynthesisError::NonFinite("estimate"));
    }
    let check = check_covariance(covariance, k)?;
    if check.min_eigenvalue <= PSD_RELATIVE_TOLERANCE * check.max_eigenvalue {
        return Err(SynthesisError::Singular {
            min_eigenvalue: check.min_eigenvalue,
            max_eigenvalue: check.max_eigenvalue,
        });
    }
    let mut symmetric = covariance.to_vec();
    for i in 0..k {
        for j in (i + 1)..k {
            let mean = 0.5 * (covariance[i * k + j] + covariance[j * k + i]);
            symmetric[i * k + j] = mean;
            symmetric[j * k + i] = mean;
        }
    }
    let l = cholesky(&symmetric, k).ok_or(SynthesisError::Singular {
        min_eigenvalue: check.min_eigenvalue,
        max_eigenvalue: check.max_eigenvalue,
    })?;
    let ones = vec![1.0; k];
    let x = cholesky_solve(&l, k, &ones);
    let denominator: f64 = x.iter().sum();
    let weights: Vec<f64> = x.iter().map(|v| v / denominator).collect();
    let estimate: f64 = weights.iter().zip(estimates).map(|(w, y)| w * y).sum();
    let residual: Vec<f64> = estimates.iter().map(|y| y - estimate).collect();
    let z = cholesky_solve(&l, k, &residual);
    let q: f64 = residual.iter().zip(&z).map(|(r, v)| r * v).sum();
    Ok(PooledEstimate {
        estimate,
        variance: 1.0 / denominator,
        weights,
        heterogeneity_q: q.max(0.0),
        degrees_of_freedom: k - 1,
    })
}

/// Inverse-variance fixed-effect pooling of independent summaries.
///
/// # Errors
/// Shape, finiteness or non-positive-variance refusals.
pub fn fixed_effect_inverse_variance(
    estimates: &[f64],
    variances: &[f64],
) -> Result<PooledEstimate, SynthesisError> {
    let k = check_summaries(estimates, variances)?;
    let precision: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
    let total: f64 = precision.iter().sum();
    let weights: Vec<f64> = precision.iter().map(|p| p / total).collect();
    let estimate: f64 = weights.iter().zip(estimates).map(|(w, y)| w * y).sum();
    let q: f64 =
        precision.iter().zip(estimates).map(|(p, y)| p * (y - estimate) * (y - estimate)).sum();
    Ok(PooledEstimate {
        estimate,
        variance: 1.0 / total,
        weights,
        heterogeneity_q: q,
        degrees_of_freedom: k - 1,
    })
}

/// Which estimator of the between-study variance produced a random-effects fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BetweenStudyMethod {
    /// Method of moments (`DerSimonian` and Laird 1986).
    DerSimonianLaird,
    /// Restricted maximum likelihood.
    RestrictedMaximumLikelihood,
}

/// A random-effects pooled estimate with its between-study variance.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomEffectsEstimate {
    /// Pooled mean effect under the random-effects weights.
    pub estimate: f64,
    /// Variance of the pooled mean, `1 / sum 1/(v_i + tau^2)`, conditional on
    /// the estimated `tau^2` (it does not carry the uncertainty of `tau^2`).
    pub variance: f64,
    /// Between-study variance estimate, truncated at zero.
    pub tau_squared: f64,
    /// Cochran's Q from the fixed-effect weights.
    pub q: f64,
    /// `k - 1`.
    pub degrees_of_freedom: usize,
    /// `max(0, (Q - df) / Q)`; `0` when `Q` is zero.
    pub i_squared: f64,
    /// Random-effects weights (sum to one).
    pub weights: Vec<f64>,
    /// Estimator of `tau^2`.
    pub method: BetweenStudyMethod,
}

fn random_effects_at(
    estimates: &[f64],
    variances: &[f64],
    tau_squared: f64,
    q: f64,
    method: BetweenStudyMethod,
) -> RandomEffectsEstimate {
    let k = estimates.len();
    let precision: Vec<f64> = variances.iter().map(|v| 1.0 / (v + tau_squared)).collect();
    let total: f64 = precision.iter().sum();
    let weights: Vec<f64> = precision.iter().map(|p| p / total).collect();
    let estimate: f64 = weights.iter().zip(estimates).map(|(w, y)| w * y).sum();
    let df = k - 1;
    let i_squared = if q > 0.0 { ((q - df as f64) / q).max(0.0) } else { 0.0 };
    RandomEffectsEstimate {
        estimate,
        variance: 1.0 / total,
        tau_squared,
        q,
        degrees_of_freedom: df,
        i_squared,
        weights,
        method,
    }
}

/// Random-effects pooling with the DerSimonian-Laird between-study variance
/// `tau^2 = max(0, (Q - (k-1)) / C)`, `C = W - sum w^2 / W`.
///
/// # Errors
/// Fewer than two studies, or shape, finiteness or non-positive-variance refusals.
pub fn random_effects_dersimonian_laird(
    estimates: &[f64],
    variances: &[f64],
) -> Result<RandomEffectsEstimate, SynthesisError> {
    let k = check_summaries(estimates, variances)?;
    if k < 2 {
        return Err(SynthesisError::TooFewStudies { needed: 2, got: k });
    }
    let fixed = fixed_effect_inverse_variance(estimates, variances)?;
    let w: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
    let sum_w: f64 = w.iter().sum();
    let sum_w2: f64 = w.iter().map(|x| x * x).sum();
    let c = sum_w - sum_w2 / sum_w;
    let tau_squared = ((fixed.heterogeneity_q - (k - 1) as f64) / c).max(0.0);
    Ok(random_effects_at(
        estimates,
        variances,
        tau_squared,
        fixed.heterogeneity_q,
        BetweenStudyMethod::DerSimonianLaird,
    ))
}

/// Restricted-likelihood score `d l / d tau^2` at `tau_squared`.
fn reml_score(estimates: &[f64], variances: &[f64], tau_squared: f64) -> f64 {
    let w: Vec<f64> = variances.iter().map(|v| 1.0 / (v + tau_squared)).collect();
    let sum_w: f64 = w.iter().sum();
    let mu: f64 = w.iter().zip(estimates).map(|(wi, y)| wi * y).sum::<f64>() / sum_w;
    let sum_w2: f64 = w.iter().map(|x| x * x).sum();
    let weighted_sq: f64 =
        w.iter().zip(estimates).map(|(wi, y)| wi * wi * (y - mu) * (y - mu)).sum();
    0.5 * (weighted_sq - sum_w + sum_w2 / sum_w)
}

/// Random-effects pooling with the restricted-maximum-likelihood `tau^2`.
///
/// The estimate is the first stationary point of the restricted likelihood
/// above zero (the score is positive at zero, negative at infinity, and
/// bisected on a doubling bracket); the boundary value zero is returned when
/// the score is not positive there.
///
/// # Errors
/// Fewer than two studies, input refusals, or [`SynthesisError::NotConverged`].
pub fn random_effects_reml(
    estimates: &[f64],
    variances: &[f64],
) -> Result<RandomEffectsEstimate, SynthesisError> {
    let k = check_summaries(estimates, variances)?;
    if k < 2 {
        return Err(SynthesisError::TooFewStudies { needed: 2, got: k });
    }
    let fixed = fixed_effect_inverse_variance(estimates, variances)?;
    let tau_squared = if reml_score(estimates, variances, 0.0) <= 0.0 {
        0.0
    } else {
        let mean = estimates.iter().sum::<f64>() / k as f64;
        let spread = estimates.iter().map(|y| (y - mean) * (y - mean)).fold(0.0_f64, f64::max);
        let mut hi = 1.0 + spread;
        let mut bracketed = false;
        for _ in 0..REML_BRACKET_DOUBLINGS {
            if reml_score(estimates, variances, hi) < 0.0 {
                bracketed = true;
                break;
            }
            hi *= 2.0;
        }
        if !bracketed {
            return Err(SynthesisError::NotConverged("restricted likelihood bracket"));
        }
        let mut lo = 0.0;
        for _ in 0..REML_BISECTIONS {
            let mid = 0.5 * (lo + hi);
            #[allow(
                clippy::float_cmp,
                reason = "bisection stalls exactly when the midpoint rounds onto an endpoint; an exact comparison is the termination test"
            )]
            let stalled = mid == lo || mid == hi;
            if stalled {
                break;
            }
            if reml_score(estimates, variances, mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    Ok(random_effects_at(
        estimates,
        variances,
        tau_squared,
        fixed.heterogeneity_q,
        BetweenStudyMethod::RestrictedMaximumLikelihood,
    ))
}

/// Prior on the common mean in hierarchical borrowing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeanPrior {
    /// Flat (improper) prior on the mean.
    Flat,
    /// Normal prior with a declared mean and variance.
    Normal {
        /// Prior mean.
        mean: f64,
        /// Prior variance (finite, positive).
        variance: f64,
    },
}

/// A declared prior on the between-study standard deviation `tau`: a finite
/// support with explicit probabilities. A continuous prior is an
/// approximation by its support; the posterior is exact for the declared one.
#[derive(Clone, Debug, PartialEq)]
pub struct TauPrior {
    /// Support points (`tau >= 0`).
    pub support: Vec<f64>,
    /// Prior probabilities, positive and summing to one.
    pub probabilities: Vec<f64>,
}

/// Posterior of the common mean and of each study effect under hierarchical
/// normal-normal borrowing.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalPosterior {
    /// Posterior mean of the common mean.
    pub mean: f64,
    /// Posterior variance of the common mean.
    pub variance: f64,
    /// Posterior mean of each study's true effect (borrowed estimate).
    pub study_means: Vec<f64>,
    /// Posterior variance of each study's true effect.
    pub study_variances: Vec<f64>,
    /// Posterior probability of each prior support point of `tau`.
    pub tau_posterior: Vec<f64>,
}

struct ConditionalPosterior {
    log_marginal: f64,
    mean: f64,
    variance: f64,
}

fn conditional_on_tau(
    estimates: &[f64],
    variances: &[f64],
    tau: f64,
    prior: MeanPrior,
) -> ConditionalPosterior {
    let t2 = tau * tau;
    let w: Vec<f64> = variances.iter().map(|v| 1.0 / (v + t2)).collect();
    let sum_w: f64 = w.iter().sum();
    let log_det: f64 = variances.iter().map(|v| (v + t2).ln()).sum();
    match prior {
        MeanPrior::Flat => {
            let sy: f64 = w.iter().zip(estimates).map(|(wi, y)| wi * y).sum();
            let syy: f64 = w.iter().zip(estimates).map(|(wi, y)| wi * y * y).sum();
            let mean = sy / sum_w;
            ConditionalPosterior {
                log_marginal: -0.5 * log_det - 0.5 * sum_w.ln() - 0.5 * (syy - sy * sy / sum_w),
                mean,
                variance: 1.0 / sum_w,
            }
        }
        MeanPrior::Normal { mean: m0, variance: v0 } => {
            let sd: f64 = w.iter().zip(estimates).map(|(wi, y)| wi * (y - m0)).sum();
            let sdd: f64 = w.iter().zip(estimates).map(|(wi, y)| wi * (y - m0) * (y - m0)).sum();
            let denom = 1.0 + v0 * sum_w;
            let precision = 1.0 / v0 + sum_w;
            ConditionalPosterior {
                log_marginal: -0.5 * log_det
                    - 0.5 * denom.ln()
                    - 0.5 * (sdd - v0 * sd * sd / denom),
                mean: m0 + sd / precision,
                variance: 1.0 / precision,
            }
        }
    }
}

/// Hierarchical normal-normal borrowing on study summaries.
///
/// Model: `y_i | theta_i ~ N(theta_i, v_i)`, `theta_i | mu, tau ~ N(mu, tau^2)`,
/// `mu` from the declared [`MeanPrior`], `tau` from the declared finite
/// [`TauPrior`]. For each support point the posterior of `mu` and of every
/// `theta_i` is normal in closed form; the posterior mean and variance are the
/// exact finite mixture over the posterior of `tau`. Point-only: the mixture
/// is a computed posterior mean and variance, not a calibrated interval.
///
/// # Errors
/// Input, prior-declaration or non-finite-arithmetic refusals.
pub fn hierarchical_normal_borrowing(
    estimates: &[f64],
    variances: &[f64],
    mean_prior: MeanPrior,
    tau_prior: &TauPrior,
) -> Result<HierarchicalPosterior, SynthesisError> {
    let k = check_summaries(estimates, variances)?;
    if k < 2 {
        return Err(SynthesisError::TooFewStudies { needed: 2, got: k });
    }
    if let MeanPrior::Normal { mean, variance } = mean_prior {
        if !mean.is_finite() || !variance.is_finite() || variance <= 0.0 {
            return Err(SynthesisError::InvalidPrior(
                "mean prior needs a finite mean and positive variance",
            ));
        }
    }
    let n = tau_prior.support.len();
    if n == 0 || n > MAX_TAU_SUPPORT || tau_prior.probabilities.len() != n {
        return Err(SynthesisError::InvalidPrior(
            "tau prior support and probabilities disagree or exceed the bound",
        ));
    }
    if tau_prior.support.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(SynthesisError::InvalidPrior("tau support must be finite and nonnegative"));
    }
    if tau_prior.probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Err(SynthesisError::InvalidPrior("tau probabilities must be finite and positive"));
    }
    let total: f64 = tau_prior.probabilities.iter().sum();
    if (total - 1.0).abs() > PRIOR_SUM_TOLERANCE {
        return Err(SynthesisError::InvalidPrior("tau probabilities must sum to one"));
    }
    let conditionals: Vec<ConditionalPosterior> = tau_prior
        .support
        .iter()
        .map(|&tau| conditional_on_tau(estimates, variances, tau, mean_prior))
        .collect();
    let log_posterior: Vec<f64> = conditionals
        .iter()
        .zip(&tau_prior.probabilities)
        .map(|(c, p)| p.ln() + c.log_marginal)
        .collect();
    let peak = log_posterior.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return Err(SynthesisError::NotConverged("marginal likelihood is not finite"));
    }
    let unnormalized: Vec<f64> = log_posterior.iter().map(|l| (l - peak).exp()).collect();
    let norm: f64 = unnormalized.iter().sum();
    let tau_posterior: Vec<f64> = unnormalized.iter().map(|u| u / norm).collect();

    let mut mean = 0.0;
    let mut second = 0.0;
    for (pi, c) in tau_posterior.iter().zip(&conditionals) {
        mean += pi * c.mean;
        second += pi * (c.variance + c.mean * c.mean);
    }
    let mut study_means = vec![0.0; k];
    let mut study_second = vec![0.0; k];
    for ((pi, c), &tau) in tau_posterior.iter().zip(&conditionals).zip(&tau_prior.support) {
        let t2 = tau * tau;
        for i in 0..k {
            let b = t2 / (t2 + variances[i]);
            let m = b * estimates[i] + (1.0 - b) * c.mean;
            let v = b * variances[i] + (1.0 - b) * (1.0 - b) * c.variance;
            study_means[i] += pi * m;
            study_second[i] += pi * (v + m * m);
        }
    }
    let study_variances: Vec<f64> =
        (0..k).map(|i| (study_second[i] - study_means[i] * study_means[i]).max(0.0)).collect();
    Ok(HierarchicalPosterior {
        mean,
        variance: (second - mean * mean).max(0.0),
        study_means,
        study_variances,
        tau_posterior,
    })
}

/// Extremes of the pooled estimate and its variance over a declared range of a
/// common pairwise correlation.
///
/// This is an assumption range. It is never a point estimate, a confidence or
/// credible interval, or a posterior, and it does not convert unknown
/// dependence into independence: it reports what the pooled number would be
/// under each correlation in the declared range and only the extremes.
#[derive(Clone, Debug, PartialEq)]
pub struct DependenceEnvelope {
    /// Number of studies.
    pub studies: usize,
    /// Declared lower correlation bound.
    pub rho_lo: f64,
    /// Declared upper correlation bound.
    pub rho_hi: f64,
    /// Smallest pooled estimate over the range.
    pub estimate_min: f64,
    /// Correlation at which the smallest pooled estimate occurs.
    pub rho_at_estimate_min: f64,
    /// Largest pooled estimate over the range.
    pub estimate_max: f64,
    /// Correlation at which the largest pooled estimate occurs.
    pub rho_at_estimate_max: f64,
    /// Smallest pooled variance over the range.
    pub variance_min: f64,
    /// Correlation at which the smallest pooled variance occurs.
    pub rho_at_variance_min: f64,
    /// Largest pooled variance over the range.
    pub variance_max: f64,
    /// Correlation at which the largest pooled variance occurs.
    pub rho_at_variance_max: f64,
}

impl DependenceEnvelope {
    /// The claim label: always [`DEPENDENCE_ENVELOPE_LABEL`].
    #[must_use]
    pub const fn label(&self) -> &'static str {
        DEPENDENCE_ENVELOPE_LABEL
    }
}

struct EquicorrelationTerms {
    k: f64,
    b: f64,
    p: f64,
    m: f64,
    mp: f64,
}

impl EquicorrelationTerms {
    fn new(estimates: &[f64], standard_errors: &[f64]) -> Self {
        let k = estimates.len() as f64;
        let a: f64 = standard_errors.iter().map(|s| 1.0 / s).sum();
        let b: f64 = standard_errors.iter().map(|s| 1.0 / (s * s)).sum();
        let p: f64 = estimates.iter().zip(standard_errors).map(|(y, s)| y / (s * s)).sum();
        let q: f64 = estimates.iter().zip(standard_errors).map(|(y, s)| y / s).sum();
        Self { k, b, p, m: (k - 1.0) * b - a * a, mp: (k - 1.0) * p - a * q }
    }

    fn validate(&self) -> Result<(), SynthesisError> {
        if ![self.k, self.b, self.p, self.m, self.mp].into_iter().all(f64::is_finite) {
            return Err(SynthesisError::NonFinite("computed equicorrelation terms"));
        }
        if self.b <= 0.0 {
            return Err(SynthesisError::NonPositiveVariance("computed study precision"));
        }
        Ok(())
    }

    /// `1' S^-1 1 (1 - rho) (1 + (k-1) rho)` for `S = D R(rho) D`.
    fn denominator(&self, rho: f64) -> f64 {
        self.m.mul_add(rho, self.b)
    }

    fn estimate(&self, rho: f64) -> f64 {
        self.mp.mul_add(rho, self.p) / self.denominator(rho)
    }

    fn variance(&self, rho: f64) -> f64 {
        (1.0 - rho) * (rho.mul_add(self.k - 1.0, 1.0)) / self.denominator(rho)
    }

    /// Stationary points of the variance: roots of
    /// `-(k-1) m rho^2 - 2 (k-1) b rho + ((k-2) b - m)`.
    fn variance_stationary_points(&self) -> Result<Vec<f64>, SynthesisError> {
        let km1 = self.k - 1.0;
        let qa = -km1 * self.m;
        let qb = -2.0 * km1 * self.b;
        let qc = (self.k - 2.0).mul_add(self.b, -self.m);
        if ![qa, qb, qc].into_iter().all(f64::is_finite) {
            return Err(SynthesisError::NonFinite("variance stationary-point coefficients"));
        }
        if qa.abs() <= 1e-14 * qb.abs() {
            if qb == 0.0 {
                return Ok(Vec::new());
            }
            let root = -qc / qb;
            return if root.is_finite() {
                Ok(vec![root])
            } else {
                Err(SynthesisError::NonFinite("variance stationary point"))
            };
        }
        let discriminant = qb * qb - 4.0 * qa * qc;
        if !discriminant.is_finite() {
            return Err(SynthesisError::NonFinite("variance stationary-point discriminant"));
        }
        if discriminant < 0.0 {
            return Ok(Vec::new());
        }
        let root = discriminant.sqrt();
        let t = -0.5 * (qb + qb.signum() * root);
        let mut out = vec![t / qa];
        if t != 0.0 {
            out.push(qc / t);
        }
        if out.iter().any(|root| !root.is_finite()) {
            return Err(SynthesisError::NonFinite("variance stationary point"));
        }
        Ok(out)
    }
}

type EnvelopeExtremes = ((f64, f64), (f64, f64));

fn finite_extremes(
    points: &[f64],
    initial_rho: f64,
    f: impl Fn(f64) -> f64,
) -> Result<EnvelopeExtremes, SynthesisError> {
    let mut min = (f64::INFINITY, initial_rho);
    let mut max = (f64::NEG_INFINITY, initial_rho);
    for &rho in points {
        let value = f(rho);
        if !value.is_finite() {
            return Err(SynthesisError::NonFinite("computed dependence envelope"));
        }
        if value < min.0 {
            min = (value, rho);
        }
        if value > max.0 {
            max = (value, rho);
        }
    }
    Ok((min, max))
}

/// Pooled estimate and variance at one common correlation `rho`, by the
/// closed form for equicorrelated summaries `S = D R(rho) D`.
///
/// # Errors
/// Fewer than two studies, input refusals, or `rho` outside the positive
/// definite range `(-1/(k-1), 1)`.
pub fn equicorrelated_gls(
    estimates: &[f64],
    standard_errors: &[f64],
    rho: f64,
) -> Result<(f64, f64), SynthesisError> {
    let k = check_equicorrelated_inputs(estimates, standard_errors)?;
    if !rho.is_finite() || rho >= 1.0 || rho <= -1.0 / (k as f64 - 1.0) {
        return Err(SynthesisError::InvalidCorrelationRange(
            "correlation must lie strictly inside (-1/(k-1), 1)",
        ));
    }
    let terms = EquicorrelationTerms::new(estimates, standard_errors);
    terms.validate()?;
    let denominator = terms.denominator(rho);
    let estimate = terms.estimate(rho);
    let variance = terms.variance(rho);
    if !(denominator.is_finite()
        && denominator > 0.0
        && estimate.is_finite()
        && variance.is_finite())
    {
        return Err(SynthesisError::NonFinite("computed equicorrelated GLS result"));
    }
    Ok((estimate, variance))
}

fn check_equicorrelated_inputs(
    estimates: &[f64],
    standard_errors: &[f64],
) -> Result<usize, SynthesisError> {
    let k = estimates.len();
    if k < 2 {
        return Err(SynthesisError::TooFewStudies { needed: 2, got: k });
    }
    if standard_errors.len() != k {
        return Err(SynthesisError::Shape("estimates and standard errors differ in length"));
    }
    if k > MAX_STUDIES {
        return Err(SynthesisError::Shape("more studies than the synthesis bound"));
    }
    if estimates.iter().any(|v| !v.is_finite()) {
        return Err(SynthesisError::NonFinite("estimate"));
    }
    if standard_errors.iter().any(|v| !v.is_finite()) {
        return Err(SynthesisError::NonFinite("standard error"));
    }
    if standard_errors.iter().any(|&v| v <= 0.0) {
        return Err(SynthesisError::NonPositiveVariance("standard error"));
    }
    Ok(k)
}

/// Pooled estimate and variance over a declared range of one common pairwise
/// correlation, returned as an exact structural envelope.
///
/// Studies with standard errors `s_i` and common correlation `rho` have
/// covariance `S = D R(rho) D`, `R = (1-rho) I + rho 11'`. With
/// `A = sum 1/s`, `B = sum 1/s^2`, `P = sum y/s^2`, `Q = sum y/s`,
/// `m = (k-1) B - A^2` and `m_P = (k-1) P - A Q` the generalized least squares
/// results are `mu(rho) = (P + m_P rho) / (B + m rho)` and
/// `V(rho) = (1 - rho)(1 + (k-1) rho) / (B + m rho)`. `mu` is a Moebius
/// function of `rho` and therefore monotone on the range (its derivative has
/// the constant sign of `m_P B - m P`), so its extremes are at the endpoints.
/// `V` can have an interior extremum: its derivative has the sign of the
/// quadratic `-(k-1) m rho^2 - 2 (k-1) B rho + ((k-2) B - m)`, whose real
/// roots inside the range are evaluated beside the endpoints. The extremes
/// are exact (no grid).
///
/// # Errors
/// Fewer than two studies, input refusals, a degenerate range (`lo >= hi`) or
/// a range leaving the positive definite interval `(-1/(k-1), 1)`.
pub fn pooled_dependence_envelope(
    estimates: &[f64],
    standard_errors: &[f64],
    rho_lo: f64,
    rho_hi: f64,
) -> Result<DependenceEnvelope, SynthesisError> {
    let k = check_equicorrelated_inputs(estimates, standard_errors)?;
    if !rho_lo.is_finite() || !rho_hi.is_finite() {
        return Err(SynthesisError::InvalidCorrelationRange("bounds must be finite"));
    }
    if rho_lo >= rho_hi {
        return Err(SynthesisError::InvalidCorrelationRange(
            "the range needs lo < hi; a single correlation is a covariance declaration",
        ));
    }
    if rho_hi >= 1.0 || rho_lo <= -1.0 / (k as f64 - 1.0) {
        return Err(SynthesisError::InvalidCorrelationRange(
            "the range must lie strictly inside the positive definite interval (-1/(k-1), 1)",
        ));
    }
    let terms = EquicorrelationTerms::new(estimates, standard_errors);
    terms.validate()?;

    let estimate_points = [rho_lo, rho_hi];
    // The estimate has no interior stationary point: its derivative numerator
    // `m_P B - m P` is constant in rho. A zero constant makes it flat, and the
    // endpoints already give the common value.
    let mut variance_points = vec![rho_lo, rho_hi];
    for root in terms.variance_stationary_points()? {
        if root.is_finite() && root > rho_lo && root < rho_hi {
            variance_points.push(root);
        }
    }
    let (est_min, est_max) = finite_extremes(&estimate_points, rho_lo, |r| terms.estimate(r))?;
    let (var_min, var_max) = finite_extremes(&variance_points, rho_lo, |r| terms.variance(r))?;
    if ![est_min.0, est_min.1, est_max.0, est_max.1, var_min.0, var_min.1, var_max.0, var_max.1]
        .into_iter()
        .all(f64::is_finite)
        || estimate_points.into_iter().chain(variance_points.iter().copied()).any(|rho| {
            let denominator = terms.denominator(rho);
            !denominator.is_finite() || denominator <= 0.0
        })
    {
        return Err(SynthesisError::NonFinite("computed dependence envelope"));
    }
    Ok(DependenceEnvelope {
        studies: k,
        rho_lo,
        rho_hi,
        estimate_min: est_min.0,
        rho_at_estimate_min: est_min.1,
        estimate_max: est_max.0,
        rho_at_estimate_max: est_max.1,
        variance_min: var_min.0,
        rho_at_variance_min: var_min.1,
        variance_max: var_max.0,
        rho_at_variance_max: var_max.1,
    })
}

#[cfg(test)]
mod dependence_envelope_tests {
    use super::*;

    #[test]
    fn finite_extrema_reducer_does_not_hide_nan_endpoints() {
        let result =
            finite_extremes(&[0.0, 1.0], 0.0, |rho| if rho == 0.0 { 1.0 } else { f64::NAN });
        assert!(matches!(result, Err(SynthesisError::NonFinite(_))));
    }
}
