//! Monte Carlo (numerical) error of decision functionals.
//!
//! Every function here estimates the *numerical* error of a quantity computed from
//! draws of a supplied law. None of it measures how wrong the supplied law is, and no
//! interval is built: the outputs are standard errors and effective sample sizes
//! labelled with the method that produced them.
//!
//! Methods (each verified against an independent oracle in the tests):
//!
//! * means, threshold probabilities, differences: the sample variance for serially
//!   independent draws, non-overlapping **batch means** for dependent streams
//!   (effective sample size `b * s^2 / v_b`, capped at `n`), the self-normalized
//!   **delta method** with Kish's effective sample size for importance weights;
//! * quantile: the **Maritz-Jarrett** (1978) exact bootstrap standard error of the
//!   `r`-th order statistic, `r = ceil(n p)`, computed from incomplete-beta weights
//!   rather than by resampling; **batch quantiles** for dependent streams;
//! * `CVaR`: the standard error from the **influence function** of the
//!   Rockafellar-Uryasev functional, `IF(u) = (min(u, q) - q) / alpha + q - CVaR`,
//!   estimated at the sample quantile `q`; batch means of the influence series for
//!   dependent streams;
//! * differences under common random numbers: the variance of the paired influence
//!   series, so shared noise cancels exactly as far as it is correlated; across
//!   independent streams the variances add.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::kish_ess;
use antecedent_stats::ln_gamma;
use serde::{Deserialize, Serialize};

use super::draws::WithinStreamDependence;
use super::{MIN_DRAWS, ReliabilityError};

/// How many non-overlapping batches a dependent stream is cut into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchPlan {
    /// `floor(sqrt(n))` batches, at least two.
    SquareRoot,
    /// A fixed batch count, at least two and at most `n`.
    Fixed(u32),
}

impl BatchPlan {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
    )]
    pub(crate) fn batches(self, n: usize) -> Result<usize, ReliabilityError> {
        let b = match self {
            Self::SquareRoot => ((n as f64).sqrt().floor() as usize).max(2),
            Self::Fixed(b) => b as usize,
        };
        if b < 2 || b > n {
            return Err(ReliabilityError::invalid(
                "batch_plan",
                format!("{b} batches for {n} draws; need 2..=n"),
            ));
        }
        Ok(b)
    }
}

/// Which side of a threshold counts as exceeding it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdSide {
    /// `utility > threshold`.
    Above,
    /// `utility <= threshold`.
    AtOrBelow,
}

/// Which tail a conditional value at risk averages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskTail {
    /// The mean of the smallest `alpha` fraction of utilities (the downside).
    Lower,
    /// The mean of the largest `alpha` fraction (the tail of a loss).
    Upper,
}

/// A functional of the utility distribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Functional {
    /// The expected utility.
    ExpectedUtility,
    /// The probability that the utility lies on one side of a threshold.
    ThresholdProbability {
        /// Utility threshold (finite).
        threshold: f64,
        /// Which side is the event.
        side: ThresholdSide,
    },
    /// The `p`-quantile, `p` in `(0, 1)`.
    Quantile {
        /// Probability level.
        p: f64,
    },
    /// Conditional value at risk at tail mass `alpha` in `(0, 1]`.
    Cvar {
        /// Tail mass averaged.
        alpha: f64,
        /// Which tail.
        tail: RiskTail,
    },
}

impl Functional {
    /// Stable label of the functional family (the calibration coordinate).
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::ExpectedUtility => "expected_utility",
            Self::ThresholdProbability { .. } => "threshold_probability",
            Self::Quantile { .. } => "quantile",
            Self::Cvar { .. } => "cvar",
        }
    }

    /// Check the declared parameters.
    ///
    /// # Errors
    /// A non-finite threshold or a level outside its open or half-open range.
    pub fn validate(&self) -> Result<(), ReliabilityError> {
        let ok = match self {
            Self::ExpectedUtility => true,
            Self::ThresholdProbability { threshold, .. } => threshold.is_finite(),
            Self::Quantile { p } => p.is_finite() && *p > 0.0 && *p < 1.0,
            Self::Cvar { alpha, .. } => alpha.is_finite() && *alpha > 0.0 && *alpha <= 1.0,
        };
        if ok {
            Ok(())
        } else {
            Err(ReliabilityError::invalid(
                "functional",
                "threshold must be finite; quantile level in (0,1); cvar alpha in (0,1]",
            ))
        }
    }
}

/// How a standard error was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandardErrorMethod {
    /// Sample variance of serially independent draws.
    IidSampleVariance,
    /// Non-overlapping batch means of a dependent stream.
    BatchMeans {
        /// Number of batches.
        batches: u32,
        /// Draws per batch.
        batch_len: u32,
    },
    /// Self-normalized importance weights, delta-method variance, Kish ESS.
    SelfNormalizedWeights,
    /// Exact bootstrap variance of an order statistic (Maritz-Jarrett).
    MaritzJarrett,
    /// Variance of per-batch order statistics of a dependent stream.
    BatchQuantiles {
        /// Number of batches.
        batches: u32,
        /// Draws per batch.
        batch_len: u32,
    },
    /// Influence-function variance of a risk measure, serially independent draws.
    InfluenceFunction,
    /// Batch means of the influence series of a risk measure.
    InfluenceFunctionBatchMeans {
        /// Number of batches.
        batches: u32,
        /// Draws per batch.
        batch_len: u32,
    },
    /// Variances of two independent streams added.
    IndependentStreamsAdded,
}

/// A Monte Carlo point estimate with its numerical error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McEstimate {
    /// The point estimate.
    pub value: f64,
    /// Estimated standard error of the point estimate due to finite draws only.
    pub standard_error: f64,
    /// Effective number of independent draws the standard error corresponds to.
    pub effective_sample_size: f64,
    /// Draws used.
    pub n_draws: u64,
    /// How the standard error was obtained.
    pub method: StandardErrorMethod,
}

/// A functional's Monte Carlo estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionalEstimate {
    /// The functional estimated.
    pub functional: Functional,
    /// Its estimate and numerical error.
    pub estimate: McEstimate,
}

/// Whether two utility series came from one shared simulation or independent ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairCoupling {
    /// Draw `i` of both series is the same random state (common random numbers).
    SharedStream,
    /// The two series are independent streams.
    IndependentStreams,
}

/// A difference of two candidates' functionals with its numerical error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferenceEstimate {
    /// The functional differenced (`A - B`).
    pub functional: Functional,
    /// `value(A) - value(B)`.
    pub value: f64,
    /// Standard error of the difference due to finite draws.
    pub standard_error: f64,
    /// Effective sample size of the difference. `None` for quantile differences on a
    /// shared stream: the batch estimator has no per-draw variance decomposition.
    pub effective_sample_size: Option<f64>,
    /// Standard error of `A` alone.
    pub standard_error_a: f64,
    /// Standard error of `B` alone.
    pub standard_error_b: f64,
    /// How the two series were coupled.
    pub coupling: PairCoupling,
    /// `se_diff^2 / (se_a^2 + se_b^2)`: below one the shared stream removed variance
    /// relative to independent streams. `None` for independent streams.
    pub variance_ratio_vs_independent: Option<f64>,
    /// Implied correlation of the two estimators, `(se_a^2 + se_b^2 - se_diff^2) /
    /// (2 se_a se_b)`. `None` for independent streams or a zero standard error.
    pub implied_correlation: Option<f64>,
    /// How the standard error was obtained.
    pub method: StandardErrorMethod,
}

/// Mean, standard error and effective sample size of one series.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeriesStats {
    pub mean: f64,
    pub standard_error: f64,
    pub ess: f64,
    pub method: StandardErrorMethod,
}

fn welford(values: &[f64]) -> (f64, f64) {
    let (mut mean, mut m2) = (0.0_f64, 0.0_f64);
    for (i, &x) in values.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i as f64 + 1.0);
        m2 += delta * (x - mean);
    }
    let n = values.len();
    (mean, if n > 1 { m2 / (n as f64 - 1.0) } else { 0.0 })
}

fn finite_checked(stats: SeriesStats) -> Result<SeriesStats, ReliabilityError> {
    if stats.mean.is_finite() && stats.standard_error.is_finite() && stats.ess.is_finite() {
        Ok(stats)
    } else {
        Err(ReliabilityError::Numerical { message: "non-finite Monte Carlo error".into() })
    }
}

/// Mean and standard error of a series under the declared dependence.
#[allow(
    clippy::float_cmp,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
pub(crate) fn series_stats(
    values: &[f64],
    weights: Option<&[f64]>,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
) -> Result<SeriesStats, ReliabilityError> {
    let n = values.len();
    if n < MIN_DRAWS || weights.is_some_and(|w| w.len() != n) {
        return Err(ReliabilityError::invalid("draw_count", "too few or misaligned draws"));
    }
    match (dependence, weights) {
        (WithinStreamDependence::Unknown, _) => {
            Err(ReliabilityError::DependenceUnknown { what: "within_stream_serial" })
        }
        (WithinStreamDependence::Autocorrelated, Some(_)) => {
            Err(ReliabilityError::RouteNotSupported {
                message: "weighted estimate on an autocorrelated stream".into(),
            })
        }
        (_, Some(w)) => {
            let sum_w: f64 = w.iter().sum();
            let mean = values.iter().zip(w).map(|(y, wi)| y * wi).sum::<f64>() / sum_w;
            let numerator: f64 =
                values.iter().zip(w).map(|(y, wi)| (wi * (y - mean)).powi(2)).sum();
            finite_checked(SeriesStats {
                mean,
                standard_error: numerator.sqrt() / sum_w,
                ess: kish_ess(w),
                method: StandardErrorMethod::SelfNormalizedWeights,
            })
        }
        (WithinStreamDependence::Independent, None) => {
            let (mean, variance) = welford(values);
            finite_checked(SeriesStats {
                mean,
                standard_error: (variance / n as f64).sqrt(),
                ess: n as f64,
                method: StandardErrorMethod::IidSampleVariance,
            })
        }
        (WithinStreamDependence::Autocorrelated, None) => batch_means(values, plan),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
fn batch_means(values: &[f64], plan: BatchPlan) -> Result<SeriesStats, ReliabilityError> {
    let n = values.len();
    let b = plan.batches(n)?;
    let m = n / b;
    let used = b * m;
    let means: Vec<f64> =
        (0..b).map(|k| values[k * m..(k + 1) * m].iter().sum::<f64>() / m as f64).collect();
    let (_, batch_variance) = welford(&means);
    let (_, draw_variance) = welford(&values[..used]);
    let ess = if batch_variance > 0.0 {
        (b as f64 * draw_variance / batch_variance).clamp(1.0, n as f64)
    } else {
        n as f64
    };
    finite_checked(SeriesStats {
        mean: values.iter().sum::<f64>() / n as f64,
        standard_error: (batch_variance / b as f64).sqrt(),
        ess,
        method: StandardErrorMethod::BatchMeans { batches: b as u32, batch_len: m as u32 },
    })
}

/// One-based order index `ceil(p n)` clamped to `1..=n`, robust to `p n` landing a few
/// ulps above an integer.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
fn order_index(p: f64, n: usize) -> usize {
    let target = p * n as f64;
    let k = (target - 1e-9 * target.max(1.0)).ceil();
    (k as usize).clamp(1, n)
}

fn sorted_copy(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    // Equal total-order f64 keys have identical bits, so stability cannot
    // change any output; the in-place sort needs no hidden merge buffer.
    sorted.sort_unstable_by(f64::total_cmp);
    sorted
}

// ---- Regularized incomplete beta, Lentz continued fraction -----------------------

fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITERATIONS: usize = 200_000;
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let guard = |v: f64| if v.abs() < TINY { TINY } else { v };
    let mut c = 1.0_f64;
    let mut d = 1.0 / guard(1.0 - qab * x / qap);
    let mut h = d;
    for step in 1..=MAX_ITERATIONS {
        let m = step as f64;
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 / guard(1.0 + even * d);
        c = guard(1.0 + even / c);
        h *= d * c;
        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 / guard(1.0 + odd * d);
        c = guard(1.0 + odd / c);
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`.
fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_continued_fraction(1.0 - x, b, a) / b
    }
}

/// Maritz-Jarrett standard error of the `r`-th order statistic (one-based) of the
/// sorted sample: the exact bootstrap standard deviation, from the resampled order
/// statistic's distribution `P(X*_(r) = x_(i)) = I_{i/n}(r, n-r+1) - I_{(i-1)/n}(r, n-r+1)`.
///
/// Mass further than twelve standard deviations of the index from `r` is dropped and
/// the remaining weights renormalized.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
pub(crate) fn maritz_jarrett_se(sorted: &[f64], r: usize) -> f64 {
    let n = sorted.len();
    let nf = n as f64;
    let (a, b) = (r as f64, (n - r + 1) as f64);
    let index_sd = (a * b / (nf + 1.0)).sqrt();
    let half = (12.0 * index_sd).ceil() as usize + 5;
    let lo = r.saturating_sub(half).max(1);
    let hi = (r + half).min(n);
    let centre = sorted[r - 1];
    let mut previous = regularized_beta((lo - 1) as f64 / nf, a, b);
    let (mut total, mut m1, mut m2) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in lo..=hi {
        let current = regularized_beta(i as f64 / nf, a, b);
        let weight = (current - previous).max(0.0);
        previous = current;
        let d = sorted[i - 1] - centre;
        total += weight;
        m1 += weight * d;
        m2 += weight * d * d;
    }
    if total <= 0.0 {
        return 0.0;
    }
    let mean = m1 / total;
    (m2 / total - mean * mean).max(0.0).sqrt()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
fn quantile_estimate(
    values: &[f64],
    weights: Option<&[f64]>,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    p: f64,
) -> Result<McEstimate, ReliabilityError> {
    let n = values.len();
    if n < MIN_DRAWS {
        return Err(ReliabilityError::invalid("draw_count", "too few draws"));
    }
    if weights.is_some() {
        return Err(ReliabilityError::RouteNotSupported {
            message: "weighted quantile: no weighted order-statistic standard error is licensed"
                .into(),
        });
    }
    let sorted = sorted_copy(values);
    let r = order_index(p, n);
    let value = sorted[r - 1];
    let iid_se = maritz_jarrett_se(&sorted, r);
    match dependence {
        WithinStreamDependence::Unknown => {
            Err(ReliabilityError::DependenceUnknown { what: "within_stream_serial" })
        }
        WithinStreamDependence::Independent => Ok(McEstimate {
            value,
            standard_error: iid_se,
            effective_sample_size: n as f64,
            n_draws: n as u64,
            method: StandardErrorMethod::MaritzJarrett,
        }),
        WithinStreamDependence::Autocorrelated => {
            let b = plan.batches(n)?;
            let m = n / b;
            let quantiles = batch_quantiles(values, b, m, p);
            let (_, variance) = welford(&quantiles);
            let se = (variance / b as f64).sqrt();
            let ess = if se > 0.0 {
                (n as f64 * iid_se * iid_se / (se * se)).clamp(1.0, n as f64)
            } else {
                n as f64
            };
            Ok(McEstimate {
                value,
                standard_error: se,
                effective_sample_size: ess,
                n_draws: n as u64,
                method: StandardErrorMethod::BatchQuantiles {
                    batches: b as u32,
                    batch_len: m as u32,
                },
            })
        }
    }
}

fn batch_quantiles(values: &[f64], b: usize, m: usize, p: f64) -> Vec<f64> {
    (0..b)
        .map(|k| {
            let sorted = sorted_copy(&values[k * m..(k + 1) * m]);
            sorted[order_index(p, m) - 1]
        })
        .collect()
}

/// Lower-tail `CVaR` of the sample and the sample quantile it is anchored at:
/// `(S_{k-1} + (alpha n - (k-1)) x_(k)) / (alpha n)`, `k = ceil(alpha n)`.
fn cvar_lower(values: &[f64], alpha: f64) -> (f64, f64) {
    let n = values.len();
    let sorted = sorted_copy(values);
    let k = order_index(alpha, n);
    let zeta = sorted[k - 1];
    let head: f64 = sorted[..k - 1].iter().sum();
    let mass = alpha * n as f64;
    ((head + (mass - (k as f64 - 1.0)) * zeta) / mass, zeta)
}

/// A mean-like functional's point value (`None` for the series mean) and its series.
type MeanLikeSeries = (Option<f64>, Vec<f64>);

/// `(value, series)` of a mean-like functional. `value` is `None` when the point
/// estimate is the (possibly weighted) series mean.
fn mean_like(
    values: &[f64],
    weights: Option<&[f64]>,
    functional: &Functional,
) -> Result<Option<MeanLikeSeries>, ReliabilityError> {
    Ok(match functional {
        Functional::ExpectedUtility => Some((None, values.to_vec())),
        Functional::ThresholdProbability { threshold, side } => {
            let hit = |u: f64| match side {
                ThresholdSide::Above => u > *threshold,
                ThresholdSide::AtOrBelow => u <= *threshold,
            };
            Some((None, values.iter().map(|&u| f64::from(u8::from(hit(u)))).collect()))
        }
        Functional::Cvar { alpha, tail } => {
            if weights.is_some() {
                return Err(ReliabilityError::RouteNotSupported {
                    message: "weighted cvar: no weighted influence-function error is licensed"
                        .into(),
                });
            }
            let sign = match tail {
                RiskTail::Lower => 1.0,
                RiskTail::Upper => -1.0,
            };
            let signed: Vec<f64> = values.iter().map(|&u| sign * u).collect();
            let (cvar, zeta) = cvar_lower(&signed, *alpha);
            let series = signed.iter().map(|&u| sign * u.min(zeta) / alpha).collect();
            Some((Some(sign * cvar), series))
        }
        Functional::Quantile { .. } => None,
    })
}

fn influence_method(functional: &Functional, method: StandardErrorMethod) -> StandardErrorMethod {
    match (functional, method) {
        (Functional::Cvar { .. }, StandardErrorMethod::IidSampleVariance) => {
            StandardErrorMethod::InfluenceFunction
        }
        (Functional::Cvar { .. }, StandardErrorMethod::BatchMeans { batches, batch_len }) => {
            StandardErrorMethod::InfluenceFunctionBatchMeans { batches, batch_len }
        }
        (_, other) => other,
    }
}

/// Estimate one functional of one candidate's utilities with its Monte Carlo error.
///
/// `weights` are optional importance weights (mean-like functionals only);
/// `dependence` is the declared serial dependence of the draws.
///
/// # Errors
/// An invalid functional, too few draws, undeclared dependence, or a weighted
/// quantile/`CVaR` request (no weighted standard error is licensed).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
pub fn estimate(
    values: &[f64],
    weights: Option<&[f64]>,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    functional: &Functional,
) -> Result<FunctionalEstimate, ReliabilityError> {
    functional.validate()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ReliabilityError::invalid("non_finite_utility", "utilities must be finite"));
    }
    let estimate = if let Some((value, series)) = mean_like(values, weights, functional)? {
        let stats = series_stats(&series, weights, dependence, plan)?;
        McEstimate {
            value: value.unwrap_or(stats.mean),
            standard_error: stats.standard_error,
            effective_sample_size: stats.ess,
            n_draws: values.len() as u64,
            method: influence_method(functional, stats.method),
        }
    } else {
        let Functional::Quantile { p } = functional else {
            return Err(ReliabilityError::invalid("functional", "unreachable quantile arm"));
        };
        quantile_estimate(values, weights, dependence, plan, *p)?
    };
    Ok(FunctionalEstimate { functional: functional.clone(), estimate })
}

/// Estimate `functional(A) - functional(B)` with its Monte Carlo error.
///
/// On a [`PairCoupling::SharedStream`] the two series must be aligned draw by draw (the
/// caller has already established that draw `i` is one random state); the error is the
/// variance of the paired influence series, so common random numbers cancel as far as
/// they are correlated. On [`PairCoupling::IndependentStreams`] the variances add and
/// the series may differ in length.
///
/// # Errors
/// Misaligned shared series, or any refusal of [`estimate`].
#[allow(
    clippy::too_many_lines,
    reason = "one coupling dispatch whose arms share the result constructor"
)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
pub fn estimate_difference(
    a: &[f64],
    b: &[f64],
    weights: Option<&[f64]>,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    functional: &Functional,
    coupling: PairCoupling,
) -> Result<DifferenceEstimate, ReliabilityError> {
    let ea = estimate(a, weights, dependence, plan, functional)?.estimate;
    let eb = estimate(b, weights, dependence, plan, functional)?.estimate;
    let value = ea.value - eb.value;
    let finish = |standard_error: f64,
                  ess: Option<f64>,
                  method: StandardErrorMethod,
                  ratio: Option<f64>,
                  correlation: Option<f64>| DifferenceEstimate {
        functional: functional.clone(),
        value,
        standard_error,
        effective_sample_size: ess,
        standard_error_a: ea.standard_error,
        standard_error_b: eb.standard_error,
        coupling,
        variance_ratio_vs_independent: ratio,
        implied_correlation: correlation,
        method,
    };
    let (va, vb) = (ea.standard_error.powi(2), eb.standard_error.powi(2));
    if coupling == PairCoupling::IndependentStreams {
        let se = (va + vb).sqrt();
        let ess = if se > 0.0 {
            (va * ea.effective_sample_size + vb * eb.effective_sample_size) / (se * se)
        } else {
            ea.effective_sample_size.min(eb.effective_sample_size)
        };
        return Ok(finish(se, Some(ess), StandardErrorMethod::IndependentStreamsAdded, None, None));
    }
    if a.len() != b.len() {
        return Err(ReliabilityError::JointLawRequired {
            reason: "shared_stream_series_of_different_length",
        });
    }
    let (se, ess, method) = if let Functional::Quantile { p } = functional {
        if weights.is_some() {
            return Err(ReliabilityError::RouteNotSupported {
                message: "weighted quantile difference".into(),
            });
        }
        let batches = plan.batches(a.len())?;
        let m = a.len() / batches;
        let qa = batch_quantiles(a, batches, m, *p);
        let qb = batch_quantiles(b, batches, m, *p);
        let paired: Vec<f64> = qa.iter().zip(&qb).map(|(x, y)| x - y).collect();
        let (_, variance) = welford(&paired);
        (
            (variance / batches as f64).sqrt(),
            None,
            StandardErrorMethod::BatchQuantiles { batches: batches as u32, batch_len: m as u32 },
        )
    } else {
        let series_a = mean_like(a, weights, functional)?.map(|(_, s)| s);
        let series_b = mean_like(b, weights, functional)?.map(|(_, s)| s);
        let (Some(sa), Some(sb)) = (series_a, series_b) else {
            return Err(ReliabilityError::invalid("functional", "no influence series"));
        };
        let paired: Vec<f64> = sa.iter().zip(&sb).map(|(x, y)| x - y).collect();
        let stats = series_stats(&paired, weights, dependence, plan)?;
        (stats.standard_error, Some(stats.ess), influence_method(functional, stats.method))
    };
    let independent = va + vb;
    let ratio = (independent > 0.0).then(|| se * se / independent);
    let correlation = (ea.standard_error > 0.0 && eb.standard_error > 0.0)
        .then(|| (independent - se * se) / (2.0 * ea.standard_error * eb.standard_error));
    Ok(finish(se, ess, method, ratio, correlation))
}

/// Existing estimator with its concrete temporary buffers admitted by the caller.
pub(crate) fn estimate_in_context(
    values: &[f64],
    weights: Option<&[f64]>,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    functional: &Functional,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<FunctionalEstimate, ReliabilityError> {
    functional.validate()?;
    let slots = super::workspace::estimate_slots(values.len(), dependence, plan, functional)?;
    let _scratch = super::workspace::reserve::<f64>(ctx, slots)?;
    estimate(values, weights, dependence, plan, functional)
}

#[expect(
    clippy::too_many_arguments,
    reason = "private contextual adapter preserves the existing paired scientific inputs and adds the actual caller context"
)]
pub(crate) fn estimate_difference_in_context(
    a: &[f64],
    b: &[f64],
    weights: Option<&[f64]>,
    dependence: WithinStreamDependence,
    plan: BatchPlan,
    functional: &Functional,
    coupling: PairCoupling,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<DifferenceEstimate, ReliabilityError> {
    functional.validate()?;
    let slots = if coupling == PairCoupling::IndependentStreams {
        super::workspace::estimate_slots(a.len(), dependence, plan, functional)?
            .max(super::workspace::estimate_slots(b.len(), dependence, plan, functional)?)
    } else {
        super::workspace::difference_slots(a.len(), dependence, plan, functional)?
    };
    let _scratch = super::workspace::reserve::<f64>(ctx, slots)?;
    estimate_difference(a, b, weights, dependence, plan, functional, coupling)
}

#[allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "exact tie and degenerate-variance tests on finite draws; counts are bounded by the carrier limits"
)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Exact bootstrap variance of the `r`-th order statistic by enumerating every
    /// ordered resample of size `n`.
    fn enumerated_bootstrap_se(values: &[f64], r: usize) -> f64 {
        let n = values.len();
        let total = n.pow(n as u32);
        let (mut s1, mut s2) = (0.0, 0.0);
        for code in 0..total {
            let mut c = code;
            let mut sample: Vec<f64> = (0..n)
                .map(|_| {
                    let v = values[c % n];
                    c /= n;
                    v
                })
                .collect();
            sample.sort_by(f64::total_cmp);
            s1 += sample[r - 1];
            s2 += sample[r - 1] * sample[r - 1];
        }
        let mean = s1 / total as f64;
        (s2 / total as f64 - mean * mean).sqrt()
    }

    #[test]
    fn maritz_jarrett_equals_the_enumerated_bootstrap() {
        let values = [1.0, 2.0, 4.0, 8.0, 16.0, 0.5];
        let sorted = sorted_copy(&values);
        for r in 1..=values.len() {
            let exact = enumerated_bootstrap_se(&sorted, r);
            let got = maritz_jarrett_se(&sorted, r);
            assert!((got - exact).abs() < 1e-8, "r={r}: {got} vs {exact}");
        }
    }

    #[test]
    fn regularized_beta_matches_binomial_tail() {
        // I_x(r, n-r+1) = P(Bin(n, x) >= r).
        let (n, r, x) = (12_usize, 5_usize, 0.37_f64);
        let mut tail = 0.0;
        for k in r..=n {
            let ln_c = ln_gamma(n as f64 + 1.0)
                - ln_gamma(k as f64 + 1.0)
                - ln_gamma((n - k) as f64 + 1.0);
            tail += (ln_c + k as f64 * x.ln() + (n - k) as f64 * (1.0 - x).ln()).exp();
        }
        assert!((regularized_beta(x, r as f64, (n - r + 1) as f64) - tail).abs() < 1e-12);
    }

    #[test]
    fn order_index_is_stable_at_integer_products() {
        assert_eq!(order_index(0.1, 100), 10);
        assert_eq!(order_index(0.5, 11), 6);
        assert_eq!(order_index(0.999, 10), 10);
    }
}
