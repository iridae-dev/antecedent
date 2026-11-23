//! Detection that depends on the evidence available for a unit.
//!
//! A true condition is recorded as present (`D_i = 1`) with probability `d(c_i)` when it is
//! present, where `c_i` is a covariate such as the number of records, visits or documents the
//! unit has, and with the false-positive rate `f` when it is absent. With a presence
//! probability `pi` that does not depend on `c`,
//! `P(D_i = 1) = pi d(c_i) + (1 - pi) f`, and
//!
//! `pi = (sum D_i - n f) / (sum d(c_i) - n f)`.
//!
//! * A **fully declared** detection function (a logistic function of `c`, or a piecewise-linear
//!   table) point-identifies `pi`. The variance is the Bernoulli sampling variance plus the
//!   uncertainty of the declared function's parameters and of `f`.
//! * A **partially declared** function is a monotone envelope: bounds `[lo, hi]` of `d` at
//!   anchor covariate values and a direction. The envelope gives each unit the tightest
//!   bounds the monotone constraint and the anchors allow, and `pi` is then only bounded:
//!   `sum d` can take any value between the sums of the two bounds, so the bounds
//!   `A / G(hi)` and `A / G(lo)` (with `A = sum D_i - n f`, `G = sum d - n f`) are sharp for
//!   the declaration.
//!
//! The identification check is the mean detection net of false positives,
//! `mean(d) - f`, which must exceed a declared floor: it plays the role of Youden's index.

use super::MeasurementErrorError;

/// A fully declared detection function of one covariate.
#[derive(Clone, Debug, PartialEq)]
pub enum DetectionFunction {
    /// `d(c) = 1 / (1 + exp(-(intercept + slope c)))`, with the parameter covariance.
    Logistic {
        /// Intercept.
        intercept: f64,
        /// Slope on the covariate.
        slope: f64,
        /// Variance of the intercept estimate.
        intercept_variance: f64,
        /// Covariance of the intercept and slope estimates.
        covariance: f64,
        /// Variance of the slope estimate.
        slope_variance: f64,
    },
    /// Piecewise linear through `(knots[i], values[i])`, constant beyond the end knots.
    Table {
        /// Strictly increasing covariate values.
        knots: Vec<f64>,
        /// Detection probability at each knot, each in `(0, 1]`.
        values: Vec<f64>,
    },
}

/// Direction of a monotone detection function.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectionDirection {
    /// Detection does not decrease as the covariate grows.
    Nondecreasing,
    /// Detection does not increase as the covariate grows.
    Nonincreasing,
}

/// One anchor of a partially declared detection function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionAnchor {
    /// Covariate value.
    pub covariate: f64,
    /// Lowest detection probability at that value.
    pub lower: f64,
    /// Highest detection probability at that value.
    pub upper: f64,
}

/// What is declared about the detection function.
#[derive(Clone, Debug, PartialEq)]
pub enum DetectionDeclaration {
    /// The whole function: the prevalence is point-identified.
    Full(DetectionFunction),
    /// A monotone function bounded at anchors: the prevalence is bounded.
    Partial {
        /// Monotone direction.
        direction: DetectionDirection,
        /// Anchors, with strictly increasing covariate values.
        anchors: Vec<DetectionAnchor>,
    },
}

/// A declared false-positive rate, `P(D = 1 | condition absent)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FalsePositiveRate {
    /// The rate, in `[0, 1)`.
    pub rate: f64,
    /// Variance of the rate's estimate; zero when declared exact.
    pub variance: f64,
}

impl FalsePositiveRate {
    /// No false positives, declared exact.
    pub const NONE: Self = Self { rate: 0.0, variance: 0.0 };
}

/// Recorded detections and the covariate that drives detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionData<'a> {
    /// Whether the condition was recorded as present, per unit.
    pub detected: &'a [bool],
    /// Covariate driving detection, per unit.
    pub covariate: &'a [f64],
}

/// A point-identified prevalence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionPoint {
    /// Units.
    pub units: usize,
    /// Recorded share.
    pub observed_rate: f64,
    /// Mean detection probability over the units.
    pub mean_detection: f64,
    /// Corrected prevalence, not truncated.
    pub prevalence: f64,
    /// Total delta-method variance.
    pub variance: f64,
    /// Bernoulli sampling part.
    pub variance_sampling: f64,
    /// Part due to the declared function's and the false-positive rate's uncertainty.
    pub variance_parameters: f64,
    /// The untruncated value left `[0, 1]`.
    pub outside_unit_interval: bool,
}

/// Sharp bounds on a prevalence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionBounds {
    /// Units.
    pub units: usize,
    /// Recorded share.
    pub observed_rate: f64,
    /// Mean of the lowest detection the declaration allows.
    pub mean_detection_lower: f64,
    /// Mean of the highest detection the declaration allows.
    pub mean_detection_upper: f64,
    /// Lower bound, at least zero.
    pub lower: f64,
    /// Upper bound, at most one.
    pub upper: f64,
    /// Sampling standard error of the lower bound as an estimator of its population value.
    pub lower_standard_error: f64,
    /// Sampling standard error of the upper bound.
    pub upper_standard_error: f64,
}

/// A detection-adjusted prevalence: a point or a bounded set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetectionAdjusted {
    /// The function was fully declared.
    Point(DetectionPoint),
    /// The function was partially declared.
    Bounds(DetectionBounds),
}

fn expit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

fn validate_function(function: &DetectionFunction) -> Result<(), MeasurementErrorError> {
    match function {
        DetectionFunction::Logistic {
            intercept,
            slope,
            intercept_variance,
            covariance,
            slope_variance,
        } => {
            let all_finite = [intercept, slope, intercept_variance, covariance, slope_variance]
                .iter()
                .all(|v| v.is_finite());
            if !all_finite
                || *intercept_variance < 0.0
                || *slope_variance < 0.0
                || covariance * covariance > intercept_variance * slope_variance * (1.0 + 1e-9)
            {
                return Err(MeasurementErrorError::InvalidParameter(
                    "logistic detection needs finite parameters and a valid covariance",
                ));
            }
        }
        DetectionFunction::Table { knots, values } => {
            if knots.len() < 2
                || knots.len() != values.len()
                || knots.iter().chain(values).any(|v| !v.is_finite())
                || knots.windows(2).any(|w| w[1] <= w[0])
                || values.iter().any(|v| !(*v > 0.0 && *v <= 1.0))
            {
                return Err(MeasurementErrorError::InvalidParameter(
                    "a detection table needs at least two increasing knots and values in (0, 1]",
                ));
            }
        }
    }
    Ok(())
}

fn table_value(knots: &[f64], values: &[f64], c: f64) -> f64 {
    let last = knots.len() - 1;
    if c <= knots[0] {
        return values[0];
    }
    if c >= knots[last] {
        return values[last];
    }
    let hi = knots.partition_point(|&k| k <= c);
    let lo = hi - 1;
    let t = (c - knots[lo]) / (knots[hi] - knots[lo]);
    values[lo] + t * (values[hi] - values[lo])
}

fn check_data(data: &DetectionData<'_>) -> Result<usize, MeasurementErrorError> {
    let n = data.detected.len();
    if n == 0 || data.covariate.len() != n {
        return Err(MeasurementErrorError::Shape("detections and covariate differ in length"));
    }
    if data.covariate.iter().any(|c| !c.is_finite()) {
        return Err(MeasurementErrorError::NonFinite("detection covariate"));
    }
    Ok(n)
}

fn check_false_positive(fp: &FalsePositiveRate) -> Result<(), MeasurementErrorError> {
    if !((0.0..1.0).contains(&fp.rate) && fp.variance.is_finite() && fp.variance >= 0.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "the false-positive rate must lie in [0, 1) with a non-negative variance",
        ));
    }
    Ok(())
}

fn check_floor(floor: f64) -> Result<(), MeasurementErrorError> {
    if !(floor.is_finite() && floor > 0.0 && floor < 1.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "minimum net detection must lie in (0, 1)",
        ));
    }
    Ok(())
}

/// Bernoulli sampling variance of `A / G` given per-unit detection probabilities.
fn sampling_variance(prevalence: f64, detection: &[f64], fp: f64, g: f64) -> f64 {
    let total: f64 = detection
        .iter()
        .map(|d| {
            let p = (prevalence * d + (1.0 - prevalence) * fp).clamp(0.0, 1.0);
            p * (1.0 - p)
        })
        .sum();
    total / (g * g)
}

/// Prevalence of a condition recorded with covariate-dependent detection.
///
/// A fully declared function gives a point; a monotone envelope gives sharp bounds.
///
/// # Errors
/// A shape or range violation, a mean detection net of false positives at or below
/// `min_net_detection` (the envelope's upper bound, for bounds), an inconsistent envelope, or
/// bounds that imply a prevalence above one refuse.
#[allow(
    clippy::too_many_lines,
    reason = "one flat sequence of declaration checks, then the point and the bounds assembly"
)]
pub fn detection_adjusted_prevalence(
    data: &DetectionData<'_>,
    declaration: &DetectionDeclaration,
    false_positive: &FalsePositiveRate,
    min_net_detection: f64,
) -> Result<DetectionAdjusted, MeasurementErrorError> {
    let n = check_data(data)?;
    check_false_positive(false_positive)?;
    check_floor(min_net_detection)?;
    let nf = n as f64;
    let f = false_positive.rate;
    let events = data.detected.iter().filter(|d| **d).count() as f64;
    let observed = events / nf;
    let a = events - nf * f;
    match declaration {
        DetectionDeclaration::Full(function) => {
            validate_function(function)?;
            let detection: Vec<f64> = match function {
                DetectionFunction::Logistic { intercept, slope, .. } => {
                    data.covariate.iter().map(|c| expit(intercept + slope * c)).collect()
                }
                DetectionFunction::Table { knots, values } => {
                    data.covariate.iter().map(|c| table_value(knots, values, *c)).collect()
                }
            };
            let mean_detection = detection.iter().sum::<f64>() / nf;
            let net = mean_detection - f;
            if net.is_nan() || net <= min_net_detection {
                return Err(MeasurementErrorError::YoudenTooSmall {
                    youden: net,
                    minimum: min_net_detection,
                });
            }
            let g = nf * net;
            let prevalence = a / g;
            let sampling = sampling_variance(prevalence, &detection, f, g);
            let d_fp = -nf * (1.0 - prevalence) / g;
            let mut from_parameters = d_fp * d_fp * false_positive.variance;
            if let DetectionFunction::Logistic {
                intercept_variance,
                covariance,
                slope_variance,
                ..
            } = function
            {
                let (mut g_a, mut g_b) = (0.0, 0.0);
                for (d, c) in detection.iter().zip(data.covariate) {
                    let slope_d = d * (1.0 - d);
                    g_a += slope_d;
                    g_b += slope_d * c;
                }
                // d(prevalence)/d(theta) = -prevalence * sum(d'(c)) / G.
                let (u, v) = (-prevalence * g_a / g, -prevalence * g_b / g);
                from_parameters +=
                    u * u * intercept_variance + 2.0 * u * v * covariance + v * v * slope_variance;
            }
            Ok(DetectionAdjusted::Point(DetectionPoint {
                units: n,
                observed_rate: observed,
                mean_detection,
                prevalence,
                variance: sampling + from_parameters,
                variance_sampling: sampling,
                variance_parameters: from_parameters,
                outside_unit_interval: !(0.0..=1.0).contains(&prevalence),
            }))
        }
        DetectionDeclaration::Partial { direction, anchors } => {
            if anchors.is_empty()
                || anchors.iter().any(|x| {
                    !(x.covariate.is_finite()
                        && x.lower.is_finite()
                        && x.upper.is_finite()
                        && (0.0..=1.0).contains(&x.lower)
                        && (0.0..=1.0).contains(&x.upper)
                        && x.lower <= x.upper)
                })
                || anchors.windows(2).any(|w| w[1].covariate <= w[0].covariate)
            {
                return Err(MeasurementErrorError::InvalidParameter(
                    "an envelope needs increasing anchors with 0 <= lower <= upper <= 1",
                ));
            }
            let mut lower_d = Vec::with_capacity(n);
            let mut upper_d = Vec::with_capacity(n);
            for &c in data.covariate {
                // Anchors that constrain a unit from below (or above) under the direction.
                let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
                for x in anchors {
                    let at_or_before = x.covariate <= c;
                    let at_or_after = x.covariate >= c;
                    match direction {
                        DetectionDirection::Nondecreasing => {
                            if at_or_before {
                                lo = lo.max(x.lower);
                            }
                            if at_or_after {
                                hi = hi.min(x.upper);
                            }
                        }
                        DetectionDirection::Nonincreasing => {
                            if at_or_after {
                                lo = lo.max(x.lower);
                            }
                            if at_or_before {
                                hi = hi.min(x.upper);
                            }
                        }
                    }
                }
                if lo > hi {
                    return Err(MeasurementErrorError::InvalidParameter(
                        "the declared anchors contradict the declared monotone direction",
                    ));
                }
                lower_d.push(lo);
                upper_d.push(hi);
            }
            let mean_lower = lower_d.iter().sum::<f64>() / nf;
            let mean_upper = upper_d.iter().sum::<f64>() / nf;
            let net_upper = mean_upper - f;
            if net_upper.is_nan() || net_upper <= min_net_detection {
                return Err(MeasurementErrorError::YoudenTooSmall {
                    youden: net_upper,
                    minimum: min_net_detection,
                });
            }
            let g_high = nf * net_upper;
            let g_low = nf * (mean_lower - f);
            let lower = (a / g_high).max(0.0);
            let upper = if g_low > 0.0 { (a / g_low).min(1.0) } else { 1.0 };
            let upper = upper.max(0.0);
            if lower > 1.0 {
                return Err(MeasurementErrorError::InvalidParameter(
                    "the declared detection implies a prevalence above one; it contradicts the recorded share",
                ));
            }
            let se_of = |value: f64, detection: &[f64], g: f64| {
                if g > 0.0 { sampling_variance(value, detection, f, g).sqrt() } else { 0.0 }
            };
            Ok(DetectionAdjusted::Bounds(DetectionBounds {
                units: n,
                observed_rate: observed,
                mean_detection_lower: mean_lower,
                mean_detection_upper: mean_upper,
                lower,
                upper,
                lower_standard_error: se_of(lower, &upper_d, g_high),
                upper_standard_error: se_of(upper, &lower_d, g_low),
            }))
        }
    }
}
