//! Calibration and sharpness diagnostics of predictive laws.
//!
//! Everything here is a *diagnostic*: it describes how a predictive law behaved
//! on the assessed observations. None of it is a calibration guarantee for the
//! law and none of it has a measured error rate (`calibration = "unmeasured"`):
//! no coordinate of this route has measured whole-method calibration.
//!
//! * **PIT.** The probability integral transform of the outcome under its law.
//!   For a discrete or ensemble law the transform is an interval `[F(y-), F(y)]`
//!   and the histogram spreads the case's unit mass uniformly over it (the
//!   non-randomized PIT of Czado, Gneiting and Held 2009), so no randomness
//!   enters. A calibrated law gives a uniform histogram.
//! * **Reliability.** Binned forecast probability against observed frequency of
//!   a binary event.
//! * **Coverage.** The fraction of outcomes inside the central prediction
//!   interval at each declared level, with the mean interval width (sharpness).
//! * **Drift.** A two-sample comparison of PIT values between two windows: the
//!   Kolmogorov-Smirnov distance and the population stability index. A
//!   separate route; no significance level is attached.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{norm_cdf, norm_inv};
use antecedent_stats::{QuantileRule, quantile_sorted};

use super::law::{ForecastLaw, LawKind, Outcome};
use super::refusal::{PredictiveAssessmentError, detail, invalid, unsupported};

/// The label every diagnostic carries.
pub const DIAGNOSTIC_LABEL: &str = "diagnostic";
/// The calibration status of every diagnostic of this route: not measured.
pub const DIAGNOSTIC_CALIBRATION: &str = "unmeasured";
/// Most PIT or reliability bins.
pub const MAX_BINS: usize = 1000;
/// Most interval levels.
pub const MAX_INTERVAL_LEVELS: usize = 16;
/// Pseudo-count added to every bin proportion of the population stability index.
pub const PSI_PSEUDOCOUNT: f64 = 0.5;

/// PIT interval `[F(y-), F(y)]` of one case, or `None` when the law is not a
/// law of an ordered outcome.
fn pit_interval(law: &ForecastLaw, outcome: &Outcome) -> Option<(f64, f64)> {
    match (law, outcome) {
        (ForecastLaw::Gaussian { mean, sd }, Outcome::Real(y)) => {
            let u = norm_cdf((y - mean) / sd);
            Some((u, u))
        }
        (ForecastLaw::Bernoulli { probability }, Outcome::Category(c)) => {
            Some(if *c == 0 { (0.0, 1.0 - probability) } else { (1.0 - probability, 1.0) })
        }
        (ForecastLaw::FiniteReal { support, probabilities }, Outcome::Real(y)) => {
            let mut below = 0.0;
            let mut at = 0.0;
            for (x, p) in support.iter().zip(probabilities) {
                if x < y {
                    below += p;
                }
                if x <= y {
                    at += p;
                }
            }
            Some((below.min(1.0), at.min(1.0)))
        }
        (ForecastLaw::EmpiricalSample { draws }, Outcome::Real(y)) => {
            let m = draws.len() as f64;
            let below = draws.iter().filter(|x| *x < y).count() as f64 / m;
            let at = draws.iter().filter(|x| *x <= y).count() as f64 / m;
            Some((below, at))
        }
        (ForecastLaw::PointEvaluated { cdf_below, cdf_at, .. }, Outcome::Real(_)) => {
            Some((*cdf_below, *cdf_at))
        }
        _ => None,
    }
}

/// PIT histogram, uniformity descriptors and the mid-PIT values of a predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct PitDiagnostic {
    /// Number of bins.
    pub bins: usize,
    /// Mass in each bin (cases; fractional for discrete laws), summing to the case count.
    pub counts: Vec<f64>,
    /// Largest `|count / n - 1 / bins|`.
    pub max_bin_deviation: f64,
    /// `sum (count - n / bins)^2 / (n / bins)`; descriptive, with no reference
    /// distribution claimed under dependence or for discrete laws.
    pub chi_square: f64,
    /// Kolmogorov-Smirnov distance of the mid-PIT values from `U(0, 1)`. Exact
    /// for continuous laws; for ensembles and discrete laws the mid-PIT is
    /// itself discrete and the distance has a floor of order `1 / m`.
    pub ks_distance: f64,
    /// The mid-PIT `(F(y-) + F(y)) / 2` of every case, in canonical case order.
    pub mid_pit: Vec<f64>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the scaled position is clamped below by zero and above by the bin count"
)]
fn bin_of(position: f64, bins: usize) -> usize {
    let scaled = (position * bins as f64).floor();
    if scaled <= 0.0 {
        0
    } else {
        let index = scaled as usize;
        index.min(bins - 1)
    }
}

/// PIT diagnostic over all cases, or `None` when some case's law has no ordered
/// outcome (categorical and joint laws).
pub(crate) fn pit_diagnostic(
    laws: &[ForecastLaw],
    outcomes: &[Outcome],
    bins: usize,
) -> Option<PitDiagnostic> {
    let mut counts = vec![0.0; bins];
    let mut mid = Vec::with_capacity(laws.len());
    for (law, outcome) in laws.iter().zip(outcomes) {
        let (lo, hi) = pit_interval(law, outcome)?;
        mid.push(0.5 * (lo + hi));
        if hi > lo {
            let width = hi - lo;
            for (b, count) in counts.iter_mut().enumerate() {
                let left = b as f64 / bins as f64;
                let right = (b + 1) as f64 / bins as f64;
                let overlap = (hi.min(right) - lo.max(left)).max(0.0);
                *count += overlap / width;
            }
        } else {
            counts[bin_of(lo, bins)] += 1.0;
        }
    }
    let n = laws.len() as f64;
    let expected = n / bins as f64;
    let max_bin_deviation =
        counts.iter().map(|c| (c / n - 1.0 / bins as f64).abs()).fold(0.0, f64::max);
    let chi_square = counts.iter().map(|c| (c - expected).powi(2) / expected).sum();
    let mut sorted = mid.clone();
    sorted.sort_by(f64::total_cmp);
    let mut ks_distance: f64 = 0.0;
    for (i, u) in sorted.iter().enumerate() {
        let upper = (i + 1) as f64 / n - u;
        let lower = u - i as f64 / n;
        ks_distance = ks_distance.max(upper).max(lower);
    }
    Some(PitDiagnostic { bins, counts, max_bin_deviation, chi_square, ks_distance, mid_pit: mid })
}

/// One bin of a reliability table.
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityBin {
    /// Lower edge of the forecast-probability bin.
    pub lower: f64,
    /// Upper edge.
    pub upper: f64,
    /// Cases in the bin.
    pub count: u64,
    /// Mean forecast probability in the bin (`0` for an empty bin).
    pub mean_forecast: f64,
    /// Observed frequency of the event in the bin (`0` for an empty bin).
    pub observed_frequency: f64,
}

/// Binned reliability of binary event forecasts.
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityDiagnostic {
    /// The bins, ascending.
    pub bins: Vec<ReliabilityBin>,
    /// `sum_b (n_b / n) |observed_b - forecast_b|`, a descriptive summary of the
    /// table; binning-dependent and not a calibration test.
    pub weighted_gap: f64,
}

/// Reliability table, or `None` unless every law is Bernoulli.
pub(crate) fn reliability_diagnostic(
    laws: &[ForecastLaw],
    outcomes: &[Outcome],
    bins: usize,
) -> Option<ReliabilityDiagnostic> {
    let mut count = vec![0_u64; bins];
    let mut forecast = vec![0.0; bins];
    let mut hits = vec![0.0; bins];
    for (law, outcome) in laws.iter().zip(outcomes) {
        let (ForecastLaw::Bernoulli { probability }, Outcome::Category(c)) = (law, outcome) else {
            return None;
        };
        let b = bin_of(*probability, bins);
        count[b] += 1;
        forecast[b] += probability;
        hits[b] += if *c == 0 { 0.0 } else { 1.0 };
    }
    let n = laws.len() as f64;
    let mut table = Vec::with_capacity(bins);
    let mut weighted_gap = 0.0;
    for (b, ((&cases, &forecast_sum), &hit_sum)) in
        count.iter().zip(&forecast).zip(&hits).enumerate()
    {
        let c = cases as f64;
        let (mean_forecast, observed_frequency) =
            if cases == 0 { (0.0, 0.0) } else { (forecast_sum / c, hit_sum / c) };
        weighted_gap += c / n * (observed_frequency - mean_forecast).abs();
        table.push(ReliabilityBin {
            lower: b as f64 / bins as f64,
            upper: (b + 1) as f64 / bins as f64,
            count: cases,
            mean_forecast,
            observed_frequency,
        });
    }
    Some(ReliabilityDiagnostic { bins: table, weighted_gap })
}

/// Empirical coverage of the central prediction interval at one level.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageDiagnostic {
    /// The nominal central level, in `(0, 1)`.
    pub level: f64,
    /// Fraction of outcomes inside the interval.
    pub empirical_coverage: f64,
    /// Mean interval width (sharpness at this level).
    pub mean_width: f64,
}

/// Central interval `[lower, upper]` of a law at `level`, or `None` for a law
/// with no interval (categorical, joint and provider-evaluated laws).
fn central_interval(law: &ForecastLaw, level: f64) -> Option<(f64, f64)> {
    let tail = (1.0 - level) / 2.0;
    match law {
        ForecastLaw::Gaussian { mean, sd } => {
            let z = norm_inv(1.0 - tail);
            Some((mean - z * sd, mean + z * sd))
        }
        ForecastLaw::Bernoulli { probability } => {
            let low = if (1.0 - probability) >= tail { 0.0 } else { 1.0 };
            let high = if (1.0 - probability) >= 1.0 - tail { 0.0 } else { 1.0 };
            Some((low, high))
        }
        ForecastLaw::FiniteReal { support, probabilities } => {
            let quantile = |p: f64| {
                let mut cumulative = 0.0;
                for (x, mass) in support.iter().zip(probabilities) {
                    cumulative += mass;
                    if cumulative >= p - 1e-12 {
                        return *x;
                    }
                }
                support[support.len() - 1]
            };
            Some((quantile(tail), quantile(1.0 - tail)))
        }
        ForecastLaw::EmpiricalSample { draws } => {
            let mut sorted = draws.clone();
            sorted.sort_by(f64::total_cmp);
            Some((
                quantile_sorted(&sorted, tail, QuantileRule::ExchangeableRank),
                quantile_sorted(&sorted, 1.0 - tail, QuantileRule::ExchangeableRank),
            ))
        }
        _ => None,
    }
}

/// Coverage at each level, or `None` when some case's law has no interval.
pub(crate) fn coverage_diagnostics(
    laws: &[ForecastLaw],
    outcomes: &[Outcome],
    levels: &[f64],
) -> Option<Vec<CoverageDiagnostic>> {
    let n = laws.len() as f64;
    let mut out = Vec::with_capacity(levels.len());
    for &level in levels {
        let mut covered = 0.0;
        let mut width = 0.0;
        for (law, outcome) in laws.iter().zip(outcomes) {
            let (lower, upper) = central_interval(law, level)?;
            let y = match outcome {
                Outcome::Real(y) => *y,
                Outcome::Category(c) => f64::from(*c),
                Outcome::Vector(_) => return None,
            };
            if y >= lower && y <= upper {
                covered += 1.0;
            }
            width += upper - lower;
        }
        out.push(CoverageDiagnostic {
            level,
            empirical_coverage: covered / n,
            mean_width: width / n,
        });
    }
    Some(out)
}

/// Mean predictive standard deviation (sharpness), or `None` when some case's
/// law has no variance here.
pub(crate) fn mean_predictive_sd(laws: &[ForecastLaw]) -> Option<f64> {
    let mut total = 0.0;
    for law in laws {
        total += match law {
            ForecastLaw::Gaussian { sd, .. } => *sd,
            ForecastLaw::Bernoulli { probability } => (probability * (1.0 - probability)).sqrt(),
            ForecastLaw::FiniteReal { support, probabilities } => {
                let mean: f64 = support.iter().zip(probabilities).map(|(x, p)| x * p).sum();
                let variance: f64 =
                    support.iter().zip(probabilities).map(|(x, p)| p * (x - mean).powi(2)).sum();
                variance.sqrt()
            }
            ForecastLaw::EmpiricalSample { draws } => {
                let m = draws.len() as f64;
                let mean = draws.iter().sum::<f64>() / m;
                (draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / m).sqrt()
            }
            _ => return None,
        };
    }
    Some(total / laws.len() as f64)
}

/// The calibration and sharpness diagnostics of one predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationDiagnostics {
    /// Always [`DIAGNOSTIC_LABEL`].
    pub label: &'static str,
    /// Always [`DIAGNOSTIC_CALIBRATION`].
    pub calibration: &'static str,
    /// PIT histogram, when every law is a law of an ordered outcome.
    pub pit: Option<PitDiagnostic>,
    /// Reliability table, when every law is Bernoulli.
    pub reliability: Option<ReliabilityDiagnostic>,
    /// Coverage at each declared level, when every law has an interval.
    pub coverage: Option<Vec<CoverageDiagnostic>>,
    /// Mean predictive standard deviation, when every law has one.
    pub mean_predictive_sd: Option<f64>,
    /// Drift of the PIT between the first and second half of the cases in time
    /// order; present only for serial or rolling-origin assessments, where case
    /// order is a time order. A separate labelled route: see [`distribution_drift`].
    pub drift: Option<DistributionDrift>,
}

/// Validate the diagnostic configuration.
pub(crate) fn check_configuration(
    pit_bins: usize,
    reliability_bins: usize,
    levels: &[f64],
) -> Result<(), PredictiveAssessmentError> {
    for (what, bins) in [("PIT", pit_bins), ("reliability", reliability_bins)] {
        if !(2..=MAX_BINS).contains(&bins) {
            return Err(invalid(
                detail::DIAGNOSTIC_NOT_DEFINED,
                format!("{what} bins must be between 2 and {MAX_BINS}; got {bins}"),
            ));
        }
    }
    if levels.len() > MAX_INTERVAL_LEVELS {
        return Err(unsupported(
            detail::BOUNDS_EXCEEDED,
            format!("{} interval levels declared; at most {MAX_INTERVAL_LEVELS}", levels.len()),
        ));
    }
    if levels.iter().any(|level| !level.is_finite() || *level <= 0.0 || *level >= 1.0) {
        return Err(invalid(
            detail::DIAGNOSTIC_NOT_DEFINED,
            "prediction interval levels must lie strictly inside (0, 1)",
        ));
    }
    Ok(())
}

/// The calibration diagnostics of one predictor's laws.
pub(crate) fn diagnose(
    laws: &[ForecastLaw],
    outcomes: &[Outcome],
    pit_bins: usize,
    reliability_bins: usize,
    levels: &[f64],
) -> CalibrationDiagnostics {
    CalibrationDiagnostics {
        label: DIAGNOSTIC_LABEL,
        calibration: DIAGNOSTIC_CALIBRATION,
        pit: pit_diagnostic(laws, outcomes, pit_bins),
        reliability: reliability_diagnostic(laws, outcomes, reliability_bins),
        coverage: if levels.is_empty() {
            None
        } else {
            coverage_diagnostics(laws, outcomes, levels)
        },
        mean_predictive_sd: mean_predictive_sd(laws),
        drift: None,
    }
}

/// Which families of law a diagnostic is available for, for messages.
#[must_use]
pub fn pit_available_for(kind: LawKind) -> bool {
    matches!(
        kind,
        LawKind::Gaussian
            | LawKind::Bernoulli
            | LawKind::FiniteReal
            | LawKind::EmpiricalSample
            | LawKind::PointEvaluated
    )
}

/// A two-sample comparison of PIT values between two windows.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributionDrift {
    /// Always [`DIAGNOSTIC_LABEL`].
    pub label: &'static str,
    /// Always [`DIAGNOSTIC_CALIBRATION`]: no significance level is attached.
    pub calibration: &'static str,
    /// Name of the route: `distribution_drift`.
    pub route: &'static str,
    /// Cases in the reference window.
    pub reference_cases: usize,
    /// Cases in the comparison window.
    pub comparison_cases: usize,
    /// Two-sample Kolmogorov-Smirnov distance of the PIT values.
    pub ks_distance: f64,
    /// Population stability index of the PIT histograms (equal-width bins,
    /// [`PSI_PSEUDOCOUNT`] added to every bin).
    pub population_stability_index: f64,
    /// Number of bins of the index.
    pub bins: usize,
}

/// Compare the PIT values of a comparison window against a reference window.
///
/// `reference` and `comparison` are mid-PIT values in `[0, 1]`. This is a
/// separate, labelled diagnostic route: it says whether the distribution of the
/// law's probabilistic errors moved, not by how much a decision should change,
/// and it carries no significance level.
///
/// # Errors
/// An empty window, a value outside `[0, 1]`, or a bin count outside `2..=1000`.
pub fn distribution_drift(
    reference: &[f64],
    comparison: &[f64],
    bins: usize,
) -> Result<DistributionDrift, PredictiveAssessmentError> {
    if reference.is_empty() || comparison.is_empty() {
        return Err(invalid(
            detail::DIAGNOSTIC_NOT_DEFINED,
            "a drift comparison needs observations in both windows",
        ));
    }
    if !(2..=MAX_BINS).contains(&bins) {
        return Err(invalid(
            detail::DIAGNOSTIC_NOT_DEFINED,
            format!("drift bins must be between 2 and {MAX_BINS}; got {bins}"),
        ));
    }
    if reference.iter().chain(comparison).any(|u| !u.is_finite() || *u < 0.0 || *u > 1.0) {
        return Err(invalid(
            detail::DIAGNOSTIC_NOT_DEFINED,
            "PIT values must be finite and inside [0, 1]",
        ));
    }
    let mut a = reference.to_vec();
    let mut b = comparison.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (na, nb) = (a.len() as f64, b.len() as f64);
    // Two-sample KS: walk the merged order statistics, comparing the empirical
    // distribution functions after all ties at a value are consumed.
    let (mut i, mut j) = (0_usize, 0_usize);
    let mut ks: f64 = 0.0;
    while i < a.len() || j < b.len() {
        let next = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => x.min(*y),
            (Some(x), None) => *x,
            (None, Some(y)) => *y,
            (None, None) => break,
        };
        while i < a.len() && a[i] <= next {
            i += 1;
        }
        while j < b.len() && b[j] <= next {
            j += 1;
        }
        ks = ks.max((i as f64 / na - j as f64 / nb).abs());
    }
    let mut counts_a = vec![0.0; bins];
    let mut counts_b = vec![0.0; bins];
    for u in &a {
        counts_a[bin_of(*u, bins)] += 1.0;
    }
    for u in &b {
        counts_b[bin_of(*u, bins)] += 1.0;
    }
    let pad = PSI_PSEUDOCOUNT * bins as f64;
    let mut psi = 0.0;
    for (ca, cb) in counts_a.iter().zip(&counts_b) {
        let pa = (ca + PSI_PSEUDOCOUNT) / (na + pad);
        let pb = (cb + PSI_PSEUDOCOUNT) / (nb + pad);
        psi += (pb - pa) * (pb / pa).ln();
    }
    Ok(DistributionDrift {
        label: DIAGNOSTIC_LABEL,
        calibration: DIAGNOSTIC_CALIBRATION,
        route: "distribution_drift",
        reference_cases: a.len(),
        comparison_cases: b.len(),
        ks_distance: ks,
        population_stability_index: psi,
        bins,
    })
}
