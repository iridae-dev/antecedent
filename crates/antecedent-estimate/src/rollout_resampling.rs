//! Whole-procedure resampling shared by the rollout designs: heterogeneity-robust and
//! covariate-adjusted difference in differences, the augmented panel, multi-unit synthetic
//! control and the temporal local ratio.
//!
//! Every replicate redraws the sampling units the declared design allows (whole clusters of a
//! clustered panel, whole subjects of independent trajectories, contiguous blocks of one
//! series), refits the entire procedure on the redrawn sample, and returns the full vector of
//! target quantities. The summary is the interquartile bootstrap standard error of each
//! quantity, a pointwise normal interval and a sup-t simultaneous band over the vector. No
//! repeated-sampling coverage of any interval built here has been measured.
//!
//! Randomness is keyed by the semantic replicate index ([`RngKey`]), so the replicate vectors
//! and every summary are the same at any thread count.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalRng, ChainId, ExecutionContext, FoldId, ReductionPolicy, ReplicateId, RngKey,
    SemanticOperation,
};

use crate::EstimationError;

/// Central coverage of every pointwise interval and simultaneous band built here.
pub const ROLLOUT_LEVEL: f64 = 0.95;

/// Normal quantiles of the interquartile range: `Φ⁻¹(0.75) − Φ⁻¹(0.25)`.
const NORMAL_IQR: f64 = 1.348_979_500_392_163_5;

/// Fewest finite replicates a component needs before its bootstrap standard error is reported.
const MIN_FINITE_REPLICATES: usize = 20;

/// How one replicate reweights the independent sampling units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnitWeighting {
    /// Nonparametric bootstrap: each unit's weight is its multinomial draw count.
    Resample,
    /// Weighted (multiplier) bootstrap: each unit's weight is an independent mean-one,
    /// variance-one exponential multiplier, and the procedure is refit with those weights.
    Multiplier,
}

impl UnitWeighting {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Resample => "whole_unit_bootstrap",
            Self::Multiplier => "whole_unit_exponential_multiplier",
        }
    }
}

/// Uncertainty of one target quantity from a whole-procedure resample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicateQuantity {
    /// Interquartile bootstrap standard error; `None` with too few finite replicates.
    pub standard_error: Option<f64>,
    /// Pointwise normal 95% interval `estimate ± z·se`.
    pub interval: Option<[f64; 2]>,
    /// Simultaneous 95% band `estimate ± c·se`, with `c` the sup-t critical value.
    pub band: Option<[f64; 2]>,
}

/// Summary of a whole-procedure resample over a vector of target quantities.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicateSummary {
    /// One summary per target quantity, in the order of the point vector.
    pub quantities: Vec<ReplicateQuantity>,
    /// Sup-t critical value of the simultaneous band; `None` when no band was formed.
    pub critical_value: Option<f64>,
    /// Requested replicates.
    pub requested: u32,
    /// Replicates whose refit produced a full finite vector.
    pub replicates_ok: u32,
    /// Replicates whose refit failed or produced a non-finite quantity.
    pub replicates_failed: u32,
}

/// What one reported design effect summarizes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DesignEffectKind {
    /// The scalar summary of the declared aggregation.
    Overall,
    /// One adoption cohort at one calendar period.
    GroupTime,
    /// Cohorts aggregated at one event time (period minus adoption period).
    EventTime,
    /// One adoption cohort averaged over its post-adoption periods.
    Group,
    /// Adopted cohorts aggregated at one calendar period.
    Calendar,
    /// A pre-adoption event-time contrast, reported as a falsification diagnostic.
    PreTrend,
    /// One post-intervention period of a synthetic panel.
    Period,
    /// One treated unit of a multi-unit synthetic panel.
    Unit,
}

impl DesignEffectKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Overall => "overall",
            Self::GroupTime => "group_time",
            Self::EventTime => "event_time",
            Self::Group => "group",
            Self::Calendar => "calendar",
            Self::PreTrend => "pre_trend",
            Self::Period => "period",
            Self::Unit => "unit",
        }
    }
}

/// One reported effect of a rollout design, with the uncertainty its procedure produced.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignEffect {
    /// What the effect summarizes.
    pub kind: DesignEffectKind,
    /// Adoption cohort, when the effect belongs to one.
    pub group: Option<i64>,
    /// Calendar period, when the effect belongs to one.
    pub period: Option<i64>,
    /// Event time (period minus adoption period), when the effect belongs to one.
    pub event_time: Option<i64>,
    /// Treated unit label, when the effect belongs to one unit.
    pub unit: Option<String>,
    /// Point estimate.
    pub estimate: f64,
    /// Standard error of the reported uncertainty construction, when one was formed.
    pub standard_error: Option<f64>,
    /// Pointwise 95% interval, when one was formed.
    pub interval: Option<[f64; 2]>,
    /// Simultaneous 95% band over the effects of the same kind, when one was formed.
    pub band: Option<[f64; 2]>,
    /// Treated units (or rows) behind the estimate.
    pub treated: usize,
    /// Comparison units (or rows) behind the estimate.
    pub comparison: usize,
}

impl DesignEffect {
    /// A point effect of `kind` with no uncertainty attached yet.
    #[must_use]
    pub const fn point(kind: DesignEffectKind, estimate: f64) -> Self {
        Self {
            kind,
            group: None,
            period: None,
            event_time: None,
            unit: None,
            estimate,
            standard_error: None,
            interval: None,
            band: None,
            treated: 0,
            comparison: 0,
        }
    }

    /// Attach a resampled summary.
    pub fn attach(&mut self, quantity: &ReplicateQuantity) {
        self.standard_error = quantity.standard_error;
        self.interval = quantity.interval;
        self.band = quantity.band;
    }
}

/// The weight of each of `units` sampling units in one replicate.
#[must_use]
pub fn unit_weights(weighting: UnitWeighting, units: usize, rng: &mut CausalRng) -> Vec<f64> {
    let mut weights = vec![0.0; units];
    match weighting {
        UnitWeighting::Resample => {
            for _ in 0..units {
                weights[uniform_index(rng, units)] += 1.0;
            }
        }
        UnitWeighting::Multiplier => {
            for weight in &mut weights {
                // 1 − U lies in (0, 1], so the logarithm is finite.
                *weight = -(1.0 - rng.next_f64()).ln();
            }
        }
    }
    weights
}

/// A uniform index in `0..n` (`n > 0`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the product of a [0, 1) draw and n is below n and non-negative"
)]
#[must_use]
pub fn uniform_index(rng: &mut CausalRng, n: usize) -> usize {
    ((rng.next_f64() * n as f64) as usize).min(n.saturating_sub(1))
}

/// Start indices of a moving-block resample of a series of `length` points in blocks of
/// `block` consecutive points; the concatenated blocks are truncated to `length`.
#[must_use]
pub fn moving_block_starts(length: usize, block: usize, rng: &mut CausalRng) -> Vec<usize> {
    let block = block.clamp(1, length.max(1));
    let starts = length + 1 - block;
    let count = length.div_ceil(block);
    (0..count).map(|_| uniform_index(rng, starts)).collect()
}

/// The rounded `n^{1/3}` block length of a series of `n` points, at least one.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the cube root of a usize is a small non-negative number"
)]
#[must_use]
pub fn cube_root_block_length(n: usize) -> usize {
    ((n as f64).cbrt().round() as usize).max(1)
}

/// The replicate stream of `operation` at replicate `index`.
#[must_use]
pub fn replicate_stream(operation: u64, index: usize, seed: u64) -> CausalRng {
    RngKey::new(
        SemanticOperation::new(operation),
        FoldId::new(0),
        ChainId::new(0),
        ReplicateId::new(index as u64),
        ReductionPolicy::Strict,
    )
    .open(seed)
}

/// Refit `procedure` on `replicates` keyed draws and summarize around `point`.
///
/// `procedure` receives the replicate's random stream and returns the full quantity vector of
/// the refit, or `None` when the redrawn sample cannot support the procedure (an empty
/// comparison group, a singular nuisance fit). The result buffer of `replicates × point.len()`
/// values is leased from the admission pool before it is allocated, and the study's
/// cancellation is checked before the lease and before every replicate.
///
/// # Errors
///
/// The study is cancelled, the admission pool cannot fund the replicate buffer, or `replicates`
/// is zero.
pub fn resample_procedure<F>(
    ctx: &ExecutionContext,
    operation: u64,
    replicates: u32,
    point: &[f64],
    procedure: F,
) -> Result<ReplicateSummary, EstimationError>
where
    F: Fn(&mut CausalRng) -> Option<Vec<f64>> + Sync,
{
    let draws = resample_draws(ctx, operation, replicates, point.len(), procedure)?;
    Ok(summarize_replicates(point, &draws, replicates))
}

/// The raw replicate vectors of [`resample_procedure`], each of length `width`, for a caller
/// that summarizes blocks of the vector separately. A component the refit could not form is
/// `NaN`; a replicate whose refit failed outright is `None`.
///
/// # Errors
///
/// As [`resample_procedure`].
pub fn resample_draws<F>(
    ctx: &ExecutionContext,
    operation: u64,
    replicates: u32,
    width: usize,
    procedure: F,
) -> Result<Vec<Option<Vec<f64>>>, EstimationError>
where
    F: Fn(&mut CausalRng) -> Option<Vec<f64>> + Sync,
{
    if replicates == 0 {
        return Err(EstimationError::refused(
            "option_not_applicable",
            "a whole-procedure resample needs at least one replicate; next attempt: request a \
             positive replicate count",
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(EstimationError::unsupported(
            "execution cancelled before the resampling buffers were admitted",
        ));
    }
    let _lease = crate::util::admit_array::<f64>(ctx, replicates as usize, width.max(1))?;
    let seed = ctx.rng.master_seed();
    ctx.map_indexed(replicates as usize, |index, inner| -> Result<_, EstimationError> {
        if inner.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported(
                "execution cancelled during whole-procedure resampling",
            ));
        }
        let mut rng = replicate_stream(operation, index, seed);
        Ok(procedure(&mut rng).filter(|values| values.len() == width))
    })
    .map_err(Into::into)
}

/// The components `range` of every replicate, for summarizing one block of a vector.
#[must_use]
pub fn block_draws(
    draws: &[Option<Vec<f64>>],
    range: std::ops::Range<usize>,
) -> Vec<Option<Vec<f64>>> {
    draws.iter().map(|draw| draw.as_ref().map(|values| values[range.clone()].to_vec())).collect()
}

/// Summarize replicate vectors around `point`.
///
/// A `None` replicate counts as failed. A non-finite component is left out of that component's
/// standard error, and a replicate is counted as successful, and enters the sup-t band, only
/// when every component is finite.
/// More than half of the requested replicates failing withholds a component's
/// standard error, or the joint band when the incomplete vectors exceed that fraction.
#[must_use]
pub fn summarize_replicates(
    point: &[f64],
    draws: &[Option<Vec<f64>>],
    requested: u32,
) -> ReplicateSummary {
    let width = point.len();
    let present: Vec<&Vec<f64>> =
        draws.iter().flatten().filter(|draw| draw.len() == width).collect();
    let complete: Vec<&Vec<f64>> =
        present.iter().copied().filter(|draw| draw.iter().all(|value| value.is_finite())).collect();
    let replicates_ok = u32::try_from(complete.len()).unwrap_or(u32::MAX);
    let replicates_failed = requested.saturating_sub(replicates_ok);
    let policy = crate::util::ReplicatePolicy {
        min_ok: u32::try_from(MIN_FINITE_REPLICATES).unwrap_or(u32::MAX),
        ..crate::util::ReplicatePolicy::BOOTSTRAP
    };
    let z = antecedent_stats::NORMAL_Q975;
    let standard_errors: Vec<Option<f64>> = (0..width)
        .map(|k| {
            let mut column: Vec<f64> =
                present.iter().map(|draw| draw[k]).filter(|value| value.is_finite()).collect();
            let successes = u32::try_from(column.len()).unwrap_or(u32::MAX);
            if policy.decide(requested, successes, requested.saturating_sub(successes)).is_err()
                || !point[k].is_finite()
            {
                return None;
            }
            column.sort_by(f64::total_cmp);
            let se = (quantile_sorted(&column, 0.75) - quantile_sorted(&column, 0.25)) / NORMAL_IQR;
            (se.is_finite() && se > 0.0).then_some(se)
        })
        .collect();
    let mut maxima: Vec<f64> = complete
        .iter()
        .filter_map(|draw| {
            let mut largest: Option<f64> = None;
            for k in 0..width {
                if let Some(se) = standard_errors[k] {
                    let t = (draw[k] - point[k]).abs() / se;
                    largest = Some(largest.map_or(t, |current| current.max(t)));
                }
            }
            largest
        })
        .collect();
    maxima.sort_by(f64::total_cmp);
    let critical_value = (maxima.len() >= MIN_FINITE_REPLICATES
        && policy.decide(requested, replicates_ok, replicates_failed).is_ok())
    .then(|| crate::util::monte_carlo_critical(&maxima, ROLLOUT_LEVEL))
    .filter(|value| value.is_finite())
    // The pointwise interval is `estimate ± z·se` with the normal quantile, while the
    // replicate quantile of max|t| measures the band on the replicates' own scale. With
    // light-tailed or bounded replicate distributions that quantile can fall below `z`,
    // which would make the simultaneous band narrower than a pointwise interval it must
    // contain. The band's critical value is therefore never below the pointwise one.
    .map(|value| value.max(antecedent_stats::NORMAL_Q975));
    let quantities = (0..width)
        .map(|k| {
            let se = standard_errors[k];
            ReplicateQuantity {
                standard_error: se,
                interval: se.map(|se| [point[k] - z * se, point[k] + z * se]),
                band: se
                    .zip(critical_value)
                    .map(|(se, critical)| [point[k] - critical * se, point[k] + critical * se]),
            }
        })
        .collect();
    ReplicateSummary { quantities, critical_value, requested, replicates_ok, replicates_failed }
}

/// Linear-interpolation quantile of an ascending-sorted, non-empty slice.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the position lies in [0, len - 1]"
)]
#[must_use]
pub fn quantile_sorted(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_failed_replicates_withhold_uncertainty_and_keep_counts() {
        let mut draws: Vec<Option<Vec<f64>>> = (0..20).map(|i| Some(vec![f64::from(i)])).collect();
        draws.resize(100, None);
        let summary = summarize_replicates(&[0.0], &draws, 100);
        assert_eq!((summary.replicates_ok, summary.replicates_failed), (20, 80));
        assert!(summary.quantities[0].standard_error.is_none());
        assert!(summary.quantities[0].band.is_none() && summary.critical_value.is_none());

        // Exactly half failing remains admissible under the common rule.
        let accepted = summarize_replicates(&[0.0], &draws[..40], 40);
        assert!(accepted.quantities[0].standard_error.is_some());
        assert!(accepted.quantities[0].band.is_some());

        // A usable component does not license a joint band whose other
        // component failed: distinguish component and vector success counts.
        let partial: Vec<Option<Vec<f64>>> =
            (0..60).map(|i| Some(vec![f64::from(i), f64::NAN])).collect();
        let summary = summarize_replicates(&[0.0, 0.0], &partial, 100);
        assert!(summary.quantities[0].standard_error.is_some());
        assert!(summary.quantities[1].standard_error.is_none());
        assert!(summary.critical_value.is_none());
        assert_eq!((summary.replicates_ok, summary.replicates_failed), (0, 100));
    }

    #[test]
    fn resample_weights_sum_to_the_unit_count_and_multipliers_are_positive() {
        let mut rng = replicate_stream(7, 3, 11);
        let resampled = unit_weights(UnitWeighting::Resample, 25, &mut rng);
        assert!((resampled.iter().sum::<f64>() - 25.0).abs() < 1e-12);
        let multiplied = unit_weights(UnitWeighting::Multiplier, 25, &mut rng);
        assert!(multiplied.iter().all(|weight| *weight > 0.0 && weight.is_finite()));
    }

    #[test]
    fn summaries_are_thread_count_invariant_and_bands_cover_pointwise_intervals() {
        let point = [0.0, 1.0];
        let procedure = |rng: &mut CausalRng| {
            let a = rng.next_f64() - 0.5;
            let b = rng.next_f64() - 0.5;
            Some(vec![a, 1.0 + 2.0 * b])
        };
        let serial = ExecutionContext::for_tests(5);
        let parallel = ExecutionContext::production(5, 4);
        let one = resample_procedure(&serial, 1, 400, &point, procedure).unwrap();
        let four = resample_procedure(&parallel, 1, 400, &point, procedure).unwrap();
        assert_eq!(one, four);
        assert_eq!(one.replicates_ok, 400);
        let critical = one.critical_value.unwrap();
        assert!(critical >= antecedent_stats::NORMAL_Q975);
        for quantity in &one.quantities {
            let interval = quantity.interval.unwrap();
            let band = quantity.band.unwrap();
            assert!(band[0] <= interval[0] && band[1] >= interval[1]);
        }
    }

    #[test]
    fn zero_replicates_and_cancelled_studies_are_refused() {
        let ctx = ExecutionContext::for_tests(1);
        assert!(resample_procedure(&ctx, 1, 0, &[0.0], |_| Some(vec![0.0])).is_err());
        let cancelled = ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        assert!(resample_procedure(&cancelled, 1, 10, &[0.0], |_| Some(vec![0.0])).is_err());
    }

    #[test]
    fn a_tiny_admission_pool_refuses_the_replicate_buffer_and_is_restored() {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 64);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let before = pool.available();
        assert!(resample_procedure(&ctx, 1, 1000, &[0.0; 8], |_| Some(vec![0.0; 8])).is_err());
        assert_eq!(pool.available(), before);
    }
}

/// Posterior summary of one quantity of a Bayesian-bootstrap posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorQuantity {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub sd: f64,
    /// Equal-tailed 95% credible interval.
    pub interval: [f64; 2],
    /// Simultaneous 95% credible band over the block, containing the interval.
    pub band: Option<[f64; 2]>,
}

/// Posterior summary of a block of quantities.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorSummary {
    /// One summary per quantity; `None` when the quantity has too few finite draws.
    pub quantities: Vec<Option<PosteriorQuantity>>,
    /// Critical value of the simultaneous band (at least the pointwise normal quantile).
    pub critical_value: Option<f64>,
    /// Draws with every quantity finite.
    pub draws_ok: u32,
    /// Draws without.
    pub draws_failed: u32,
}

/// Summarize posterior draws (one vector per draw) of `width` quantities: mean, standard
/// deviation, equal-tailed 95% credible intervals and a simultaneous credible band
/// `mean ± c·sd`, `c` the 95% quantile over complete draws of `max |θ − mean| / sd` and never
/// below the pointwise normal quantile, widened where needed to contain each interval.
#[must_use]
pub fn summarize_posterior(
    draws: &[Option<Vec<f64>>],
    width: usize,
    requested: u32,
) -> PosteriorSummary {
    let complete: Vec<&Vec<f64>> = draws
        .iter()
        .flatten()
        .filter(|draw| draw.len() == width && draw.iter().all(|value| value.is_finite()))
        .collect();
    let draws_ok = u32::try_from(complete.len()).unwrap_or(u32::MAX);
    let moments: Vec<Option<(f64, f64, [f64; 2])>> = (0..width)
        .map(|k| {
            if complete.len() < MIN_FINITE_REPLICATES {
                return None;
            }
            let mut column: Vec<f64> = complete.iter().map(|draw| draw[k]).collect();
            let mean = column.iter().sum::<f64>() / column.len() as f64;
            let sd = (column.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                / (column.len() - 1) as f64)
                .sqrt();
            column.sort_by(f64::total_cmp);
            let interval = [quantile_sorted(&column, 0.025), quantile_sorted(&column, 0.975)];
            Some((mean, sd, interval))
        })
        .collect();
    let mut maxima: Vec<f64> = complete
        .iter()
        .filter_map(|draw| {
            (0..width)
                .filter_map(|k| {
                    moments[k]
                        .filter(|(_, sd, _)| *sd > 0.0)
                        .map(|(mean, sd, _)| (draw[k] - mean).abs() / sd)
                })
                .reduce(f64::max)
        })
        .collect();
    maxima.sort_by(f64::total_cmp);
    let critical_value = (maxima.len() >= MIN_FINITE_REPLICATES)
        .then(|| crate::util::monte_carlo_critical(&maxima, ROLLOUT_LEVEL))
        .filter(|value| value.is_finite())
        .map(|value| value.max(antecedent_stats::NORMAL_Q975));
    let quantities = moments
        .iter()
        .map(|moment| {
            moment.map(|(mean, sd, interval)| PosteriorQuantity {
                mean,
                sd,
                interval,
                band: critical_value
                    .map(|c| [(mean - c * sd).min(interval[0]), (mean + c * sd).max(interval[1])]),
            })
        })
        .collect();
    PosteriorSummary {
        quantities,
        critical_value,
        draws_ok,
        draws_failed: requested.saturating_sub(draws_ok),
    }
}

impl DesignEffect {
    /// Replace the point with a posterior mean and attach its credible interval and band; the
    /// standard error stays empty because the interval is not a normal interval.
    pub fn attach_posterior(&mut self, quantity: Option<&PosteriorQuantity>) {
        if let Some(quantity) = quantity {
            self.estimate = quantity.mean;
            self.standard_error = None;
            self.interval = Some(quantity.interval);
            self.band = quantity.band;
        }
    }
}
