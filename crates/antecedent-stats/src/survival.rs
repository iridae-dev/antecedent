//! Native risk-set summaries for randomized survival outcomes.
//!
//! These point estimators consume an already identified randomized contrast.
//! Independent censoring and entry are scientific assumptions supplied by the
//! query; this module cannot verify them from observed follow-up.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::splitmix64;

/// Outcome summarized over randomized treatment arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalEndpoint {
    /// Kaplan–Meier survival and restricted mean survival time.
    Survival,
    /// Aalen–Johansen cumulative incidence for one event cause.
    CumulativeIncidence {
        /// Positive event-cause code whose incidence is requested.
        target_cause: i64,
    },
}

/// Subject rows of a randomized survival evaluation, aligned by row.
///
/// Event code zero means censoring.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalRows<'a> {
    /// Follow-up time per subject.
    pub duration: &'a [f64],
    /// Event cause per subject (zero is censored).
    pub event_code: &'a [i64],
    /// Randomized arm per subject.
    pub treated: &'a [bool],
    /// Left-truncation (delayed-entry) time per subject, when entry is delayed.
    pub delayed_entry: Option<&'a [f64]>,
}

/// A caller-supplied subject-specific censoring survival grid, known independently of
/// the outcomes it weights.
#[derive(Clone, Copy, Debug)]
pub struct KnownCensoring<'a> {
    /// Increasing time grid from zero through the restriction horizon.
    pub times: &'a [f64],
    /// Row-major left-limit censoring survival `G(t−) = P(C >= t)`, one value per
    /// subject and grid time. Events take priority over censoring at tied times.
    pub survival: &'a [f64],
    /// Declared positivity floor the survival values must stay above.
    pub minimum_probability: f64,
}

/// Arm curves on one event-time grid and, for survival, restricted means.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedSurvivalSummary {
    /// Event-time grid, including time zero and the restriction horizon.
    pub times: Vec<f64>,
    /// Control-arm survival or target-cause cumulative incidence.
    pub control: Vec<f64>,
    /// Treated-arm survival or target-cause cumulative incidence.
    pub treated: Vec<f64>,
    /// Control restricted mean survival time, absent for cumulative incidence.
    pub rmst_control: Option<f64>,
    /// Treated restricted mean survival time, absent for cumulative incidence.
    pub rmst_treated: Option<f64>,
    /// Smallest positive risk set at an observed event time in either arm.
    pub minimum_event_risk_set: Option<usize>,
    /// Smallest positive event risk set by control and treated arm.
    pub minimum_event_risk_set_by_arm: [Option<usize>; 2],
}

/// Pointwise, percentile subject-bootstrap intervals for the randomized contrasts.
///
/// The two treatment arms are resampled separately. A caller-supplied censoring
/// grid is held fixed with each subject: its nuisance function is not refitted.
/// These intervals do not cover the entire curve simultaneously.
#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalBootstrapIntervals {
    /// RMST treatment minus control interval, for a survival endpoint.
    pub rmst_difference: Option<[f64; 2]>,
    /// Treatment minus control curve at the restriction horizon. This is a
    /// survival difference or a target-cause cumulative-incidence difference.
    pub difference_at_tau: [f64; 2],
    /// Requested number of subject-bootstrap replicates.
    pub replicates_requested: u32,
    /// Replicates meeting the original estimator's support contract.
    pub replicates_ok: u32,
}

/// Simultaneous 95% band for the randomized treatment-minus-control curve.
///
/// The band is evaluated on the observed event-time grid, including zero and
/// the requested horizon. Its common radius comes from the 95th percentile
/// of bootstrap suprema of centered curve differences. The admitted designs
/// use independent randomized subjects and may include delayed entry with
/// unweighted risk sets or fixed known censoring survival.
#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalDifferenceBand {
    /// Observed event-time grid used to evaluate every bootstrap curve.
    pub times: Vec<f64>,
    /// Estimated treatment-minus-control curve on that grid.
    pub difference: Vec<f64>,
    /// Simultaneous lower endpoints aligned with `times`.
    pub lower: Vec<f64>,
    /// Simultaneous upper endpoints aligned with `times`.
    pub upper: Vec<f64>,
    /// Bootstrap draws whose estimator satisfied the support contract.
    pub replicates_ok: u32,
    /// Smallest arm risk set at the horizon, for bands with delayed entry.
    pub minimum_horizon_risk_set: Option<usize>,
}

/// Minimum event-time and horizon risk-set size for delayed-entry bands.
pub const DELAYED_ENTRY_BAND_MIN_RISK_SET: usize = 25;
/// Minimum observed subjects per arm for a delayed-entry band with marginal or fixed-G observation.
#[must_use]
pub const fn delayed_entry_band_min_arm(fixed_known_censoring: bool) -> usize {
    if fixed_known_censoring { 200 } else { 160 }
}

/// Visit each within-arm bootstrap draw in arm order, advancing the caller's stream once per draw.
#[doc(hidden)]
pub fn for_each_resampled_subject(
    arms: &[Vec<usize>; 2],
    rng: &mut u64,
    mut visit: impl FnMut(usize, bool),
) {
    for (arm_index, arm) in arms.iter().enumerate() {
        for _ in 0..arm.len() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "u64 PRNG output is reduced modulo arm length to a valid index"
            )]
            let i = arm[(splitmix64(rng) as usize) % arm.len()];
            visit(i, arm_index == 1);
        }
    }
}

/// Build a simultaneous curve-difference band by subject resampling within arms.
///
/// At least 80 subjects per arm and 399 valid replicates are required. The
/// pointwise scalar intervals remain a separate construction with their own
/// support and calibration.
pub fn randomized_survival_bootstrap_difference_band(
    duration: &[f64],
    event_code: &[i64],
    treated: &[bool],
    tau: f64,
    endpoint: SurvivalEndpoint,
    replicates: u32,
    seed: u64,
) -> Result<SurvivalDifferenceBand, &'static str> {
    randomized_survival_bootstrap_difference_band_with_observation(
        &SurvivalRows { duration, event_code, treated, delayed_entry: None },
        None,
        tau,
        endpoint,
        replicates,
        seed,
        || false,
    )
}

/// Simultaneous subject-resampled difference band on the observed analysis grid.
///
/// Delayed-entry bands carry entry times and any fixed known censoring probabilities
/// together with each subject in every resample. Both observed arms need at least
/// 25 at-risk subjects through the horizon and every observed event risk set must
/// meet the same threshold. The original unweighted no-entry design keeps its
/// existing admission rule.
///
/// # Errors
///
/// The observed design or its bootstrap draws cannot support the band, or the caller cancels.
pub fn randomized_survival_bootstrap_difference_band_with_observation(
    rows: &SurvivalRows<'_>,
    known_censoring: Option<&KnownCensoring<'_>>,
    tau: f64,
    endpoint: SurvivalEndpoint,
    replicates: u32,
    seed: u64,
    cancelled: impl Fn() -> bool,
) -> Result<SurvivalDifferenceBand, &'static str> {
    let SurvivalRows { duration, event_code, treated, delayed_entry } = *rows;
    if !(399..=100_000).contains(&replicates) {
        return Err("survival simultaneous band requires 399 to 100000 replicates");
    }
    if duration.len() != event_code.len()
        || duration.len() != treated.len()
        || delayed_entry.is_some_and(|entry| entry.len() != duration.len())
    {
        return Err("survival band subject arrays must be row-aligned");
    }
    if known_censoring.is_some() && delayed_entry.is_none() {
        return Err(
            "simultaneous survival band with fixed censoring weights requires delayed entry",
        );
    }
    if cancelled() {
        return Err("survival simultaneous band was cancelled");
    }
    let summarize =
        |sample: &SurvivalRows<'_>, censoring: Option<&KnownCensoring<'_>>| match censoring {
            Some(known) => randomized_survival_ipcw_summary(sample, known, tau, endpoint)
                .map(|(summary, _)| summary),
            None => randomized_survival_summary(sample, tau, endpoint),
        };
    let arms = [false, true].map(|arm| {
        treated.iter().enumerate().filter_map(|(i, &a)| (a == arm).then_some(i)).collect::<Vec<_>>()
    });
    let minimum_arm = if delayed_entry.is_some() {
        delayed_entry_band_min_arm(known_censoring.is_some())
    } else {
        80
    };
    if arms.iter().any(|arm| arm.len() < minimum_arm) {
        if delayed_entry.is_some() {
            return Err(
                "simultaneous survival band with delayed entry has insufficient subjects in an arm",
            );
        }
        return Err(
            "survival simultaneous band requires at least 80 subjects in each randomized arm",
        );
    }
    let original = summarize(rows, known_censoring)?;
    let minimum_horizon_risk_set = delayed_entry.map(|entry| {
        arms.iter()
            .map(|arm| arm.iter().filter(|&&i| duration[i] >= tau && entry[i] < tau).count())
            .min()
            .unwrap_or(0)
    });
    if let Some(entry) = delayed_entry
        && (original.minimum_event_risk_set.is_none()
            || !band_grid_has_risk_sets(
                rows,
                entry,
                &original.times,
                DELAYED_ENTRY_BAND_MIN_RISK_SET,
            ))
    {
        return Err(
            "simultaneous survival band with delayed entry requires 25 at-risk subjects through each observed event and the horizon",
        );
    }
    let difference = original
        .treated
        .iter()
        .zip(&original.control)
        .map(|(active, control)| active - control)
        .collect::<Vec<_>>();
    let mut rng = seed ^ 0x7D0C_0B41_5752_5A3D;
    let mut suprema = Vec::with_capacity(replicates as usize);
    let width = known_censoring.map_or(0, |known| known.times.len());
    for _ in 0..replicates {
        if cancelled() {
            return Err("survival simultaneous band was cancelled");
        }
        let mut d = Vec::with_capacity(duration.len());
        let mut e = Vec::with_capacity(duration.len());
        let mut a = Vec::with_capacity(duration.len());
        let mut left = delayed_entry.map(|_| Vec::with_capacity(duration.len()));
        let mut g = known_censoring.map(|known| Vec::with_capacity(known.survival.len()));
        for_each_resampled_subject(&arms, &mut rng, |i, arm| {
            d.push(duration[i]);
            e.push(event_code[i]);
            a.push(arm);
            if let (Some(entry), Some(out)) = (delayed_entry, &mut left) {
                out.push(entry[i]);
            }
            if let (Some(known), Some(out)) = (known_censoring, &mut g) {
                out.extend_from_slice(&known.survival[i * width..(i + 1) * width]);
            }
        });
        let rows = SurvivalRows {
            duration: &d,
            event_code: &e,
            treated: &a,
            delayed_entry: left.as_deref(),
        };
        let sampled_censoring = known_censoring
            .zip(g.as_deref())
            .map(|(known, survival)| KnownCensoring { survival, ..*known });
        let Ok(sample) = summarize(&rows, sampled_censoring.as_ref()) else {
            continue;
        };
        if let Some(entry) = &left
            && !band_grid_has_risk_sets(&rows, entry, &original.times, 1)
        {
            continue;
        }
        let sup = original
            .times
            .iter()
            .zip(&difference)
            .map(|(&time, &point)| {
                let position = sample.times.partition_point(|event_time| *event_time <= time);
                let j = position.saturating_sub(1);
                (sample.treated[j] - sample.control[j] - point).abs()
            })
            .fold(0.0_f64, f64::max);
        suprema.push(sup);
    }
    let ok = bootstrap_support_count(
        suprema.len(),
        replicates,
        399,
        "too many survival band draws",
        "survival simultaneous band lost more than 10% of subject draws to support failures",
    )?;
    let (lower, upper) = supremum_endpoints(&difference, &mut suprema);
    Ok(SurvivalDifferenceBand {
        times: original.times,
        difference,
        lower,
        upper,
        replicates_ok: ok,
        minimum_horizon_risk_set,
    })
}

fn band_grid_has_risk_sets(
    rows: &SurvivalRows<'_>,
    entry: &[f64],
    grid: &[f64],
    minimum_at_risk: usize,
) -> bool {
    grid.iter().skip(1).all(|&time| {
        let mut counts = [0; 2];
        for ((&duration, &left), &treated) in rows.duration.iter().zip(entry).zip(rows.treated) {
            if left < time && time <= duration {
                counts[usize::from(treated)] += 1;
            }
        }
        counts.into_iter().all(|count| count >= minimum_at_risk)
    })
}

fn supremum_endpoints(difference: &[f64], suprema: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    suprema.sort_by(f64::total_cmp);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ceil of a positive quantile position yields a small nonnegative rank index"
    )]
    let index = ((suprema.len() as f64 + 1.0) * 0.95).ceil() as usize;
    let radius = suprema[index.saturating_sub(1).min(suprema.len() - 1)];
    let lower = difference.iter().map(|point| (point - radius).max(-1.0)).collect();
    let upper = difference.iter().map(|point| (point + radius).min(1.0)).collect();
    (lower, upper)
}

/// Resample individual subjects within randomized arms and recompute the
/// complete product-limit or Aalen-Johansen estimator on each draw.
///
/// A replicate that loses required cause or horizon support is omitted. At
/// least 90% of requested draws and at least 199 valid draws are required for
/// a two-sided 95% interval; otherwise the interval is refused. The resampling
/// seed is an explicit input so results are independent of execution threads.
pub fn randomized_survival_bootstrap_intervals(
    rows: &SurvivalRows<'_>,
    known_censoring: Option<&KnownCensoring<'_>>,
    tau: f64,
    endpoint: SurvivalEndpoint,
    replicates: u32,
    seed: u64,
) -> Result<SurvivalBootstrapIntervals, &'static str> {
    randomized_survival_bootstrap_intervals_controlled(
        rows,
        known_censoring,
        tau,
        endpoint,
        replicates,
        seed,
        || false,
    )
}

/// The subject bootstrap with a cancellation checkpoint before every replicate.
///
/// # Errors
///
/// Refuses unsupported input or a cancelled calculation.
#[doc(hidden)]
pub fn randomized_survival_bootstrap_intervals_controlled(
    rows: &SurvivalRows<'_>,
    known_censoring: Option<&KnownCensoring<'_>>,
    tau: f64,
    endpoint: SurvivalEndpoint,
    replicates: u32,
    seed: u64,
    cancelled: impl Fn() -> bool,
) -> Result<SurvivalBootstrapIntervals, &'static str> {
    let SurvivalRows { duration, event_code, treated, delayed_entry } = *rows;
    if !(199..=100_000).contains(&replicates) {
        return Err("survival pointwise bootstrap requires 199 to 100000 replicates");
    }
    if duration.len() != event_code.len()
        || duration.len() != treated.len()
        || delayed_entry.is_some_and(|entry| entry.len() != duration.len())
        || known_censoring.is_some_and(|known| {
            known.survival.len() != duration.len().saturating_mul(known.times.len())
        })
    {
        return Err("survival bootstrap subject arrays must be row-aligned");
    }
    let arms = [false, true].map(|arm| {
        treated.iter().enumerate().filter_map(|(i, &a)| (a == arm).then_some(i)).collect::<Vec<_>>()
    });
    if arms.iter().any(|arm| arm.len() < 8) {
        return Err("survival bootstrap requires at least eight subjects in each randomized arm");
    }
    let grid_width = known_censoring.map_or(0, |known| known.times.len());
    let mut rng = seed ^ 0x8BB8_E832_F975_6D5D;
    let mut draws_rmst = Vec::with_capacity(replicates as usize);
    let mut draws_tau = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        if cancelled() {
            return Err("survival subject bootstrap was cancelled");
        }
        let mut d = Vec::with_capacity(duration.len());
        let mut e = Vec::with_capacity(duration.len());
        let mut a = Vec::with_capacity(duration.len());
        let mut left = delayed_entry.map(|_| Vec::with_capacity(duration.len()));
        let mut g = known_censoring.map(|_| Vec::with_capacity(duration.len() * grid_width));
        for_each_resampled_subject(&arms, &mut rng, |i, arm| {
            d.push(duration[i]);
            e.push(event_code[i]);
            a.push(arm);
            if let (Some(source), Some(out)) = (delayed_entry, left.as_mut()) {
                out.push(source[i]);
            }
            if let (Some(known), Some(out)) = (known_censoring, g.as_mut()) {
                out.extend_from_slice(&known.survival[i * grid_width..(i + 1) * grid_width]);
            }
        });
        let replicate = SurvivalRows {
            duration: &d,
            event_code: &e,
            treated: &a,
            delayed_entry: left.as_deref(),
        };
        let estimate = match known_censoring {
            Some(known) => randomized_survival_ipcw_summary(
                &replicate,
                &KnownCensoring {
                    times: known.times,
                    survival: g.as_deref().unwrap_or(&[]),
                    minimum_probability: known.minimum_probability,
                },
                tau,
                endpoint,
            )
            .map(|(summary, _)| summary),
            None => randomized_survival_summary(&replicate, tau, endpoint),
        };
        let Ok(estimate) = estimate else { continue };
        let last = estimate.treated.last().zip(estimate.control.last());
        if let Some((&treated_tau, &control_tau)) = last {
            draws_tau.push(treated_tau - control_tau);
            if let (Some(t), Some(c)) = (estimate.rmst_treated, estimate.rmst_control) {
                draws_rmst.push(t - c);
            }
        }
    }
    let ok = bootstrap_support_count(
        draws_tau.len(),
        replicates,
        199,
        "too many survival bootstrap draws",
        "survival bootstrap lost more than 10% of subject draws to support failures",
    )?;
    if endpoint == SurvivalEndpoint::Survival && draws_rmst.len() != draws_tau.len() {
        return Err("survival bootstrap lost RMST estimates");
    }
    Ok(SurvivalBootstrapIntervals {
        rmst_difference: (endpoint == SurvivalEndpoint::Survival)
            .then(|| percentile_95(&mut draws_rmst)),
        difference_at_tau: percentile_95(&mut draws_tau),
        replicates_requested: replicates,
        replicates_ok: ok,
    })
}

/// The 95% bootstrap percentile interval of `draws`: type-7 quantiles at 2.5% and 97.5%,
/// the crate's rule for resampling percentiles (`statistical_transport::percentile_interval`).
#[doc(hidden)]
pub fn percentile_95(draws: &mut [f64]) -> [f64; 2] {
    draws.sort_by(f64::total_cmp);
    let (lower, upper) =
        crate::equal_tail_interval_sorted(draws, 0.95, crate::QuantileRule::Interpolated);
    [lower, upper]
}

/// Apply the survival bootstrap's minimum count and maximum support-loss policy.
#[doc(hidden)]
pub fn bootstrap_support_count(
    draws: usize,
    requested: u32,
    minimum: u32,
    overflow_error: &'static str,
    support_error: &'static str,
) -> Result<u32, &'static str> {
    let ok = u32::try_from(draws).map_err(|_| overflow_error)?;
    if ok < minimum || ok < requested.saturating_sub(requested / 10) {
        return Err(support_error);
    }
    Ok(ok)
}

/// Compute randomized arm curves with optional left truncation.
///
/// Event code zero means censoring. `Survival` accepts only codes zero and one;
/// cumulative incidence requires at least two observed positive causes. All
/// observations must have follow-up to a common `tau` in each arm. No interval
/// is constructed by this kernel.
pub fn randomized_survival_summary(
    rows: &SurvivalRows<'_>,
    tau: f64,
    endpoint: SurvivalEndpoint,
) -> Result<RandomizedSurvivalSummary, &'static str> {
    let SurvivalRows { duration, event_code, treated, delayed_entry } = *rows;
    let n = duration.len();
    if n == 0
        || event_code.len() != n
        || treated.len() != n
        || delayed_entry.is_some_and(|entry| entry.len() != n)
    {
        return Err("duration, event, treatment, and entry must have equal nonzero length");
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err("tau must be finite and positive");
    }
    if duration.iter().any(|&time| !time.is_finite() || time < 0.0) {
        return Err("durations must be finite and nonnegative");
    }
    if let Some(entry) = delayed_entry
        && entry
            .iter()
            .zip(duration)
            .any(|(&left, &right)| !left.is_finite() || left < 0.0 || left >= right)
    {
        return Err("entry times must be finite, nonnegative, and earlier than exit");
    }
    match endpoint {
        SurvivalEndpoint::Survival => {
            if event_code.iter().any(|&code| code != 0 && code != 1) {
                return Err("survival events must be encoded as zero or one");
            }
        }
        SurvivalEndpoint::CumulativeIncidence { target_cause } => {
            if target_cause <= 0 || event_code.iter().any(|&code| code < 0) {
                return Err("event causes must be nonnegative and target cause positive");
            }
            let causes: BTreeSet<i64> =
                event_code.iter().copied().filter(|&code| code > 0).collect();
            if causes.len() < 2 || !causes.contains(&target_cause) {
                return Err(
                    "cumulative incidence requires two observed causes including the target",
                );
            }
        }
    }
    for arm in [false, true] {
        let rows: Vec<usize> = treated
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == arm).then_some(index))
            .collect();
        if rows.is_empty() {
            return Err("both randomized arms require observed units");
        }
        if rows.iter().all(|&index| duration[index] < tau) {
            return Err("tau exceeds observed follow-up in one arm");
        }
        if let Some(entry) = delayed_entry
            && rows.iter().all(|&index| entry[index] != 0.0)
        {
            return Err("RMST or incidence from zero requires a time-zero entrant in each arm");
        }
    }

    let control = arm_curve(duration, event_code, treated, false, delayed_entry, tau, endpoint)?;
    let active = arm_curve(duration, event_code, treated, true, delayed_entry, tau, endpoint)?;
    let mut times = control.0.clone();
    times.extend(active.0.iter().copied());
    times.push(tau);
    times.sort_by(f64::total_cmp);
    times.dedup();
    let align = |arm_times: &[f64], values: &[f64]| -> Vec<f64> {
        times
            .iter()
            .map(|&time| {
                let position = arm_times.partition_point(|&event_time| event_time <= time);
                values[position.saturating_sub(1)]
            })
            .collect()
    };
    Ok(RandomizedSurvivalSummary {
        control: align(&control.0, &control.1),
        treated: align(&active.0, &active.1),
        times,
        rmst_control: control.2,
        rmst_treated: active.2,
        minimum_event_risk_set: control.3.into_iter().chain(active.3).min(),
        minimum_event_risk_set_by_arm: [control.3, active.3],
    })
}

/// Evaluate arm curves with a caller supplied, subject-specific censoring survival grid.
///
/// [`KnownCensoring::survival`] is row-major with one value per subject and time;
/// each value is `P(C >= t)` so subjects censored at `t` remain in the event risk set.
/// The probabilities must be known independently of the outcomes used here; this
/// point kernel does not fit or verify a censoring model. No interval is
/// constructed by this function. Optional marginally independent left entry
/// ([`SurvivalRows::delayed_entry`]) and censoring probabilities remain attached
/// to their subject in a subject-resampling interval; conditional or informative
/// entry is not adjusted.
///
/// # Errors
///
/// Returns an error for misaligned rows or times, invalid event codes, absent
/// arm support through `tau`, or censoring probabilities that violate the
/// declared positivity and monotonicity contract.
#[allow(
    clippy::too_many_lines,
    reason = "validate the rows, then per arm build the censoring-weighted curve and risk-set diagnostics that the contrast reads"
)]
pub fn randomized_survival_ipcw_summary(
    rows: &SurvivalRows<'_>,
    censoring: &KnownCensoring<'_>,
    tau: f64,
    endpoint: SurvivalEndpoint,
) -> Result<(RandomizedSurvivalSummary, f64), &'static str> {
    let SurvivalRows { duration, event_code, treated, delayed_entry } = *rows;
    let KnownCensoring { times, survival: censoring_survival, minimum_probability } = *censoring;
    let n = duration.len();
    let m = times.len();
    if n == 0
        || event_code.len() != n
        || treated.len() != n
        || delayed_entry.is_some_and(|entry| entry.len() != n)
        || m < 2
        || censoring_survival.len()
            != n.checked_mul(m).ok_or("censoring grid dimensions overflow")?
    {
        return Err(
            "duration, event, treatment, and censoring grid must have aligned nonzero rows",
        );
    }
    if !tau.is_finite()
        || tau <= 0.0
        || !minimum_probability.is_finite()
        || minimum_probability <= 0.0
        || minimum_probability > 1.0
    {
        return Err("tau and censoring positivity floor must be finite and positive");
    }
    if times[0] != 0.0
        || (times[m - 1] - tau).abs() > 1e-10
        || times.iter().any(|time| !time.is_finite())
        || times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err("censoring times must increase from zero through tau");
    }
    if duration.iter().any(|time| !time.is_finite() || *time < 0.0)
        || duration.iter().any(|&time| {
            time <= tau && times.binary_search_by(|grid| grid.total_cmp(&time)).is_err()
        })
    {
        return Err("durations must be finite and represented on the censoring grid through tau");
    }
    if delayed_entry.is_some_and(|entry| {
        entry
            .iter()
            .zip(duration)
            .any(|(&left, &right)| !left.is_finite() || left < 0.0 || left >= right)
    }) {
        return Err("entry times must be finite, nonnegative, and earlier than exit");
    }
    match endpoint {
        SurvivalEndpoint::Survival if event_code.iter().any(|code| *code != 0 && *code != 1) => {
            return Err("survival events must be encoded as zero or one");
        }
        SurvivalEndpoint::CumulativeIncidence { target_cause } => {
            let causes: BTreeSet<i64> =
                event_code.iter().copied().filter(|code| *code > 0).collect();
            if target_cause <= 0
                || event_code.iter().any(|code| *code < 0)
                || causes.len() < 2
                || !causes.contains(&target_cause)
            {
                return Err(
                    "cumulative incidence requires two observed causes including the positive target",
                );
            }
        }
        SurvivalEndpoint::Survival => {}
    }
    let mut minimum_g = 1.0_f64;
    for row in censoring_survival.chunks_exact(m) {
        if (row[0] - 1.0).abs() > 1e-10
            || row
                .iter()
                .any(|value| !value.is_finite() || *value < minimum_probability || *value > 1.0)
            || row.windows(2).any(|pair| pair[1] > pair[0] + 1e-10)
        {
            return Err(
                "censoring survival must start at one, remain nonincreasing, and satisfy positivity",
            );
        }
        minimum_g = minimum_g.min(row.iter().copied().fold(1.0_f64, f64::min));
    }
    for arm in [false, true] {
        if !(0..n).any(|i| treated[i] == arm && duration[i] >= tau) {
            return Err("both randomized arms require observed follow-up through tau");
        }
        if delayed_entry.is_some_and(|entry| !(0..n).any(|i| treated[i] == arm && entry[i] == 0.0))
        {
            return Err("RMST or incidence from zero requires a time-zero entrant in each arm");
        }
    }
    let mut curves: [Vec<f64>; 2] = [Vec::with_capacity(m), Vec::with_capacity(m)];
    let mut restricted_means = [0.0_f64; 2];
    let mut minimum_risk_set: Option<usize> = None;
    let mut minimum_risk_set_by_arm = [None; 2];
    for (arm_index, arm) in [false, true].into_iter().enumerate() {
        let mut survival = 1.0;
        let mut incidence = 0.0;
        let mut previous_time = 0.0;
        for (j, &time) in times.iter().enumerate() {
            restricted_means[arm_index] += (time - previous_time) * survival;
            let at_risk = (0..n)
                .filter(|&i| {
                    treated[i] == arm
                        && duration[i] >= time
                        && delayed_entry.is_none_or(|entry| entry[i] < time)
                })
                .collect::<Vec<_>>();
            #[allow(
                clippy::float_cmp,
                reason = "matches rows whose recorded event time equals this exact grid time"
            )]
            let event_rows = at_risk
                .iter()
                .copied()
                .filter(|&i| event_code[i] > 0 && duration[i] == time)
                .collect::<Vec<_>>();
            if !event_rows.is_empty() {
                let weight = |i: usize| 1.0 / censoring_survival[i * m + j];
                let risk_weight = at_risk.iter().map(|&i| weight(i)).sum::<f64>();
                let event_weight = event_rows.iter().map(|&i| weight(i)).sum::<f64>();
                if !risk_weight.is_finite()
                    || risk_weight <= 0.0
                    || !event_weight.is_finite()
                    || event_weight > risk_weight * (1.0 + 1e-10)
                {
                    return Err("IPCW event weight exceeds its finite positive weighted risk set");
                }
                let hazard = (event_weight / risk_weight).min(1.0);
                if let SurvivalEndpoint::CumulativeIncidence { target_cause } = endpoint {
                    let target_weight = event_rows
                        .iter()
                        .filter(|&&i| event_code[i] == target_cause)
                        .map(|&i| weight(i))
                        .sum::<f64>();
                    incidence += survival * target_weight / risk_weight;
                }
                survival *= 1.0 - hazard;
                minimum_risk_set =
                    Some(minimum_risk_set.map_or(at_risk.len(), |old| old.min(at_risk.len())));
                minimum_risk_set_by_arm[arm_index] = Some(
                    minimum_risk_set_by_arm[arm_index]
                        .map_or(at_risk.len(), |old: usize| old.min(at_risk.len())),
                );
            }
            curves[arm_index].push(if endpoint == SurvivalEndpoint::Survival {
                survival
            } else {
                incidence
            });
            previous_time = time;
        }
    }
    Ok((
        RandomizedSurvivalSummary {
            times: times.to_vec(),
            control: curves[0].clone(),
            treated: curves[1].clone(),
            rmst_control: (endpoint == SurvivalEndpoint::Survival).then_some(restricted_means[0]),
            rmst_treated: (endpoint == SurvivalEndpoint::Survival).then_some(restricted_means[1]),
            minimum_event_risk_set: minimum_risk_set,
            minimum_event_risk_set_by_arm: minimum_risk_set_by_arm,
        },
        minimum_g,
    ))
}

type ArmCurve = (Vec<f64>, Vec<f64>, Option<f64>, Option<usize>);

fn arm_curve(
    duration: &[f64],
    event_code: &[i64],
    treated: &[bool],
    arm: bool,
    delayed_entry: Option<&[f64]>,
    tau: f64,
    endpoint: SurvivalEndpoint,
) -> Result<ArmCurve, &'static str> {
    let mut times: Vec<f64> = duration
        .iter()
        .zip(event_code)
        .zip(treated)
        .filter_map(|((&time, &cause), &assignment)| {
            (assignment == arm && cause > 0 && time <= tau).then_some(time)
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut curve_times = vec![0.0];
    let initial = if endpoint == SurvivalEndpoint::Survival { 1.0 } else { 0.0 };
    let mut values = vec![initial];
    let mut survival = 1.0;
    let mut incidence = 0.0;
    let mut previous = 0.0;
    let mut rmst = 0.0;
    let mut minimum_risk_set: Option<usize> = None;
    for time in times {
        rmst += (time - previous) * survival;
        let at_risk = duration
            .iter()
            .zip(treated)
            .enumerate()
            .filter(|&(index, (exit, assignment))| {
                *assignment == arm
                    && *exit >= time
                    && delayed_entry.is_none_or(|entry| entry[index] < time)
            })
            .count();
        if at_risk == 0 {
            return Err("observed event has an empty risk set");
        }
        let mut all_events = 0;
        let mut target_events = 0;
        #[allow(
            clippy::float_cmp,
            reason = "counts rows whose recorded exit time equals this exact event time"
        )]
        for ((&exit, &cause), &assignment) in duration.iter().zip(event_code).zip(treated) {
            if assignment == arm && exit == time && cause > 0 {
                all_events += 1;
                if endpoint == SurvivalEndpoint::Survival
                    || matches!(endpoint, SurvivalEndpoint::CumulativeIncidence { target_cause } if cause == target_cause)
                {
                    target_events += 1;
                }
            }
        }
        if all_events > at_risk {
            return Err("event count exceeds the risk set");
        }
        if endpoint == SurvivalEndpoint::Survival {
            survival *= 1.0 - all_events as f64 / at_risk as f64;
            values.push(survival);
        } else {
            incidence += survival * f64::from(target_events) / at_risk as f64;
            survival *= 1.0 - all_events as f64 / at_risk as f64;
            values.push(incidence);
        }
        curve_times.push(time);
        previous = time;
        minimum_risk_set = Some(minimum_risk_set.map_or(at_risk, |prior| prior.min(at_risk)));
    }
    rmst += (tau - previous) * survival;
    Ok((
        curve_times,
        values,
        (endpoint == SurvivalEndpoint::Survival).then_some(rmst),
        minimum_risk_set,
    ))
}
