//! Native risk-set summaries for randomized survival outcomes.
//!
//! These point estimators consume an already identified randomized contrast.
//! Independent censoring and entry are scientific assumptions supplied by the
//! query; this module cannot verify them from observed follow-up.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

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
/// of bootstrap suprema of centered curve differences. This construction is
/// currently restricted to independent randomized subjects without delayed
/// entry or fitted or fixed censoring weights.
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
}

/// Build a simultaneous curve-difference band by subject resampling within arms.
///
/// At least 80 subjects per arm and 399 valid replicates are required. The
/// pointwise scalar intervals remain a separate construction with their own
/// support and calibration. No full-curve claim follows for delayed entry,
/// conditional censoring, or observational treatment assignment.
pub fn randomized_survival_bootstrap_difference_band(
    duration: &[f64], event_code: &[i64], treated: &[bool], tau: f64,
    endpoint: SurvivalEndpoint, replicates: u32, seed: u64,
) -> Result<SurvivalDifferenceBand, &'static str> {
    if replicates < 399 || replicates > 100_000 {
        return Err("survival simultaneous band requires 399 to 100000 replicates");
    }
    if duration.len() != event_code.len() || duration.len() != treated.len() {
        return Err("survival band subject arrays must be row-aligned");
    }
    let arms = [false, true].map(|arm| {
        treated.iter().enumerate().filter_map(|(i, &a)| (a == arm).then_some(i)).collect::<Vec<_>>()
    });
    if arms.iter().any(|arm| arm.len() < 80) {
        return Err("survival simultaneous band requires at least 80 subjects in each randomized arm");
    }
    let original = randomized_survival_summary(duration, event_code, treated, None, tau, endpoint)?;
    let difference = original.treated.iter().zip(&original.control)
        .map(|(active, control)| active - control).collect::<Vec<_>>();
    let mut rng = seed ^ 0x7D0C_0B41_5752_5A3D;
    let mut suprema = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        let mut d = Vec::with_capacity(duration.len());
        let mut e = Vec::with_capacity(duration.len());
        let mut a = Vec::with_capacity(duration.len());
        for (arm_index, arm) in arms.iter().enumerate() {
            for _ in 0..arm.len() {
                let i = arm[(splitmix64(&mut rng) as usize) % arm.len()];
                d.push(duration[i]);
                e.push(event_code[i]);
                a.push(arm_index == 1);
            }
        }
        let Ok(sample) = randomized_survival_summary(&d, &e, &a, None, tau, endpoint) else { continue };
        let sup = original.times.iter().zip(&difference).map(|(&time, &point)| {
            let position = sample.times.partition_point(|event_time| *event_time <= time);
            let j = position.saturating_sub(1);
            (sample.treated[j] - sample.control[j] - point).abs()
        }).fold(0.0_f64, f64::max);
        suprema.push(sup);
    }
    let ok = u32::try_from(suprema.len()).map_err(|_| "too many survival band draws")?;
    if ok < 399 || ok < replicates.saturating_sub(replicates / 10) {
        return Err("survival simultaneous band lost more than 10% of subject draws to support failures");
    }
    suprema.sort_by(f64::total_cmp);
    let index = ((suprema.len() as f64 + 1.0) * 0.95).ceil() as usize;
    let radius = suprema[index.saturating_sub(1).min(suprema.len() - 1)];
    let lower = difference.iter().map(|point| (point - radius).max(-1.0)).collect();
    let upper = difference.iter().map(|point| (point + radius).min(1.0)).collect();
    Ok(SurvivalDifferenceBand {
        times: original.times,
        difference,
        lower,
        upper,
        replicates_ok: ok,
    })
}

/// Resample individual subjects within randomized arms and recompute the
/// complete product-limit or Aalen-Johansen estimator on each draw.
///
/// A replicate that loses required cause or horizon support is omitted. At
/// least 90% of requested draws and at least 199 valid draws are required for
/// a two-sided 95% interval; otherwise the interval is refused. The resampling
/// seed is an explicit input so results are independent of execution threads.
#[allow(clippy::too_many_arguments)]
pub fn randomized_survival_bootstrap_intervals(
    duration: &[f64],
    event_code: &[i64],
    treated: &[bool],
    delayed_entry: Option<&[f64]>,
    known_censoring: Option<(&[f64], &[f64], f64)>,
    tau: f64,
    endpoint: SurvivalEndpoint,
    replicates: u32,
    seed: u64,
) -> Result<SurvivalBootstrapIntervals, &'static str> {
    if replicates < 199 || replicates > 100_000 {
        return Err("survival pointwise bootstrap requires 199 to 100000 replicates");
    }
    if duration.len() != event_code.len() || duration.len() != treated.len()
        || delayed_entry.is_some_and(|entry| entry.len() != duration.len())
    {
        return Err("survival bootstrap subject arrays must be row-aligned");
    }
    let arms = [false, true].map(|arm| {
        treated.iter().enumerate().filter_map(|(i, &a)| (a == arm).then_some(i)).collect::<Vec<_>>()
    });
    if arms.iter().any(|arm| arm.len() < 8) {
        return Err("survival bootstrap requires at least eight subjects in each randomized arm");
    }
    let grid_width = known_censoring.map_or(0, |(grid, _, _)| grid.len());
    let mut rng = seed ^ 0x8BB8_E832_F975_6D5D;
    let mut draws_rmst = Vec::with_capacity(replicates as usize);
    let mut draws_tau = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        let mut d = Vec::with_capacity(duration.len());
        let mut e = Vec::with_capacity(duration.len());
        let mut a = Vec::with_capacity(duration.len());
        let mut left = delayed_entry.map(|_| Vec::with_capacity(duration.len()));
        let mut g = known_censoring.map(|_| Vec::with_capacity(duration.len() * grid_width));
        for (arm_index, arm) in arms.iter().enumerate() {
            for _ in 0..arm.len() {
                let i = arm[(splitmix64(&mut rng) as usize) % arm.len()];
                d.push(duration[i]);
                e.push(event_code[i]);
                a.push(arm_index == 1);
                if let (Some(source), Some(out)) = (delayed_entry, left.as_mut()) {
                    out.push(source[i]);
                }
                if let (Some((_, source, _)), Some(out)) = (known_censoring, g.as_mut()) {
                    out.extend_from_slice(&source[i * grid_width..(i + 1) * grid_width]);
                }
            }
        }
        let estimate = match known_censoring {
            Some((grid, _, floor)) => randomized_survival_ipcw_summary_with_entry(
                &d, &e, &a, left.as_deref(), grid, g.as_deref().unwrap_or(&[]), tau, floor, endpoint,
            ).map(|(summary, _)| summary),
            None => randomized_survival_summary(&d, &e, &a, left.as_deref(), tau, endpoint),
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
    let ok = u32::try_from(draws_tau.len()).map_err(|_| "too many survival bootstrap draws")?;
    if ok < 199 || ok < replicates.saturating_sub(replicates / 10) {
        return Err("survival bootstrap lost more than 10% of subject draws to support failures");
    }
    if endpoint == SurvivalEndpoint::Survival && draws_rmst.len() != draws_tau.len() {
        return Err("survival bootstrap lost RMST estimates");
    }
    Ok(SurvivalBootstrapIntervals {
        rmst_difference: (endpoint == SurvivalEndpoint::Survival).then(|| percentile_95(&mut draws_rmst)),
        difference_at_tau: percentile_95(&mut draws_tau),
        replicates_requested: replicates,
        replicates_ok: ok,
    })
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn percentile_95(draws: &mut [f64]) -> [f64; 2] {
    draws.sort_by(f64::total_cmp);
    let n = draws.len();
    let lower = ((n as f64 + 1.0) * 0.025).ceil() as usize;
    let upper = ((n as f64 + 1.0) * 0.975).ceil() as usize;
    [draws[lower.saturating_sub(1).min(n - 1)], draws[upper.saturating_sub(1).min(n - 1)]]
}

/// Compute randomized arm curves with optional left truncation.
///
/// Event code zero means censoring. `Survival` accepts only codes zero and one;
/// cumulative incidence requires at least two observed positive causes. All
/// observations must have follow-up to a common `tau` in each arm. No interval
/// is constructed by this kernel.
pub fn randomized_survival_summary(
    duration: &[f64],
    event_code: &[i64],
    treated: &[bool],
    delayed_entry: Option<&[f64]>,
    tau: f64,
    endpoint: SurvivalEndpoint,
) -> Result<RandomizedSurvivalSummary, &'static str> {
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
    if let Some(entry) = delayed_entry {
        if entry
            .iter()
            .zip(duration)
            .any(|(&left, &right)| !left.is_finite() || left < 0.0 || left >= right)
        {
            return Err("entry times must be finite, nonnegative, and earlier than exit");
        }
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
        if let Some(entry) = delayed_entry {
            if rows.iter().all(|&index| entry[index] != 0.0) {
                return Err("RMST or incidence from zero requires a time-zero entrant in each arm");
            }
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
/// `censoring_survival` is row-major with one value per subject and time. The
/// probabilities must be known independently of the outcomes used here; this
/// point kernel does not fit or verify a censoring model. No interval is
/// constructed by this function.
///
/// # Errors
///
/// Returns an error for misaligned rows or times, invalid event codes, absent
/// arm support through `tau`, or censoring probabilities that violate the
/// declared positivity and monotonicity contract.
pub fn randomized_survival_ipcw_summary(
    duration: &[f64],
    event_code: &[i64],
    treated: &[bool],
    times: &[f64],
    censoring_survival: &[f64],
    tau: f64,
    minimum_probability: f64,
    endpoint: SurvivalEndpoint,
) -> Result<(RandomizedSurvivalSummary, f64), &'static str> {
    randomized_survival_ipcw_summary_with_entry(
        duration, event_code, treated, None, times, censoring_survival, tau,
        minimum_probability, endpoint,
    )
}

/// Evaluate a fixed-known-censoring curve with marginally independent left entry.
/// Entry and censoring probabilities remain attached to their subject in a
/// subject-resampling interval. Conditional or informative entry is not adjusted.
#[allow(clippy::too_many_arguments)]
pub fn randomized_survival_ipcw_summary_with_entry(
    duration: &[f64],
    event_code: &[i64],
    treated: &[bool],
    delayed_entry: Option<&[f64]>,
    times: &[f64],
    censoring_survival: &[f64],
    tau: f64,
    minimum_probability: f64,
    endpoint: SurvivalEndpoint,
) -> Result<(RandomizedSurvivalSummary, f64), &'static str> {
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
    if delayed_entry.is_some_and(|entry| entry.iter().zip(duration).any(|(&left, &right)| {
        !left.is_finite() || left < 0.0 || left >= right
    })) {
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
        if delayed_entry.is_some_and(|entry| !(0..n).any(|i| treated[i] == arm && entry[i] == 0.0)) {
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
            let at_risk = (0..n).filter(|&i| {
                treated[i] == arm && duration[i] >= time
                    && delayed_entry.is_none_or(|entry| entry[i] < time)
            }).collect::<Vec<_>>();
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
            incidence += survival * target_events as f64 / at_risk as f64;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_bootstrap_known_truth_deterministic_and_refuses_thin_arms() {
        let mut durations = vec![3.0; 80];
        let mut events = vec![0; 80];
        let treated = (0..80).map(|i| i >= 40).collect::<Vec<_>>();
        for i in 0..40 {
            if i % 5 == 0 { durations[i] = 1.0; events[i] = 1; }
        }
        for i in 40..80 {
            if i % 10 == 0 { durations[i] = 1.0; events[i] = 1; }
        }
        let first = randomized_survival_bootstrap_intervals(
            &durations, &events, &treated, None, None, 3.0,
            SurvivalEndpoint::Survival, 399, 17,
        ).unwrap();
        let repeated = randomized_survival_bootstrap_intervals(
            &durations, &events, &treated, None, None, 3.0,
            SurvivalEndpoint::Survival, 399, 17,
        ).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.replicates_ok, 399);
        assert!(first.rmst_difference.unwrap()[0] <= 0.2);
        assert!(first.rmst_difference.unwrap()[1] >= 0.2);
        assert!(randomized_survival_bootstrap_intervals(
            &durations[..8], &events[..8], &treated[..8], None, None, 3.0,
            SurvivalEndpoint::Survival, 399, 17,
        ).is_err());
        assert!(randomized_survival_bootstrap_intervals(
            &durations, &events, &treated, None, None, 3.0,
            SurvivalEndpoint::Survival, 198, 17,
        ).is_err());
        assert!(randomized_survival_bootstrap_intervals(
            &durations, &events, &treated, Some(&vec![0.0; 80]), None, 3.0,
            SurvivalEndpoint::Survival, 399, 17,
        ).is_err());
    }

    #[test]
    fn competing_risk_bootstrap_carries_fixed_known_g_with_subjects() {
        let mut durations = vec![3.0; 80];
        let mut events = vec![0; 80];
        let treated = (0..80).map(|i| i >= 40).collect::<Vec<_>>();
        let mut g = Vec::with_capacity(80 * 3);
        for i in 0..80 {
            if i % 5 == 0 { durations[i] = 1.0; events[i] = 1; }
            else if i % 7 == 0 { durations[i] = 1.0; events[i] = 2; }
            g.extend_from_slice(&[1.0, if i % 2 == 0 { 0.8 } else { 0.9 }, 0.7]);
        }
        let result = randomized_survival_bootstrap_intervals(
            &durations, &events, &treated, None,
            Some((&[0.0, 1.0, 3.0], &g, 0.01)), 3.0,
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 }, 399, 19,
        ).unwrap();
        assert!(result.rmst_difference.is_none());
        assert_eq!(result.replicates_ok, 399);
        assert!(result.difference_at_tau[0] <= 0.0 && result.difference_at_tau[1] >= 0.0);
    }

    #[test]
    fn repeated_sampling_pointwise_bootstrap_covers_simple_survival_truth() {
        // Two-arm discrete event-time law: P(T=1) is .30 versus .15, otherwise
        // T>tau=3. The true RMST and S(3) differences are +.30 and +.15.
        let mut state = 0xBE22_D4E5_95D2_6B63;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(240);
            let mut events = Vec::with_capacity(240);
            let mut treated = Vec::with_capacity(240);
            for i in 0..240 {
                let arm = i >= 120;
                let u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                let failed = u < if arm { 0.15 } else { 0.30 };
                durations.push(if failed { 1.0 } else { 3.0 });
                events.push(i64::from(failed));
                treated.push(arm);
            }
            let intervals = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, None, None, 3.0,
                SurvivalEndpoint::Survival, 299, trial + 700,
            ).unwrap();
            let rmst = intervals.rmst_difference.unwrap();
            covered_rmst += u32::from(rmst[0] <= 0.30 && 0.30 <= rmst[1]);
            let tau = intervals.difference_at_tau;
            covered_tau += u32::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        // At p=.95, 400 trials have a binomial standard error about .011.
        // The .90 floor allows Monte Carlo fluctuation while rejecting material
        // undercoverage for this declared data-generating law.
        assert!(covered_rmst >= 360, "RMST coverage: {covered_rmst}/400");
        assert!(covered_tau >= 360, "S(tau) coverage: {covered_tau}/400");
    }

    #[test]
    fn delayed_entry_subject_bootstrap_covers_left_truncated_survival_truth() {
        // Entry is independent of the source-population event time. A subject
        // entering at one with an event at one is unobserved; those entering
        // at zero identify the first jump, and later risk sets use both entry
        // cohorts. The two arm laws imply S(3) and RMST differences of .15
        // and .25 respectively.
        let mut state = 0xD3E1_A7E0_95D2_6B63;
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut entries = Vec::with_capacity(320);
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 160 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let event_u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let (first, second) = if arm { (0.20, 0.15) } else { (0.30, 0.20) };
                    let (exit, event) = if event_u < first {
                        (1.0, 1)
                    } else if event_u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    if entry >= exit { continue; }
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                    entries.push(entry);
                    observed += 1;
                }
            }
            let intervals = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, Some(&entries), None, 3.0,
                SurvivalEndpoint::Survival, 299, trial + 2_100,
            ).unwrap();
            let rmst = intervals.rmst_difference.unwrap();
            covered_rmst += u32::from(rmst[0] <= 0.25 && 0.25 <= rmst[1]);
            let tau = intervals.difference_at_tau;
            covered_tau += u32::from(tau[0] <= 0.15 && 0.15 <= tau[1]);
        }
        eprintln!("left-truncated survival coverage: RMST {covered_rmst}/400, S(tau) {covered_tau}/400");
        assert!(covered_rmst >= 360, "left-truncated RMST coverage: {covered_rmst}/400");
        assert!(covered_tau >= 360, "left-truncated S(tau) coverage: {covered_tau}/400");
    }

    #[test]
    fn delayed_entry_competing_risk_bootstrap_covers_cif_truth() {
        // Entry at zero or one is independent of event time and cause. People
        // entering at one after a time-one event are absent from the observed
        // sample. Target-cause masses are .15 + .15 in control and .10 + .10
        // in treatment, so the source-population CIF difference at three is
        // -.10; the competing-cause masses are .10 at each event time.
        let mut state = 0xD3E1_C1F0_95D2_6B63;
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut entries = Vec::with_capacity(320);
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 160 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let first_target = if arm { 0.10 } else { 0.15 };
                    let second_target = if arm { 0.10 } else { 0.15 };
                    let (exit, event) = if u < first_target {
                        (1.0, 1)
                    } else if u < first_target + 0.10 {
                        (1.0, 2)
                    } else if u < first_target + 0.10 + second_target {
                        (2.0, 1)
                    } else if u < first_target + 0.20 + second_target {
                        (2.0, 2)
                    } else {
                        (3.0, 0)
                    };
                    if entry >= exit { continue; }
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                    entries.push(entry);
                    observed += 1;
                }
            }
            let interval = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, Some(&entries), None, 3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                299, trial + 8_100,
            ).unwrap().difference_at_tau;
            covered += u32::from(interval[0] <= -0.10 && -0.10 <= interval[1]);
        }
        eprintln!("left-truncated competing-risk CIF coverage: {covered}/400");
        assert!(covered >= 360, "left-truncated CIF coverage: {covered}/400");
    }

    #[test]
    fn delayed_entry_fixed_g_bootstrap_covers_survival_truth() {
        // Independent entry, independently randomized treatment, and known
        // stratum-specific censoring probabilities. Censoring can occur after
        // time one or two; its survival grid is fixed with each subject.
        let mut state = 0xD3E1_F1E0_95D2_6B63;
        let times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut covered_rmst = 0;
        let mut covered_tau = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(400);
            let mut events = Vec::with_capacity(400);
            let mut treated = Vec::with_capacity(400);
            let mut entries = Vec::with_capacity(400);
            let mut g = Vec::with_capacity(400 * times.len());
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 200 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let (first, second) = if arm { (0.20, 0.15) } else { (0.30, 0.20) };
                    let (event_time, event): (f64, i64) = if u < first {
                        (1.0, 1)
                    } else if u < first + second {
                        (2.0, 1)
                    } else {
                        (3.0, 0)
                    };
                    let high_censoring = splitmix64(&mut state) & 1 == 1;
                    let censor_hazard = if high_censoring { 0.20 } else { 0.10 };
                    let c1 = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let c2 = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let censor_time = if c1 < censor_hazard { 1.5 }
                        else if c2 < censor_hazard { 2.5 } else { 3.0 };
                    let exit = event_time.min(censor_time);
                    if entry >= exit { continue; }
                    durations.push(exit);
                    events.push(if event_time <= censor_time { event } else { 0 });
                    treated.push(arm);
                    entries.push(entry);
                    g.extend_from_slice(&[
                        1.0, 1.0, 1.0, 1.0 - censor_hazard,
                        1.0 - censor_hazard, (1.0 - censor_hazard).powi(2),
                    ]);
                    observed += 1;
                }
            }
            let estimate = randomized_survival_ipcw_summary_with_entry(
                &durations, &events, &treated, Some(&entries), &times, &g, 3.0,
                0.01, SurvivalEndpoint::Survival,
            ).unwrap().0;
            assert!(estimate.rmst_control.is_some());
            let interval = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, Some(&entries),
                Some((&times, &g, 0.01)), 3.0, SurvivalEndpoint::Survival,
                299, trial + 9_100,
            ).unwrap();
            let rmst = interval.rmst_difference.unwrap();
            covered_rmst += u32::from(rmst[0] <= 0.25 && 0.25 <= rmst[1]);
            covered_tau += u32::from(interval.difference_at_tau[0] <= 0.15
                && 0.15 <= interval.difference_at_tau[1]);
        }
        eprintln!("left-entry fixed-G survival coverage: RMST {covered_rmst}/400, S(tau) {covered_tau}/400");
        assert!(covered_rmst >= 360, "left-entry fixed-G RMST coverage: {covered_rmst}/400");
        assert!(covered_tau >= 360, "left-entry fixed-G S(tau) coverage: {covered_tau}/400");
    }

    #[test]
    fn delayed_entry_fixed_g_bootstrap_covers_competing_incidence_truth() {
        let mut state = 0xD3E1_F1CF_95D2_6B63;
        let times = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(400);
            let mut events = Vec::with_capacity(400);
            let mut treated = Vec::with_capacity(400);
            let mut entries = Vec::with_capacity(400);
            let mut g = Vec::with_capacity(400 * times.len());
            for arm in [false, true] {
                let mut observed = 0;
                while observed < 200 {
                    let entry = (splitmix64(&mut state) & 1) as f64;
                    let u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let target = if arm { 0.10 } else { 0.15 };
                    let (event_time, event): (f64, i64) = if u < target {
                        (1.0, 1)
                    } else if u < target + 0.10 {
                        (1.0, 2)
                    } else if u < 2.0 * target + 0.10 {
                        (2.0, 1)
                    } else if u < 2.0 * target + 0.20 {
                        (2.0, 2)
                    } else {
                        (3.0, 0)
                    };
                    let censor_hazard = if splitmix64(&mut state) & 1 == 1 { 0.20 } else { 0.10 };
                    let c1 = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let c2 = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let censor_time = if c1 < censor_hazard { 1.5 }
                        else if c2 < censor_hazard { 2.5 } else { 3.0 };
                    let exit = event_time.min(censor_time);
                    if entry >= exit { continue; }
                    durations.push(exit);
                    events.push(if event_time <= censor_time { event } else { 0 });
                    treated.push(arm);
                    entries.push(entry);
                    g.extend_from_slice(&[
                        1.0, 1.0, 1.0, 1.0 - censor_hazard,
                        1.0 - censor_hazard, (1.0 - censor_hazard).powi(2),
                    ]);
                    observed += 1;
                }
            }
            let interval = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, Some(&entries),
                Some((&times, &g, 0.01)), 3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                299, trial + 10_100,
            ).unwrap().difference_at_tau;
            covered += u32::from(interval[0] <= -0.10 && -0.10 <= interval[1]);
        }
        eprintln!("left-entry fixed-G competing-risk CIF coverage: {covered}/400");
        assert!(covered >= 360, "left-entry fixed-G CIF coverage: {covered}/400");
    }

    #[test]
    fn simultaneous_survival_difference_band_covers_entire_event_grid() {
        let mut state = 0xB41D_95A1_5EED_2026;
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            for arm in [false, true] {
                for _ in 0..160 {
                    let u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let (first, second) = if arm { (0.15, 0.10) } else { (0.30, 0.20) };
                    let (exit, event) = if u < first { (1.0, 1) }
                        else if u < first + second { (2.0, 1) }
                        else { (3.0, 0) };
                    durations.push(exit);
                    events.push(event);
                    treated.push(arm);
                }
            }
            let band = randomized_survival_bootstrap_difference_band(
                &durations, &events, &treated, 3.0,
                SurvivalEndpoint::Survival, 399, trial + 4_400,
            ).unwrap();
            let covers = band.times.iter().enumerate().all(|(i, &time)| {
                let truth = if time < 1.0 { 0.0 } else if time < 2.0 { 0.15 } else { 0.25 };
                band.lower[i] <= truth && truth <= band.upper[i]
            });
            covered += u32::from(covers);
        }
        eprintln!("simultaneous survival-grid band coverage: {covered}/400");
        assert!(covered >= 360, "simultaneous survival-grid coverage: {covered}/400");
        assert!(randomized_survival_bootstrap_difference_band(
            &[1.0, 3.0, 1.0, 3.0], &[1, 0, 1, 0], &[false, false, true, true],
            3.0, SurvivalEndpoint::Survival, 399, 7,
        ).unwrap_err().contains("80 subjects"));
        let duration = (0..160).map(|i| if i % 5 == 0 { 1.0 } else { 3.0 }).collect::<Vec<_>>();
        let event = duration.iter().map(|time| i64::from(*time == 1.0)).collect::<Vec<_>>();
        let treated = (0..160).map(|i| i >= 80).collect::<Vec<_>>();
        let first = randomized_survival_bootstrap_difference_band(
            &duration, &event, &treated, 3.0, SurvivalEndpoint::Survival, 399, 42,
        ).unwrap();
        assert_eq!(first, randomized_survival_bootstrap_difference_band(
            &duration, &event, &treated, 3.0, SurvivalEndpoint::Survival, 399, 42,
        ).unwrap());
    }

    #[test]
    fn simultaneous_competing_incidence_band_covers_entire_event_grid() {
        let mut state = 0xC1F0_95A1_5EED_2026;
        let mut covered = 0;
        for trial in 0..400 {
            let mut durations = Vec::with_capacity(320);
            let mut causes = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            for arm in [false, true] {
                for _ in 0..160 {
                    let u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                    let (first_target, second_target) = if arm { (0.30, 0.15) } else { (0.20, 0.10) };
                    let (exit, cause) = if u < first_target { (1.0, 1) }
                        else if u < first_target + 0.10 { (1.0, 2) }
                        else if u < first_target + 0.10 + second_target { (2.0, 1) }
                        else if u < first_target + 0.20 + second_target { (2.0, 2) }
                        else { (3.0, 0) };
                    durations.push(exit);
                    causes.push(cause);
                    treated.push(arm);
                }
            }
            let band = randomized_survival_bootstrap_difference_band(
                &durations, &causes, &treated, 3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
                399, trial + 6_400,
            ).unwrap();
            let covers = band.times.iter().enumerate().all(|(i, &time)| {
                let truth = if time < 1.0 { 0.0 } else if time < 2.0 { 0.10 } else { 0.15 };
                band.lower[i] <= truth && truth <= band.upper[i]
            });
            covered += u32::from(covers);
        }
        eprintln!("simultaneous competing-incidence grid band coverage: {covered}/400");
        assert!(covered >= 360, "simultaneous competing-incidence coverage: {covered}/400");
    }

    #[test]
    fn repeated_sampling_fixed_g_bootstrap_covers_survival_truth() {
        // Known censoring is deliberately heterogeneous and held with the
        // resampled subject. Event risk is randomized and independent of G.
        let mut state = 0xE043_DA75_88C3_9839;
        let mut covered = 0;
        for trial in 0..240 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut g = Vec::with_capacity(320 * 4);
            for i in 0..320 {
                let arm = i >= 160;
                let keep_probability = if i % 2 == 0 { 0.9 } else { 0.7 };
                let censor_u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                let event_u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                let censored = censor_u >= keep_probability;
                let failed = event_u < if arm { 0.15 } else { 0.30 };
                durations.push(if censored { 0.5 } else if failed { 1.0 } else { 3.0 });
                events.push(i64::from(!censored && failed));
                treated.push(arm);
                g.extend_from_slice(&[1.0, keep_probability, keep_probability, keep_probability]);
            }
            let interval = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, None,
                Some((&[0.0, 0.5, 1.0, 3.0], &g, 0.01)), 3.0,
                SurvivalEndpoint::Survival, 299, trial + 1700,
            ).unwrap().rmst_difference.unwrap();
            covered += u32::from(interval[0] <= 0.30 && 0.30 <= interval[1]);
        }
        assert!(covered >= 216, "fixed-G IPCW RMST coverage: {covered}/240");
    }

    #[test]
    fn repeated_sampling_competing_risk_bootstrap_covers_cif_truth() {
        let mut state = 0x8120_D48E_70BA_19D3;
        let mut covered = 0;
        for trial in 0..240 {
            let mut durations = Vec::with_capacity(320);
            let mut events = Vec::with_capacity(320);
            let mut treated = Vec::with_capacity(320);
            let mut g = Vec::with_capacity(320 * 5);
            for i in 0..320 {
                let arm = i >= 160;
                let keep_probability = if i % 2 == 0 { 0.9 } else { 0.7 };
                let censor_u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                let cause_u = (splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
                let censored = censor_u >= keep_probability;
                let first_target = if arm { 0.10 } else { 0.20 };
                let first_competing = first_target + 0.10;
                let later_target = first_competing + 0.10;
                let later_competing = later_target + 0.10;
                let (event_time, cause) = if cause_u < first_target {
                    (1.0, 1)
                } else if cause_u < first_competing {
                    (1.0, 2)
                } else if cause_u < later_target {
                    (2.0, 1)
                } else if cause_u < later_competing {
                    (2.0, 2)
                } else {
                    (3.0, 0)
                };
                durations.push(if censored { 0.5 } else { event_time });
                events.push(if censored { 0 } else { cause });
                treated.push(arm);
                g.extend_from_slice(&[1.0, keep_probability, keep_probability, keep_probability, keep_probability]);
            }
            let interval = randomized_survival_bootstrap_intervals(
                &durations, &events, &treated, None,
                Some((&[0.0, 0.5, 1.0, 2.0, 3.0], &g, 0.01)), 3.0,
                SurvivalEndpoint::CumulativeIncidence { target_cause: 1 }, 299, trial + 2700,
            ).unwrap().difference_at_tau;
            covered += u32::from(interval[0] <= -0.10 && -0.10 <= interval[1]);
        }
        assert!(covered >= 216, "fixed-G competing-risk CIF coverage: {covered}/240");
    }

    #[test]
    fn known_censoring_ipcw_survival_matches_unweighted_truth_when_g_is_one() {
        let (summary, minimum_g) = randomized_survival_ipcw_summary(
            &[1.0, 2.0, 2.0, 2.0],
            &[1, 0, 0, 0],
            &[false, false, true, true],
            &[0.0, 1.0, 2.0],
            &[1.0; 12],
            2.0,
            0.1,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        assert_eq!(summary.control, [1.0, 0.5, 0.5]);
        assert_eq!(summary.treated, [1.0, 1.0, 1.0]);
        assert_eq!(summary.rmst_control, Some(1.5));
        assert_eq!(summary.rmst_treated, Some(2.0));
        assert_eq!(minimum_g, 1.0);
    }

    #[test]
    fn known_censoring_ipcw_refuses_positivity_failure() {
        let mut g = [1.0; 12];
        g[4] = 0.05;
        assert!(
            randomized_survival_ipcw_summary(
                &[1.0, 2.0, 2.0, 2.0],
                &[1, 0, 0, 0],
                &[false, false, true, true],
                &[0.0, 1.0, 2.0],
                &g,
                2.0,
                0.1,
                SurvivalEndpoint::Survival,
            )
            .is_err()
        );
    }

    #[test]
    fn known_censoring_ipcw_competing_risk_known_truth() {
        let (summary, _) = randomized_survival_ipcw_summary(
            &[1.0, 2.0, 2.0, 1.0, 2.0, 2.0],
            &[1, 2, 0, 1, 2, 0],
            &[false, false, false, true, true, true],
            &[0.0, 1.0, 2.0],
            &[1.0; 18],
            2.0,
            0.1,
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
        )
        .unwrap();
        assert_eq!(summary.control, [0.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(summary.treated, [0.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(summary.rmst_control, None);
        assert_eq!(summary.rmst_treated, None);
    }

    #[test]
    fn randomized_rmst_and_survival_known_truth() {
        let result = randomized_survival_summary(
            &[1.0, 2.0, 2.0, 2.0],
            &[1, 0, 0, 0],
            &[false, false, true, true],
            None,
            2.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        assert_eq!(result.times, [0.0, 1.0, 2.0]);
        assert_eq!(result.control, [1.0, 0.5, 0.5]);
        assert_eq!(result.treated, [1.0, 1.0, 1.0]);
        assert_eq!(result.rmst_control, Some(1.5));
        assert_eq!(result.rmst_treated, Some(2.0));
    }

    #[test]
    fn delayed_entry_risk_sets_and_competing_incidence_known_truth() {
        let survival = randomized_survival_summary(
            &[1.0, 2.0, 2.0, 2.0],
            &[1, 0, 0, 0],
            &[false, false, true, true],
            Some(&[0.0, 0.5, 0.0, 0.5]),
            2.0,
            SurvivalEndpoint::Survival,
        )
        .unwrap();
        assert_eq!(survival.rmst_control, Some(1.5));
        assert_eq!(survival.minimum_event_risk_set, Some(2));

        let incidence = randomized_survival_summary(
            &[1.0, 2.0, 1.0, 2.0],
            &[1, 2, 1, 1],
            &[false, false, true, true],
            None,
            2.0,
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
        )
        .unwrap();
        assert_eq!(incidence.control, [0.0, 0.5, 0.5]);
        assert_eq!(incidence.treated, [0.0, 0.5, 1.0]);
        assert_eq!(incidence.rmst_control, None);
    }

    #[test]
    fn refuses_unsupported_risk_sets_and_codes() {
        assert!(
            randomized_survival_summary(
                &[1.0, 2.0],
                &[1, 2],
                &[false, true],
                None,
                2.0,
                SurvivalEndpoint::Survival,
            )
            .is_err()
        );
        assert!(
            randomized_survival_summary(
                &[1.0, 2.0],
                &[1, 0],
                &[false, true],
                Some(&[0.0, 2.0]),
                2.0,
                SurvivalEndpoint::Survival,
            )
            .is_err()
        );
    }
}
