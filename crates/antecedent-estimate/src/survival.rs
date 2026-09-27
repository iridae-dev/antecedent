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
/// point kernel does not fit or verify a censoring model. Delayed entry is not
/// supported by this weighted estimator. No interval is constructed.
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
    let n = duration.len();
    let m = times.len();
    if n == 0
        || event_code.len() != n
        || treated.len() != n
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
            let at_risk =
                (0..n).filter(|&i| treated[i] == arm && duration[i] >= time).collect::<Vec<_>>();
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
