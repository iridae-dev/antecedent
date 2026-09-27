//! Point-only randomized survival summaries with right censoring.
// SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Retained randomized survival or competing-risk result from Study execution.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct SurvivalSection {
    pub times: Vec<f64>,
    pub control: Vec<f64>,
    pub treated: Vec<f64>,
    pub rmst_control: Option<f64>,
    pub rmst_treated: Option<f64>,
    pub target_cause: Option<i64>,
    pub tau: f64,
    pub minimum_event_risk_set: Option<usize>,
    pub uncertainty: String,
    pub rmst_difference_interval: Option<Vec<f64>>,
    pub difference_at_tau_interval: Option<Vec<f64>>,
    pub bootstrap_replicates_requested: Option<u32>,
    pub bootstrap_replicates_ok: Option<u32>,
    pub censoring_survival_provenance: Option<String>,
    pub difference_band: Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, u32)>,
    pub band_unavailable_reason: Option<String>,
}

impl From<&antecedent::SurvivalEstimate> for SurvivalSection {
    fn from(value: &antecedent::SurvivalEstimate) -> Self {
        Self {
            times: value.times.to_vec(),
            control: value.control.to_vec(),
            treated: value.treated.to_vec(),
            rmst_control: value.rmst_control,
            rmst_treated: value.rmst_treated,
            target_cause: value.target_cause,
            tau: value.tau,
            minimum_event_risk_set: value.minimum_event_risk_set,
            uncertainty: value.uncertainty.to_string(),
            rmst_difference_interval: value.rmst_difference_interval.map(|interval| interval.to_vec()),
            difference_at_tau_interval: value.difference_at_tau_interval.map(|interval| interval.to_vec()),
            bootstrap_replicates_requested: value.bootstrap_replicates_requested,
            bootstrap_replicates_ok: value.bootstrap_replicates_ok,
            censoring_survival_provenance: value.censoring_survival_provenance.as_ref().map(ToString::to_string),
            difference_band: value.difference_band.as_ref().map(|band| (
                band.times.to_vec(), band.difference.to_vec(), band.lower.to_vec(),
                band.upper.to_vec(), band.replicates_ok,
            )),
            band_unavailable_reason: value.band_unavailable_reason.as_ref().map(ToString::to_string),
        }
    }
}

fn km_curve(times: &[f64], events: &[bool], tau: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let mut event_times: Vec<f64> = times
        .iter()
        .zip(events)
        .filter_map(|(&time, &event)| (event && time <= tau).then_some(time))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup_by(|a, b| *a == *b);

    let mut survival = 1.0;
    let mut previous = 0.0;
    let mut rmst = 0.0;
    let mut curve_times = vec![0.0];
    let mut curve_survival = vec![1.0];
    for time in event_times {
        rmst += (time - previous) * survival;
        let at_risk = times.iter().filter(|&&duration| duration >= time).count();
        let failures = times
            .iter()
            .zip(events)
            .filter(|(duration, event)| **duration == time && **event)
            .count();
        survival *= 1.0 - failures as f64 / at_risk as f64;
        curve_times.push(time);
        curve_survival.push(survival);
        previous = time;
    }
    rmst += (tau - previous) * survival;
    (curve_times, curve_survival, rmst)
}

fn delayed_entry_km_curve(
    times: &[f64],
    entry: &[f64],
    events: &[bool],
    tau: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut event_times: Vec<f64> = times
        .iter()
        .zip(events)
        .filter_map(|(&time, &event)| (event && time <= tau).then_some(time))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup_by(|a, b| *a == *b);

    let mut survival = 1.0;
    let mut previous = 0.0;
    let mut rmst = 0.0;
    let mut curve_times = vec![0.0];
    let mut curve_survival = vec![1.0];
    for time in event_times {
        rmst += (time - previous) * survival;
        let at_risk = times
            .iter()
            .zip(entry)
            .filter(|(duration, entered)| **entered < time && **duration >= time)
            .count();
        let failures = times
            .iter()
            .zip(events)
            .filter(|(duration, event)| **duration == time && **event)
            .count();
        survival *= 1.0 - failures as f64 / at_risk as f64;
        curve_times.push(time);
        curve_survival.push(survival);
        previous = time;
    }
    rmst += (tau - previous) * survival;
    (curve_times, curve_survival, rmst)
}

fn delayed_entry_cif_curve(
    times: &[f64],
    entry: &[f64],
    causes: &[i64],
    tau: f64,
    target_cause: i64,
) -> (Vec<f64>, Vec<f64>) {
    let mut event_times: Vec<f64> = times
        .iter()
        .zip(causes)
        .filter_map(|(&time, &cause)| (cause > 0 && time <= tau).then_some(time))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup_by(|a, b| *a == *b);

    let mut survival = 1.0;
    let mut incidence = 0.0;
    let mut curve_times = vec![0.0];
    let mut curve_incidence = vec![0.0];
    for time in event_times {
        let at_risk = times
            .iter()
            .zip(entry)
            .filter(|(duration, entered)| **entered < time && **duration >= time)
            .count();
        let all_events = times
            .iter()
            .zip(causes)
            .filter(|(duration, cause)| **duration == time && **cause > 0)
            .count();
        let target_events = times
            .iter()
            .zip(causes)
            .filter(|(duration, cause)| **duration == time && **cause == target_cause)
            .count();
        incidence += survival * target_events as f64 / at_risk as f64;
        survival *= 1.0 - all_events as f64 / at_risk as f64;
        curve_times.push(time);
        curve_incidence.push(incidence);
    }
    (curve_times, curve_incidence)
}

fn validate_delayed_entry(
    duration: &[f64],
    entry: &[f64],
    treated: &[bool],
    tau: f64,
) -> PyResult<()> {
    if duration.is_empty() || entry.len() != duration.len() || treated.len() != duration.len() {
        return Err(PyValueError::new_err(
            "duration, delayed_entry, and treatment vectors must have equal non-zero length",
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(PyValueError::new_err("tau must be finite and positive"));
    }
    if duration.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(PyValueError::new_err("durations must be finite and non-negative"));
    }
    if entry.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(PyValueError::new_err("delayed_entry times must be finite and non-negative"));
    }
    if entry.iter().zip(duration).any(|(entered, observed)| entered >= observed) {
        return Err(PyValueError::new_err(
            "delayed_entry must be strictly earlier than each observed duration",
        ));
    }
    for arm in [false, true] {
        let indices: Vec<usize> = treated
            .iter()
            .enumerate()
            .filter_map(|(i, &is_treated)| (is_treated == arm).then_some(i))
            .collect();
        if indices.is_empty() {
            return Err(PyValueError::new_err(
                "delayed-entry survival requires units in both treatment arms",
            ));
        }
        if !indices.iter().any(|&i| entry[i] == 0.0) {
            return Err(PyValueError::new_err(
                "delayed-entry RMST from time zero requires at least one time-zero entrant in each arm",
            ));
        }
        if !indices.iter().any(|&i| duration[i] >= tau) {
            return Err(PyValueError::new_err(
                "tau must not exceed the observed follow-up horizon in either arm",
            ));
        }
    }
    Ok(())
}

/// IPCW weighted product-limit curves using caller supplied censoring survival.
#[pyfunction]
fn randomized_survival_ipcw(
    duration: PyReadonlyArray1<'_, f64>,
    event_observed: Vec<bool>,
    treated: Vec<bool>,
    times: Vec<f64>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
    tau: f64,
    minimum_probability: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, Option<usize>, Option<usize>, f64)> {
    let duration = duration.as_array();
    let g = censoring_survival.as_array();
    let n = duration.len();
    if n == 0 || event_observed.len() != n || treated.len() != n || g.nrows() != n {
        return Err(PyValueError::new_err(
            "duration, event_observed, treatment, and censoring survival rows must have equal non-zero length",
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(PyValueError::new_err("tau must be finite and positive"));
    }
    if !minimum_probability.is_finite() || minimum_probability <= 0.0 || minimum_probability > 1.0 {
        return Err(PyValueError::new_err("minimum_probability must be finite and in (0, 1]"));
    }
    if duration.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(PyValueError::new_err("durations must be finite and non-negative"));
    }
    if times.len() < 2
        || times.len() != g.ncols()
        || times[0] != 0.0
        || (times[times.len() - 1] - tau).abs() > 1e-10
        || times.iter().any(|time| !time.is_finite())
        || times.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(PyValueError::new_err(
            "times must be a strictly increasing grid from zero through tau and match censoring_survival columns",
        ));
    }
    if (0..n).any(|i| duration[i] <= tau && !times.iter().any(|time| *time == duration[i])) {
        return Err(PyValueError::new_err(
            "times must include every observed duration at or before tau",
        ));
    }
    let mut minimum_g = 1.0_f64;
    for i in 0..n {
        let row = g.row(i);
        if row
            .iter()
            .any(|value| !value.is_finite() || *value < minimum_probability || *value > 1.0)
        {
            return Err(PyValueError::new_err(
                "censoring survival values violate the declared positivity floor or lie outside (0, 1]",
            ));
        }
        if (row[0] - 1.0).abs() > 1e-10
            || row.windows(2).into_iter().any(|pair| pair[1] > pair[0] + 1e-10)
        {
            return Err(PyValueError::new_err(
                "censoring survival rows must start at one and be non-increasing",
            ));
        }
        minimum_g = minimum_g.min(row.iter().copied().fold(1.0_f64, f64::min));
    }
    for arm in [false, true] {
        if !(0..n).any(|i| treated[i] == arm) {
            return Err(PyValueError::new_err(
                "IPCW survival requires units in both treatment arms",
            ));
        }
        if !(0..n).any(|i| treated[i] == arm && duration[i] >= tau) {
            return Err(PyValueError::new_err(
                "tau must not exceed the observed follow-up horizon in either arm",
            ));
        }
    }
    let event_codes = event_observed.iter().map(|event| i64::from(*event)).collect::<Vec<_>>();
    let (summary, kernel_minimum_g) =
        antecedent_estimate::survival::randomized_survival_ipcw_summary(
            &duration.to_vec(),
            &event_codes,
            &treated,
            &times,
            &g.iter().copied().collect::<Vec<_>>(),
            tau,
            minimum_probability,
            antecedent_estimate::survival::SurvivalEndpoint::Survival,
        )
        .map_err(PyValueError::new_err)?;
    debug_assert!((minimum_g - kernel_minimum_g).abs() <= 1e-10);
    Ok((
        summary.times,
        summary.control,
        summary.treated,
        summary.rmst_control.expect("survival summary has RMST"),
        summary.rmst_treated.expect("survival summary has RMST"),
        summary.minimum_event_risk_set_by_arm[0],
        summary.minimum_event_risk_set_by_arm[1],
        kernel_minimum_g,
    ))
}

/// IPCW Aalen-Johansen cumulative incidence with caller-supplied censoring survival.
#[pyfunction]
fn randomized_cumulative_incidence_ipcw(
    duration: PyReadonlyArray1<'_, f64>,
    event_cause: Vec<i64>,
    treated: Vec<bool>,
    times: Vec<f64>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
    tau: f64,
    target_cause: i64,
    minimum_probability: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Option<usize>, Option<usize>, f64)> {
    let duration = duration.as_array();
    let g = censoring_survival.as_array();
    let n = duration.len();
    if n == 0 || event_cause.len() != n || treated.len() != n || g.nrows() != n {
        return Err(PyValueError::new_err(
            "duration, event_cause, treatment, and censoring survival rows must have equal non-zero length",
        ));
    }
    if !tau.is_finite() || tau <= 0.0 || target_cause <= 0 {
        return Err(PyValueError::new_err(
            "tau must be positive and target_cause must be a positive event code",
        ));
    }
    if !minimum_probability.is_finite() || minimum_probability <= 0.0 || minimum_probability > 1.0 {
        return Err(PyValueError::new_err("minimum_probability must be finite and in (0, 1]"));
    }
    if duration.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(PyValueError::new_err("durations must be finite and non-negative"));
    }
    if event_cause.iter().any(|&cause| cause < 0) {
        return Err(PyValueError::new_err(
            "event_cause codes must be non-negative; zero denotes censoring",
        ));
    }
    let causes: std::collections::BTreeSet<i64> =
        event_cause.iter().copied().filter(|&c| c > 0).collect();
    if causes.len() < 2 || !causes.contains(&target_cause) {
        return Err(PyValueError::new_err(
            "at least two event causes must be observed and target_cause must be among them",
        ));
    }
    if times.len() < 2
        || times.len() != g.ncols()
        || times[0] != 0.0
        || (times[times.len() - 1] - tau).abs() > 1e-10
        || times.iter().any(|t| !t.is_finite())
        || times.windows(2).any(|w| w[1] <= w[0])
    {
        return Err(PyValueError::new_err(
            "times must be a strictly increasing grid from zero through tau and match censoring_survival columns",
        ));
    }
    if (0..n).any(|i| duration[i] <= tau && !times.iter().any(|t| *t == duration[i])) {
        return Err(PyValueError::new_err(
            "times must include every observed duration at or before tau",
        ));
    }
    let mut minimum_g = 1.0_f64;
    for i in 0..n {
        let row = g.row(i);
        if row.iter().any(|v| !v.is_finite() || *v < minimum_probability || *v > 1.0) {
            return Err(PyValueError::new_err(
                "censoring survival values violate the declared positivity floor or lie outside (0, 1]",
            ));
        }
        if (row[0] - 1.0).abs() > 1e-10 || row.windows(2).into_iter().any(|w| w[1] > w[0] + 1e-10) {
            return Err(PyValueError::new_err(
                "censoring survival rows must start at one and be non-increasing",
            ));
        }
        minimum_g = minimum_g.min(row.iter().copied().fold(1.0_f64, f64::min));
    }
    for arm in [false, true] {
        if !(0..n).any(|i| treated[i] == arm)
            || !(0..n).any(|i| treated[i] == arm && duration[i] >= tau)
        {
            return Err(PyValueError::new_err(
                "IPCW cumulative incidence requires both arms observed through tau",
            ));
        }
    }
    let (summary, kernel_minimum_g) =
        antecedent_estimate::survival::randomized_survival_ipcw_summary(
            &duration.to_vec(),
            &event_cause,
            &treated,
            &times,
            &g.iter().copied().collect::<Vec<_>>(),
            tau,
            minimum_probability,
            antecedent_estimate::survival::SurvivalEndpoint::CumulativeIncidence { target_cause },
        )
        .map_err(PyValueError::new_err)?;
    debug_assert!((minimum_g - kernel_minimum_g).abs() <= 1e-10);
    Ok((
        summary.times,
        summary.control,
        summary.treated,
        summary.minimum_event_risk_set_by_arm[0],
        summary.minimum_event_risk_set_by_arm[1],
        kernel_minimum_g,
    ))
}

/// Two-arm randomized Kaplan-Meier curves and restricted mean survival time.
#[pyfunction]
fn randomized_survival(
    duration: PyReadonlyArray1<'_, f64>,
    event_observed: Vec<bool>,
    treated: Vec<bool>,
    tau: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> {
    let duration = duration.as_array();
    let n = duration.len();
    if n == 0 || event_observed.len() != n || treated.len() != n {
        return Err(PyValueError::new_err(
            "duration, event_observed, and treated must have equal non-zero length",
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(PyValueError::new_err("tau must be finite and positive"));
    }
    if duration.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(PyValueError::new_err("durations must be finite and non-negative"));
    }
    let mut t0 = Vec::new();
    let mut e0 = Vec::new();
    let mut t1 = Vec::new();
    let mut e1 = Vec::new();
    for i in 0..n {
        if treated[i] {
            t1.push(duration[i]);
            e1.push(event_observed[i]);
        } else {
            t0.push(duration[i]);
            e0.push(event_observed[i]);
        }
    }
    if t0.is_empty() || t1.is_empty() {
        return Err(PyValueError::new_err(
            "randomized survival requires observed units in both treatment arms",
        ));
    }
    if t0.iter().copied().fold(0.0_f64, f64::max) < tau
        || t1.iter().copied().fold(0.0_f64, f64::max) < tau
    {
        return Err(PyValueError::new_err(
            "tau must not exceed the observed follow-up horizon in either arm",
        ));
    }
    let (times0, values0, rmst0) = km_curve(&t0, &e0, tau);
    let (times1, values1, rmst1) = km_curve(&t1, &e1, tau);

    // Return step-function values on the union of event times, with zero included.
    let mut grid = times0.clone();
    grid.extend(times1.iter().copied());
    grid.push(tau);
    grid.sort_by(f64::total_cmp);
    grid.dedup_by(|a, b| *a == *b);
    let step_values = |arm_times: &[f64], arm_values: &[f64]| -> Vec<f64> {
        grid.iter()
            .map(|&time| {
                let index = arm_times.partition_point(|&event_time| event_time <= time);
                arm_values[index.saturating_sub(1)]
            })
            .collect()
    };
    // At a time-zero event, the post-event value is the final duplicate.
    let aligned0 = step_values(&times0, &values0);
    let aligned1 = step_values(&times1, &values1);
    Ok((grid, aligned0, aligned1, rmst0, rmst1))
}

/// Left-truncated two-arm randomized Kaplan-Meier curves and RMST.
#[pyfunction]
fn randomized_survival_delayed_entry(
    duration: PyReadonlyArray1<'_, f64>,
    delayed_entry: PyReadonlyArray1<'_, f64>,
    event_observed: Vec<bool>,
    treated: Vec<bool>,
    tau: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> {
    let duration = duration.as_array().to_vec();
    let entry = delayed_entry.as_array().to_vec();
    validate_delayed_entry(&duration, &entry, &treated, tau)?;
    if event_observed.len() != duration.len() {
        return Err(PyValueError::new_err(
            "duration, delayed_entry, event_observed, and treatment vectors must have equal length",
        ));
    }
    let mut t0 = Vec::new();
    let mut l0 = Vec::new();
    let mut e0 = Vec::new();
    let mut t1 = Vec::new();
    let mut l1 = Vec::new();
    let mut e1 = Vec::new();
    for i in 0..duration.len() {
        if treated[i] {
            t1.push(duration[i]);
            l1.push(entry[i]);
            e1.push(event_observed[i]);
        } else {
            t0.push(duration[i]);
            l0.push(entry[i]);
            e0.push(event_observed[i]);
        }
    }
    let (times0, values0, rmst0) = delayed_entry_km_curve(&t0, &l0, &e0, tau);
    let (times1, values1, rmst1) = delayed_entry_km_curve(&t1, &l1, &e1, tau);
    let mut grid = times0.clone();
    grid.extend(times1.clone());
    grid.push(tau);
    grid.sort_by(f64::total_cmp);
    grid.dedup_by(|a, b| *a == *b);
    let align = |arm_times: &[f64], arm_values: &[f64]| -> Vec<f64> {
        grid.iter()
            .map(|&time| {
                let index = arm_times.partition_point(|&event_time| event_time <= time);
                arm_values[index.saturating_sub(1)]
            })
            .collect()
    };
    let aligned0 = align(&times0, &values0);
    let aligned1 = align(&times1, &values1);
    Ok((grid, aligned0, aligned1, rmst0, rmst1))
}

fn cumulative_incidence_curve(
    times: &[f64],
    causes: &[i64],
    tau: f64,
    target_cause: i64,
) -> (Vec<f64>, Vec<f64>) {
    let mut event_times: Vec<f64> = times
        .iter()
        .zip(causes)
        .filter_map(|(&time, &cause)| (cause > 0 && time <= tau).then_some(time))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup_by(|a, b| *a == *b);

    let mut survival = 1.0;
    let mut incidence = 0.0;
    let mut curve_times = vec![0.0];
    let mut curve_incidence = vec![0.0];
    for time in event_times {
        let at_risk = times.iter().filter(|&&duration| duration >= time).count();
        let all_events = times
            .iter()
            .zip(causes)
            .filter(|(duration, cause)| **duration == time && **cause > 0)
            .count();
        let target_events = times
            .iter()
            .zip(causes)
            .filter(|(duration, cause)| **duration == time && **cause == target_cause)
            .count();
        incidence += survival * target_events as f64 / at_risk as f64;
        survival *= 1.0 - all_events as f64 / at_risk as f64;
        curve_times.push(time);
        curve_incidence.push(incidence);
    }
    (curve_times, curve_incidence)
}

/// Cause-specific cumulative incidence in a two-arm randomized study.
#[pyfunction]
fn randomized_cumulative_incidence(
    duration: PyReadonlyArray1<'_, f64>,
    event_cause: Vec<i64>,
    treated: Vec<bool>,
    tau: f64,
    target_cause: i64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let duration = duration.as_array();
    let n = duration.len();
    if n == 0 || event_cause.len() != n || treated.len() != n {
        return Err(PyValueError::new_err(
            "duration, event_cause, and treated must have equal non-zero length",
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(PyValueError::new_err("tau must be finite and positive"));
    }
    if target_cause <= 0 {
        return Err(PyValueError::new_err("target_cause must be a positive event code"));
    }
    if duration.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(PyValueError::new_err("durations must be finite and non-negative"));
    }
    if event_cause.iter().any(|&cause| cause < 0) {
        return Err(PyValueError::new_err(
            "event_cause codes must be non-negative; zero denotes censoring",
        ));
    }
    let distinct_causes: std::collections::BTreeSet<i64> =
        event_cause.iter().copied().filter(|&cause| cause > 0).collect();
    if distinct_causes.len() < 2 {
        return Err(PyValueError::new_err(
            "cumulative incidence requires at least two observed event-cause codes",
        ));
    }
    if !distinct_causes.contains(&target_cause) {
        return Err(PyValueError::new_err(
            "target_cause must occur among the observed event-cause codes",
        ));
    }

    let mut t0 = Vec::new();
    let mut c0 = Vec::new();
    let mut t1 = Vec::new();
    let mut c1 = Vec::new();
    for i in 0..n {
        if treated[i] {
            t1.push(duration[i]);
            c1.push(event_cause[i]);
        } else {
            t0.push(duration[i]);
            c0.push(event_cause[i]);
        }
    }
    if t0.is_empty() || t1.is_empty() {
        return Err(PyValueError::new_err(
            "randomized cumulative incidence requires observed units in both treatment arms",
        ));
    }
    if t0.iter().copied().fold(0.0_f64, f64::max) < tau
        || t1.iter().copied().fold(0.0_f64, f64::max) < tau
    {
        return Err(PyValueError::new_err(
            "tau must not exceed the observed follow-up horizon in either arm",
        ));
    }
    let (times0, values0) = cumulative_incidence_curve(&t0, &c0, tau, target_cause);
    let (times1, values1) = cumulative_incidence_curve(&t1, &c1, tau, target_cause);
    let mut grid = times0.clone();
    grid.extend(times1.iter().copied());
    grid.push(tau);
    grid.sort_by(f64::total_cmp);
    grid.dedup_by(|a, b| *a == *b);
    let align = |arm_times: &[f64], arm_values: &[f64]| -> Vec<f64> {
        grid.iter()
            .map(|&time| {
                let index = arm_times.partition_point(|&event_time| event_time <= time);
                arm_values[index.saturating_sub(1)]
            })
            .collect()
    };
    Ok((grid.clone(), align(&times0, &values0), align(&times1, &values1)))
}

/// Left-truncated competing-risk cumulative incidence in randomized arms.
#[pyfunction]
fn randomized_cumulative_incidence_delayed_entry(
    duration: PyReadonlyArray1<'_, f64>,
    delayed_entry: PyReadonlyArray1<'_, f64>,
    event_cause: Vec<i64>,
    treated: Vec<bool>,
    tau: f64,
    target_cause: i64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let duration = duration.as_array().to_vec();
    let entry = delayed_entry.as_array().to_vec();
    validate_delayed_entry(&duration, &entry, &treated, tau)?;
    if event_cause.len() != duration.len() {
        return Err(PyValueError::new_err(
            "duration, delayed_entry, event_cause, and treatment vectors must have equal length",
        ));
    }
    if target_cause <= 0 {
        return Err(PyValueError::new_err("target_cause must be a positive event code"));
    }
    if event_cause.iter().any(|&cause| cause < 0) {
        return Err(PyValueError::new_err(
            "event_cause codes must be non-negative; zero denotes censoring",
        ));
    }
    let distinct_causes: std::collections::BTreeSet<i64> =
        event_cause.iter().copied().filter(|&cause| cause > 0).collect();
    if distinct_causes.len() < 2 {
        return Err(PyValueError::new_err(
            "cumulative incidence requires at least two observed event-cause codes",
        ));
    }
    if !distinct_causes.contains(&target_cause) {
        return Err(PyValueError::new_err(
            "target_cause must occur among the observed event-cause codes",
        ));
    }
    let mut t0 = Vec::new();
    let mut l0 = Vec::new();
    let mut c0 = Vec::new();
    let mut t1 = Vec::new();
    let mut l1 = Vec::new();
    let mut c1 = Vec::new();
    for i in 0..duration.len() {
        if treated[i] {
            t1.push(duration[i]);
            l1.push(entry[i]);
            c1.push(event_cause[i]);
        } else {
            t0.push(duration[i]);
            l0.push(entry[i]);
            c0.push(event_cause[i]);
        }
    }
    let (times0, values0) = delayed_entry_cif_curve(&t0, &l0, &c0, tau, target_cause);
    let (times1, values1) = delayed_entry_cif_curve(&t1, &l1, &c1, tau, target_cause);
    let mut grid = times0.clone();
    grid.extend(times1.clone());
    grid.push(tau);
    grid.sort_by(f64::total_cmp);
    grid.dedup_by(|a, b| *a == *b);
    let align = |arm_times: &[f64], arm_values: &[f64]| -> Vec<f64> {
        grid.iter()
            .map(|&time| {
                let index = arm_times.partition_point(|&event_time| event_time <= time);
                arm_values[index.saturating_sub(1)]
            })
            .collect()
    };
    let aligned0 = align(&times0, &values0);
    let aligned1 = align(&times1, &values1);
    Ok((grid, aligned0, aligned1))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SurvivalSection>()?;
    module.add_function(wrap_pyfunction!(randomized_survival_ipcw, module)?)?;
    module.add_function(wrap_pyfunction!(randomized_cumulative_incidence_ipcw, module)?)?;
    module.add_function(wrap_pyfunction!(randomized_survival, module)?)?;
    module.add_function(wrap_pyfunction!(randomized_survival_delayed_entry, module)?)?;
    module.add_function(wrap_pyfunction!(randomized_cumulative_incidence, module)?)?;
    module
        .add_function(wrap_pyfunction!(randomized_cumulative_incidence_delayed_entry, module)?)?;
    Ok(())
}
