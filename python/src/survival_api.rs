//! Point-only randomized survival summaries with right censoring.
// SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Simultaneous difference band: times, difference, lower, upper, ok replicates,
/// and the exact graphless band license (`Some("licensed")`) when the published
/// band and arm support match a licensed band row.
type DifferenceBand = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, u32, Option<String>);
/// IPCW survival curves: times, control, treated, RMSTs, per-arm minimum event
/// risk sets, and the realized censoring-survival floor.
type IpcwSurvival = (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, Option<usize>, Option<usize>, f64);
/// IPCW cumulative incidence: times, control, treated, per-arm minimum event
/// risk sets, and the realized censoring-survival floor.
type IpcwIncidence = (Vec<f64>, Vec<f64>, Vec<f64>, Option<usize>, Option<usize>, f64);

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
    pub assignment_counts: (usize, usize),
    pub censoring_survival_provenance: Option<String>,
    pub difference_band: Option<DifferenceBand>,
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
            rmst_difference_interval: value
                .rmst_difference_interval
                .map(|interval| interval.to_vec()),
            difference_at_tau_interval: value
                .difference_at_tau_interval
                .map(|interval| interval.to_vec()),
            bootstrap_replicates_requested: value.bootstrap_replicates_requested,
            bootstrap_replicates_ok: value.bootstrap_replicates_ok,
            assignment_counts: (value.assignment_counts[0], value.assignment_counts[1]),
            censoring_survival_provenance: value
                .censoring_survival_provenance
                .as_ref()
                .map(ToString::to_string),
            difference_band: value.difference_band.as_ref().map(|band| {
                (
                    band.times.to_vec(),
                    band.difference.to_vec(),
                    band.lower.to_vec(),
                    band.upper.to_vec(),
                    band.replicates_ok,
                    band.support_status.map(|status| status.as_str().to_string()),
                )
            }),
            band_unavailable_reason: value
                .band_unavailable_reason
                .as_ref()
                .map(ToString::to_string),
        }
    }
}

/// IPCW weighted product-limit curves using caller supplied censoring survival.
#[pyfunction]
#[allow(
    clippy::float_cmp,
    reason = "grid membership tests match exact recorded durations against exact grid times, not approximate quantities"
)]
fn randomized_survival_ipcw(
    duration: PyReadonlyArray1<'_, f64>,
    event_observed: Vec<bool>,
    treated: Vec<bool>,
    times: Vec<f64>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
    tau: f64,
    minimum_probability: f64,
) -> PyResult<IpcwSurvival> {
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
#[allow(
    clippy::float_cmp,
    reason = "grid membership tests match exact recorded durations against exact grid times, not approximate quantities"
)]
fn randomized_cumulative_incidence_ipcw(
    duration: PyReadonlyArray1<'_, f64>,
    event_cause: Vec<i64>,
    treated: Vec<bool>,
    times: Vec<f64>,
    censoring_survival: PyReadonlyArray2<'_, f64>,
    tau: f64,
    target_cause: i64,
    minimum_probability: f64,
) -> PyResult<IpcwIncidence> {
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SurvivalSection>()?;
    module.add_function(wrap_pyfunction!(randomized_survival_ipcw, module)?)?;
    module.add_function(wrap_pyfunction!(randomized_cumulative_incidence_ipcw, module)?)?;
    Ok(())
}
