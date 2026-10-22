//! Typed boundaries of two posteriors that have no same-target counterpart.
//!
//! * The temporal response-curve posterior is a tempered generalized posterior; the ordinary
//!   reading needs the innovation law of the series in the likelihood
//!   (`temporal_curve.ordinary_posterior_not_licensed`).
//! * A trajectory survival contrast under a prescribed time-varying regime has a conditional
//!   subject-bootstrap interval, and no credible interval: that needs a joint likelihood for
//!   the confounders, treatment, censoring and hazard.
//!
//! The numbers and every refusal, with the next attempt, are Rust's.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::posterior_boundaries::{
    SerialErrorModel, TrajectoryRows, require_ordinary_temporal_curve_posterior,
    trajectory_hazard_credible_interval, trajectory_survival_bootstrap,
    trajectory_survival_contrast,
};
use pyo3::prelude::*;

use crate::{detach_catch, py_err, value_err};

/// Subject-major trajectories as Python passes them.
struct Trajectories {
    subjects: usize,
    periods: usize,
    treatment: Vec<bool>,
    event: Vec<bool>,
    at_risk: Vec<bool>,
    treatment_probability: Vec<f64>,
    censoring_weight: Vec<f64>,
    minimum_probability: f64,
}

impl Trajectories {
    fn rows(&self) -> TrajectoryRows<'_> {
        TrajectoryRows {
            subjects: self.subjects,
            periods: self.periods,
            treatment: &self.treatment,
            event: &self.event,
            at_risk: &self.at_risk,
            treatment_probability: &self.treatment_probability,
            censoring_weight: &self.censoring_weight,
            minimum_probability: self.minimum_probability,
        }
    }
}

/// Refuse the ordinary-likelihood reading of a temporal response-curve posterior unless the
/// caller attests independent innovations (`"unmodeled"` or `"independent_innovations"`).
#[pyfunction]
fn require_ordinary_temporal_curve_posterior_native(serial_error_model: &str) -> PyResult<()> {
    let serial = match serial_error_model {
        "unmodeled" => SerialErrorModel::Unmodeled,
        "independent_innovations" => SerialErrorModel::IndependentInnovations,
        other => {
            return Err(value_err(format!(
                "serial_error_model must be 'unmodeled' or 'independent_innovations', got {other:?}"
            )));
        }
    };
    require_ordinary_temporal_curve_posterior(serial).map_err(py_err)
}

/// Weighted product-limit contrast of two regimes: `(survival difference, restricted-mean
/// difference)`.
#[pyfunction]
#[pyo3(signature = (subjects, periods, treatment, event, at_risk, treatment_probability,
    censoring_weight, minimum_probability, active, control))]
#[allow(clippy::too_many_arguments)]
fn trajectory_contrast_native(
    py: Python<'_>,
    subjects: usize,
    periods: usize,
    treatment: Vec<bool>,
    event: Vec<bool>,
    at_risk: Vec<bool>,
    treatment_probability: Vec<f64>,
    censoring_weight: Vec<f64>,
    minimum_probability: f64,
    active: Vec<bool>,
    control: Vec<bool>,
) -> PyResult<(f64, f64)> {
    let data = Trajectories {
        subjects,
        periods,
        treatment,
        event,
        at_risk,
        treatment_probability,
        censoring_weight,
        minimum_probability,
    };
    detach_catch(py, move || {
        let contrast =
            trajectory_survival_contrast(&data.rows(), &active, &control).map_err(py_err)?;
        Ok((contrast.survival_difference, contrast.restricted_mean_difference))
    })
}

/// Percentile subject-bootstrap intervals conditional on the supplied weights:
/// `(survival lower, survival upper, rmst lower, rmst upper, requested, ok)`.
#[pyfunction]
#[pyo3(signature = (subjects, periods, treatment, event, at_risk, treatment_probability,
    censoring_weight, minimum_probability, active, control, replicates, seed))]
#[allow(clippy::too_many_arguments)]
fn trajectory_bootstrap_native(
    py: Python<'_>,
    subjects: usize,
    periods: usize,
    treatment: Vec<bool>,
    event: Vec<bool>,
    at_risk: Vec<bool>,
    treatment_probability: Vec<f64>,
    censoring_weight: Vec<f64>,
    minimum_probability: f64,
    active: Vec<bool>,
    control: Vec<bool>,
    replicates: u32,
    seed: u64,
) -> PyResult<(f64, f64, f64, f64, u32, u32)> {
    let data = Trajectories {
        subjects,
        periods,
        treatment,
        event,
        at_risk,
        treatment_probability,
        censoring_weight,
        minimum_probability,
    };
    detach_catch(py, move || {
        let intervals =
            trajectory_survival_bootstrap(&data.rows(), &active, &control, replicates, seed)
                .map_err(py_err)?;
        let [survival_low, survival_high] = intervals.survival_difference;
        let [rmst_low, rmst_high] = intervals.restricted_mean_difference;
        Ok((
            survival_low,
            survival_high,
            rmst_low,
            rmst_high,
            intervals.replicates_requested,
            intervals.replicates_ok,
        ))
    })
}

/// A credible interval for the trajectory survival contrast: always a typed refusal naming
/// the missing joint likelihood and the next attempt.
#[pyfunction]
#[pyo3(signature = (subjects, periods, treatment, event, at_risk, treatment_probability,
    censoring_weight, minimum_probability, active, control))]
#[allow(clippy::too_many_arguments)]
fn trajectory_credible_interval_native(
    py: Python<'_>,
    subjects: usize,
    periods: usize,
    treatment: Vec<bool>,
    event: Vec<bool>,
    at_risk: Vec<bool>,
    treatment_probability: Vec<f64>,
    censoring_weight: Vec<f64>,
    minimum_probability: f64,
    active: Vec<bool>,
    control: Vec<bool>,
) -> PyResult<(f64, f64)> {
    let data = Trajectories {
        subjects,
        periods,
        treatment,
        event,
        at_risk,
        treatment_probability,
        censoring_weight,
        minimum_probability,
    };
    detach_catch(py, move || {
        let [lower, upper] =
            trajectory_hazard_credible_interval(&data.rows(), &active, &control).map_err(py_err)?;
        Ok((lower, upper))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        require_ordinary_temporal_curve_posterior_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(trajectory_contrast_native, module)?)?;
    module.add_function(wrap_pyfunction!(trajectory_bootstrap_native, module)?)?;
    module.add_function(wrap_pyfunction!(trajectory_credible_interval_native, module)?)?;
    Ok(())
}
