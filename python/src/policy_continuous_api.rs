//! Direct Python access to the shared conditional-dose estimator.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type DoseResponsePoint = (String, f64, f64, usize, f64, f64, f64, f64);

#[pyfunction]
#[pyo3(signature = (outcome, dose, groups, dose_density, target_doses, bandwidth, *, min_local_support=3))]
fn conditional_dose_response(
    outcome: PyReadonlyArray1<'_, f64>,
    dose: PyReadonlyArray1<'_, f64>,
    groups: Vec<String>,
    dose_density: PyReadonlyArray1<'_, f64>,
    target_doses: Vec<f64>,
    bandwidth: f64,
    min_local_support: usize,
) -> PyResult<Vec<DoseResponsePoint>> {
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let dose = dose.as_array().iter().copied().collect::<Vec<_>>();
    let density = dose_density.as_array().iter().copied().collect::<Vec<_>>();
    antecedent_estimate::continuous_dose::conditional_dose_response(
        &y, &dose, &groups, &density, &target_doses, bandwidth, min_local_support,
    )
    .map(|points| points.into_iter().map(|point| (
        point.baseline_group, point.target_dose, point.response, point.local_rows,
        point.effective_sample_size, point.minimum_dose_density,
        point.maximum_normalized_weight, point.local_outcome_sd,
    )).collect())
    .map_err(PyValueError::new_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(conditional_dose_response, module)?)?;
    Ok(())
}
