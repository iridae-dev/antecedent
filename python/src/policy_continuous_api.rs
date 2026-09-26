//! Local conditional dose-response utility using caller-supplied dose densities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;

/// (stratum, target dose, response, local rows, effective N, min density,
/// max normalized weight, descriptive local outcome SD).
type DoseResponsePoint = (String, f64, f64, usize, f64, f64, f64, f64);

/// Estimate local conditional dose responses using a triangular kernel and
/// inverse known/external density weights. No policy value is estimated.
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
    let n = y.len();
    if n == 0 || dose.len() != n || groups.len() != n || density.len() != n {
        return Err(PyValueError::new_err(
            "outcome, dose, groups, and dose density must have equal non-zero length",
        ));
    }
    if target_doses.is_empty() || target_doses.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("target_doses must be non-empty and finite"));
    }
    if !bandwidth.is_finite() || bandwidth <= 0.0 || min_local_support < 2 {
        return Err(PyValueError::new_err(
            "bandwidth must be positive and min_local_support must be at least two",
        ));
    }
    if y.iter().chain(&dose).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes and doses must be finite"));
    }
    if density.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(PyValueError::new_err(
            "continuous-dose positivity failure: supplied dose densities must be finite and positive",
        ));
    }
    if groups.iter().any(String::is_empty) {
        return Err(PyValueError::new_err("baseline group labels must be non-empty strings"));
    }
    let strata: BTreeSet<String> = groups.iter().cloned().collect();
    let mut output = Vec::with_capacity(strata.len() * target_doses.len());
    for stratum in strata {
        for target in &target_doses {
            let mut weighted_outcomes = Vec::new();
            let mut weights = Vec::new();
            let mut local_densities = Vec::new();
            for i in 0..n {
                let scaled_distance = (dose[i] - target) / bandwidth;
                if groups[i] != stratum || scaled_distance.abs() >= 1.0 {
                    continue;
                }
                let kernel = 0.75 * (1.0 - scaled_distance.powi(2));
                let weight = kernel / density[i];
                if !weight.is_finite() || weight <= 0.0 {
                    return Err(PyValueError::new_err(
                        "dose-response weights overflowed or lost positivity",
                    ));
                }
                weighted_outcomes.push(y[i]);
                weights.push(weight);
                local_densities.push(density[i]);
            }
            if weights.len() < min_local_support {
                return Err(PyValueError::new_err(format!(
                    "dose-response support failure: stratum {stratum:?} at dose {target} has {} local rows, below min_local_support={min_local_support}",
                    weights.len()
                )));
            }
            let sum_weight = weights.iter().sum::<f64>();
            let sum_weight_squared = weights.iter().map(|weight| weight.powi(2)).sum::<f64>();
            let response = weights
                .iter()
                .zip(&weighted_outcomes)
                .map(|(weight, outcome)| weight * outcome)
                .sum::<f64>()
                / sum_weight;
            let effective_n = sum_weight.powi(2) / sum_weight_squared;
            let local_sd = (weights
                .iter()
                .zip(&weighted_outcomes)
                .map(|(weight, outcome)| weight * (outcome - response).powi(2))
                .sum::<f64>()
                / sum_weight)
                .sqrt();
            let max_normalized_weight =
                weights.iter().map(|weight| weight / sum_weight).fold(0.0_f64, f64::max);
            let minimum_density = local_densities.iter().copied().fold(f64::INFINITY, f64::min);
            if !response.is_finite() || !effective_n.is_finite() || !local_sd.is_finite() {
                return Err(PyValueError::new_err(
                    "dose-response estimate overflowed finite precision",
                ));
            }
            output.push((
                stratum.clone(),
                *target,
                response,
                weights.len(),
                effective_n,
                minimum_density,
                max_normalized_weight,
                local_sd,
            ));
        }
    }
    Ok(output)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(conditional_dose_response, module)?)?;
    Ok(())
}
