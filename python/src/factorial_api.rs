//! Native estimates for a two-factor 2×2 randomized factorial design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::detach_catch;

#[pyclass(skip_from_py_object)]
struct FactorialNativeResult {
    #[pyo3(get)]
    cell_means: Vec<f64>,
    #[pyo3(get)]
    cell_support: Vec<usize>,
    #[pyo3(get)]
    factor_a_effect: f64,
    #[pyo3(get)]
    factor_b_effect: f64,
    #[pyo3(get)]
    interaction_effect: f64,
    #[pyo3(get)]
    factor_a_variance_bound: f64,
    #[pyo3(get)]
    factor_b_variance_bound: f64,
    #[pyo3(get)]
    interaction_variance_bound: f64,
}

/// Estimate finite-row cell means and factorial contrasts under independent
/// Bernoulli assignment of two binary factors.
#[pyfunction]
fn estimate_factorial_2x2(
    py: Python<'_>,
    outcome: PyReadonlyArray1<'_, f64>,
    factor_a: Vec<bool>,
    factor_b: Vec<bool>,
    probability_a: PyReadonlyArray1<'_, f64>,
    probability_b: PyReadonlyArray1<'_, f64>,
) -> PyResult<FactorialNativeResult> {
    let outcome = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let probability_a = probability_a.as_array().iter().copied().collect::<Vec<_>>();
    let probability_b = probability_b.as_array().iter().copied().collect::<Vec<_>>();
    let fit = detach_catch(py, move || {
        antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2(
            &outcome,
            &factor_a,
            &factor_b,
            &probability_a,
            &probability_b,
        )
        .map_err(crate::value_err)
    })?;
    Ok(FactorialNativeResult {
        cell_means: fit.cell_means,
        cell_support: fit.cell_support,
        factor_a_effect: fit.factor_a_effect,
        factor_b_effect: fit.factor_b_effect,
        interaction_effect: fit.interaction_effect,
        factor_a_variance_bound: fit.factor_a_variance_bound,
        factor_b_variance_bound: fit.factor_b_variance_bound,
        interaction_variance_bound: fit.interaction_variance_bound,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FactorialNativeResult>()?;
    module.add_function(wrap_pyfunction!(estimate_factorial_2x2, module)?)?;
    Ok(())
}
