//! Native estimates for a two-factor 2×2 randomized factorial design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    outcome: PyReadonlyArray1<'_, f64>,
    factor_a: Vec<bool>,
    factor_b: Vec<bool>,
    probability_a: PyReadonlyArray1<'_, f64>,
    probability_b: PyReadonlyArray1<'_, f64>,
) -> PyResult<FactorialNativeResult> {
    let y = outcome.as_array();
    let pa = probability_a.as_array();
    let pb = probability_b.as_array();
    let n = y.len();
    if n == 0 || factor_a.len() != n || factor_b.len() != n || pa.len() != n || pb.len() != n {
        return Err(PyValueError::new_err(
            "outcome, assignments, and probability vectors must have equal non-zero length",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    if pa.iter().chain(pb.iter()).any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
        return Err(PyValueError::new_err(
            "both factor assignment probabilities must be strictly between zero and one",
        ));
    }

    let mut sums = [0.0; 4];
    let mut variance_terms = [0.0; 4];
    let mut support = [0_usize; 4];
    for i in 0..n {
        let a = usize::from(factor_a[i]);
        let b = usize::from(factor_b[i]);
        let cell = 2 * a + b;
        let qa = if a == 1 { pa[i] } else { 1.0 - pa[i] };
        let qb = if b == 1 { pb[i] } else { 1.0 - pb[i] };
        let q = qa * qb;
        support[cell] += 1;
        sums[cell] += y[i] / q;
        variance_terms[cell] += (1.0 - q) * y[i].powi(2) / q;
    }
    if support.contains(&0) {
        return Err(PyValueError::new_err(
            "factorial positivity failure: every treatment cell needs observed support",
        ));
    }
    let means = sums.map(|sum| sum / n as f64);
    let variances = variance_terms.map(|term| term / (n as f64).powi(2));
    let factor_a_weights = [-0.5, -0.5, 0.5, 0.5];
    let factor_b_weights = [-0.5, 0.5, -0.5, 0.5];
    let interaction_weights = [1.0, -1.0, -1.0, 1.0];
    let contrast = |weights: [f64; 4]| {
        means.iter().zip(weights).map(|(mean, weight)| mean * weight).sum::<f64>()
    };
    // Young's inequality gives Var(sum_j w_j X_j) <= 4 sum_j w_j^2 Var(X_j).
    let variance_bound = |weights: [f64; 4]| {
        4.0 * variances
            .iter()
            .zip(weights)
            .map(|(variance, weight)| variance * weight.powi(2))
            .sum::<f64>()
    };
    Ok(FactorialNativeResult {
        cell_means: means.to_vec(),
        cell_support: support.to_vec(),
        factor_a_effect: contrast(factor_a_weights),
        factor_b_effect: contrast(factor_b_weights),
        interaction_effect: contrast(interaction_weights),
        factor_a_variance_bound: variance_bound(factor_a_weights),
        factor_b_variance_bound: variance_bound(factor_b_weights),
        interaction_variance_bound: variance_bound(interaction_weights),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FactorialNativeResult>()?;
    module.add_function(wrap_pyfunction!(estimate_factorial_2x2, module)?)?;
    Ok(())
}
