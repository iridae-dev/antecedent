//! Python boundary for sharp binary-IV ATE bounds.

use antecedent_identify::{BinaryIvLaw, binary_iv_ate_bounds};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::CausalIdentifyError;

/// Sharp Balke–Pearl bounds on ``E[Y(1)-Y(0)]`` from the observed binary-IV law.
///
/// ``cells`` is a 2×4 nested list: one arm per instrument value ``Z∈{0,1}``,
/// with cell order ``(Y,D)=(0,0),(1,0),(0,1),(1,1)``.
#[pyfunction(name = "binary_iv_ate_bounds")]
fn binary_iv_ate_bounds_py(cells: Vec<Vec<f64>>) -> PyResult<(f64, f64)> {
    if cells.len() != 2 || cells.iter().any(|arm| arm.len() != 4) {
        return Err(PyValueError::new_err(
            "cells must be a 2x4 nested list: one arm per instrument value, \
             cells (Y,D)=(0,0),(1,0),(0,1),(1,1)",
        ));
    }
    let law = BinaryIvLaw {
        cells: [
            [cells[0][0], cells[0][1], cells[0][2], cells[0][3]],
            [cells[1][0], cells[1][1], cells[1][2], cells[1][3]],
        ],
    };
    let bounds = binary_iv_ate_bounds(law)
        .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
    Ok((bounds.lower, bounds.upper))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_iv_ate_bounds_py, m)?)?;
    Ok(())
}
