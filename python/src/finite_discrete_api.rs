//! Python bridge for the finite-discrete adjustment contrast of a caller's frequency table.
//!
//! Python hands over the eight-cell frequency table (cell `4t + 2z + y` for treatment `t`,
//! adjuster `z` and outcome `y`) and weighted posterior draws of the same joint. The exact plan,
//! the plug-in and posterior cells and every refusal are Rust's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::finite_discrete_contrast;
use pyo3::prelude::*;

use crate::transport_common::execution_context;
use crate::{PyCancellationToken, detach_catch, py_err, value_err};

/// Plug-in effect, posterior mean, equal-tailed lower and upper quantiles, rows, plans compiled.
type ContrastParts = (f64, f64, f64, f64, u64, u64);

fn cells(values: &[u64], what: &str) -> PyResult<[u64; 8]> {
    <[u64; 8]>::try_from(values)
        .map_err(|_| value_err(format!("{what} must have exactly 8 cells, got {}", values.len())))
}

/// The adjustment contrast `sum_z P(z) P(y | t, z)` of a frequency table with its posterior.
#[pyfunction]
#[pyo3(signature = (counts, draws, cancel=None))]
fn finite_discrete_adjustment_contrast(
    py: Python<'_>,
    counts: Vec<u64>,
    draws: Vec<(Vec<u64>, f64)>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<ContrastParts> {
    let table = cells(&counts, "counts")?;
    let mut weighted = Vec::with_capacity(draws.len());
    for (draw, weight) in &draws {
        weighted.push((cells(draw, "each posterior draw")?, *weight));
    }
    detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let contrast = finite_discrete_contrast(table, weighted, &ctx).map_err(py_err)?;
        Ok((
            contrast.plug_in_effect,
            contrast.posterior_mean,
            contrast.posterior_lower,
            contrast.posterior_upper,
            contrast.sample_size,
            contrast.plan_compilations,
        ))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(finite_discrete_adjustment_contrast, module)?)?;
    Ok(())
}
