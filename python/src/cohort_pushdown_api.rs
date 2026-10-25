//! Python bridge for cohort-checked column projection and filter pushdown.
//!
//! Python hands over which cells of each column are observed and the closure of columns the
//! analysis requires. The complete-case cohort, the projection and filter rules, and the check
//! that the rows a pushdown delivers are the declared cohort are Rust's; a refusal raises a
//! `ValueError` naming the column that must stay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::file_table::{checked_projection, checked_push_filter};
use antecedent_data::column_closure::{
    ClosureInputs, ColumnClosure, ColumnFilter, ColumnRole, ObservationFrame,
};
use pyo3::prelude::*;

use crate::value_err_code;

/// Columns by duty: target, validators, support, weights, splits, complete-case.
type ClosureDuties = (Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>);

fn refuse(message: impl std::fmt::Display) -> PyErr {
    value_err_code(antecedent_core::reason_code!("invalid_argument"), message.to_string())
}

fn roles(names: Vec<String>) -> Vec<ColumnRole> {
    names.iter().map(ColumnRole::new).collect()
}

fn closure_of(duties: ClosureDuties) -> ColumnClosure {
    let (target, validators, support, weights, splits, complete_case) = duties;
    let mut inputs = ClosureInputs::new(roles(target));
    for column in roles(validators) {
        inputs = inputs.validator(column);
    }
    for column in roles(support) {
        inputs = inputs.support(column);
    }
    for column in roles(weights) {
        inputs = inputs.weight(column);
    }
    for column in roles(splits) {
        inputs = inputs.split(column);
    }
    for column in roles(complete_case) {
        inputs = inputs.complete_case(column);
    }
    inputs.close()
}

fn frame_of(observed: Vec<(String, Vec<bool>)>) -> PyResult<ObservationFrame> {
    ObservationFrame::try_new(
        observed.into_iter().map(|(name, mask)| (ColumnRole::new(name), mask)).collect(),
    )
    .map_err(refuse)
}

/// Project to `keep` and check the cohort it delivers is the closure's complete-case cohort.
/// Returns the delivered rows.
#[pyfunction]
fn cohort_checked_projection(
    closure: ClosureDuties,
    observed: Vec<(String, Vec<bool>)>,
    keep: Vec<String>,
) -> PyResult<Vec<usize>> {
    let closure = closure_of(closure);
    let frame = frame_of(observed)?;
    let cohort = checked_projection(&closure, &frame, &roles(keep)).map_err(refuse)?;
    Ok(cohort.rows().to_vec())
}

/// Push a row filter read off `column` and check it drops no row of the closure's complete-case
/// cohort. Returns the delivered rows.
#[pyfunction]
fn cohort_checked_push_filter(
    closure: ClosureDuties,
    observed: Vec<(String, Vec<bool>)>,
    column: &str,
    drop_row: Vec<bool>,
) -> PyResult<Vec<usize>> {
    let closure = closure_of(closure);
    let frame = frame_of(observed)?;
    let filter = ColumnFilter::new(ColumnRole::new(column), drop_row);
    let cohort = checked_push_filter(&closure, &frame, &filter).map_err(refuse)?;
    Ok(cohort.rows().to_vec())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(cohort_checked_projection, module)?)?;
    module.add_function(wrap_pyfunction!(cohort_checked_push_filter, module)?)?;
    Ok(())
}
