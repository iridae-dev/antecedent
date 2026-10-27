//! Python read model of the support matrix: the licensed cells and the report of one cell.
//!
//! The report is the Rust classification of a `query:graph:structure:inference:validation`
//! coordinate with its refusal text and, for a cell that is not licensed, the licensed
//! neighbors that change explicit premises (never an automatic fallback). Listing and
//! reporting read the generated matrix only; nothing is prepared, fitted or scanned.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::support::{
    CellStatus, cell_coordinate, classify, licensed_neighbors, licensed_support_cells,
    refused_message, support_cell_from_coordinate,
};
use pyo3::prelude::*;

use crate::value_err_code;

/// Every licensed coordinate, in matrix order.
#[pyfunction]
fn support_licensed_cells() -> Vec<String> {
    licensed_support_cells().map(cell_coordinate).collect()
}

/// Classification, blocker text and licensed neighbors of one coordinate, as JSON.
///
/// `message` is the blocker a refused or not-applicable cell reports and `null` for a
/// licensed cell. A coordinate that is malformed or names an axis value outside the
/// matrix raises `invalid_argument`.
#[pyfunction]
fn support_cell_report(coordinate: &str) -> PyResult<String> {
    let cell = support_cell_from_coordinate(coordinate).ok_or_else(|| {
        value_err_code(
            antecedent_core::reason_code!("invalid_argument"),
            format!(
                "support cell {coordinate:?} is not a query:graph:structure:inference:validation \
                 coordinate over the matrix axes"
            ),
        )
    })?;
    let status = classify(cell);
    let message = match status {
        CellStatus::Licensed => None,
        CellStatus::NotApplicable { reason } | CellStatus::Allowlisted { reason, .. } => {
            Some(reason)
        }
        CellStatus::Refused => Some(refused_message(cell)),
    };
    let report = serde_json::json!({
        "coordinate": cell_coordinate(cell),
        "status": status.as_str(),
        "message": message,
        "neighbors": antecedent::host::neighbor_records(&licensed_neighbors(cell)),
    });
    serde_json::to_string(&report).map_err(|error| crate::value_err(error.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(support_licensed_cells, m)?)?;
    m.add_function(wrap_pyfunction!(support_cell_report, m)?)?;
    Ok(())
}
