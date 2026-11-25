//! Python binding for the read-only winning-design projection.
//!
//! The caller supplies the candidate id, rank and sample size a ranking already stored.
//! This module does not rank and does not call `design.consume`.

use antecedent::analysis::winning_design::{project_stored_winner, StoredDesignRow};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{value_err, value_err_display};

const MAX_ROWS_JSON: usize = 1_048_576;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowsWire {
    rows: Vec<RowWire>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowWire {
    candidate_id: String,
    rank: u64,
    n: u64,
}

#[derive(Debug, Serialize)]
struct DeclarationWire {
    candidate_id: String,
    n: u64,
}

/// Copy the rank-0 candidate id and sample size from stored ranking rows.
#[pyfunction]
fn project_winning_design(rows_json: &str) -> PyResult<String> {
    if rows_json.len() > MAX_ROWS_JSON {
        return Err(value_err("winning design rows exceed the accepted size"));
    }
    let wire: RowsWire = serde_json::from_str(rows_json)
        .map_err(|_| value_err("winning design rows are not the stored ranking"))?;
    let declaration = project_stored_winner(wire.rows.iter().map(|row| StoredDesignRow {
        candidate_id: row.candidate_id.as_str(),
        rank: row.rank,
        n: row.n,
    }))
    .map_err(value_err_display)?;
    serde_json::to_string(&DeclarationWire {
        candidate_id: declaration.candidate_id,
        n: declaration.n,
    })
    .map_err(|_| value_err("winning design rows are not the stored ranking"))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(project_winning_design, m)?)
}
