//! Python bridge for composition across the horizons of one temporal claim.
//!
//! Python hands over the joint draws of each horizon (one column per horizon, row `r` of every
//! column one realization that refit all horizons together) and the declaration of the
//! composition. The horizon-indexed coordinate check, the aligned arithmetic and every refusal
//! are Rust's. Columns that were fit one by one carry no common alignment identity and are
//! never paired by index.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;

use antecedent::analysis::horizon_composition::{
    ComposedHorizons, HorizonDraws, compose_horizons, compose_horizons_with,
};
use antecedent_contract::PropagationOperation;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use serde_json::json;

use crate::{py_execution_context, py_msg};

fn summary_json(composed: &ComposedHorizons) -> String {
    json!({
        "coordinate": composed.coordinate.id(),
        "horizons": composed.coordinate.horizons,
        "route": composed.plan.route.as_str(),
        "alignment_id": composed.plan.alignment_id,
        "label": composed.label,
        "n": composed.spread.n,
        "mean": composed.spread.mean,
        "variance": composed.spread.variance,
        "standard_deviation": composed.spread.standard_deviation,
        "mean_standard_error": composed.spread.mean_standard_error,
        "draws": composed.draws,
    })
    .to_string()
}

/// Compose the horizons of one temporal claim from their joint draws.
///
/// `operation` is a propagation operation name. A linear one takes `coefficients` (one per
/// horizon); a nonlinear one takes `map`, a callable of one realization's horizon values that
/// returns a float. Returns the composed coordinate (carrying the horizon axis), the aligned
/// route, the spread of the composed draws and the draws, as JSON.
#[pyfunction]
#[pyo3(signature = (horizons, columns, evidence_id, alignment_id, operation, *, coefficients=None, map=None, seed=0))]
fn compose_horizon_draws(
    horizons: Vec<u32>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    evidence_id: String,
    alignment_id: String,
    operation: &str,
    coefficients: Option<Vec<f64>>,
    map: Option<Bound<'_, PyAny>>,
    seed: u64,
) -> PyResult<String> {
    let operation = PropagationOperation::from_name(operation)
        .ok_or_else(|| crate::value_err(format!("unknown propagation operation {operation:?}")))?;
    if horizons.len() != columns.len() {
        return Err(crate::value_err("give one draw column per horizon"));
    }
    let columns: Vec<Vec<f64>> = columns.iter().map(|c| c.as_array().to_vec()).collect();
    let center = columns.iter().map(|c| c.iter().sum::<f64>() / c.len().max(1) as f64).collect();
    let draws = HorizonDraws { horizons, center, columns, alignment_id, evidence_id };
    let refusal = |error: antecedent_contract::ExternalRefusal| {
        py_msg(format!("{}: {} ({})", error.code, error.detail, error.remedy.unwrap_or("")))
    };
    let composed = match (coefficients, map) {
        (Some(coefficients), None) => {
            compose_horizons(&draws, &coefficients, operation).map_err(refusal)?
        }
        (None, Some(map)) => {
            if !map.is_callable() {
                return Err(crate::value_err("map must be callable"));
            }
            let failure: RefCell<Option<PyErr>> = RefCell::new(None);
            let call = |values: &[f64]| -> f64 {
                match map.call1((values.to_vec(),)).and_then(|out| out.extract::<f64>()) {
                    Ok(value) => value,
                    Err(error) => {
                        failure.borrow_mut().get_or_insert(error);
                        f64::NAN
                    }
                }
            };
            let ctx = py_execution_context(seed, 1);
            let result = compose_horizons_with(&draws, operation, &call, &ctx);
            if let Some(error) = failure.into_inner() {
                return Err(error);
            }
            result.map_err(refusal)?
        }
        _ => {
            return Err(crate::value_err(
                "give coefficients for a linear composition or map for a nonlinear one, not both",
            ));
        }
    };
    Ok(summary_json(&composed))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compose_horizon_draws, m)?)
}
