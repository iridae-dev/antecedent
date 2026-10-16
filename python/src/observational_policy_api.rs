//! Bounded Python bridge for observational policy value.
//!
//! Python supplies the outcome, treatment and adjustment columns and a small JSON declaration
//! (support, objective, evaluation, the policy's actions or the held-out fraction, folds,
//! floor, seed); the nuisance fits, the learning and cross-fitting, every refusal and the
//! self-describing report are Rust's. A refusal raises the typed exception with its reason
//! code.

use crate::bridge::check_declared_size;
use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observational_policy::{
    ObservationalPolicyRequest, ObservationalPolicySpecWire, ObservationalPolicyValue,
};
use pyo3::prelude::*;

use crate::{detach_catch, py_err, value_err, with_reason_code};

/// Largest declaration JSON accepted; it carries the policy's per-row actions.
const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;

/// Estimate the declared policy's value and return the report as JSON.
///
/// `adjustment_columns[j]` is the column named `adjustment_names[j]`.
#[pyfunction]
fn evaluate_observational_policy(
    py: Python<'_>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    adjustment_names: Vec<String>,
    adjustment_columns: Vec<Vec<f64>>,
    declaration_json: &str,
) -> PyResult<String> {
    check_declared_size(declaration_json, MAX_JSON_BYTES, "observational policy value")?;
    let spec: ObservationalPolicySpecWire =
        serde_json::from_str(declaration_json).map_err(|e| {
            with_reason_code(
                value_err(format!("policy_value.invalid_request: {e}")),
                antecedent_core::reason_code!("invalid_argument"),
            )
        })?;
    if adjustment_names.len() != adjustment_columns.len() {
        return Err(with_reason_code(
            value_err("policy_value.invalid_request: a column is missing for a name"),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    let request = ObservationalPolicyRequest {
        outcome,
        treatment,
        adjustment: adjustment_names
            .into_iter()
            .zip(adjustment_columns)
            .map(|(name, values)| AdjustmentColumn { name, values })
            .collect(),
        spec,
    };
    detach_catch(py, move || {
        let report = ObservationalPolicyValue::evaluate(&request).map_err(py_err)?;
        serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_observational_policy, m)?)
}
