//! Temporal-response support from Rust (`TemporalResponseSpec::support`).
//!
//! Python must read these values. It may duplicate checks; it must not
//! define the policy.

use antecedent::query::{TemporalPolicy, TemporalResponseSpec};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) const DEFAULT_TREATMENT_LAG: u32 = TemporalResponseSpec::DEFAULT_TREATMENT_LAG;
pub(crate) const DEFAULT_POLICY: &str = TemporalResponseSpec::DEFAULT_POLICY;

pub(crate) fn policy_at_lag(tag: impl AsRef<str>, treatment_lag: u32) -> PyResult<TemporalPolicy> {
    let at = -i32::try_from(treatment_lag)
        .map_err(|_| crate::value_err("treatment_lag does not fit in i32"))?;
    let tag = tag.as_ref().to_ascii_lowercase();
    TemporalResponseSpec::parse_policy(&tag, at).map_err(|e| crate::value_err(e.to_string()))
}

/// Machine-readable [`TemporalResponseSpec`] support for the Python facade.
#[pyfunction]
fn temporal_response_spec(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let support = TemporalResponseSpec::support();
    let d = PyDict::new(py);
    d.set_item("max_horizons", support.max_horizons)?;
    d.set_item("max_cells", support.max_cells)?;
    d.set_item("allowed_policies", support.allowed_policies)?;
    d.set_item("default_policy", support.default_policy)?;
    d.set_item("default_treatment_lag", support.default_treatment_lag)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(temporal_response_spec, m)?)?;
    Ok(())
}
