//! The pieces every declared-JSON artifact binding shares: the byte-bound
//! check on a declaration and the mapping of a serialization failure.
//!
//! Each binding keeps its own public `MAX_*_BYTES` limit and passes it in, so
//! the limits stay per-artifact while the error spelling lives here once.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Refuse a declared JSON text longer than `limit` bytes.
///
/// The message is `"{what} declaration is too large"`; `what` names the
/// artifact as its binding always spelled it.
pub(crate) fn check_declared_size(text: &str, limit: usize, what: &str) -> PyResult<()> {
    if text.len() > limit {
        return Err(PyValueError::new_err(format!("{what} declaration is too large")));
    }
    Ok(())
}

/// A failure that can only mean the bytes do not describe what they claim.
pub(crate) fn serialization(e: impl std::fmt::Display) -> PyErr {
    crate::CausalSerializationError::new_err(e.to_string())
}

/// The wire JSON of one structured external refusal.
///
/// Every `ExternalRefusal` a binding returns to Python is spelled here once.
pub(crate) fn external_refusal_json(value: &antecedent_core::ExternalRefusal) -> String {
    serde_json::json!({
        "code": value.code,
        "stage": value.stage,
        "detail": value.detail,
        "offending": value.offending,
        "expected": value.expected,
        "supplied": value.supplied,
        "remedy": value.remedy,
    })
    .to_string()
}
