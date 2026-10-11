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
/// Every `ExternalRefusal` a binding returns to Python is spelled here once,
/// through the same `RefusalWire` the external-binding bridge uses, so the
/// missing operation of a capability refusal (`capability`) reaches Python.
pub(crate) fn external_refusal_json(value: &antecedent_core::ExternalRefusal) -> String {
    serde_json::to_string(&antecedent_io::external_binding_wire::RefusalWire::from(value.clone()))
        .expect("a refusal of strings and options always serialises")
}

#[cfg(test)]
mod tests {
    use super::external_refusal_json;
    use antecedent_core::{ExternalCapability, ExternalRefusal};

    fn refusal(capability: Option<ExternalCapability>) -> ExternalRefusal {
        ExternalRefusal {
            code: antecedent_core::reason_code!("external_capability_missing"),
            stage: "negotiate",
            detail: "provider_capability_negotiation.capability_missing".to_owned(),
            offending: None,
            expected: None,
            supplied: None,
            capability,
            remedy: None,
        }
    }

    #[test]
    fn the_missing_operation_reaches_the_wire() {
        let json = external_refusal_json(&refusal(Some(ExternalCapability::Cdf)));
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["capability"], "cdf");
        assert_eq!(value["code"], "external_capability_missing");
    }

    #[test]
    fn a_refusal_without_one_carries_a_null_capability() {
        let json = external_refusal_json(&refusal(None));
        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert!(value.get("capability").is_some_and(serde_json::Value::is_null));
    }
}
