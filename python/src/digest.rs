//! Shared parsing for Python-facing semantic digest declarations.

use antecedent::contract::evidence_ledger::parse_digest_hex;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;

pub(crate) fn digest(value: &str, what: &str) -> PyResult<SemanticDigest> {
    parse_digest_hex(value)
        .ok_or_else(|| crate::value_err(format!("{what} must be 64 hexadecimal characters")))
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    use super::digest;

    #[test]
    fn digest_parser_accepts_hex_case_and_reports_invalid_input() {
        Python::initialize();
        Python::attach(|py| {
            let lower = "abcdef0123456789".repeat(4);
            let upper = lower.to_ascii_uppercase();
            let mixed = "aBcDeF0123456789".repeat(4);
            let expected = digest(&lower, "identity").unwrap();
            assert_eq!(digest(&upper, "identity").unwrap(), expected);
            assert_eq!(digest(&mixed, "identity").unwrap(), expected);
            for invalid in ["a".repeat(63), "a".repeat(65), format!("{}g", "a".repeat(63))] {
                let error = digest(&invalid, "identity").unwrap_err();
                assert!(error.is_instance_of::<crate::CausalValidateError>(py));
                assert_eq!(
                    error.value(py).to_string(),
                    "identity must be 64 hexadecimal characters"
                );
            }
        });
    }
}
