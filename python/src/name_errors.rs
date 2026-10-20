//! Shared refusal for an unrecognised name passed from Python.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use pyo3::prelude::*;

/// The validation error raised when `name` is not a known `what`.
pub(crate) fn unknown_name(what: &str, name: &str) -> PyErr {
    crate::value_err_code(reason_code!("invalid_argument"), format!("unknown {what} `{name}`"))
}
