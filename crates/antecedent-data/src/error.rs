//! Data-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use antecedent_core::VariableId;

/// Errors from data construction, lookup, or materialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DataError {
    /// Schema/data length mismatch.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
        /// Context.
        context: &'static str,
    },
    /// Unknown variable in this table.
    UnknownVariable {
        /// Requested id.
        id: VariableId,
    },
    /// Column type does not match the requested view.
    TypeMismatch {
        /// Variable id.
        id: VariableId,
        /// Expected type label.
        expected: &'static str,
    },
    /// Invalid validity bitmap length.
    InvalidValidity {
        /// Explanation.
        message: &'static str,
    },
    /// Row selection produced an empty sample.
    EmptySelection {
        /// Explanation.
        context: &'static str,
    },
    /// Temporal gather requires a complete series (no missing values or masked rows).
    IncompleteSeries {
        /// Offending variable, when the gap is column-specific.
        id: Option<VariableId>,
        /// Explanation.
        message: &'static str,
    },
    /// Invalid argument (split policy, configuration, etc.).
    InvalidArgument {
        /// Explanation.
        message: String,
    },
    /// Underlying schema error.
    Schema(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual, context } => {
                write!(f, "{context}: expected length {expected}, got {actual}")
            }
            Self::UnknownVariable { id } => write!(f, "unknown variable {id}"),
            Self::TypeMismatch { id, expected } => {
                write!(f, "variable {id} is not of type {expected}")
            }
            Self::InvalidValidity { message } => write!(f, "invalid validity: {message}"),
            Self::EmptySelection { context } => write!(f, "empty selection: {context}"),
            Self::IncompleteSeries { id, message } => match id {
                Some(id) => write!(f, "incomplete series (variable {id}): {message}"),
                None => write!(f, "incomplete series: {message}"),
            },
            Self::InvalidArgument { message } => write!(f, "invalid argument: {message}"),
            Self::Schema(msg) => write!(f, "schema error: {msg}"),
        }
    }
}

impl std::error::Error for DataError {}
