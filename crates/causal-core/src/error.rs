//! Schema construction errors for `causal-core`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Errors raised while building or looking up schema elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchemaError {
    /// Two variables were declared with the same name.
    DuplicateVariableName {
        /// Conflicting name.
        name: String,
    },
    /// Name lookup failed at an API boundary.
    UnknownVariableName {
        /// Requested name.
        name: String,
    },
    /// Dense variable ID is outside the schema.
    UnknownVariableId {
        /// Requested raw id.
        id: u32,
    },
    /// Schema exceeded the maximum number of variables (`u32::MAX`).
    TooManyVariables,
    /// A categorical / ordinal variable lacked a category domain.
    MissingCategoryDomain {
        /// Variable name that required a domain.
        name: String,
    },
    /// A non-categorical variable was given a category domain.
    UnexpectedCategoryDomain {
        /// Variable name that must not carry a domain.
        name: String,
    },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateVariableName { name } => {
                write!(f, "duplicate variable name `{name}`")
            }
            Self::UnknownVariableName { name } => {
                write!(f, "unknown variable name `{name}`")
            }
            Self::UnknownVariableId { id } => write!(f, "unknown variable id {id}"),
            Self::TooManyVariables => write!(f, "schema exceeds maximum variable count"),
            Self::MissingCategoryDomain { name } => {
                write!(f, "variable `{name}` requires a category domain")
            }
            Self::UnexpectedCategoryDomain { name } => {
                write!(f, "variable `{name}` must not have a category domain")
            }
        }
    }
}

impl std::error::Error for SchemaError {}
