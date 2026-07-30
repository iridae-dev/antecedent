//! Schema construction errors for `antecedent-core`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Errors raised while building or looking up schema elements.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SchemaError {
    /// Two variables were declared with the same name.
    #[error("duplicate variable name `{name}`")]
    DuplicateVariableName {
        /// Conflicting name.
        name: String,
    },
    /// Name lookup failed at an API boundary.
    #[error("unknown variable name `{name}`")]
    UnknownVariableName {
        /// Requested name.
        name: String,
    },
    /// Dense variable ID is outside the schema.
    #[error("unknown variable id {id}")]
    UnknownVariableId {
        /// Requested raw id.
        id: u32,
    },
    /// Schema exceeded the maximum number of variables (`u32::MAX`).
    #[error("schema exceeds maximum variable count")]
    TooManyVariables,
    /// A categorical / ordinal variable lacked a category domain.
    #[error("variable `{name}` requires a category domain")]
    MissingCategoryDomain {
        /// Variable name that required a domain.
        name: String,
    },
    /// A non-categorical variable was given a category domain.
    #[error("variable `{name}` must not have a category domain")]
    UnexpectedCategoryDomain {
        /// Variable name that must not carry a domain.
        name: String,
    },
}
