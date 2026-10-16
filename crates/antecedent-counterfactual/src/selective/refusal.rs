//! Typed scientific refusals of cross-world execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{ExternalRefusal, StructuredRefusal};
use thiserror::Error;

/// Stage reported by the refusals of this module.
pub const EXECUTE_STAGE: &str = "execute";

/// A structured refusal plus the prerequisites that would license the request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrossWorldRefusal {
    /// Registered code, stage, namespaced detail, offending subject and remedy.
    pub refusal: StructuredRefusal,
    /// The named prerequisites that are not met (empty when none applies).
    pub missing: Vec<String>,
}

/// Cross-world execution errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CrossWorldExecutionError {
    /// A typed refusal.
    #[error("cross-world execution refused: {}", .0.refusal.detail)]
    Refused(Box<CrossWorldRefusal>),
    /// Cooperative cancellation was observed at a bounded work boundary.
    #[error("cancelled during cross-world execution")]
    Cancelled,
}

impl CrossWorldExecutionError {
    /// The structured refusal, when the error is a refusal.
    #[must_use]
    pub fn refusal(&self) -> Option<&StructuredRefusal> {
        match self {
            Self::Refused(r) => Some(&r.refusal),
            Self::Cancelled => None,
        }
    }

    /// The unmet prerequisites a refusal names.
    #[must_use]
    pub fn missing(&self) -> &[String] {
        match self {
            Self::Refused(r) => &r.missing,
            Self::Cancelled => &[],
        }
    }
}

pub(crate) fn refuse(
    code: &'static str,
    detail: &str,
    offending: Option<String>,
    remedy: &'static str,
) -> CrossWorldExecutionError {
    refuse_missing(code, detail, offending, remedy, Vec::new())
}

pub(crate) fn refuse_missing(
    code: &'static str,
    detail: &str,
    offending: Option<String>,
    remedy: &'static str,
    missing: Vec<String>,
) -> CrossWorldExecutionError {
    CrossWorldExecutionError::Refused(Box::new(CrossWorldRefusal {
        refusal: StructuredRefusal {
            code,
            stage: EXECUTE_STAGE,
            detail: detail.to_owned(),
            offending,
            expected: None,
            supplied: None,
            capability: None,
            remedy: Some(remedy),
        },
        missing,
    }))
}

pub(crate) fn invalid(detail: &str, offending: impl Into<String>) -> CrossWorldExecutionError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        Some(offending.into()),
        "supply a well-formed declaration; the offending field is named",
    )
}

pub(crate) fn from_external(refusal: ExternalRefusal) -> CrossWorldExecutionError {
    CrossWorldExecutionError::Refused(Box::new(CrossWorldRefusal { refusal, missing: Vec::new() }))
}

pub(crate) fn poll(
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), CrossWorldExecutionError> {
    if ctx.cancellation.is_cancelled() { Err(CrossWorldExecutionError::Cancelled) } else { Ok(()) }
}
