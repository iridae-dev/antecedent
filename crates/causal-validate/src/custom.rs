//! Object-safe custom effect validators.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::ExecutionContext;

use crate::common::{RefutationProblem, RefutationReport};
use crate::error::ValidationError;

/// Slow-path custom effect validator (Python / user callbacks).
///
/// Distinct from [`crate::Validator`] which uses associated types and is not dyn-safe.
pub trait CustomEffectValidator: Send + Sync {
    /// Stable name written into [`RefutationReport::refuter`].
    fn name(&self) -> &str;

    /// Run the check against a prepared refutation problem.
    ///
    /// # Errors
    ///
    /// Validation / applicability failures.
    fn validate(
        &self,
        problem: &RefutationProblem<'_>,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError>;
}
