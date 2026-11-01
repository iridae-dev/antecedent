//! Counterfactual evaluation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_model::ModelError;
use thiserror::Error;

/// Counterfactual errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CounterfactualError {
    /// Model / shape issue.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Missing factual values required for abduction.
    #[error("missing factual: {message}")]
    MissingFactual {
        /// Variable.
        message: String,
    },
    /// Nested interventions not allowed.
    #[error("nested counterfactuals not enabled")]
    NestedNotAllowed,
    /// Abduction is ill-posed or not exact inversion: the observed values do not
    /// determine each unit's exogenous terms (a deterministic mechanism given
    /// varying data, or a posterior or prior-drawn exogenous term). This is not a
    /// model-adequacy test: an invertible mechanism always regenerates the data it
    /// abduces from.
    #[error("abduction is not exact inversion: {message}")]
    AbductionNotExact {
        /// What failed to invert.
        message: String,
    },
    /// Cooperative cancellation was observed between evaluation steps.
    #[error("cancelled during cross-world evaluation")]
    Cancelled,
    /// Numerical failure.
    #[error("numerical error: {message}")]
    Numerical {
        /// Context.
        message: String,
    },
}

impl CounterfactualError {
    /// Ad-hoc model message.
    #[must_use]
    pub fn model_msg(message: impl Into<String>) -> Self {
        Self::Model(ModelError::Unsupported { message: message.into() })
    }
}

/// Admit `cells` `f64` cells (checked, 8 bytes each) from the context's pool before they are
/// allocated. Hold the lease for as long as the buffer lives.
///
/// # Errors
///
/// [`CounterfactualError::Cancelled`] when the context is cancelled; a
/// [`ModelError::ResourceLimit`] when the size overflows or the pool cannot admit it.
pub(crate) fn admit_cells(
    ctx: &antecedent_core::ExecutionContext,
    what: &'static str,
    cells: Option<usize>,
) -> Result<antecedent_core::execution::AdmissionLease, CounterfactualError> {
    use antecedent_core::execution::AdmissionError;
    let limit = ctx.memory_limit_bytes().unwrap_or(u64::MAX);
    let refused = |required_bytes| {
        CounterfactualError::Model(ModelError::ResourceLimit {
            message: what,
            required_bytes,
            limit_bytes: limit,
        })
    };
    let bytes = cells
        .and_then(|c| c.checked_mul(8))
        .and_then(|b| u64::try_from(b).ok())
        .ok_or_else(|| refused(u64::MAX))?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => CounterfactualError::Cancelled,
        _ => refused(bytes),
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use antecedent_core::ExecutionContext;

    #[test]
    fn cells_are_admitted_before_allocation_and_refusals_keep_their_class() {
        let ctx = ExecutionContext::for_tests(0).with_memory_limit(Some(80));
        assert!(admit_cells(&ctx, "test", Some(10)).is_ok());
        for cells in [Some(11), None, Some(usize::MAX)] {
            assert!(matches!(
                admit_cells(&ctx, "test", cells),
                Err(CounterfactualError::Model(ModelError::ResourceLimit { .. }))
            ));
        }
        let held = admit_cells(&ctx, "test", Some(10)).unwrap();
        assert!(admit_cells(&ctx, "test", Some(1)).is_err(), "the lease holds its bytes");
        drop(held);
        ctx.cancellation.cancel();
        assert_eq!(admit_cells(&ctx, "test", Some(1)).unwrap_err(), CounterfactualError::Cancelled);
    }
}
