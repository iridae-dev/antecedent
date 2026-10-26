//! Overlap-policy gates and structured refusals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Require [`OverlapPolicy::ExplicitOverride`] (linear / IV / RD / front-door / GLM paths).
pub(crate) fn require_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if overlap != OverlapPolicy::ExplicitOverride {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// Refuse [`OverlapPolicy::ExplicitOverride`] (propensity / AIPW paths — positivity mandatory).
pub(crate) fn refuse_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if matches!(overlap, OverlapPolicy::ExplicitOverride) {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// A structured refusal whose message is `"{detail}: {message}"`: `detail` is the namespaced
/// reason literal a caller keys on, `message` the human text.
pub(crate) fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}
