//! Shared estimation helpers (Phase 4 SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use causal_stats::StatsError;

use crate::adjustment::OverlapPolicy;
use crate::error::EstimationError;

/// Map a stats-layer error into [`EstimationError::Stats`].
#[allow(clippy::needless_pass_by_value)] // StatsError is small / owned at call sites
pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::Stats(e.to_string())
}

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

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
pub(crate) fn sample_std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    var.sqrt()
}
