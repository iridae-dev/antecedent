//! Shared validation for temporal link and propensity designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::EstimationError;

pub(crate) fn positive_finite(value: f64, message: &str) -> Result<(), EstimationError> {
    if value.is_finite() && value > 0.0 { Ok(()) } else { Err(EstimationError::data_msg(message)) }
}

#[allow(clippy::float_cmp, reason = "the intercept column is coded exactly one")]
pub(crate) fn column_is_ones(column: &[f64]) -> bool {
    column.iter().all(|&value| value == 1.0)
}

#[allow(clippy::float_cmp, reason = "binary outcomes are coded exactly zero or one")]
pub(crate) fn binary_outcomes(y: &[f64], message: &str) -> Result<(), EstimationError> {
    if y.iter().all(|&value| value == 0.0 || value == 1.0) {
        Ok(())
    } else {
        Err(EstimationError::data_msg(message))
    }
}

pub(crate) fn finite_column(name: &str, values: &[f64]) -> Result<(), EstimationError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(EstimationError::data_msg(format!("{name} values must be finite")))
    }
}
