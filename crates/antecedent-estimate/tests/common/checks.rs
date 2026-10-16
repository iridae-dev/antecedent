//! Shared absolute comparisons and coded-refusal unpacking for estimate integration tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::EstimationError;

/// `|actual - expected| <= 1e-9`.
#[allow(dead_code, reason = "some integration targets use only the coded-refusal helper")]
pub fn close_abs_1e9(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

/// `(code, message)` of a [`EstimationError::Refused`].
///
/// # Panics
///
/// Panics when `error` is not a coded refusal.
pub fn coded_refusal(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}
