//! Shared checks for observation weights used by weighted fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Whether every weight is finite and non-negative; zero, including negative zero, is allowed.
pub(crate) fn finite_nonnegative(weights: &[f64]) -> bool {
    weights.iter().all(|weight| weight.is_finite() && *weight >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_weights_preserve_fit_specific_refusal_and_boundaries() {
        assert!(finite_nonnegative(&[]));
        assert!(finite_nonnegative(&[-0.0, 0.0, f64::MAX]));
        assert!(!finite_nonnegative(&[f64::NAN]));
        assert!(!finite_nonnegative(&[f64::INFINITY]));
        assert!(!finite_nonnegative(&[-f64::MIN_POSITIVE]));
    }
}
