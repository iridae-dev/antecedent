//! Shared checks for observation weights used by weighted fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;

/// Whether every weight is finite and non-negative; zero, including negative zero, is allowed.
pub(crate) fn finite_nonnegative(weights: &[f64]) -> bool {
    weights.iter().all(|weight| weight.is_finite() && *weight >= 0.0)
}

pub(crate) fn validate_observation_weights(
    weights: Option<&[f64]>,
    nrows: usize,
    message: &'static str,
) -> Result<(), StatsError> {
    if let Some(w) = weights {
        let sum: f64 = w.iter().sum();
        if w.len() != nrows || !finite_nonnegative(w) || !sum.is_finite() || sum <= 0.0 {
            return Err(StatsError::Shape { message });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_weights_preserve_fit_specific_refusal_and_boundaries() {
        let check = |w, n, message| validate_observation_weights(w, n, message);
        assert!(finite_nonnegative(&[]));
        assert!(finite_nonnegative(&[-0.0, 0.0, f64::MAX]));
        assert!(!finite_nonnegative(&[f64::NAN]));
        assert!(!finite_nonnegative(&[f64::INFINITY]));
        assert!(!finite_nonnegative(&[-f64::MIN_POSITIVE]));
        assert!(check(None, 3, "gam").is_ok());
        assert!(check(Some(&[0.0, 1.0]), 2, "gam").is_ok());
        for weights in [&[0.0, 0.0][..], &[f64::NAN, 1.0], &[-1.0, 2.0], &[f64::MAX, f64::MAX]] {
            assert_eq!(check(Some(weights), 2, "gam"), Err(StatsError::Shape { message: "gam" }));
            assert_eq!(
                check(Some(weights), 2, "multinomial"),
                Err(StatsError::Shape { message: "multinomial" })
            );
        }
        assert_eq!(check(Some(&[1.0]), 2, "gam"), Err(StatsError::Shape { message: "gam" }));
    }
}
