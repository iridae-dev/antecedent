//! Time-uniform confidence sequence for the mean of a `[0, 1]`-bounded arm.
//!
//! Let `X_1, X_2, ...` be the outcomes of one arm in the order that arm was
//! pulled, independent with common mean `mu` and bounded in `[0, 1]`. By
//! Hoeffding's lemma `X_t - mu` is sub-Gaussian with variance proxy
//! `sigma^2 = 1/4`. With `S_m = sum_{t <= m} (X_t - mu)`, the normal mixture
//! over `lambda ~ N(0, tau^2)` of `exp(lambda S_m - lambda^2 m sigma^2 / 2)`
//! is a nonnegative supermartingale with initial value one (Robbins 1970;
//! Howard, Ramdas, `McAuliffe` and Sekhon 2021, normal mixture boundary), so by
//! Ville's inequality, with probability at least `1 - alpha`, simultaneously
//! for all `m >= 1`,
//!
//! ```text
//! |S_m| < sqrt( (1 + m sigma^2 tau^2) / tau^2 * ln( (1 + m sigma^2 tau^2) / alpha^2 ) ).
//! ```
//!
//! Parameterizing `tau^2 = 1 / (sigma^2 n0)` makes `n0` the sample size at
//! which the boundary is tightest and gives the radius of the mean
//! `sigma sqrt((m + n0) ln((m + n0) / (n0 alpha^2))) / m`. The sequence is the
//! running intersection of those intervals with `[0, 1]`.
//!
//! The guarantee is a finite-time theorem under the stated premise (the
//! observations of the arm, in pull order, are independent with a common mean:
//! the "stack of rewards" model of a stochastic bandit, with outcome delays
//! that do not depend on the outcomes). It is *not* a measured coverage
//! record of any whole procedure, and the callers label the sequence
//! uncalibrated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AdaptiveInferenceError as Error;

/// Most observations one sequence accepts.
pub const MAX_SEQUENCE_OBSERVATIONS: usize = 50_000_000;

/// Sub-Gaussian standard deviation proxy of a `[0, 1]` variable (Hoeffding).
const SUB_GAUSSIAN_SD: f64 = 0.5;
const CANCEL_STRIDE: usize = 4096;

/// The confidence sequence of one arm at its final sample size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmSequence {
    /// Observations of the arm.
    pub count: u64,
    /// Sample mean; `NaN` when the arm has no observation, in which case the
    /// sequence is the whole unit interval.
    pub mean: f64,
    /// Lower end of the running intersection.
    pub lower: f64,
    /// Upper end of the running intersection.
    pub upper: f64,
}

fn check_level(alpha: f64, tuning_samples: f64) -> Result<(), Error> {
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter("the miscoverage level must lie in (0, 1)"));
    }
    if !(tuning_samples.is_finite() && tuning_samples > 0.0) {
        return Err(Error::InvalidParameter("the tuning sample size must be finite and positive"));
    }
    Ok(())
}

/// Radius of the mixture-Hoeffding interval for the mean after `count`
/// observations, at two-sided miscoverage `alpha`, tuned for `tuning_samples`.
///
/// # Errors
/// `count` of zero, `alpha` outside `(0, 1)`, or a non-positive tuning size.
pub fn mixture_hoeffding_radius(count: u64, alpha: f64, tuning_samples: f64) -> Result<f64, Error> {
    check_level(alpha, tuning_samples)?;
    if count == 0 {
        return Err(Error::Shape("a radius needs at least one observation"));
    }
    let m = count as f64;
    let total = m + tuning_samples;
    let boundary =
        SUB_GAUSSIAN_SD * (total * (total / (tuning_samples * alpha * alpha)).ln()).sqrt();
    Ok(boundary / m)
}

/// The running-intersection confidence sequence of a bounded arm, evaluated
/// at the final observation, from outcomes in pull order.
///
/// # Errors
/// An outcome outside `[0, 1]` or not finite, a bad level, more than
/// [`MAX_SEQUENCE_OBSERVATIONS`] observations, cancellation, or an empty
/// intersection (evidence that the bounded independent premise failed).
pub fn bounded_mean_confidence_sequence(
    values: &[f64],
    alpha: f64,
    tuning_samples: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<ArmSequence, Error> {
    check_level(alpha, tuning_samples)?;
    if values.len() > MAX_SEQUENCE_OBSERVATIONS {
        return Err(Error::Limit("more than fifty million observations"));
    }
    if values.iter().any(|x| !(x.is_finite() && (0.0..=1.0).contains(x))) {
        return Err(Error::InvalidParameter("a bounded outcome must lie in [0, 1]"));
    }
    if values.is_empty() {
        return Ok(ArmSequence { count: 0, mean: f64::NAN, lower: 0.0, upper: 1.0 });
    }
    let (mut lower, mut upper, mut sum) = (0.0_f64, 1.0_f64, 0.0_f64);
    for (index, x) in values.iter().enumerate() {
        if index % CANCEL_STRIDE == 0 && cancelled() {
            return Err(Error::Cancelled);
        }
        sum += x;
        let count = index as u64 + 1;
        let mean = sum / count as f64;
        let radius = mixture_hoeffding_radius(count, alpha, tuning_samples)?;
        lower = lower.max(mean - radius);
        upper = upper.min(mean + radius);
    }
    if lower > upper {
        return Err(Error::Numerical(
            "the intersected confidence sequence is empty: the bounded independent premise failed",
        ));
    }
    let count = values.len() as u64;
    Ok(ArmSequence { count, mean: sum / count as f64, lower, upper })
}

/// Bounds for the difference of two arm means, `a - b`, from two sequences
/// each constructed at `alpha / 2` (a union bound over the two arms).
#[must_use]
pub fn contrast_bounds(a: &ArmSequence, b: &ArmSequence) -> (f64, f64) {
    (a.lower - b.upper, a.upper - b.lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_matches_the_closed_form() {
        // sigma = 1/2, n0 = 100, alpha = 0.05, m = 100:
        // 0.5 sqrt(200 ln(200 / (100 * 0.0025))) / 100.
        let r = mixture_hoeffding_radius(100, 0.05, 100.0).unwrap();
        let expected = 0.5 * (200.0_f64 * (800.0_f64).ln()).sqrt() / 100.0;
        assert!((r - expected).abs() < 1e-15);
        assert!((r * 100.0 - 18.281_974_356_819_244).abs() < 1e-9);
    }

    #[test]
    fn a_value_outside_the_unit_interval_is_refused() {
        let outcome = bounded_mean_confidence_sequence(&[0.2, 1.2], 0.05, 10.0, &|| false);
        assert!(matches!(outcome, Err(Error::InvalidParameter(_))));
    }
}
