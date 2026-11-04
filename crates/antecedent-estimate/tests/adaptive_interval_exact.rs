//! The interval constructions behind the measured adaptive-inference cells against closed forms.
//!
//! Beta posteriors with an integer shape of one have an explicit quantile function, so the
//! equal-tailed credible interval is known without any numerical inverse: Beta(1, 1) is the
//! uniform law, Beta(2, 1) has CDF x^2 and Beta(1, 2) has CDF 1 - (1 - x)^2. The Wald interval
//! is the estimate plus and minus the standard normal quantile times the standard error, with
//! the quantiles 1.959963984540054 (95%) and 1.6448536269514722 (90%). The Welch difference of
//! two small arms is computed by hand.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::adaptive_measured_interval::{
    beta_credible_interval, repeated_confidence_interval, weighted_wald_interval, welch_difference,
};

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol, "{what}: got {got}, want {want}");
}

#[test]
fn the_uniform_posterior_interval_is_the_tail_probabilities() {
    let (lo, hi) = beta_credible_interval(1.0, 1.0, 0.95).expect("interval");
    close(lo, 0.025, 1e-12, "Beta(1,1) lower at 95%");
    close(hi, 0.975, 1e-12, "Beta(1,1) upper at 95%");
    let (lo, hi) = beta_credible_interval(1.0, 1.0, 0.9).expect("interval");
    close(lo, 0.05, 1e-12, "Beta(1,1) lower at 90%");
    close(hi, 0.95, 1e-12, "Beta(1,1) upper at 90%");
}

#[test]
fn a_one_shape_posterior_interval_is_the_explicit_quantile() {
    // Beta(2, 1): F(x) = x^2, so the p-quantile is sqrt(p).
    let (lo, hi) = beta_credible_interval(2.0, 1.0, 0.95).expect("interval");
    close(lo, 0.025_f64.sqrt(), 1e-9, "Beta(2,1) lower");
    close(hi, 0.975_f64.sqrt(), 1e-9, "Beta(2,1) upper");
    // Beta(1, 2): F(x) = 1 - (1 - x)^2, so the p-quantile is 1 - sqrt(1 - p).
    let (lo, hi) = beta_credible_interval(1.0, 2.0, 0.95).expect("interval");
    close(lo, 1.0 - 0.975_f64.sqrt(), 1e-9, "Beta(1,2) lower");
    close(hi, 1.0 - 0.025_f64.sqrt(), 1e-9, "Beta(1,2) upper");
}

#[test]
fn a_degenerate_or_unsupported_request_has_no_interval() {
    assert!(beta_credible_interval(0.0, 1.0, 0.95).is_none());
    assert!(beta_credible_interval(1.0, 1.0, 1.0).is_none());
    assert!(beta_credible_interval(1.0, 1.0, 0.0).is_none());
    assert!(weighted_wald_interval(0.3, -0.01, 0.95).is_none());
    assert!(weighted_wald_interval(f64::NAN, 0.04, 0.95).is_none());
    assert!(weighted_wald_interval(0.3, 0.04, 1.0).is_none());
}

#[test]
fn the_wald_interval_is_the_estimate_plus_and_minus_z_standard_errors() {
    // Variance 0.04 is a standard error of 0.2.
    let (lo, hi) = weighted_wald_interval(0.3, 0.04, 0.95).expect("interval");
    close(lo, 0.3 - 1.959_963_984_540_054 * 0.2, 1e-7, "95% lower");
    close(hi, 0.3 + 1.959_963_984_540_054 * 0.2, 1e-7, "95% upper");
    let (lo, hi) = weighted_wald_interval(-1.0, 0.25, 0.9).expect("interval");
    close(lo, -1.0 - 1.644_853_626_951_472_2 * 0.5, 1e-7, "90% lower");
    close(hi, -1.0 + 1.644_853_626_951_472_2 * 0.5, 1e-7, "90% upper");
    // A zero variance is the point.
    assert_eq!(weighted_wald_interval(0.7, 0.0, 0.95), Some((0.7, 0.7)));
}

#[test]
fn the_repeated_interval_is_the_look_boundary_times_the_standard_error() {
    assert_eq!(repeated_confidence_interval(1.0, 0.5, 2.0), Some((0.0, 2.0)));
    assert!(repeated_confidence_interval(1.0, 0.0, 2.0).is_none());
    assert!(repeated_confidence_interval(1.0, 0.5, -2.0).is_none());
}

#[test]
fn the_welch_difference_of_two_small_arms_is_the_hand_value() {
    // Control 1, 2, 3: mean 2, variance 1, variance of the mean 1/3.
    // Treatment 2, 4, 6: mean 4, variance 4, variance of the mean 4/3.
    let (difference, se) =
        welch_difference(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).expect("difference");
    close(difference, 2.0, 1e-15, "difference");
    close(se, (5.0_f64 / 3.0).sqrt(), 1e-15, "standard error");
    assert!(welch_difference(&[1.0], &[2.0, 4.0]).is_none());
    assert!(welch_difference(&[1.0, 1.0], &[2.0, 2.0]).is_none(), "no spread, no standard error");
}
