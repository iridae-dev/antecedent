//! The E-value against its defining minimum and the standardized-difference conversion.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{e_value_from_risk_ratio, smd_to_risk_ratio};

/// The E-value is defined by a minimum: the smallest `E` such that confounder associations
/// `RR_EU = RR_UD = E` give the Ding-VanderWeele bound `RR_EU RR_UD / (RR_EU + RR_UD - 1) = RR`.
/// With equal strengths the bound is `E^2 / (2E - 1)`, so the closed form must invert it
/// exactly, and any weaker equal-strength confounder must leave the bound below `RR`.
#[test]
fn evalue_inverts_the_confounding_bound_and_is_its_smallest_equal_strength_root() {
    let bound = |e: f64| e / (2.0 - 1.0 / e);
    for &rr in &[1.0 + 1e-12, 1.000_001, 1.05, 1.5, 2.0, 3.7, 10.0, 123.456, 1e6, 1e30, 1e100] {
        let e = e_value_from_risk_ratio(rr);
        assert!(e >= 1.0);
        assert!((bound(e) / rr - 1.0).abs() < 4e-15, "rr={rr}: bound {} e {e}", bound(e));
        // Smallest root: a slightly weaker confounder cannot reach rr (bound is increasing).
        if rr > 1.001 {
            assert!(bound(1.0 + (e - 1.0) * (1.0 - 1e-6)) < rr, "rr={rr}");
        }
        // Protective effects use the reciprocal.
        assert!((e_value_from_risk_ratio(1.0 / rr) - e).abs() <= 4e-15 * e, "rr={rr}");
    }
    assert_eq!(e_value_from_risk_ratio(1.0), 1.0);
    assert!(e_value_from_risk_ratio(f64::INFINITY).is_infinite());
    assert!(e_value_from_risk_ratio(0.0).is_infinite());
}

/// The conversion `RR = exp(0.91 d)` agrees with a multiprecision evaluation at the
/// largest standardized differences that stay finite, and overflows to an infinite E-value
/// (not a NaN) beyond them.
#[test]
fn standardized_difference_conversion_is_finite_then_infinite_never_nan() {
    let e = e_value_from_risk_ratio(smd_to_risk_ratio(5.0));
    // exp(4.55) = 94.63240...; E = RR + sqrt(RR (RR - 1)) = 188.8...
    let rr = (0.91_f64 * 5.0).exp();
    assert!((e - (rr + (rr * (rr - 1.0)).sqrt())).abs() < 1e-12 * e);
    let huge = e_value_from_risk_ratio(smd_to_risk_ratio(1.0e4));
    assert!(huge.is_infinite() && !huge.is_nan());
}
