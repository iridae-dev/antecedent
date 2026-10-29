//! Population moments of aligned draw columns and the moment ranges of a declared correlation
//! range, against exact rational arithmetic
//! (`oracle_data/transported_composition_reference.py`).
//!
//! These are the two numeric routes of the joint-law module that no other joint-law record covers;
//! the linear, mixture, envelope, aligned-functional and interval routes are reviewed with their
//! own records. The paired columns include offsets of 2^30 (where a one-pass variance formula
//! loses every digit), a single row, constant and perfectly anti-correlated columns. The ranges
//! include negative and zero weights, a zero standard deviation, ranges touching -1 and 1, means of
//! size 2^20 and a range one part in 1024 wide. The refusals of the range (reversed or oversized
//! range, a negative or non-finite moment) are executed as well.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{MomentSummary, correlation_range_envelope, paired_moments};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/transported_composition_reference.json"))
        .expect("frozen reference")
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn fs(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(f).collect()
}

fn near(got: f64, want: f64, scale: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * (scale + want.abs()), "{what}: {got} vs {want}");
}

#[test]
fn paired_moments_match_the_exact_definitions() {
    for case in reference()["paired"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let (x, y) = (fs(&case["x"]), fs(&case["y"]));
        let got = paired_moments(&x, &y).expect("moments");
        assert_eq!(got.n, x.len());
        near(got.mean_x, f(&case["mean_x"]), 1.0, name);
        near(got.mean_y, f(&case["mean_y"]), 1.0, name);
        near(got.var_x, f(&case["var_x"]), 1.0, name);
        near(got.var_y, f(&case["var_y"]), 1.0, name);
        near(got.covariance, f(&case["covariance"]), 1.0, name);
    }
    assert!(format!("{:?}", paired_moments(&[], &[]).err()).contains("a draw column needs at least one row"), "a draw column needs at least one row");
    assert!(format!("{:?}", paired_moments(&[1.0], &[1.0, 2.0]).err()).contains("aligned columns differ in length"), "aligned columns differ in length");
    assert!(format!("{:?}", paired_moments(&[1.0, f64::NAN], &[1.0, 2.0]).err()).contains("NonFinite"), "NonFinite");
}

#[test]
fn correlation_ranges_match_the_exact_endpoint_values() {
    for case in reference()["ranges"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let x = MomentSummary { mean: f(&case["x_mean"]), standard_deviation: f(&case["x_sd"]) };
        let y = MomentSummary { mean: f(&case["y_mean"]), standard_deviation: f(&case["y_sd"]) };
        let got = correlation_range_envelope(
            x,
            y,
            f(&case["a"]),
            f(&case["b"]),
            f(&case["lo"]),
            f(&case["hi"]),
        )
        .expect("a range");
        for (label, bounds, want) in [
            ("product moment", got.product_moment, &case["product_moment"]),
            ("covariance", got.covariance, &case["covariance"]),
            ("linear variance", got.linear_variance, &case["linear_variance"]),
        ] {
            let w = fs(want);
            near(bounds.lower, w[0], 1.0, &format!("{name} {label} lower"));
            near(bounds.upper, w[1], 1.0, &format!("{name} {label} upper"));
        }
    }
    let m = MomentSummary { mean: 0.0, standard_deviation: 1.0 };
    assert!(format!("{:?}", correlation_range_envelope(m, m, 1.0, 1.0, 0.5, 0.5).err()).contains("a single correlation is a covariance declaration"), "a single correlation");
    assert!(format!("{:?}", correlation_range_envelope(m, m, 1.0, 1.0, 0.6, 0.5).err()).contains("a single correlation is a covariance declaration"), "a reversed range");
    assert!(format!("{:?}", correlation_range_envelope(m, m, 1.0, 1.0, -1.5, 0.5).err()).contains("a single correlation is a covariance declaration"), "a single correlation is a covariance declaration");
    assert!(format!("{:?}", correlation_range_envelope(m, m, 1.0, 1.0, 0.0, 1.5).err()).contains("a single correlation is a covariance declaration"), "a single correlation is a covariance declaration");
    let negative = MomentSummary { mean: 0.0, standard_deviation: -1.0 };
    assert!(format!("{:?}", correlation_range_envelope(negative, m, 1.0, 1.0, 0.0, 0.5).err()).contains("standard deviation is negative"), "standard deviation is negative");
    let nan = MomentSummary { mean: f64::NAN, standard_deviation: 1.0 };
    assert!(format!("{:?}", correlation_range_envelope(nan, m, 1.0, 1.0, 0.0, 0.5).err()).contains("NonFinite"), "NonFinite");
}
