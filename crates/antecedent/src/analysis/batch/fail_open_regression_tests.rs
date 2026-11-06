use super::*;

#[test]
fn a_nan_q_value_is_never_the_winner_and_order_does_not_matter() {
    assert_eq!(argmin_finite(&[f64::NAN, 0.3, 0.1, f64::NAN]), Some(2));
    assert_eq!(argmin_finite(&[0.1, f64::NAN, 0.3]), Some(0));
    assert_eq!(argmin_finite(&[f64::NAN, f64::INFINITY]), None);
}

#[test]
fn degenerate_se_is_undefined_not_a_discovery() {
    // `se == 0` with a nonzero point estimate is a broken/degenerate estimate
    // (e.g. a constant influence column), not infinite precision.
    assert!(two_sided_p(5.0, 0.0).is_nan());
    // `se == NaN` (e.g. `sqrt` of a negative covariance entry) is likewise undefined.
    assert!(two_sided_p(5.0, f64::NAN).is_nan());
    // a genuinely testable claim is unaffected.
    let p = two_sided_p(3.0, 1.0);
    assert!(p.is_finite() && p > 0.0 && p < 1.0);
}

#[test]
fn wald_p_value_is_the_two_sided_standard_normal_tail() {
    // Phi^-1(0.975) = 1.959963984540054 gives a two-sided tail of 0.05.
    assert!((two_sided_p(1.959_963_984_540_054, 1.0) - 0.05).abs() < 1e-12);
    // Sign of the estimate does not matter and the scale is value / se.
    assert!((two_sided_p(-3.0, 2.0) - two_sided_p(1.5, 1.0)).abs() < 1e-15);
    // Far tail stays positive and monotone instead of collapsing early.
    assert!(two_sided_p(10.0, 1.0) > 0.0 && two_sided_p(10.0, 1.0) < two_sided_p(9.0, 1.0));
}

#[test]
fn screen_row_bitmap_rejects_duplicates_and_out_of_range_in_one_pass() {
    use antecedent_data::GlobalId;
    let ids = |v: &[u64]| v.iter().map(|&i| GlobalId::new(i)).collect::<Vec<_>>();
    assert_eq!(screen_row_bitmap(&ids(&[0, 2, 9]), 10), Some(vec![0b0000_0101, 0b0000_0010]));
    assert_eq!(screen_row_bitmap(&ids(&[1, 1]), 10), None);
    assert_eq!(screen_row_bitmap(&ids(&[10]), 10), None);
}

#[test]
fn degenerate_se_is_excluded_from_bh_by_instead_of_ranked_first() {
    // One degenerate claim (se=0, nonzero point estimate) next to one ordinary
    // claim (se=1). Before the fix `two_sided_p(5.0, 0.0) == 0.0`, which BH/BY
    // treat as the strongest possible discovery (adjusted p = q = 0).
    let degenerate_p = two_sided_p(5.0, 0.0);
    let ordinary_p = two_sided_p(3.0, 1.0);
    let p_values = [degenerate_p, ordinary_p];
    let bh = antecedent_stats::benjamini_hochberg(&p_values);
    let by = antecedent_stats::benjamini_yekutieli(&p_values);
    assert!(bh[0].is_nan(), "degenerate claim must not receive an adjusted p-value, got {}", bh[0]);
    assert!(by[0].is_nan(), "degenerate claim must not receive an adjusted q-value, got {}", by[0]);
    // the ordinary claim's family size stays 2 (conservative fill of the
    // invalid test as p=1), it is not silently shrunk to 1.
    let expected_bh = (2.0 * ordinary_p).min(1.0);
    let expected_by = (3.0 * ordinary_p).min(1.0); // m=2 harmonic factor H_2 = 1.5
    assert!((bh[1] - expected_bh).abs() < 1e-12, "bh={} expected={}", bh[1], expected_bh);
    assert!((by[1] - expected_by).abs() < 1e-12, "by={} expected={}", by[1], expected_by);
}
